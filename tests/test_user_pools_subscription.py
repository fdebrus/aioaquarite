"""Tests for the users/{uid}.pools subscription.

Covers both the low-level ``AquariteClient.subscribe_user_pools`` fan-out
and the resilient supervisor wrapper.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from aioaquarite.auth import AquariteAuth
from aioaquarite.client import AquariteClient
from aioaquarite.exceptions import AquariteError
from aioaquarite.subscription import ResilientUserPoolsSubscription


# ── low-level subscribe_user_pools ────────────────────────────────────────


async def _wait_for(
    predicate: "Callable[[], bool]", timeout: float = 2.0
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


from ._fakes import (
    HOLD,
    FakeAsyncClient,
    FakeGapic,
    doc_change,
    doc_path,
    target_current,
    target_no_change,
)


def _make_authenticated_auth(fs_client: FakeAsyncClient) -> AquariteAuth:
    """Build an AquariteAuth that hands back the fake async client.

    No real network or sign-in; ``get_async_client`` is patched directly.
    """
    auth = AquariteAuth(MagicMock(), "user@example.com", "hunter2")
    auth.tokens = {
        "idToken": "fake-id-token",
        "refreshToken": "rt",
        "expiresIn": "3600",
        "localId": "uid-abc",
    }

    async def _get_async_client() -> FakeAsyncClient:
        return fs_client

    auth.get_async_client = _get_async_client  # type: ignore[method-assign]
    return auth


def test_subscribe_user_pools_invokes_callback_with_pool_ids() -> None:
    async def _run() -> None:
        gapic = FakeGapic()
        fs_client = FakeAsyncClient(gapic)
        user_doc = doc_path(fs_client, "users", "uid-abc")
        gapic.scripts = [
            [
                doc_change(
                    {"pools": ["pool-a", "pool-b", "pool-c"]}, name=user_doc
                ),
                target_current(),
                HOLD,
            ]
        ]
        client = AquariteClient(_make_authenticated_auth(fs_client))

        received: list[list[str]] = []
        watch = await client.subscribe_user_pools(received.append)
        assert watch is not None

        # Right document was targeted, on the right database.
        assert fs_client.document_calls == [("users", "uid-abc")]
        request = gapic.captured_requests[0]
        assert request.database == fs_client._database_string
        assert list(request.add_target.documents.documents) == [user_doc]

        assert received == [["pool-a", "pool-b", "pool-c"]]
        await watch.aclose()

    asyncio.run(_run())


def test_subscribe_user_pools_empty_pools_list_yields_empty_list() -> None:
    async def _run() -> None:
        gapic = FakeGapic()
        fs_client = FakeAsyncClient(gapic)
        user_doc = doc_path(fs_client, "users", "uid-abc")
        gapic.scripts = [
            [
                # Document with no 'pools' key at all.
                doc_change({"name": "someone"}, name=user_doc),
                target_current(),
                # Document with an explicit empty list.
                doc_change({"pools": []}, name=user_doc),
                target_no_change(resume_token=b"t1"),
                HOLD,
            ]
        ]
        client = AquariteClient(_make_authenticated_auth(fs_client))

        received: list[list[str]] = []
        watch = await client.subscribe_user_pools(received.append)

        await _wait_for(lambda: len(received) >= 2)
        assert received == [[], []]
        await watch.aclose()

    asyncio.run(_run())


def test_subscribe_user_pools_fans_out_across_multiple_snapshots() -> None:
    async def _run() -> None:
        gapic = FakeGapic()
        fs_client = FakeAsyncClient(gapic)
        user_doc = doc_path(fs_client, "users", "uid-abc")
        gapic.scripts = [
            [
                doc_change({"pools": ["a"]}, name=user_doc),
                target_current(),
                doc_change({"pools": ["a", "b"]}, name=user_doc),  # pool added
                target_no_change(resume_token=b"t1"),
                doc_change({"pools": ["b"]}, name=user_doc),  # pool removed
                target_no_change(resume_token=b"t2"),
                HOLD,
            ]
        ]
        client = AquariteClient(_make_authenticated_auth(fs_client))

        received: list[list[str]] = []
        watch = await client.subscribe_user_pools(received.append)

        await _wait_for(lambda: len(received) >= 3)
        assert received == [["a"], ["a", "b"], ["b"]]
        await watch.aclose()

    asyncio.run(_run())


# ── ResilientUserPoolsSubscription ────────────────────────────────────────
#
# These mirror tests/test_subscription.py; the supervisor is now shared in
# the base class so the same correctness properties must hold for the
# user-pools variant.


from ._fakes import FakeTaskWatch as _FakeWatch


class _FakeAuth:
    """Mirrors AquariteAuth: one rotation bumps a counter every caller reads."""

    def __init__(self) -> None:
        self.expiring = False
        self.refresh_on_next = False
        self.raise_next: Exception | None = None
        self.get_client_calls = 0
        self.token_generation = 0

    def is_token_expiring(self) -> bool:
        return self.expiring

    def calculate_sleep_duration(self) -> float:
        return 0.005

    async def get_client(self) -> tuple[object, int]:
        self.get_client_calls += 1
        if self.raise_next is not None:
            err = self.raise_next
            self.raise_next = None
            raise err
        # Only the first caller after a rotation performs it, exactly as
        # _ensure_fresh_clients does under its lock. Everyone still sees the
        # new generation.
        if self.refresh_on_next:
            self.refresh_on_next = False
            self.token_generation += 1
        return object(), self.token_generation


class _FakeClient:
    def __init__(self) -> None:
        self.auth = _FakeAuth()
        self.watches: list[_FakeWatch] = []
        self.callbacks: list[Callable[[list[str]], None]] = []
        self.subscribe_event = asyncio.Event()

    async def subscribe_user_pools(
        self, callback: Callable[[list[str]], None]
    ) -> _FakeWatch:
        watch = _FakeWatch()
        self.watches.append(watch)
        self.callbacks.append(callback)
        self.subscribe_event.set()
        return watch


def _fast_sub(
    client: _FakeClient,
    callback: Callable[[list[str]], None],
) -> ResilientUserPoolsSubscription:
    return ResilientUserPoolsSubscription(
        client,  # type: ignore[arg-type]
        callback,
        initial_backoff=0.005,
        max_backoff=0.05,
        health_check_interval=0.005,
    )


def test_token_rotation_triggers_resubscribe() -> None:
    async def _run() -> None:
        client = _FakeClient()
        received: list[list[str]] = []
        sub = _fast_sub(client, received.append)
        await sub._start()
        assert len(client.watches) == 1
        client.callbacks[0](["pool-a"])

        client.auth.refresh_on_next = True
        await _wait_for(lambda: len(client.watches) >= 2)

        assert client.watches[0].unsubscribed
        client.callbacks[-1](["pool-a", "pool-b"])
        assert received == [["pool-a"], ["pool-a", "pool-b"]]

        await sub.aclose()

    asyncio.run(_run())


def test_transient_error_triggers_reconnect() -> None:
    async def _run() -> None:
        client = _FakeClient()
        sub = _fast_sub(client, lambda _data: None)
        await sub._start()
        assert len(client.watches) == 1

        client.auth.raise_next = AquariteError("network blip")
        await _wait_for(lambda: len(client.watches) >= 2)

        assert sub._task is not None and not sub._task.done()
        await sub.aclose()

    asyncio.run(_run())


def test_aclose_is_idempotent_and_clean() -> None:
    caplog_records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            caplog_records.append(record)

    handler = _Handler(level=logging.WARNING)
    logging.getLogger("aioaquarite.subscription").addHandler(handler)
    try:

        async def _run() -> None:
            client = _FakeClient()
            sub = _fast_sub(client, lambda _data: None)
            await sub._start()
            await sub.aclose()
            await sub.aclose()  # second close: no-op
            assert client.watches[0].unsubscribed
            await asyncio.sleep(0.02)

        asyncio.run(_run())
    finally:
        logging.getLogger("aioaquarite.subscription").removeHandler(handler)

    warnings_or_worse = [r for r in caplog_records if r.levelno >= logging.WARNING]
    assert warnings_or_worse == [], (
        "aclose should not emit warnings/errors: "
        f"{[r.getMessage() for r in warnings_or_worse]}"
    )


def test_aclose_before_start_is_safe() -> None:
    async def _run() -> None:
        client = _FakeClient()
        sub = _fast_sub(client, lambda _data: None)
        await sub.aclose()
        assert client.watches == []

    asyncio.run(_run())


def test_public_surface_exports_user_pools_helpers() -> None:
    from aioaquarite import AquariteClient, ResilientUserPoolsSubscription

    assert hasattr(AquariteClient, "subscribe_user_pools")
    assert hasattr(AquariteClient, "subscribe_user_pools_resilient")
    assert ResilientUserPoolsSubscription is not None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
