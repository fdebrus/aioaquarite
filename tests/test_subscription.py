"""Tests for ResilientPoolSubscription."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import pytest

from aioaquarite.exceptions import AquariteError
from aioaquarite.subscription import ResilientPoolSubscription


class _FakeWatch:
    def __init__(self) -> None:
        self.unsubscribed = False

    def unsubscribe(self) -> None:
        self.unsubscribed = True


class _FakeAuth:
    def __init__(self) -> None:
        self.expiring = False
        self.refresh_on_next = False
        self.raise_next: Exception | None = None
        self.get_client_calls = 0

    def is_token_expiring(self) -> bool:
        return self.expiring

    def calculate_sleep_duration(self) -> float:
        return 0.005

    async def get_client(self) -> tuple[object, bool]:
        self.get_client_calls += 1
        if self.raise_next is not None:
            err = self.raise_next
            self.raise_next = None
            raise err
        refreshed = self.refresh_on_next
        self.refresh_on_next = False
        return object(), refreshed


class _FakeClient:
    def __init__(self) -> None:
        self.auth = _FakeAuth()
        self.watches: list[_FakeWatch] = []
        self.callbacks: list[Callable[[dict[str, Any]], None]] = []
        self.subscribe_event = asyncio.Event()

    async def subscribe_pool(
        self, pool_id: str, callback: Callable[[dict[str, Any]], None]
    ) -> _FakeWatch:
        watch = _FakeWatch()
        self.watches.append(watch)
        self.callbacks.append(callback)
        self.subscribe_event.set()
        return watch


async def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


def _fast_sub(
    client: _FakeClient,
    callback: Callable[[dict[str, Any]], None],
) -> ResilientPoolSubscription:
    return ResilientPoolSubscription(
        client,  # type: ignore[arg-type]
        "pool1",
        callback,
        initial_backoff=0.005,
        max_backoff=0.05,
        health_check_interval=0.005,
    )


def test_token_rotation_triggers_resubscribe() -> None:
    async def _run() -> None:
        client = _FakeClient()
        received: list[dict[str, Any]] = []
        sub = _fast_sub(client, received.append)
        await sub._start()
        assert len(client.watches) == 1
        client.callbacks[0]({"v": 1})

        client.auth.refresh_on_next = True
        await _wait_for(lambda: len(client.watches) >= 2)

        assert client.watches[0].unsubscribed
        client.callbacks[-1]({"v": 2})
        assert received == [{"v": 1}, {"v": 2}]

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
            # Let any stray scheduled callbacks run.
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
        # Never call _start; aclose must not raise.
        await sub.aclose()
        assert client.watches == []

    asyncio.run(_run())


def test_low_level_subscribe_pool_unchanged() -> None:
    # The 0.3.x surface must still be available.
    from aioaquarite import AquariteClient

    assert hasattr(AquariteClient, "subscribe_pool")
    assert hasattr(AquariteClient, "subscribe_pool_resilient")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
