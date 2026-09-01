"""Protocol-level tests for the native async Firestore watch.

These drive :class:`AsyncDocumentWatch` against a fake GAPIC ``listen()``
replaying real ``ListenResponse`` protos, so the single-document assembly
(coalescing, consistency points, resume tokens, RESET/REMOVE handling,
shutdown) is exercised exactly as the wire protocol presents it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from aioaquarite._watch import AsyncDocumentWatch
from aioaquarite.client import AquariteClient
from aioaquarite.exceptions import ConnectionError

from ._fakes import (
    HOLD,
    FakeAsyncClient,
    FakeGapic,
    doc_change,
    target_current,
    target_no_change,
    target_remove,
    target_reset,
)

POOL_ID = "pool-1"
DOC_PATH = "projects/test-proj/databases/(default)/documents/pools/pool-1"

SAMPLE = {
    "main": {"temperature": 25.5, "hasPH": True, "version": 825},
    "light": {"status": 1},
    "name": "Backyard",
}
SAMPLE_V2 = {"light": {"status": 0}, "name": "Backyard"}

FAKE_DOC = doc_change(SAMPLE)
FAKE_CURRENT = target_current()
FAKE_HOLD = HOLD


async def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


def _watch(
    scripts: list[list[Any]],
    callback: Callable[[dict[str, Any]], None],
    *,
    store: dict[str, bytes] | None = None,
) -> tuple[AsyncDocumentWatch, FakeGapic]:
    gapic = FakeGapic()
    gapic.scripts = scripts
    watch = AsyncDocumentWatch(
        FakeAsyncClient(gapic),  # type: ignore[arg-type]
        DOC_PATH,
        callback,
        resume_tokens=store if store is not None else {},
        label="pool test",
    )
    return watch, gapic


def _pool_client(scripts: list[list[Any]]) -> tuple[AquariteClient, FakeGapic]:
    """An AquariteClient whose auth serves the fake async Firestore client."""
    gapic = FakeGapic()
    gapic.scripts = scripts
    auth = MagicMock()
    auth.get_async_client = AsyncMock(return_value=FakeAsyncClient(gapic))
    auth.tokens = {"idToken": "t", "localId": "uid-abc"}
    return AquariteClient(auth), gapic


# ── emission semantics ─────────────────────────────────────────────────


def test_document_change_then_current_emits_decoded_dict() -> None:
    """One consistent snapshot, decoded exactly as to_dict() would."""

    async def _run() -> None:
        received: list[dict[str, Any]] = []
        watch, gapic = _watch([[FAKE_DOC, FAKE_CURRENT, HOLD]], received.append)
        await watch.start()

        assert received == [SAMPLE]
        # Types survive the proto round-trip, not just values.
        assert type(received[0]["main"]["temperature"]) is float
        assert type(received[0]["main"]["hasPH"]) is bool
        assert type(received[0]["main"]["version"]) is int

        request = gapic.captured_requests[0]
        assert request.database == "projects/test-proj/databases/(default)"
        assert list(request.add_target.documents.documents) == [DOC_PATH]
        assert request.add_target.resume_token == b""
        assert (
            "google-cloud-resource-prefix",
            "projects/test-proj/databases/(default)",
        ) in list(gapic.captured_metadata[0])

        await watch.aclose()

    asyncio.run(_run())


def test_changes_before_current_are_coalesced() -> None:
    """Nothing may be emitted before the server marks the target caught up."""

    async def _run() -> None:
        received: list[dict[str, Any]] = []
        watch, _ = _watch(
            [[doc_change(SAMPLE), doc_change(SAMPLE_V2), FAKE_CURRENT, HOLD]],
            received.append,
        )
        await watch.start()

        assert received == [SAMPLE_V2]
        await watch.aclose()

    asyncio.run(_run())


def test_change_after_current_waits_for_consistency_marker() -> None:
    """A post-CURRENT change is stashed until a NO_CHANGE delimits it."""

    async def _run() -> None:
        received: list[dict[str, Any]] = []
        watch, _ = _watch(
            [[FAKE_DOC, FAKE_CURRENT, doc_change(SAMPLE_V2), HOLD]],
            received.append,
        )
        await watch.start()
        await asyncio.sleep(0.02)  # let the stashed change sit

        assert received == [SAMPLE]  # v2 has no consistency point yet
        await watch.aclose()

    asyncio.run(_run())


def test_no_change_after_current_emits_pending() -> None:
    async def _run() -> None:
        received: list[dict[str, Any]] = []
        watch, _ = _watch(
            [
                [
                    FAKE_DOC,
                    FAKE_CURRENT,
                    doc_change(SAMPLE_V2),
                    target_no_change(resume_token=b"tok-1"),
                    HOLD,
                ]
            ],
            received.append,
        )
        await watch.start()
        await _wait_for(lambda: len(received) >= 2)

        assert received == [SAMPLE, SAMPLE_V2]
        await watch.aclose()

    asyncio.run(_run())


def test_foreign_target_ids_are_ignored() -> None:
    async def _run() -> None:
        received: list[dict[str, Any]] = []
        watch, _ = _watch(
            [
                [
                    doc_change(SAMPLE_V2, target_ids=(999,)),  # not ours
                    FAKE_DOC,
                    FAKE_CURRENT,
                    HOLD,
                ]
            ],
            received.append,
        )
        await watch.start()

        assert received == [SAMPLE]
        await watch.aclose()

    asyncio.run(_run())


# ── resume tokens ──────────────────────────────────────────────────────


def test_resume_token_captured_and_sent_on_reconnect() -> None:
    async def _run() -> None:
        store: dict[str, bytes] = {}
        watch1, gapic1 = _watch(
            [
                [
                    FAKE_DOC,
                    FAKE_CURRENT,
                    target_no_change(resume_token=b"tok-1"),
                    HOLD,
                ]
            ],
            lambda _d: None,
            store=store,
        )
        await watch1.start()
        await _wait_for(lambda: store.get(DOC_PATH) == b"tok-1")
        await watch1.aclose()

        # The next watch of the same document resumes where this one stopped.
        watch2, gapic2 = _watch(
            [[FAKE_CURRENT, HOLD]], lambda _d: None, store=store
        )
        await watch2.start()
        assert gapic2.captured_requests[0].add_target.resume_token == b"tok-1"
        await watch2.aclose()

    asyncio.run(_run())


def test_reset_drops_state_and_resume_token() -> None:
    """After RESET nothing stashed survives and the token is invalid."""

    async def _run() -> None:
        store: dict[str, bytes] = {}
        received: list[dict[str, Any]] = []
        watch, _ = _watch(
            [
                [
                    FAKE_DOC,
                    FAKE_CURRENT,
                    target_no_change(resume_token=b"tok-1"),
                    target_reset(),
                    doc_change(SAMPLE_V2),
                    FAKE_CURRENT,
                    HOLD,
                ]
            ],
            received.append,
            store=store,
        )
        await watch.start()
        await _wait_for(lambda: len(received) >= 2)

        # The world was re-sent after the reset and emitted at the new
        # CURRENT; the resume token from before the reset is gone, so a
        # reconnect would re-add the target without one.
        assert received == [SAMPLE, SAMPLE_V2]
        assert store == {}
        await watch.aclose()

        watch2, gapic2 = _watch(
            [[FAKE_CURRENT, HOLD]], lambda _d: None, store=store
        )
        await watch2.start()
        assert gapic2.captured_requests[0].add_target.resume_token == b""
        await watch2.aclose()

    asyncio.run(_run())


# ── failure model ──────────────────────────────────────────────────────


def test_stream_end_finishes_task_with_error() -> None:
    """A listen stream ending is a failure the supervisor must see."""

    async def _run() -> None:
        watch, _ = _watch([[FAKE_DOC, FAKE_CURRENT]], lambda _d: None)
        await watch.start()
        await _wait_for(lambda: watch.done)

        assert isinstance(watch.task.exception(), ConnectionError)

    asyncio.run(_run())


def test_stream_exception_surfaces_on_the_task() -> None:
    async def _run() -> None:
        boom = RuntimeError("gRPC channel closed")
        watch, _ = _watch([[FAKE_DOC, FAKE_CURRENT, boom]], lambda _d: None)
        await watch.start()
        await _wait_for(lambda: watch.done)

        assert watch.task.exception() is boom

    asyncio.run(_run())


def test_start_surfaces_a_stream_that_dies_before_current() -> None:
    async def _run() -> None:
        watch, _ = _watch([[RuntimeError("denied")]], lambda _d: None)
        with pytest.raises(RuntimeError, match="denied"):
            await watch.start()

    asyncio.run(_run())


def test_start_times_out_without_current() -> None:
    async def _run() -> None:
        watch, _ = _watch([[HOLD]], lambda _d: None)
        with pytest.raises(ConnectionError, match="consistent snapshot"):
            await watch.start(ready_timeout=0.05)
        assert watch.done

    asyncio.run(_run())


def test_target_remove_fails_the_watch() -> None:
    async def _run() -> None:
        watch, _ = _watch(
            [[FAKE_DOC, FAKE_CURRENT, target_remove(code=7, message="denied")]],
            lambda _d: None,
        )
        await watch.start()
        await _wait_for(lambda: watch.done)

        exc = watch.task.exception()
        assert isinstance(exc, ConnectionError)
        assert "denied" in str(exc)

    asyncio.run(_run())


def test_callback_exception_does_not_kill_the_stream() -> None:
    async def _run() -> None:
        calls = 0

        def _bad(_data: dict[str, Any]) -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("consumer bug")

        watch, _ = _watch(
            [
                [
                    FAKE_DOC,
                    FAKE_CURRENT,
                    doc_change(SAMPLE_V2),
                    target_no_change(resume_token=b"t"),
                    HOLD,
                ]
            ],
            _bad,
        )
        await watch.start()
        await _wait_for(lambda: calls >= 2)

        assert not watch.done
        await watch.aclose()

    asyncio.run(_run())


# ── shutdown ───────────────────────────────────────────────────────────


def test_aclose_cancels_promptly_with_no_callback_after() -> None:
    async def _run() -> None:
        received: list[dict[str, Any]] = []
        watch, _ = _watch(
            [[FAKE_DOC, FAKE_CURRENT, doc_change(SAMPLE_V2), HOLD]],
            received.append,
        )
        await watch.start()
        assert received == [SAMPLE]

        await watch.aclose()
        assert watch.done
        await asyncio.sleep(0.02)
        assert received == [SAMPLE]

        # Idempotent.
        await watch.aclose()

    asyncio.run(_run())


def test_no_emission_after_unsubscribe() -> None:
    """unsubscribe() takes effect immediately, even for a snapshot already
    scheduled on the loop — the emit guard, not just task cancellation."""

    async def _run() -> None:
        received: list[dict[str, Any]] = []
        watch, _ = _watch([[FAKE_DOC, FAKE_CURRENT, HOLD]], received.append)
        await watch.start()
        assert received == [SAMPLE]

        watch.unsubscribe()
        # A consistent snapshot the loop had already picked up must be
        # dropped, not delivered to a consumer that just unsubscribed.
        watch._emit(SAMPLE_V2)
        assert received == [SAMPLE]
        await watch.aclose()

    asyncio.run(_run())


def test_unsubscribe_is_synchronous_and_stops_the_task() -> None:
    async def _run() -> None:
        watch, _ = _watch([[FAKE_DOC, FAKE_CURRENT, HOLD]], lambda _d: None)
        await watch.start()

        watch.unsubscribe()  # no await
        await _wait_for(lambda: watch.done)
        assert watch.task.cancelled()

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
