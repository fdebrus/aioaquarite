"""Tests for write/snapshot reconciliation (aioaquarite._pending).

Each test is named after the invariant it guards (see the module
docstring of :mod:`aioaquarite._pending`). They drive the real
:class:`AquariteClient` through a fake listen stream (live ``Feed``
pushes of real ``ListenResponse`` protos) and a fake REST endpoint
(``send_command`` is the library's single REST write boundary).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aioaquarite import _pending
from aioaquarite.client import AquariteClient
from aioaquarite.exceptions import AquariteError, CommandError

from ._fakes import (
    FakeAsyncClient,
    FakeGapic,
    Feed,
    doc_change,
    target_current,
    target_no_change,
)

POOL_ID = "pool-1"
PATH = "light.status"


async def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


class _Harness:
    """A subscribed client with a live push feed and a fake REST side."""

    def __init__(
        self, client: AquariteClient, watch: Any, feed: Feed, received: list[dict[str, Any]]
    ) -> None:
        self.client = client
        self.watch = watch
        self.feed = feed
        self.received = received

    def push(self, data: dict[str, Any]) -> None:
        """Deliver one consistent snapshot through the listen stream."""
        self.feed.push(doc_change(data))
        self.feed.push(target_no_change(resume_token=b"rt"))

    async def pushed(self, data: dict[str, Any]) -> dict[str, Any]:
        """Push a snapshot and return what the subscriber received for it."""
        before = len(self.received)
        self.push(data)
        await _wait_for(lambda: len(self.received) > before)
        return self.received[-1]

    def light(self, entry: dict[str, Any]) -> Any:
        return entry.get("light", {}).get("status")

    async def aclose(self) -> None:
        await self.watch.aclose()


async def _start(initial: dict[str, Any]) -> _Harness:
    feed = Feed()
    gapic = FakeGapic()
    gapic.scripts = [[doc_change(initial), target_current(), feed]]
    auth = MagicMock()
    auth.get_async_client = AsyncMock(return_value=FakeAsyncClient(gapic))
    auth.tokens = {"idToken": "t", "localId": "uid-abc"}
    client = AquariteClient(auth)
    client.send_command = AsyncMock()  # type: ignore[method-assign]
    received: list[dict[str, Any]] = []
    watch = await client.subscribe_pool(POOL_ID, received.append)
    assert received == [initial]
    received.clear()
    return _Harness(client, watch, feed, received)


# ── invariants 1–3: ordered queue, in-order confirmation, newest overlay ──


def test_plain_toggle_stale_snapshot_suppressed() -> None:
    """Write ON; a stale OFF snapshot is suppressed; the confirming ON is
    accepted; a later real OFF applies immediately."""

    async def _run() -> None:
        h = await _start({"light": {"status": 0}})
        await h.client.set_value(POOL_ID, PATH, 1)
        # The acknowledged write is mirrored into the stored pool data.
        assert h.client.get_pool_data(POOL_ID)["light"]["status"] == 1

        assert h.light(await h.pushed({"light": {"status": 0}})) == 1  # stale echo
        assert h.light(await h.pushed({"light": {"status": 1}})) == 1  # confirms
        assert h.light(await h.pushed({"light": {"status": 0}})) == 0  # real change
        await h.aclose()

    asyncio.run(_run())


def test_snapshot_confirms_writes_in_order_only() -> None:
    """Rapid ON/OFF: a pre-ON echo carrying OFF does not lift protection,
    and the delayed ON confirmation cannot flip the state back."""

    async def _run() -> None:
        h = await _start({"light": {"status": 0}})
        await h.client.set_value(POOL_ID, PATH, 1)
        await h.client.set_value(POOL_ID, PATH, 0)

        assert h.light(await h.pushed({"light": {"status": 0}})) == 0  # pre-ON echo
        assert h.light(await h.pushed({"light": {"status": 1}})) == 0  # ON confirms
        assert h.light(await h.pushed({"light": {"status": 0}})) == 0  # OFF confirms
        assert h.light(await h.pushed({"light": {"status": 1}})) == 1  # real change
        await h.aclose()

    asyncio.run(_run())


def test_overlay_carries_newest_pending_value() -> None:
    """With two writes pending, snapshots carry the newest, not the head's
    value and not the snapshot's own."""

    async def _run() -> None:
        h = await _start({"light": {"status": 0}})
        await h.client.set_value(POOL_ID, PATH, 1)
        await h.client.set_value(POOL_ID, PATH, 2)

        assert h.light(await h.pushed({"light": {"status": 7}})) == 2
        await h.aclose()

    asyncio.run(_run())


def test_values_agree_tolerates_firestore_variants() -> None:
    """A confirming snapshot may carry "1"/True for a written 1 — on
    coercion-mapped paths and on unmapped ones alike."""

    async def _run() -> None:
        h = await _start({"light": {"status": 0}})
        await h.client.set_value(POOL_ID, PATH, 1)

        assert h.light(await h.pushed({"light": {"status": "1"}})) == "1"  # confirms
        # Protection lifted: a later disagreeing push sticks.
        assert h.light(await h.pushed({"light": {"status": 0}})) == 0

        # An unmapped path has no coercion map entry to normalise the
        # remote value, so the tolerant comparison itself must cope.
        await h.client.set_value(POOL_ID, "custom.level", 5)
        assert (await h.pushed({"custom": {"level": "5"}}))["custom"]["level"] == "5"
        assert (await h.pushed({"custom": {"level": 9}}))["custom"]["level"] == 9
        await h.aclose()

    asyncio.run(_run())


# ── invariant 4: per-write aging ──────────────────────────────────────────


def test_each_write_ages_out_on_its_own_timestamp() -> None:
    async def _run() -> None:
        clock = {"now": 100.0}
        with patch.object(_pending, "monotonic", side_effect=lambda: clock["now"]):
            h = await _start({"light": {"status": 0}})
            await h.client.set_value(POOL_ID, PATH, 1)
            clock["now"] = 104.0
            await h.client.set_value(POOL_ID, PATH, 0)

            # The ON write has aged out, but the newer OFF keeps its full
            # TTL: the remote on may not override it yet.
            clock["now"] = 100.0 + _pending.PENDING_WRITE_TTL_SECONDS + 1.0
            assert h.light(await h.pushed({"light": {"status": 1}})) == 0

            # Once the OFF has aged too, the remote change wins.
            clock["now"] = 104.0 + _pending.PENDING_WRITE_TTL_SECONDS + 1.0
            assert h.light(await h.pushed({"light": {"status": 1}})) == 1
            await h.aclose()

    asyncio.run(_run())


def test_sustained_writing_cannot_keep_old_entries_alive() -> None:
    """Aging must be per entry, not per queue: while newer writes keep
    arriving, an expired head whose echo never comes must still die, or the
    queue would suppress real remote changes for as long as writing
    continues."""

    async def _run() -> None:
        clock = {"now": 100.0}
        with patch.object(_pending, "monotonic", side_effect=lambda: clock["now"]):
            h = await _start({"light": {"status": 0}})
            await h.client.set_value(POOL_ID, PATH, 1)  # echo never comes
            clock["now"] = 105.0
            await h.client.set_value(POOL_ID, PATH, 2)
            clock["now"] = 109.0
            await h.client.set_value(POOL_ID, PATH, 3)

            # Head (1) is dead; this pass prunes it and may confirm nothing.
            clock["now"] = 111.0
            assert h.light(await h.pushed({"light": {"status": 2}})) == 3
            # Now 2 is the head and this snapshot confirms it in order.
            clock["now"] = 112.0
            assert h.light(await h.pushed({"light": {"status": 2}})) == 3
            clock["now"] = 113.0
            assert h.light(await h.pushed({"light": {"status": 3}})) == 3
            # Queue empty: the real remote change applies well before the
            # newest write's own TTL would have allowed under queue-aging.
            clock["now"] = 114.0
            assert h.light(await h.pushed({"light": {"status": 9}})) == 9
            await h.aclose()

    asyncio.run(_run())


# ── invariant 5: a pruning snapshot cannot confirm the new head ───────────


def test_snapshot_that_pruned_cannot_confirm_the_new_head() -> None:
    """With ON (expired) and OFF queued, a stale pre-write echo carrying
    OFF prunes the ON but must not also confirm the OFF — the delayed ON
    echo would then flip the state back."""

    async def _run() -> None:
        clock = {"now": 100.0}
        with patch.object(_pending, "monotonic", side_effect=lambda: clock["now"]):
            h = await _start({"light": {"status": 0}})
            await h.client.set_value(POOL_ID, PATH, 1)
            clock["now"] = 104.0
            await h.client.set_value(POOL_ID, PATH, 0)

            clock["now"] = 100.0 + _pending.PENDING_WRITE_TTL_SECONDS + 1.0
            assert h.light(await h.pushed({"light": {"status": 0}})) == 0  # prunes ON
            # The delayed ON echo: the OFF write is newer and unconfirmed.
            assert h.light(await h.pushed({"light": {"status": 1}})) == 0
            await h.aclose()

    asyncio.run(_run())


# ── invariant 6: identical repeats coalesce ───────────────────────────────


def test_identical_repeated_write_coalesces_into_the_tail() -> None:
    """Two identical commands hold one pending entry: the single coalesced
    echo confirms it, and the next real push applies immediately."""

    async def _run() -> None:
        h = await _start({"light": {"status": 0}})
        await h.client.set_value(POOL_ID, PATH, 1)
        await h.client.set_value(POOL_ID, PATH, 1)

        assert h.light(await h.pushed({"light": {"status": 1}})) == 1  # confirms
        # A second entry would suppress this real OFF until the TTL.
        assert h.light(await h.pushed({"light": {"status": 0}})) == 0
        await h.aclose()

    asyncio.run(_run())


# ── invariant 7: pre-queued sequences ─────────────────────────────────────


def test_failed_first_send_discards_both_queued_writes() -> None:
    """A pulse whose first send fails unwinds both entries, so the pushes
    reflecting the cloud's real state are not suppressed; the truth is
    re-fetched."""

    async def _run() -> None:
        h = await _start({"light": {"status": 1}})
        fetch = AsyncMock(return_value={"light": {"status": 1}})
        h.client._fetch_pool_document = fetch  # type: ignore[method-assign]
        h.client.send_command.side_effect = CommandError("boom")

        with pytest.raises(CommandError):
            await h.client.pulse(POOL_ID, PATH, 0, 1, 0.0)

        await _wait_for(lambda: fetch.await_count >= 1)  # reconcile ran
        # Nothing pending any more: a real OFF push applies immediately.
        assert h.light(await h.pushed({"light": {"status": 0}})) == 0
        await h.aclose()

    asyncio.run(_run())


def test_failed_discard_rearms_expiry_from_surviving_tail() -> None:
    """Discarding the pulse's writes re-arms expiry from the survivor's own
    timestamp, not the discarded writes' fresher ones."""

    async def _run() -> None:
        clock = {"now": 100.0}
        with patch.object(_pending, "monotonic", side_effect=lambda: clock["now"]):
            h = await _start({"light": {"status": 0}})
            await h.client.set_value(POOL_ID, PATH, 1)  # survivor at t=100

            clock["now"] = 104.0
            h.client.send_command.side_effect = CommandError("boom")

            delays: list[float] = []
            loop = asyncio.get_running_loop()
            real_call_later = loop.call_later

            def _spy(delay: float, cb: Any, *args: Any) -> Any:
                if getattr(cb, "__name__", "") == "_expire":
                    delays.append(delay)
                return real_call_later(delay, cb, *args)

            with patch.object(loop, "call_later", side_effect=_spy):
                with pytest.raises(CommandError):
                    await h.client.pulse(POOL_ID, PATH, 0, 1, 0.0)

            ttl = _pending.PENDING_WRITE_TTL_SECONDS
            # Two records, then one re-arm per discard; the last re-arm runs
            # from the survivor's own timestamp (t=100), not the pulse's.
            assert delays == [ttl, ttl, ttl, ttl - 4.0]
            await h.aclose()

    asyncio.run(_run())


def test_late_send_success_restarts_that_writes_ttl() -> None:
    """The pulse's final write is protected from its own send's round trip,
    not from queueing: with sends costing wall time, a stale echo past the
    queue-time TTL is still overlaid with the final value."""

    async def _run() -> None:
        clock = {"now": 100.0}
        with patch.object(_pending, "monotonic", side_effect=lambda: clock["now"]):
            h = await _start({"light": {"status": 1}})

            async def _send_costs_time(_payload: dict[str, Any]) -> None:
                clock["now"] += 3.0

            h.client.send_command.side_effect = _send_costs_time
            await h.client.pulse(POOL_ID, PATH, 0, 1, 0.0)

            # Past the queue-time TTL, within the TTL of the on send itself.
            clock["now"] = 100.0 + _pending.PENDING_WRITE_TTL_SECONDS + 1.0
            assert h.light(await h.pushed({"light": {"status": 0}})) == 1
            await h.aclose()

    asyncio.run(_run())


# ── invariant 8: writers sharing a path are serialised ────────────────────


def test_write_lock_serialises_a_writer_with_the_pulse() -> None:
    """A write during the pulse waits for it, so queue order == wire order."""

    async def _run() -> None:
        h = await _start({"light": {"status": 1}})
        release = asyncio.Event()
        wire: list[Any] = []

        async def _hold_first_off(payload: dict[str, Any]) -> None:
            wire.append(payload["changes"])
            if len(wire) == 1:
                await release.wait()

        h.client.send_command.side_effect = _hold_first_off

        pulse_task = asyncio.ensure_future(
            h.client.pulse(POOL_ID, PATH, 0, 1, 0.0)
        )
        await _wait_for(lambda: len(wire) == 1)

        async def _user_off() -> None:
            async with h.client.write_lock(POOL_ID, PATH):
                await h.client.set_value(POOL_ID, PATH, 0)

        off_task = asyncio.ensure_future(_user_off())
        await asyncio.sleep(0.05)
        assert len(wire) == 1  # the write waited instead of interleaving

        release.set()
        await pulse_task
        await off_task
        assert len(wire) == 3  # off, on, then the user's off

        # Confirmations in wire order settle on the user's final off.
        assert h.light(await h.pushed({"light": {"status": 0}})) == 0
        assert h.light(await h.pushed({"light": {"status": 1}})) == 0
        assert h.light(await h.pushed({"light": {"status": 0}})) == 0
        await h.aclose()

    asyncio.run(_run())


# ── invariant 9: expiry reconciles with the truth ─────────────────────────


def test_expiry_redelivers_raw_snapshot_then_fetches() -> None:
    """TTL expiry without confirmation drops the overlay, re-delivers the
    last raw snapshot immediately (no availability flap), then delivers an
    authoritative fetch."""

    async def _run() -> None:
        with patch.object(_pending, "PENDING_WRITE_TTL_SECONDS", 0.05):
            h = await _start({"light": {"status": 0}})
            fetch = AsyncMock(return_value={"light": {"status": 0}, "fresh": 1})
            h.client._fetch_pool_document = fetch  # type: ignore[method-assign]

            await h.client.set_value(POOL_ID, PATH, 1)
            assert h.received and h.light(h.received[-1]) == 1  # ack delivery
            h.received.clear()
            await _wait_for(lambda: len(h.received) >= 2)

            # First the last raw snapshot un-overlaid, then the fetch result.
            assert h.received[0] == {"light": {"status": 0}}
            assert h.received[1].get("fresh") == 1
            await h.aclose()

    asyncio.run(_run())


def test_expiry_redelivery_keeps_other_paths_overlays() -> None:
    """Expiry of one path must not strip another path's live protection."""

    async def _run() -> None:
        clock = {"now": 100.0}
        with (
            patch.object(_pending, "monotonic", side_effect=lambda: clock["now"]),
            patch.object(_pending, "PENDING_WRITE_TTL_SECONDS", 10.0),
        ):
            h = await _start(
                {"light": {"status": 0}, "filtration": {"intel": {"temp": 24}}}
            )
            fetch = AsyncMock(
                return_value={
                    "light": {"status": 0},
                    "filtration": {"intel": {"temp": 24}},
                }
            )
            h.client._fetch_pool_document = fetch  # type: ignore[method-assign]

            await h.client.set_value(POOL_ID, PATH, 1)
            clock["now"] = 109.0
            await h.client.set_value(POOL_ID, "filtration.intel.temp", 27)
            h.received.clear()  # drop the ack deliveries

            # Fire the light write's expiry by hand at its own deadline.
            clock["now"] = 111.0
            h.client._pending(POOL_ID)._expire(PATH)
            await _wait_for(lambda: len(h.received) >= 2)

            # Redelivery: light back to raw truth, temp still protected.
            assert h.received[0]["light"]["status"] == 0
            assert h.received[0]["filtration"]["intel"]["temp"] == 27
            # The authoritative fetch result is protected the same way.
            assert h.received[1]["filtration"]["intel"]["temp"] == 27
            await h.aclose()

    asyncio.run(_run())


def test_expiry_redelivery_cannot_confirm_other_paths() -> None:
    """The re-read of the last raw snapshot is stale by definition: letting
    it confirm another path's head would hand a later stale echo the right
    to flip that path."""

    async def _run() -> None:
        clock = {"now": 100.0}
        with patch.object(_pending, "monotonic", side_effect=lambda: clock["now"]):
            # The last raw snapshot carries light=1.
            h = await _start({"light": {"status": 1}})
            await h.client.set_value(POOL_ID, "main.other", 5)
            # Rapid toggle, still fresh at expiry time: 1 then 0 queued;
            # the raw snapshot agrees with the head.
            clock["now"] = 109.0
            await h.client.set_value(POOL_ID, PATH, 1)
            await h.client.set_value(POOL_ID, PATH, 0)
            h.received.clear()  # drop the ack deliveries

            # The reconcile fetch hangs: only the redelivery pass and the
            # pushes below may touch the queues.
            async def _hanging_fetch(_pool_id: str) -> dict[str, Any]:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            h.client._fetch_pool_document = _hanging_fetch  # type: ignore[method-assign]

            # Expire only the unrelated path, forcing a raw redelivery.
            clock["now"] = 100.0 + _pending.PENDING_WRITE_TTL_SECONDS
            h.client._pending(POOL_ID)._expire("main.other")
            await _wait_for(lambda: len(h.received) >= 1)
            assert h.light(h.received[-1]) == 0  # overlaid, not confirmed

            # With the head wrongly confirmed, this echo pair would end
            # delivering 1; in order it must settle on the pending 0.
            assert h.light(await h.pushed({"light": {"status": 0}})) == 0
            assert h.light(await h.pushed({"light": {"status": 1}})) == 0
            await h.aclose()

    asyncio.run(_run())


def test_reconcile_fetch_failure_retries_with_backoff() -> None:
    async def _run() -> None:
        with (
            patch.object(_pending, "PENDING_WRITE_TTL_SECONDS", 0.05),
            patch.object(_pending, "RECONCILE_RETRY_INITIAL", 0.03),
        ):
            h = await _start({"light": {"status": 0}})
            fetch = AsyncMock(
                side_effect=[
                    AquariteError("cloud down"),
                    AquariteError("still down"),
                    {"light": {"status": 0}, "healed": 1},
                ]
            )
            h.client._fetch_pool_document = fetch  # type: ignore[method-assign]

            await h.client.set_value(POOL_ID, PATH, 1)
            await _wait_for(lambda: h.received and h.received[-1].get("healed") == 1)
            assert fetch.await_count == 3
            await h.aclose()

    asyncio.run(_run())


def test_snapshot_cancels_in_flight_reconcile_fetch() -> None:
    """A snapshot proves the connection; a reconcile fetch hanging
    mid-flight must be cancelled, not allowed to deliver its stale read."""

    async def _run() -> None:
        with patch.object(_pending, "PENDING_WRITE_TTL_SECONDS", 0.05):
            h = await _start({"light": {"status": 0}})
            release = asyncio.Event()
            started = asyncio.Event()

            async def _hanging_stale_fetch(_pool_id: str) -> dict[str, Any]:
                started.set()
                await release.wait()
                return {"light": {"status": 0}, "stale": 1}

            h.client._fetch_pool_document = _hanging_stale_fetch  # type: ignore[method-assign]
            await h.client.set_value(POOL_ID, PATH, 1)
            await _wait_for(started.is_set)  # expiry fired, fetch hangs

            latest = await h.pushed({"light": {"status": 1}})
            release.set()
            await asyncio.sleep(0.2)

            # The push won; the fetch's stale read was never delivered.
            assert h.received[-1] is latest
            await h.aclose()

    asyncio.run(_run())


def test_snapshot_cancels_armed_reconcile_retry() -> None:
    """A snapshot landing between retries disarms the pending retry."""

    async def _run() -> None:
        with (
            patch.object(_pending, "PENDING_WRITE_TTL_SECONDS", 0.05),
            patch.object(_pending, "RECONCILE_RETRY_INITIAL", 0.08),
        ):
            h = await _start({"light": {"status": 0}})
            fetch = AsyncMock(side_effect=AquariteError("cloud down"))
            h.client._fetch_pool_document = fetch  # type: ignore[method-assign]

            await h.client.set_value(POOL_ID, PATH, 1)
            await _wait_for(lambda: fetch.await_count == 1)  # retry armed

            await h.pushed({"light": {"status": 1}})
            await asyncio.sleep(0.3)
            assert fetch.await_count == 1  # the armed retry never fired
            await h.aclose()

    asyncio.run(_run())


# ── invariant 10: in-flight manual fetches cannot regress newer state ─────


def test_in_flight_fetch_superseded_by_snapshot() -> None:
    async def _run() -> None:
        h = await _start({"light": {"status": 1}})
        release = asyncio.Event()

        async def _slow_stale_fetch(_pool_id: str) -> dict[str, Any]:
            await release.wait()
            return {"light": {"status": 0}}  # read before the push's change

        h.client._fetch_pool_document = _slow_stale_fetch  # type: ignore[method-assign]
        fetch_task = asyncio.ensure_future(h.client.fetch_pool_data(POOL_ID))
        await asyncio.sleep(0.02)

        await h.pushed({"light": {"status": 1}})
        release.set()

        assert (await fetch_task)["light"]["status"] == 1  # the newer state
        await h.aclose()

    asyncio.run(_run())


def test_failed_in_flight_fetch_superseded_by_snapshot() -> None:
    """A late fetch failure must not shadow data a push already delivered."""

    async def _run() -> None:
        h = await _start({"light": {"status": 1}})
        release = asyncio.Event()

        async def _slow_failing_fetch(_pool_id: str) -> dict[str, Any]:
            await release.wait()
            raise AquariteError("late failure")

        h.client._fetch_pool_document = _slow_failing_fetch  # type: ignore[method-assign]
        fetch_task = asyncio.ensure_future(h.client.fetch_pool_data(POOL_ID))
        await asyncio.sleep(0.02)

        await h.pushed({"light": {"status": 1}})
        release.set()

        assert (await fetch_task)["light"]["status"] == 1  # no raise
        await h.aclose()

    asyncio.run(_run())


def test_fetch_failure_without_newer_state_still_raises() -> None:
    async def _run() -> None:
        h = await _start({"light": {"status": 1}})
        h.client._fetch_pool_document = AsyncMock(  # type: ignore[method-assign]
            side_effect=AquariteError("cloud down")
        )
        with pytest.raises(AquariteError):
            await h.client.fetch_pool_data(POOL_ID)
        await h.aclose()

    asyncio.run(_run())


def test_manual_fetch_result_is_overlaid() -> None:
    """A manual fetch during the echo window must not regress the write."""

    async def _run() -> None:
        h = await _start({"light": {"status": 0}})
        await h.client.set_value(POOL_ID, PATH, 1)
        h.client._fetch_pool_document = AsyncMock(  # type: ignore[method-assign]
            return_value={"light": {"status": 0}}  # pre-write read
        )
        data = await h.client.fetch_pool_data(POOL_ID)
        assert data["light"]["status"] == 1
        await h.aclose()

    asyncio.run(_run())


# ── the pulse ─────────────────────────────────────────────────────────────


def test_pulse_never_delivers_the_intermediate_off() -> None:
    """Ack deliveries and echoes landing inside the pulse delay are all
    overlaid with the final on, so subscribers never see the transient off.

    Values are captured at callback time: ack deliveries hand out the
    live pool-data dict, so inspecting stored entries afterwards would
    miss a transient value that a later delivery overwrote in place.
    """

    async def _run() -> None:
        h = await _start({"light": {"status": 1}})
        seen: list[Any] = []
        h.client._pool_subscribers[POOL_ID] = lambda data: seen.append(
            h.light(data)
        )

        async def _echo_during_off(payload: dict[str, Any]) -> None:
            if '"status": 0' in payload["changes"]:
                h.push({"light": {"status": 1}})  # stale pre-pulse echo
                h.push({"light": {"status": 0}})  # the off echo

        h.client.send_command.side_effect = _echo_during_off
        await h.client.pulse(POOL_ID, PATH, 0, 1, 0.05)
        await _wait_for(lambda: len(seen) >= 4)  # 2 acks + 2 echoes

        assert seen == [1] * len(seen)
        assert h.client.get_pool_data(POOL_ID)["light"]["status"] == 1
        await h.aclose()

    asyncio.run(_run())


def test_acknowledged_write_is_delivered_immediately() -> None:
    """The cloud ack is the moment consumers must reflect a write — not
    the Firestore echo seconds later. The delivery carries the overlaid
    pool data."""

    async def _run() -> None:
        h = await _start({"light": {"status": 0}})
        await h.client.set_value(POOL_ID, PATH, 1)

        assert h.received and h.light(h.received[-1]) == 1
        await h.aclose()

    asyncio.run(_run())


def test_unsent_prequeued_value_never_enters_the_payload_cache() -> None:
    """record_pending must not mirror: a value the cloud never accepted
    cannot be promoted into the next command payload on that branch, and
    a failed sequence leaves the cache clean."""

    async def _run() -> None:
        h = await _start({"light": {"status": 1}})
        h.client.record_pending(POOL_ID, PATH, 0)

        assert h.client.get_pool_data(POOL_ID)["light"]["status"] == 1
        assert h.received == []  # queueing is silent too

        h.client.discard_pending(POOL_ID, PATH)
        assert h.client.get_pool_data(POOL_ID)["light"]["status"] == 1
        await h.aclose()

    asyncio.run(_run())


def test_pulse_survives_a_stale_pre_pulse_push() -> None:
    async def _run() -> None:
        h = await _start({"light": {"status": 1}})
        await h.client.pulse(POOL_ID, PATH, 0, 1, 0.0)

        # Stale pre-pulse push still carrying on: confirms nothing.
        assert h.light(await h.pushed({"light": {"status": 1}})) == 1
        # The pulse's off echo must not flicker the state off.
        assert h.light(await h.pushed({"light": {"status": 0}})) == 1
        # The final on echo confirms; a later real push then sticks.
        assert h.light(await h.pushed({"light": {"status": 1}})) == 1
        assert h.light(await h.pushed({"light": {"status": 0}})) == 0
        await h.aclose()

    asyncio.run(_run())


def test_pulse_failed_final_send_reconciles_consumed_off() -> None:
    """The off echo confirmed inside the delay, so the queued on is the
    overlay; discarding it alone would freeze the on state with no push
    coming — the reconcile fetch must restore the real off."""

    async def _run() -> None:
        h = await _start({"light": {"status": 1}})
        fetch = AsyncMock(return_value={"light": {"status": 0}})
        h.client._fetch_pool_document = fetch  # type: ignore[method-assign]

        async def _echo_then_fail(payload: dict[str, Any]) -> None:
            if '"status": 0' in payload["changes"]:
                h.push({"light": {"status": 0}})
                return
            raise CommandError("boom")

        h.client.send_command.side_effect = _echo_then_fail
        with pytest.raises(CommandError):
            await h.client.pulse(POOL_ID, PATH, 0, 1, 0.05)

        await _wait_for(
            lambda: bool(h.received) and h.light(h.received[-1]) == 0
        )
        await h.aclose()

    asyncio.run(_run())


# ── lifecycle and transparency ────────────────────────────────────────────


def test_snapshots_pass_through_unmodified_without_writes() -> None:
    """Zero-code-change guarantee: with nothing pending, delivery is the
    decoded snapshot itself."""

    async def _run() -> None:
        h = await _start({"light": {"status": 0}})
        sample = {"light": {"status": 1}, "main": {"temperature": 25.5}}
        assert await h.pushed(sample) == sample
        await h.aclose()

    asyncio.run(_run())


def test_resilient_aclose_releases_pending_state() -> None:
    """Closing the resilient subscription cancels expiry timers and the
    reconcile, so nothing fires into a closed consumer."""

    async def _run() -> None:
        with patch.object(_pending, "PENDING_WRITE_TTL_SECONDS", 0.05):
            feed = Feed()
            gapic = FakeGapic()
            gapic.scripts = [[doc_change({"light": {"status": 0}}), target_current(), feed]]
            auth = MagicMock()
            auth.get_async_client = AsyncMock(return_value=FakeAsyncClient(gapic))
            auth.tokens = {"idToken": "t", "localId": "uid-abc"}
            auth.token_generation = 0
            auth.is_token_expiring = MagicMock(return_value=False)
            auth.calculate_sleep_duration = MagicMock(return_value=30.0)
            auth.get_client = AsyncMock(return_value=(object(), 0))
            client = AquariteClient(auth)
            client.send_command = AsyncMock()  # type: ignore[method-assign]
            fetch = AsyncMock(return_value={"light": {"status": 0}})
            client._fetch_pool_document = fetch  # type: ignore[method-assign]

            received: list[dict[str, Any]] = []
            sub = await client.subscribe_pool_resilient(POOL_ID, received.append)
            await client.set_value(POOL_ID, PATH, 1)
            await sub.aclose()
            received.clear()

            await asyncio.sleep(0.2)
            fetch.assert_not_awaited()  # no expiry reconcile after close
            assert received == []

    asyncio.run(_run())


def test_resilient_subscription_delivers_through_the_overlay() -> None:
    """The overlay applies on the resilient path too, across its wiring."""

    async def _run() -> None:
        feed = Feed()
        gapic = FakeGapic()
        gapic.scripts = [[doc_change({"light": {"status": 0}}), target_current(), feed]]
        auth = MagicMock()
        auth.get_async_client = AsyncMock(return_value=FakeAsyncClient(gapic))
        auth.tokens = {"idToken": "t", "localId": "uid-abc"}
        auth.token_generation = 0
        auth.is_token_expiring = MagicMock(return_value=False)
        auth.calculate_sleep_duration = MagicMock(return_value=30.0)
        auth.get_client = AsyncMock(return_value=(object(), 0))
        client = AquariteClient(auth)
        client.send_command = AsyncMock()  # type: ignore[method-assign]

        received: list[dict[str, Any]] = []
        sub = await client.subscribe_pool_resilient(POOL_ID, received.append)
        received.clear()
        await client.set_value(POOL_ID, PATH, 1)

        feed.push(doc_change({"light": {"status": 0}}))  # stale echo
        feed.push(target_no_change(resume_token=b"rt"))
        await _wait_for(lambda: len(received) >= 1)
        assert received[-1]["light"]["status"] == 1
        await sub.aclose()

    asyncio.run(_run())


def test_mirror_replaces_scalar_intermediate_nodes() -> None:
    """A write below a scalar node replaces it instead of crashing."""

    async def _run() -> None:
        h = await _start({"light": "scalar"})
        await h.client.set_value(POOL_ID, PATH, 1)
        assert h.client.get_pool_data(POOL_ID)["light"]["status"] == 1
        await h.aclose()

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
