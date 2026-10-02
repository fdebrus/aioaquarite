"""Write/snapshot reconciliation for acknowledged cloud writes.

Hayward's cloud acknowledges a REST command (HTTP 200) several seconds
before the Firestore document reflects it. A snapshot emitted in that
window genuinely carries the pre-write state; no timestamp or ordering
can tell such an echo from a real external change. This module keeps the
heuristic that both Home Assistant integrations previously carried in
their coordinators:

* every acknowledged write is queued per ``(pool, path)`` with its own
  timestamp;
* a snapshot confirms queued writes **in order** — the head is popped
  only when the snapshot agrees with it (tolerantly, since Firestore
  returns int/str/bool variants), and never while an earlier write is
  still pending;
* while anything is pending for a path, delivered data carries the
  **newest** pending value at that path (overlay), not the snapshot's;
* each write ages out on its own timestamp
  (:data:`PENDING_WRITE_TTL_SECONDS`), and a snapshot that just pruned
  an expired write cannot confirm the new head on the same pass — it may
  itself be a pre-write echo;
* expiry without confirmation re-delivers the last raw snapshot
  un-overlaid for that path (last-known truth, no availability flap) and
  then runs an authoritative fetch, retrying failures with backoff; a
  new snapshot cancels the pending fetch or retry.

Everything here runs on the event loop — the library has had no threads
since 0.12 — so the only locks are ``asyncio`` ones.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Any

from ._coercion import normalise as _normalise

_LOGGER = logging.getLogger(__name__)

# Fallback for when a confirming Firestore push never arrives (controller
# offline, command dropped); a confirming snapshot normally clears the
# pending entry well before this. Module constant by design — not a
# consumer-facing knob.
PENDING_WRITE_TTL_SECONDS = 10.0

# Backoff for the authoritative fetch after a pending write expired
# unconfirmed: first retry delay, doubling up to the cap.
RECONCILE_RETRY_INITIAL = 2.0
RECONCILE_RETRY_MAX = 60.0


def _set_path(data: dict[str, Any], value_path: str, value: Any) -> None:
    """Write value into data at a dot-notation path, creating dicts as needed.

    A non-dict intermediate node is replaced: the cloud acknowledged a
    write below it, so whatever scalar sat there is stale.
    """
    keys = value_path.split(".")
    target: dict[str, Any] = data
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    target[keys[-1]] = value


def _get_path(data: dict[str, Any], value_path: str) -> Any:
    """Read a nested value, normalised per the coercion map; None if absent."""
    val: Any = data
    try:
        for key in value_path.split("."):
            val = val[key]
    except (KeyError, TypeError):
        return None
    return _normalise(value_path, val, None)


def _values_agree(remote: Any, pending: Any) -> bool:
    """Compare values tolerantly: Firestore returns int/str/bool variants."""
    if remote == pending:
        return True
    try:
        return float(remote) == float(pending)
    except (TypeError, ValueError):
        return False


@dataclass
class _PendingWrite:
    """One acknowledged (or pre-queued) write awaiting its Firestore echo."""

    value: Any
    written_at: float
    # Pre-queued sequence entries (record_pending) start unsent; the send
    # that later succeeds claims the oldest unsent entry instead of
    # appending a duplicate.
    sent: bool


class PendingWriteReconciler:
    """Per-pool pending-write queues and snapshot reconciliation.

    Owned by :class:`~aioaquarite.AquariteClient`, one per pool. The
    client wires three hooks in:

    * ``fetch`` — coroutine returning the pool document (authoritative
      read, used by the expiry reconcile);
    * ``get_data`` — the client's stored pool data, mirrored into by
      writes so command payloads build on the acknowledged state;
    * ``deliver`` — stores delivered data and forwards it to the pool's
      subscriber callback, if any.
    """

    def __init__(
        self,
        label: str,
        fetch: Callable[[], Awaitable[dict[str, Any]]],
        get_data: Callable[[], dict[str, Any] | None],
        deliver: Callable[[dict[str, Any]], None],
    ) -> None:
        self._label = label
        self._fetch = fetch
        self._get_data = get_data
        self._deliver = deliver
        self._writes: dict[str, list[_PendingWrite]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._expiry_handles: dict[str, asyncio.TimerHandle] = {}
        self._retry_handle: asyncio.TimerHandle | None = None
        self._retry_delay = RECONCILE_RETRY_INITIAL
        self._reconcile_task: asyncio.Task[None] | None = None
        self._last_raw: dict[str, Any] | None = None
        # Bumped by every snapshot and every reconcile-fetch delivery; an
        # in-flight manual fetch that observes a bump must yield to the
        # newer state instead of publishing its stale read.
        self.generation = 0

    # ── write-side API (called with the event loop running) ────────────

    def lock(self, value_path: str) -> asyncio.Lock:
        """The lock serialising writers of a path.

        Writers sharing a path must keep the pending-queue order identical
        to the wire order, or confirmations would overlay values the
        controller no longer has.
        """
        return self._locks.setdefault(value_path, asyncio.Lock())

    def record(self, value_path: str, value: Any, *, sent: bool) -> None:
        """Queue a write (acknowledged, or pre-queued for a sequence).

        Coalesces an identical repeat into the existing tail — an
        idempotent repeat causes no second document transition, so a
        second entry would wait for a confirmation that never comes.
        The written value is mirrored into the stored pool data so
        subsequent command payloads build on it; nothing is delivered.
        """
        writes = self._writes.setdefault(value_path, [])
        now = monotonic()
        # Age out entries by their own timestamp so sustained writing
        # cannot grow the queue without bound.
        while writes and now - writes[0].written_at >= PENDING_WRITE_TTL_SECONDS:
            writes.pop(0)
        if writes and writes[-1].value == value:
            writes[-1].written_at = now
            writes[-1].sent = writes[-1].sent or sent
        else:
            writes.append(_PendingWrite(value, now, sent))
        if (data := self._get_data()) is not None:
            _set_path(data, value_path, value)
        self._arm_expiry(value_path)

    def on_write_success(self, updates: dict[str, Any]) -> None:
        """Account for one acknowledged command covering these paths.

        For each path, the oldest unsent pre-queued entry is claimed —
        that entry *is* the write this send carried, since sequences hold
        the path lock and send in queue order. A claimed tail restarts
        its TTL so the window covers the round trip of the actual send,
        not of queueing. With nothing pre-queued, the value is recorded
        as an ordinary acknowledged write.
        """
        for value_path, value in updates.items():
            self._claim_or_record(value_path, value)

    def _claim_or_record(self, value_path: str, value: Any) -> None:
        writes = self._writes.get(value_path, [])
        now = monotonic()
        while writes and now - writes[0].written_at >= PENDING_WRITE_TTL_SECONDS:
            writes.pop(0)
        if not writes:
            self._clear(value_path)
        for write in writes:
            if not write.sent:
                # If a snapshot confirmed the queued head early (the
                # document already held that value, so the send was an
                # idempotent no-op), this claims the next queued entry
                # instead. That only marks it sent ahead of its own send,
                # which the later send then treats as a tail refresh —
                # the outcomes converge.
                write.sent = True
                if write is writes[-1]:
                    write.written_at = monotonic()
                    self._arm_expiry(value_path)
                return
        self.record(value_path, value, sent=True)

    def refresh(self, value_path: str) -> None:
        """Restart the newest pending write's protection window.

        For a write queued ahead of time (a pulse's final on): its TTL
        must cover the Firestore round trip from the actual send, not
        from queueing, or expiry could publish stale data just before
        the confirming push lands.
        """
        writes = self._writes.get(value_path)
        if not writes:
            return
        writes[-1].written_at = monotonic()
        self._arm_expiry(value_path)

    def discard(self, value_path: str) -> None:
        """Drop the newest pending write for a path.

        For unwinding a pre-queued write whose send failed: it must not
        keep suppressing the snapshots that reflect what the cloud really
        has. The expiry timer was armed by the newer, discarded write, so
        it is re-armed from the surviving tail's own timestamp.
        """
        writes = self._writes.get(value_path)
        if not writes:
            return
        writes.pop()
        if not writes:
            self._clear(value_path)
            return
        remaining = max(
            0.0,
            writes[-1].written_at + PENDING_WRITE_TTL_SECONDS - monotonic(),
        )
        self._arm_expiry(value_path, delay=remaining)

    # ── delivery side ──────────────────────────────────────────────────

    def on_snapshot(self, data: dict[str, Any]) -> None:
        """Reconcile and deliver one Firestore snapshot.

        A snapshot is authoritative: it supersedes any in-flight manual
        fetch (generation bump) and cancels a pending reconcile fetch or
        retry.
        """
        self.generation += 1
        self.cancel_reconcile()
        self._last_raw = deepcopy(data)
        self._deliver(self.overlay(data, confirm=True))

    def overlay(
        self, data: dict[str, Any], *, confirm: bool
    ) -> dict[str, Any]:
        """Reconcile ``data`` (in place) against the pending queues.

        With ``confirm``, a queue head that agrees with the data is
        popped — in order only, and never on the same pass that pruned an
        expired write, because such a snapshot may itself be a pre-write
        echo. Afterwards every path with pending writes carries its
        newest pending value.
        """
        now = monotonic()
        for value_path, writes in list(self._writes.items()):
            pruned = False
            while (
                writes
                and now - writes[0].written_at >= PENDING_WRITE_TTL_SECONDS
            ):
                writes.pop(0)
                pruned = True
            if not writes:
                self._clear(value_path)
                continue
            remote_value = _get_path(data, value_path)
            if (
                confirm
                and not pruned
                and _values_agree(remote_value, writes[0].value)
            ):
                writes.pop(0)
                if not writes:
                    self._clear(value_path)
                    continue
            _set_path(data, value_path, writes[-1].value)
        return data

    # ── expiry and the authoritative reconcile fetch ───────────────────

    def _arm_expiry(self, value_path: str, *, delay: float | None = None) -> None:
        if (handle := self._expiry_handles.pop(value_path, None)) is not None:
            handle.cancel()
        self._expiry_handles[value_path] = asyncio.get_running_loop().call_later(
            PENDING_WRITE_TTL_SECONDS if delay is None else delay,
            self._expire,
            value_path,
        )

    def _expire(self, value_path: str) -> None:
        """TTL fired without a confirming snapshot: reconcile with truth.

        The overlay is dropped and the last raw snapshot is re-delivered
        immediately — last-known truth, keeping consumers' entities
        available instead of flapping — then an authoritative fetch runs.
        """
        self._expiry_handles.pop(value_path, None)
        if self._writes.pop(value_path, None) is None:
            return
        _LOGGER.debug(
            "%s: pending write for %s expired unconfirmed; reconciling",
            self._label,
            value_path,
        )
        if self._last_raw is not None:
            # Other paths may still be inside their own TTL window: keep
            # their overlays, but never let this stale re-read confirm
            # anything.
            self._deliver(self.overlay(deepcopy(self._last_raw), confirm=False))
        self.start_reconcile()

    def start_reconcile(self) -> None:
        """Launch the authoritative fetch unless one is already running."""
        if self._retry_handle is not None:
            self._retry_handle.cancel()
            self._retry_handle = None
        if self._reconcile_task is not None and not self._reconcile_task.done():
            return
        self._reconcile_task = asyncio.get_running_loop().create_task(
            self._reconcile(), name=f"aioaquarite-reconcile-{self._label}"
        )

    async def _reconcile(self) -> None:
        """Fetch authoritative data, retrying failures with backoff.

        A failure only logs and re-arms — the listen stream itself is
        alive, so this never touches ``on_health``. A new snapshot (or a
        successful manual fetch) cancels the retry.
        """
        try:
            data = await self._fetch()
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 — background task must survive
            delay = self._retry_delay
            self._retry_delay = min(self._retry_delay * 2, RECONCILE_RETRY_MAX)
            _LOGGER.warning(
                "%s: reconcile fetch failed (%s); retrying in %.1fs",
                self._label,
                err,
                delay,
            )
            self._retry_handle = asyncio.get_running_loop().call_later(
                delay, self.start_reconcile
            )
            return
        self._retry_delay = RECONCILE_RETRY_INITIAL
        self.generation += 1
        self._last_raw = deepcopy(data)
        self._deliver(self.overlay(data, confirm=True))

    def note_authoritative_fetch(self, data: dict[str, Any]) -> None:
        """Record a successful manual fetch as the new last-known truth.

        A manual fetch is as authoritative as a push: the pending
        reconcile retry would only refetch, so it is cancelled.
        """
        self.cancel_reconcile()
        self._last_raw = deepcopy(data)

    def cancel_reconcile(self) -> None:
        """Stop the reconcile: the pending retry and the in-flight task.

        The in-flight task matters too: its late result could overwrite
        an authoritative snapshot with older data, and its late failure
        would re-arm the retry against fresh data.
        """
        if self._retry_handle is not None:
            self._retry_handle.cancel()
            self._retry_handle = None
        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            self._reconcile_task = None
        self._retry_delay = RECONCILE_RETRY_INITIAL

    # ── housekeeping ───────────────────────────────────────────────────

    def _clear(self, value_path: str) -> None:
        self._writes.pop(value_path, None)
        if (handle := self._expiry_handles.pop(value_path, None)) is not None:
            handle.cancel()

    def close(self) -> None:
        """Drop all pending state and cancel every timer and task."""
        for handle in self._expiry_handles.values():
            handle.cancel()
        self._expiry_handles.clear()
        self._writes.clear()
        self.cancel_reconcile()
        self._last_raw = None
