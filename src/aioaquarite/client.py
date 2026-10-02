"""Async API client for the Hayward Aquarite pool system."""

import asyncio
import json
import logging
from copy import deepcopy
from typing import Any, Callable, MutableMapping

import aiohttp

from ._coercion import normalise as _normalise
from ._pending import PendingWriteReconciler
from ._watch import AsyncDocumentWatch
from .auth import AquariteAuth
from .const import DEFAULT_HTTP_TIMEOUT, HAYWARD_REST_API
from .exceptions import CommandError, ConnectionError
from .subscription import (
    DEFAULT_HEALTH_CHECK_INTERVAL,
    DEFAULT_INITIAL_BACKOFF,
    DEFAULT_MAX_BACKOFF,
    ResilientPoolSubscription,
    ResilientUserPoolsSubscription,
)

_LOGGER = logging.getLogger(__name__)


class AquariteClient:
    """Aquarite API client for interacting with the Hayward cloud."""

    def __init__(self, auth: AquariteAuth) -> None:
        self._auth = auth
        self._pool_data: dict[str, dict[str, Any]] = {}
        self._branch_locks: dict[tuple[str, tuple[str, ...]], asyncio.Lock] = {}
        # Write/snapshot reconciliation, one reconciler per pool: pending
        # acknowledged writes, their expiry timers, and the authoritative
        # reconcile fetch. See aioaquarite._pending for the model.
        self._reconcilers: dict[str, PendingWriteReconciler] = {}
        self._pool_subscribers: dict[str, Callable[[dict[str, Any]], None]] = {}
        # Firestore resume tokens, keyed by document path. Successive
        # watches of the same document resume where the last one stopped
        # instead of replaying from scratch.
        self._resume_tokens: dict[str, bytes] = {}

    @property
    def auth(self) -> AquariteAuth:
        """Return the auth handler."""
        return self._auth

    def set_pool_data(self, pool_id: str, data: dict[str, Any]) -> None:
        """Store current pool data (used for building command payloads)."""
        self._pool_data[pool_id] = data

    def get_pool_data(self, pool_id: str) -> dict[str, Any] | None:
        """Return stored pool data."""
        return self._pool_data.get(pool_id)

    def _pending(self, pool_id: str) -> PendingWriteReconciler:
        """The pool's write/snapshot reconciler, created on first use."""
        reconciler = self._reconcilers.get(pool_id)
        if reconciler is None:
            reconciler = PendingWriteReconciler(
                f"pool {pool_id}",
                lambda: self._fetch_pool_document(pool_id),
                lambda: self._pool_data.get(pool_id),
                lambda data: self._deliver_pool_data(pool_id, data),
            )
            self._reconcilers[pool_id] = reconciler
        return reconciler

    def _deliver_pool_data(self, pool_id: str, data: dict[str, Any]) -> None:
        """Store reconciled data and forward it to the pool's subscriber."""
        self._pool_data[pool_id] = data
        callback = self._pool_subscribers.get(pool_id)
        if callback is None:
            return
        try:
            callback(data)
        except Exception:  # noqa: BLE001 — a consumer bug must not kill delivery
            _LOGGER.exception("pool %s: data callback failed", pool_id)

    def _release_pending(self, pool_id: str) -> None:
        """Drop the pool's reconciler and subscriber on final close."""
        reconciler = self._reconcilers.pop(pool_id, None)
        if reconciler is not None:
            reconciler.close()
        self._pool_subscribers.pop(pool_id, None)

    async def get_pools(self) -> dict[str, str]:
        """Fetch all pools for the authenticated user.

        Returns a mapping of pool_id -> pool_name.
        """
        client = await self._auth.get_async_client()
        assert self._auth.tokens is not None
        user_doc = await client.collection("users").document(
            self._auth.tokens["localId"]
        ).get()
        user_dict = user_doc.to_dict() or {}

        pools: dict[str, str] = {}
        for pool_id in user_dict.get("pools", []):
            pool_doc = await client.collection("pools").document(pool_id).get()
            pool_dict = pool_doc.to_dict()
            if pool_dict:
                name = pool_dict.get("form", {}).get("name", "Unknown")
                if "names" in pool_dict.get("form", {}) and pool_dict["form"]["names"]:
                    name = pool_dict["form"]["names"][0].get("name", name)
                pools[pool_id] = name
        return pools

    async def fetch_pool_data(self, pool_id: str) -> dict[str, Any]:
        """Fetch the full pool document from Firestore.

        The result is reconciled against pending acknowledged writes the
        same way snapshots are, and a snapshot (or reconcile fetch) that
        lands while this fetch is in flight supersedes it: the newer
        state is returned instead of the stale read — on the failure
        path too, so a late error cannot shadow data that already
        arrived.
        """
        reconciler = self._pending(pool_id)
        generation = reconciler.generation
        try:
            data = await self._fetch_pool_document(pool_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            if generation != reconciler.generation:
                return self._pool_data.get(pool_id) or {}
            raise
        if generation != reconciler.generation:
            return self._pool_data.get(pool_id) or {}
        reconciler.note_authoritative_fetch(data)
        merged = reconciler.overlay(data, confirm=True)
        self._pool_data[pool_id] = merged
        return merged

    async def _fetch_pool_document(self, pool_id: str) -> dict[str, Any]:
        """Read the raw pool document from Firestore (no reconciliation)."""
        client = await self._auth.get_async_client()
        pool_doc = await client.collection("pools").document(pool_id).get()
        data: dict[str, Any] = pool_doc.to_dict() or {}
        return data

    async def subscribe_pool(
        self, pool_id: str, callback: Callable[[dict[str, Any]], None]
    ) -> AsyncDocumentWatch:
        """Subscribe to real-time Firestore updates for a pool.

        Args:
            pool_id: The pool document ID.
            callback: Called with the pool data dict on each snapshot,
                **on the event loop** (not from a thread — since 0.12.0
                the listener is a native async task).

        Returns:
            An :class:`AsyncDocumentWatch`; call ``unsubscribe()`` (or
            ``await aclose()``) on it to stop listening.
        """
        client = await self._auth.get_async_client()
        doc_ref = client.collection("pools").document(pool_id)

        reconciler = self._pending(pool_id)
        self._pool_subscribers[pool_id] = callback
        watch = AsyncDocumentWatch(
            client,
            doc_ref._document_path,
            reconciler.on_snapshot,
            resume_tokens=self._resume_tokens,
            label=f"pool {pool_id}",
        )
        await watch.start()
        _LOGGER.debug("Firestore subscription active for %s", pool_id)
        return watch

    async def subscribe_pool_resilient(
        self,
        pool_id: str,
        callback: Callable[[dict[str, Any]], None],
        *,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        health_check_interval: float | None = DEFAULT_HEALTH_CHECK_INTERVAL,
        on_health: Callable[[bool], None] | None = None,
    ) -> ResilientPoolSubscription:
        """Subscribe to a pool with automatic token refresh and reconnect.

        Returns a :class:`ResilientPoolSubscription` handle; call
        ``await handle.aclose()`` to stop the subscription. The callback is
        invoked on the event loop — see
        :class:`ResilientPoolSubscription` for details.
        """
        sub = ResilientPoolSubscription(
            self,
            pool_id,
            callback,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            health_check_interval=health_check_interval,
            on_health=on_health,
        )
        await sub._start()
        return sub

    async def subscribe_user_pools(
        self, callback: Callable[[list[str]], None]
    ) -> AsyncDocumentWatch:
        """Subscribe to the user document's ``pools`` list.

        The callback receives the current ``list[str]`` of pool IDs every
        time ``users/{uid}`` changes, **on the event loop** (not from a
        thread — since 0.12.0 the listener is a native async task). Use
        this to detect pool additions or removals in the Hayward app
        without polling :meth:`get_pools`.

        Returns an :class:`AsyncDocumentWatch`; call ``unsubscribe()``
        (or ``await aclose()``) on it to stop listening.
        """
        client = await self._auth.get_async_client()
        assert self._auth.tokens is not None
        doc_ref = client.collection("users").document(
            self._auth.tokens["localId"]
        )

        def _on_data(data: dict[str, Any]) -> None:
            callback(list(data.get("pools", [])))

        watch = AsyncDocumentWatch(
            client,
            doc_ref._document_path,
            _on_data,
            resume_tokens=self._resume_tokens,
            label="user pools",
        )
        await watch.start()
        _LOGGER.debug("Firestore user-pools subscription active")
        return watch

    async def subscribe_user_pools_resilient(
        self,
        callback: Callable[[list[str]], None],
        *,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        health_check_interval: float | None = DEFAULT_HEALTH_CHECK_INTERVAL,
        on_health: Callable[[bool], None] | None = None,
    ) -> ResilientUserPoolsSubscription:
        """Subscribe to the user's pool list with token refresh and reconnect.

        Returns a :class:`ResilientUserPoolsSubscription` handle; call
        ``await handle.aclose()`` to stop the subscription. The callback
        is invoked on the event loop — see
        :class:`ResilientUserPoolsSubscription` for details.
        """
        sub = ResilientUserPoolsSubscription(
            self,
            callback,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            health_check_interval=health_check_interval,
            on_health=on_health,
        )
        await sub._start()
        return sub

    async def get_pool_stats(
        self,
        pool_id: str,
        type_: str,
        period: int,
    ) -> list[list[dict[str, Any]]]:
        """Fetch a stored sample series for a pool from ``/getStats``.

        Hits the Hayward cloud function ``getStats`` (a Firebase Cloud
        Function). The endpoint requires the user's Firebase id token and
        returns whatever time-series the Aquarite backend has retained for
        the requested metric.

        Args:
            pool_id: Pool document ID, the same value used as
                ``uuid`` in the cloud command payload.
            type_: Metric selector. Verified type values on firmware A50
                (observed May 2026) are ``ph``, ``rx``, ``temp``, ``cl``,
                ``cd``, ``filtration`` and ``aux1`` through ``aux4``. The
                strings ``light``, ``production`` and ``salt`` also appear
                in the web app source and may be populated on different
                hardware variants. Unrecognised values currently return
                HTTP 200 with timestamps but no ``field`` entries.
            period: Required by the cloud function — requests without it
                are rejected with HTTP 405. The Hayward backend currently
                appears to ignore the value semantically and always returns
                roughly the last 30 days of samples at ~10-minute
                granularity. The web app sends ``14``; ``30`` is a safe
                default for callers who just want the full window.

        Returns:
            The raw decoded payload — a list of series, each series being a
            list of point dicts. A point dict has the shape
            ``{"field": <value>, "seconds": <utc_unix>}`` for recognised
            metric types; the ``field`` key may be absent if the device
            had no value to report (or the type is unknown to the
            backend). The outer list typically contains a single series.

            Field encodings observed:
              - ``ph``: integer pH × 100 (``885`` → 8.85).
              - ``rx``: ORP in millivolts (integer).
              - ``temp``: water temperature in °C (float).
              - ``cl`` / ``cd``: probe reading; ``0`` when no probe is
                fitted.
              - ``filtration`` / ``aux1`` ... ``aux4``: ``0`` / ``1`` for
                off / on.

        Raises:
            ConnectionError: On a transport failure or timeout, including
                during the auth token refresh.
            CommandError: If the cloud function returns a non-2xx status.
        """
        try:
            client, _ = await self._auth.get_client()
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise ConnectionError(f"Auth client refresh failed: {err}") from err
        assert self._auth.tokens is not None
        headers = {
            "Authorization": f"Bearer {self._auth.tokens['idToken']}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {"uuid": pool_id, "type": type_, "period": period}
        try:
            async with self._auth._session.post(
                f"{HAYWARD_REST_API}getStats",
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=DEFAULT_HTTP_TIMEOUT),
            ) as response:
                _LOGGER.debug(
                    "getStats pool_id=%s type=%s period=%s -> %s",
                    pool_id,
                    type_,
                    period,
                    response.status,
                )
                if response.status >= 400:
                    raise CommandError(
                        f"getStats failed with status {response.status}"
                    )
                data: list[list[dict[str, Any]]] = await response.json()
                return data
        except aiohttp.ClientError as err:
            raise ConnectionError(f"HTTP transport error: {err}") from err
        except asyncio.TimeoutError as err:
            raise ConnectionError(f"Request timed out: {err}") from err

    async def get_server_date(self) -> dict[str, Any]:
        """Fetch the cloud function's current date from ``/getServerDate``.

        Useful for sanity-checking clock drift between the local host and
        the Hayward backend. The endpoint is unauthenticated. The cloud
        function returns ``{"date": "YYMMDD"}`` (e.g. ``"260529"`` for
        29 May 2026) — exact shape preserved.

        Raises:
            ConnectionError: On a transport failure or timeout.
            CommandError: If the cloud function returns a non-2xx status.
        """
        try:
            async with self._auth._session.get(
                f"{HAYWARD_REST_API}getServerDate",
                timeout=aiohttp.ClientTimeout(total=DEFAULT_HTTP_TIMEOUT),
            ) as response:
                _LOGGER.debug("getServerDate -> %s", response.status)
                if response.status >= 400:
                    raise CommandError(
                        f"getServerDate failed with status {response.status}"
                    )
                data: dict[str, Any] = await response.json()
                return data
        except aiohttp.ClientError as err:
            raise ConnectionError(f"HTTP transport error: {err}") from err
        except asyncio.TimeoutError as err:
            raise ConnectionError(f"Request timed out: {err}") from err

    async def send_command(self, data: dict[str, Any]) -> None:
        """Send a command to the Hayward cloud REST API."""
        try:
            client, _ = await self._auth.get_client()
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise ConnectionError(
                f"Auth client refresh failed: {err}"
            ) from err
        assert self._auth.tokens is not None
        headers = {"Authorization": f"Bearer {self._auth.tokens['idToken']}"}

        try:
            async with self._auth._session.post(
                f"{HAYWARD_REST_API}/sendPoolCommand",
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as response:
                _LOGGER.debug(
                    "sendPoolCommand operation=%s pool_id=%s -> %s",
                    data.get("operation"),
                    data.get("poolId"),
                    response.status,
                )
                if response.status >= 400:
                    raise CommandError(
                        f"Command failed with status {response.status}"
                    )
        except aiohttp.ClientError as err:
            raise ConnectionError(
                f"HTTP transport error: {err}"
            ) from err
        except asyncio.TimeoutError as err:
            raise ConnectionError(
                f"Request timed out: {err}"
            ) from err

    async def set_value(
        self, pool_id: str, value_path: str, value: Any
    ) -> None:
        """Set a single value on the pool device via REST API.

        Thin wrapper over :meth:`set_values`.
        """
        await self.set_values(pool_id, {value_path: value})

    async def set_values(
        self, pool_id: str, updates: dict[str, Any]
    ) -> None:
        """Set several values of one command branch as a single command.

        The command payload carries the whole branch rebuilt from the
        stored pool data, so all paths must resolve to the same branch:
        the same top-level key and, for deep paths (4+ segments), the
        same second-level key. Mixing branches raises ``ValueError``.

        On success the stored pool data is updated to match, so a
        subsequent command builds its payload on the new state instead
        of a snapshot-stale one (which would silently revert it).

        Concurrent commands for the same branch are serialised: the
        payload is built from the stored document, so two callers writing
        different fields at once would otherwise each send a branch that
        predates the other and revert it.
        """
        if not updates:
            raise ValueError("updates must not be empty")
        signatures = {self._branch_signature(path) for path in updates}
        if len(signatures) != 1:
            raise ValueError(
                "updates must target a single command branch, got "
                f"{sorted(signatures)}"
            )

        lock = self._branch_locks.setdefault(
            (pool_id, next(iter(signatures))), asyncio.Lock()
        )
        async with lock:
            await self._async_send_branch(pool_id, updates)

    async def _async_send_branch(
        self, pool_id: str, updates: dict[str, Any]
    ) -> None:
        """Build and send one branch command; caller holds the branch lock."""
        pool_data = self._pool_data.get(pool_id)
        if not pool_data:
            raise RuntimeError("Pool data not available; fetch data first.")

        current_config = self._extract_branch(pool_data, next(iter(updates)))

        effective = dict(updates)
        if "hidro.cloration_enabled" in effective:
            enabled = effective["hidro.cloration_enabled"]
            effective["hidro.cloration_enabled"] = 1 if enabled else 0
            effective["hidro.reduction"] = 1 if enabled else 0
            effective["hidro.disable"] = 1

        for path, value in effective.items():
            self._set_in_dict(current_config, path, value)

        payload = {
            "gateway": pool_data.get("wifi"),
            "poolId": pool_id,
            "operation": "WRP",
            "changes": json.dumps(current_config),
            "source": "web",
        }
        _LOGGER.debug("set_values pool_id=%s updates=%s", pool_id, effective)
        await self.send_command(payload)

        # The cloud acknowledged the write: record every written path as
        # pending so snapshots that predate it cannot flicker consumers'
        # state, and mirror the values into the stored pool data so the
        # next command payload is built on the acknowledged state.
        self._pending(pool_id).on_write_success(effective)

    # ── write/snapshot reconciliation helpers ────────────────

    def write_lock(self, pool_id: str, value_path: str) -> asyncio.Lock:
        """The lock serialising writers of one value path.

        Writers sharing a path (an entity and a pulse sequence, say)
        must keep the pending-write order identical to the wire order,
        or confirmations would overlay values the controller no longer
        has. Hold it across a multi-send sequence; single ``set_value``
        calls need no extra locking (the per-branch command lock already
        serialises them).
        """
        return self._pending(pool_id).lock(value_path)

    def record_pending(self, pool_id: str, value_path: str, value: Any) -> None:
        """Queue a value as pending **without sending it**.

        For pre-queued write sequences (a pulse queues its off and on
        before the first send): the Firestore echoes are then confirmed
        in order, and any snapshot landing mid-sequence is overlaid with
        the final value instead of a transient one. The send that later
        succeeds claims the queued entry — ``set_value`` will not record
        a duplicate — and restarts its TTL at that moment. Call under
        :meth:`write_lock`.
        """
        self._pending(pool_id).record(value_path, value, sent=False)

    def discard_pending(self, pool_id: str, value_path: str) -> None:
        """Drop the newest pending write for a path.

        For unwinding a pre-queued write whose send failed: it must not
        keep suppressing the snapshots that reflect what the cloud
        really has. Expiry re-arms from the surviving tail's own
        timestamp. Consider :meth:`reconcile` afterwards if an echo may
        already have been consumed mid-sequence.
        """
        self._pending(pool_id).discard(value_path)

    def refresh_pending(self, pool_id: str, value_path: str) -> None:
        """Restart the newest pending write's TTL window.

        For a write queued long before its send: the window must cover
        the round trip of the actual send, not of queueing.
        """
        self._pending(pool_id).refresh(value_path)

    def reconcile(self, pool_id: str) -> None:
        """Fetch authoritative pool data in the background and deliver it.

        Used after a write sequence failed halfway: the local prediction
        is then unreliable, so the truth is fetched and delivered to the
        pool's subscriber. Failures retry with backoff; a new snapshot
        cancels the retry.
        """
        self._pending(pool_id).start_reconcile()

    async def pulse(
        self,
        pool_id: str,
        value_path: str,
        off_value: Any,
        on_value: Any,
        delay: float,
    ) -> None:
        """Power-cycle one value without ever exposing the transient state.

        Sends ``off_value``, waits ``delay`` seconds, then sends
        ``on_value`` — the sequence a pool LED fixture needs to advance
        its colour. Both writes are queued as pending before the first
        send, so their Firestore echoes are confirmed in order and a
        snapshot landing inside the delay is delivered carrying the
        final ``on_value``: subscribers never see the intermediate
        ``off_value``. A failed send discards the writes the cloud never
        acknowledged and reconciles with an authoritative fetch; the
        final write's TTL is restarted after its own send so the window
        covers that round trip.

        The sequence holds :meth:`write_lock` for the path throughout,
        so concurrent writers queue behind it instead of interleaving.
        """
        reconciler = self._pending(pool_id)
        async with reconciler.lock(value_path):
            reconciler.record(value_path, off_value, sent=False)
            reconciler.record(value_path, on_value, sent=False)
            try:
                await self.set_value(pool_id, value_path, off_value)
            except BaseException:
                # Neither write reached the cloud; a snapshot consumed
                # mid-send may have been overlaid with a queued value
                # that just got discarded, so fetch the truth.
                reconciler.discard(value_path)
                reconciler.discard(value_path)
                reconciler.start_reconcile()
                raise
            try:
                await asyncio.sleep(delay)
                await self.set_value(pool_id, value_path, on_value)
            except BaseException:
                reconciler.discard(value_path)
                reconciler.start_reconcile()
                raise
            reconciler.refresh(value_path)

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _branch_signature(path: str) -> tuple[str, ...]:
        """Return the command-branch identity for a dot-notation path.

        Mirrors :meth:`_extract_branch`: deep paths (4+ segments) send
        only the two top levels, so their branch identity includes the
        second key.
        """
        keys = path.split(".")
        if len(keys) >= 4:
            return (keys[0], keys[1])
        return (keys[0],)

    @staticmethod
    def _set_in_dict(
        data_dict: MutableMapping[str, Any], path: str, value: Any
    ) -> None:
        """Set a value in a nested dict using dot-notation path.

        A non-dict intermediate node is replaced: the write targets a key
        below it, so whatever scalar the cloud had there is stale.
        """
        keys = path.split(".")
        for key in keys[:-1]:
            child = data_dict.get(key)
            if not isinstance(child, dict):
                child = {}
                data_dict[key] = child
            data_dict = child
        data_dict[keys[-1]] = value

    @staticmethod
    def _extract_branch(
        data: MutableMapping[str, Any], path: str
    ) -> dict[str, Any]:
        """Deep-clone the relevant branch of the data structure.

        For deep paths (4+ segments, e.g. relays.relay1.info.onoff),
        extract only 2 levels deep to send just the target branch.
        """
        keys = path.split(".")
        root_key = keys[0]
        if len(keys) >= 4:
            second_key = keys[1]
            root_data = data.get(root_key, {})
            return {root_key: {second_key: deepcopy(root_data.get(second_key, {}))}}
        return {root_key: deepcopy(data.get(root_key, {}))}

    @staticmethod
    def get_value(data: dict[str, Any], path: str, default: Any = None) -> Any:
        """Get a nested value from pool data using dot-notation path.

        Values for fields known to be numeric or boolean are normalised
        to native Python types regardless of how the Hayward cloud
        encoded them (e.g. ``"747"`` → ``747``, ``"1"`` → ``True``).
        Unmapped paths are returned unchanged. Missing keys and values
        that cannot be coerced both return ``default``; the latter also
        logs a WARNING. See :mod:`aioaquarite._coercion` for the
        path → type map.
        """
        if not data:
            return default
        keys = path.split(".")
        val: Any = data
        try:
            for key in keys:
                val = val[key]
        except (KeyError, TypeError):
            return default
        return _normalise(path, val, default)
