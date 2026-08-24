"""Async API client for the Hayward Aquarite pool system."""

import asyncio
import json
import logging
from copy import deepcopy
from typing import Any, Callable, MutableMapping

import aiohttp
from google.cloud.firestore_v1.watch import Watch

from ._coercion import normalise as _normalise
from .auth import AquariteAuth
from .const import HAYWARD_REST_API
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

    async def get_pools(self) -> dict[str, str]:
        """Fetch all pools for the authenticated user.

        Returns a mapping of pool_id -> pool_name.
        """
        client, _ = await self._auth.get_client()
        assert self._auth.tokens is not None
        user_doc = await asyncio.to_thread(
            client.collection("users")
            .document(self._auth.tokens["localId"])
            .get
        )
        user_dict = user_doc.to_dict() or {}  # type: ignore[union-attr]

        pools: dict[str, str] = {}
        for pool_id in user_dict.get("pools", []):
            pool_doc = await asyncio.to_thread(
                client.collection("pools").document(pool_id).get
            )
            pool_dict = pool_doc.to_dict()  # type: ignore[union-attr]
            if pool_dict:
                name = pool_dict.get("form", {}).get("name", "Unknown")
                if "names" in pool_dict.get("form", {}) and pool_dict["form"]["names"]:
                    name = pool_dict["form"]["names"][0].get("name", name)
                pools[pool_id] = name
        return pools

    async def fetch_pool_data(self, pool_id: str) -> dict[str, Any]:
        """Fetch the full pool document from Firestore."""
        client, _ = await self._auth.get_client()
        pool_doc = await asyncio.to_thread(
            client.collection("pools").document(pool_id).get
        )
        data: dict[str, Any] = pool_doc.to_dict() or {}  # type: ignore[union-attr]
        self._pool_data[pool_id] = data
        return data

    async def subscribe_pool(
        self, pool_id: str, callback: Callable[[dict[str, Any]], None]
    ) -> Watch:
        """Subscribe to real-time Firestore updates for a pool.

        Args:
            pool_id: The pool document ID.
            callback: Called with the pool data dict on each snapshot.

        Returns:
            A Watch object; call ``unsubscribe()`` on it to stop listening.
        """
        client, _ = await self._auth.get_client()
        doc_ref = client.collection("pools").document(pool_id)

        def on_snapshot(
            doc_snapshot: list[Any], changes: list[Any], read_time: Any
        ) -> None:
            for doc in doc_snapshot:
                data: dict[str, Any] = doc.to_dict()
                self._pool_data[pool_id] = data
                callback(data)

        watch: Watch = await asyncio.to_thread(doc_ref.on_snapshot, on_snapshot)
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
    ) -> ResilientPoolSubscription:
        """Subscribe to a pool with automatic token refresh and reconnect.

        Returns a :class:`ResilientPoolSubscription` handle; call
        ``await handle.aclose()`` to stop the subscription. The callback is
        invoked from the Firestore background thread — see
        :class:`ResilientPoolSubscription` for details.
        """
        sub = ResilientPoolSubscription(
            self,
            pool_id,
            callback,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            health_check_interval=health_check_interval,
        )
        await sub._start()
        return sub

    async def subscribe_user_pools(
        self, callback: Callable[[list[str]], None]
    ) -> Watch:
        """Subscribe to the user document's ``pools`` list.

        The callback receives the current ``list[str]`` of pool IDs every
        time ``users/{uid}`` changes. Use this to detect pool additions
        or removals in the Hayward app without polling
        :meth:`get_pools`.

        Returns a Watch object; call ``unsubscribe()`` on it to stop
        listening. The callback runs on the Firestore background thread.
        """
        client, _ = await self._auth.get_client()
        assert self._auth.tokens is not None
        doc_ref = client.collection("users").document(
            self._auth.tokens["localId"]
        )

        def on_snapshot(
            doc_snapshot: list[Any], changes: list[Any], read_time: Any
        ) -> None:
            for doc in doc_snapshot:
                data: dict[str, Any] = doc.to_dict() or {}
                callback(list(data.get("pools", [])))

        watch: Watch = await asyncio.to_thread(doc_ref.on_snapshot, on_snapshot)
        _LOGGER.debug("Firestore user-pools subscription active")
        return watch

    async def subscribe_user_pools_resilient(
        self,
        callback: Callable[[list[str]], None],
        *,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        health_check_interval: float | None = DEFAULT_HEALTH_CHECK_INTERVAL,
    ) -> ResilientUserPoolsSubscription:
        """Subscribe to the user's pool list with token refresh and reconnect.

        Returns a :class:`ResilientUserPoolsSubscription` handle; call
        ``await handle.aclose()`` to stop the subscription. The callback
        runs on the Firestore background thread — see
        :class:`ResilientUserPoolsSubscription` for details.
        """
        sub = ResilientUserPoolsSubscription(
            self,
            callback,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            health_check_interval=health_check_interval,
        )
        await sub._start()
        return sub

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
                _LOGGER.debug("Command sent. Status: %s", response.status)
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
        """
        if not updates:
            raise ValueError("updates must not be empty")
        signatures = {self._branch_signature(path) for path in updates}
        if len(signatures) != 1:
            raise ValueError(
                "updates must target a single command branch, got "
                f"{sorted(signatures)}"
            )

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
        _LOGGER.debug(
            "set_values updates=%s changes=%s",
            effective,
            json.dumps(current_config, indent=2, default=str),
        )
        await self.send_command(payload)

        # Mirror the accepted changes into the stored pool data so the
        # next command payload is built from the state the cloud now has.
        for path, value in effective.items():
            self._set_in_dict(pool_data, path, value)

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
        """Set a value in a nested dict using dot-notation path."""
        keys = path.split(".")
        for key in keys[:-1]:
            data_dict = data_dict.setdefault(key, {})
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
