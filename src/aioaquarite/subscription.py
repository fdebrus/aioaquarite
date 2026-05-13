"""Resilient pool subscription that owns token refresh and reconnect.

Wraps :meth:`AquariteClient.subscribe_pool` with a supervisor task that:

* refreshes the auth token before it expires and resubscribes the watch
  when a new token is minted,
* periodically health-checks the connection and reconnects on any
  exception other than :class:`asyncio.CancelledError`,
* applies exponential backoff (default 10s → 600s) between reconnect
  attempts and resets the delay on success.

The user callback is invoked from the Firestore background thread (same
threading model as :meth:`AquariteClient.subscribe_pool`). Callers running
on an asyncio loop should hand data back via ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from google.cloud.firestore_v1.watch import Watch

    from .client import AquariteClient

_LOGGER = logging.getLogger(__name__)

DEFAULT_INITIAL_BACKOFF = 10.0
DEFAULT_MAX_BACKOFF = 600.0
DEFAULT_HEALTH_CHECK_INTERVAL = 60.0


class ResilientPoolSubscription:
    """A pool subscription that auto-refreshes tokens and reconnects."""

    def __init__(
        self,
        client: AquariteClient,
        pool_id: str,
        callback: Callable[[dict[str, Any]], None],
        *,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        health_check_interval: float | None = DEFAULT_HEALTH_CHECK_INTERVAL,
    ) -> None:
        self._client = client
        self._pool_id = pool_id
        self._callback = callback
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._health_check_interval = health_check_interval
        self._watch: Watch | None = None
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def pool_id(self) -> str:
        """The pool document ID this subscription is bound to."""
        return self._pool_id

    async def _start(self) -> None:
        """Perform the initial subscribe and launch the supervisor."""
        await self._do_subscribe()
        self._task = asyncio.create_task(
            self._supervise(), name=f"aioaquarite-sub-{self._pool_id}"
        )

    async def _do_subscribe(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._watch = await self._client.subscribe_pool(
                self._pool_id, self._callback
            )

    async def _do_unsubscribe(self) -> None:
        async with self._lock:
            watch = self._watch
            self._watch = None
        if watch is not None:
            await asyncio.to_thread(watch.unsubscribe)

    async def _resubscribe(self, reason: str) -> None:
        _LOGGER.debug(
            "Resubscribing pool %s (reason: %s)", self._pool_id, reason
        )
        await self._do_unsubscribe()
        await self._do_subscribe()

    def _next_sleep(self) -> float:
        auth = self._client.auth
        sleep_for = auth.calculate_sleep_duration()
        if self._health_check_interval is not None:
            sleep_for = min(sleep_for, self._health_check_interval)
        return sleep_for

    async def _supervise(self) -> None:
        backoff = self._initial_backoff
        auth = self._client.auth
        while True:
            try:
                await asyncio.sleep(self._next_sleep())

                if auth.is_token_expiring():
                    _LOGGER.debug(
                        "Token expiring, refreshing (pool %s)", self._pool_id
                    )
                _, refreshed = await auth.get_client()
                if refreshed:
                    await self._resubscribe("token refreshed")
                backoff = self._initial_backoff
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 — match HA's broad catch
                _LOGGER.warning(
                    "Pool %s connection issue: %s; reconnecting in %.1fs",
                    self._pool_id,
                    err,
                    backoff,
                )
                try:
                    await asyncio.sleep(backoff)
                    await self._resubscribe("after connection error")
                    backoff = self._initial_backoff
                except asyncio.CancelledError:
                    raise
                except Exception as err2:  # noqa: BLE001
                    _LOGGER.warning(
                        "Pool %s reconnect failed: %s; will retry",
                        self._pool_id,
                        err2,
                    )
                    backoff = min(backoff * 2, self._max_backoff)

    async def aclose(self) -> None:
        """Stop the subscription. Idempotent; safe at any setup stage."""
        if self._closed:
            return
        self._closed = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._do_unsubscribe()
