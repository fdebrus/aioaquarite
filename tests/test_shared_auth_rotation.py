"""Every subscription on an account must survive a token rotation.

One :class:`AquariteAuth` is shared by every subscription belonging to an
account: one per pool, plus the ``users/{uid}.pools`` listener. A rotation
used to be reported as a per-call "a refresh just happened" boolean, so only
whichever supervisor won the auth lock learned about it and resubscribed.
The others kept a watch bound to credentials that were about to expire, and
went silently dead the moment they did — no exception, no health transition,
just entities frozen on their last values until some later rotation happened
to be won by that supervisor.

These tests pin the counter-based contract that replaced the boolean.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from aioaquarite.auth import AquariteAuth
from aioaquarite.subscription import (
    ResilientPoolSubscription,
    ResilientUserPoolsSubscription,
)


from ._fakes import FakeTaskWatch as _FakeWatch


class _SharedAuth:
    """Mirrors AquariteAuth._ensure_fresh_clients, lock and all.

    The rotation is performed by exactly one caller, but the generation it
    produces is readable by every caller — that is the whole point.
    """

    def __init__(self) -> None:
        self.expiring = False
        self.token_generation = 0
        self.rotations = 0
        self._rotate_pending = False
        self._lock = asyncio.Lock()

    def rotate_on_next_check(self) -> None:
        self._rotate_pending = True
        self.expiring = True

    def is_token_expiring(self) -> bool:
        return self.expiring

    def calculate_sleep_duration(self) -> float:
        return 0.005

    async def get_client(self) -> tuple[object, int]:
        async with self._lock:
            if self._rotate_pending:
                self._rotate_pending = False
                self.expiring = False
                self.token_generation += 1
                self.rotations += 1
            return object(), self.token_generation


class _SharedClient:
    """One client per account, handing the same auth to every subscription."""

    def __init__(self, auth: _SharedAuth) -> None:
        self.auth = auth
        self.pool_watches: dict[str, list[_FakeWatch]] = {}
        self.user_pool_watches: list[_FakeWatch] = []

    async def subscribe_pool(
        self, pool_id: str, callback: Callable[[dict[str, Any]], None]
    ) -> _FakeWatch:
        watch = _FakeWatch()
        self.pool_watches.setdefault(pool_id, []).append(watch)
        return watch

    async def subscribe_user_pools(
        self, callback: Callable[[list[str]], None]
    ) -> _FakeWatch:
        watch = _FakeWatch()
        self.user_pool_watches.append(watch)
        return watch


async def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


def test_concurrent_callers_all_observe_one_rotation() -> None:
    """The loser of the auth lock must still learn that the token rotated.

    This is the contract the boolean broke: it handed "a refresh happened"
    to whichever caller performed it and False to everyone else, so a pool
    listener could hold credentials it never knew had been replaced.
    """

    async def _run() -> None:
        auth = AquariteAuth(MagicMock(), "user@example.com", "hunter2")
        auth.tokens = {"idToken": "t0", "refreshToken": "rt", "expiresIn": "3600"}
        # Already inside the refresh buffer, so the next check rotates.
        auth.expiry = datetime.datetime.now(datetime.UTC)

        refreshes = 0

        async def _fake_refresh_token() -> None:
            nonlocal refreshes
            refreshes += 1
            auth.expiry = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
                hours=1
            )
            auth._update_firestore_client()

        auth.refresh_token = _fake_refresh_token  # type: ignore[method-assign]

        with (
            patch("aioaquarite.auth.Credentials"),
            patch("aioaquarite.auth.FirestoreClient"),
            patch("aioaquarite.auth.AsyncFirestoreClient"),
        ):
            # Stand the clients up first so _ensure_fresh_clients skips sign-in.
            auth._update_firestore_client()
            before = auth.token_generation

            first, second = await asyncio.gather(
                auth.get_client(), auth.get_client()
            )

        assert refreshes == 1, "only one caller should perform the rotation"
        assert first[1] == second[1], "both callers must read the same generation"
        assert first[1] != before, "and it must differ from the pre-rotation value"

    asyncio.run(_run())


@pytest.mark.parametrize(
    "pool_count",
    [1, 2, 3],
    ids=["one-pool", "two-pools", "three-pools"],
)
def test_one_rotation_resubscribes_every_subscription(pool_count: int) -> None:
    """A single rotation must rebuild every watch, not just the winner's."""

    async def _run() -> None:
        auth = _SharedAuth()
        client = _SharedClient(auth)
        pool_ids = [f"pool-{i}" for i in range(pool_count)]

        pools = [
            ResilientPoolSubscription(
                client,  # type: ignore[arg-type]
                pool_id,
                lambda _data: None,
                initial_backoff=0.01,
                max_backoff=0.02,
                health_check_interval=0.005,
            )
            for pool_id in pool_ids
        ]
        user_pools = ResilientUserPoolsSubscription(
            client,  # type: ignore[arg-type]
            lambda _ids: None,
            initial_backoff=0.01,
            max_backoff=0.02,
            health_check_interval=0.005,
        )
        for sub in (*pools, user_pools):
            await sub._start()

        assert all(len(client.pool_watches[p]) == 1 for p in pool_ids)
        assert len(client.user_pool_watches) == 1

        auth.rotate_on_next_check()

        await _wait_for(
            lambda: all(len(client.pool_watches[p]) == 2 for p in pool_ids)
            and len(client.user_pool_watches) == 2
        )

        # Exactly one rotation happened, however many supervisors observed it.
        assert auth.rotations == 1
        # The stale watches were released rather than leaked.
        assert all(client.pool_watches[p][0].unsubscribed for p in pool_ids)
        assert client.user_pool_watches[0].unsubscribed

        for sub in (*pools, user_pools):
            await sub.aclose()

    asyncio.run(_run())


def test_settled_generation_does_not_resubscribe_again() -> None:
    """After catching up, a subscription must go quiet until the next rotation.

    Guards the other failure mode of a counter: recording the wrong value and
    resubscribing on every tick, which would hammer Firestore.
    """

    async def _run() -> None:
        auth = _SharedAuth()
        client = _SharedClient(auth)
        sub = ResilientPoolSubscription(
            client,  # type: ignore[arg-type]
            "pool-a",
            lambda _data: None,
            initial_backoff=0.01,
            max_backoff=0.02,
            health_check_interval=0.005,
        )
        await sub._start()

        auth.rotate_on_next_check()
        await _wait_for(lambda: len(client.pool_watches["pool-a"]) == 2)

        # Many supervisor ticks pass with no rotation; the count must hold.
        await asyncio.sleep(0.2)
        assert len(client.pool_watches["pool-a"]) == 2

        await sub.aclose()

    asyncio.run(_run())
