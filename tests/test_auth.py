"""Tests for AquariteAuth's client lifecycle (0.13.2 cleanup).

Since 0.13.2 the auth layer builds only the async Firestore client; the
synchronous ``google.cloud.firestore_v1.Client`` is never constructed,
and ``ensure_fresh_token()`` replaces ``get_client()`` for callers that
only need the token kept fresh and the rotation generation.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import aioaquarite
from aioaquarite.auth import AquariteAuth

_TOKEN_PAYLOAD = {
    "idToken": "id-token-1",
    "refreshToken": "refresh-token-1",
    "expiresIn": "3600",
}
_REFRESH_PAYLOAD = {
    "id_token": "id-token-2",
    "refresh_token": "refresh-token-2",
    "expires_in": "3600",
}


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status = 200
        self._payload = payload

    async def json(self, content_type: Any = None) -> dict[str, Any]:
        return self._payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeSession:
    """aiohttp session double: records posts, serves scripted payloads."""

    def __init__(self) -> None:
        self.posts: list[str] = []

    def post(self, url: str, **_kwargs: Any) -> _FakeResponse:
        self.posts.append(url)
        if "signInWithPassword" in url:
            return _FakeResponse(dict(_TOKEN_PAYLOAD))
        return _FakeResponse(dict(_REFRESH_PAYLOAD))


def _auth() -> tuple[AquariteAuth, _FakeSession]:
    session = _FakeSession()
    return AquariteAuth(session, "user@example.com", "pw"), session  # type: ignore[arg-type]


def test_synchronous_firestore_client_is_never_constructed() -> None:
    """The library builds only the async client — the sync one is gone."""

    async def _run() -> None:
        auth, _session = _auth()
        with (
            patch("aioaquarite.auth.Credentials"),
            patch("aioaquarite.auth.AsyncClient") as async_cls,
            patch("google.cloud.firestore_v1.Client") as sync_cls,
        ):
            await auth.ensure_fresh_token()
            await auth.get_async_client()
            auth.close()

        async_cls.assert_called_once()
        sync_cls.assert_not_called()

    asyncio.run(_run())


def test_ensure_fresh_token_authenticates_on_first_call() -> None:
    async def _run() -> None:
        auth, session = _auth()
        with (
            patch("aioaquarite.auth.Credentials"),
            patch("aioaquarite.auth.AsyncClient"),
        ):
            generation = await auth.ensure_fresh_token()

        assert any("signInWithPassword" in url for url in session.posts)
        assert generation == 1
        assert auth.tokens is not None and auth.tokens["idToken"] == "id-token-1"

    asyncio.run(_run())


def test_ensure_fresh_token_is_a_noop_while_the_token_is_fresh() -> None:
    async def _run() -> None:
        auth, session = _auth()
        with (
            patch("aioaquarite.auth.Credentials"),
            patch("aioaquarite.auth.AsyncClient"),
        ):
            first = await auth.ensure_fresh_token()
            posts = len(session.posts)
            second = await auth.ensure_fresh_token()

        assert second == first  # same generation, nothing rotated
        assert len(session.posts) == posts  # and no network traffic

    asyncio.run(_run())


def test_ensure_fresh_token_refreshes_an_expiring_token() -> None:
    async def _run() -> None:
        auth, session = _auth()
        with (
            patch("aioaquarite.auth.Credentials"),
            patch("aioaquarite.auth.AsyncClient") as async_cls,
        ):
            first = await auth.ensure_fresh_token()
            # Push the token into the refresh buffer.
            auth.expiry = datetime.datetime.now(datetime.UTC)
            second = await auth.ensure_fresh_token()

        assert second == first + 1  # the rotation bumped the generation
        assert auth.tokens is not None and auth.tokens["idToken"] == "id-token-2"
        assert session.posts[-1].startswith("https://securetoken")
        assert async_cls.call_count == 2  # a fresh client per rotation

    asyncio.run(_run())


def test_close_closes_the_async_client_and_is_idempotent() -> None:
    async def _run() -> None:
        auth, _session = _auth()
        fake_client = MagicMock()
        with (
            patch("aioaquarite.auth.Credentials"),
            patch("aioaquarite.auth.AsyncClient", return_value=fake_client),
        ):
            await auth.ensure_fresh_token()

        auth.close()
        fake_client.close.assert_called_once()
        auth.close()  # second close: no-op, no raise
        fake_client.close.assert_called_once()

    asyncio.run(_run())


def test_watch_reexport_is_gone() -> None:
    """The vestigial sync-Watch re-export was for the archived hayward-ha;
    its import was also the last place the library loaded Google's
    thread-based watch module."""
    with pytest.raises(ImportError):
        from aioaquarite import Watch  # noqa: F401

    for name in aioaquarite.__all__:
        assert getattr(aioaquarite, name, None) is not None, name
    assert "Watch" not in aioaquarite.__all__


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
