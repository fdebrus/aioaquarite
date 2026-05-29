"""Tests for AquariteClient cloud-function helpers: get_pool_stats,
get_server_date."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest

from aioaquarite.auth import AquariteAuth
from aioaquarite.client import AquariteClient
from aioaquarite.const import HAYWARD_REST_API
from aioaquarite.exceptions import CommandError


# ── fakes ──────────────────────────────────────────────────────────────


class _FakeResponse:
    """Minimal async context manager mimicking aiohttp's ClientResponse."""

    def __init__(
        self,
        *,
        status: int = 200,
        payload: Any = None,
    ) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def json(self) -> Any:
        return self._payload


class _FakeSession:
    """Captures the POST/GET call args and returns a canned response."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        # Recorded args for assertions:
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        json: Any = None,
        headers: dict[str, str] | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> _FakeResponse:
        self.post_calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return self._response

    def get(
        self,
        url: str,
        *,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> _FakeResponse:
        self.get_calls.append({"url": url, "timeout": timeout})
        return self._response


def _make_auth(session: Any, *, with_token: bool = True) -> AquariteAuth:
    """Build an AquariteAuth wrapping a fake session, optionally
    pre-populated with a token so request-side tests can run without
    going through the live sign-in flow."""
    auth = AquariteAuth(session, "user@example.com", "hunter2")
    if with_token:
        auth.tokens = {"idToken": "fake-id-token", "refreshToken": "rt", "expiresIn": "3600"}
    return auth


# ── get_pool_stats ─────────────────────────────────────────────────────


SAMPLE_PH_PAYLOAD: list[list[dict[str, Any]]] = [
    [
        {"field": 885, "seconds": 1777447486},
        {"field": 886, "seconds": 1777448147},
        {"field": 884, "seconds": 1777448807},
    ]
]


def test_get_pool_stats_posts_expected_payload_and_returns_decoded() -> None:
    """Happy path: correct URL, body, headers; raw decoded payload returned."""

    async def _run() -> None:
        response = _FakeResponse(status=200, payload=SAMPLE_PH_PAYLOAD)
        session = _FakeSession(response)
        auth = _make_auth(session)
        client = AquariteClient(auth)

        result = await client.get_pool_stats("pool-uuid-1", "ph", 14)

        assert result == SAMPLE_PH_PAYLOAD
        assert len(session.post_calls) == 1
        call = session.post_calls[0]
        assert call["url"] == f"{HAYWARD_REST_API}getStats"
        assert call["json"] == {"uuid": "pool-uuid-1", "type": "ph", "period": 14}
        assert call["headers"] == {
            "Authorization": "Bearer fake-id-token",
            "Content-Type": "application/json",
        }
        assert call["timeout"] is not None

    asyncio.run(_run())


def test_get_pool_stats_raises_command_error_on_http_error() -> None:
    async def _run() -> None:
        response = _FakeResponse(status=500, payload=None)
        session = _FakeSession(response)
        auth = _make_auth(session)
        client = AquariteClient(auth)

        with pytest.raises(CommandError, match="500"):
            await client.get_pool_stats("pool-uuid-1", "ph", 30)

    asyncio.run(_run())


def test_get_pool_stats_raises_runtime_error_when_unauthenticated() -> None:
    async def _run() -> None:
        session = _FakeSession(_FakeResponse(status=200, payload=[[]]))
        auth = _make_auth(session, with_token=False)
        client = AquariteClient(auth)

        with pytest.raises(RuntimeError, match="authenticate"):
            await client.get_pool_stats("pool-uuid-1", "ph", 30)

        # Pre-check failed before the network was touched.
        assert session.post_calls == []

    asyncio.run(_run())


def test_get_pool_stats_handles_unknown_type_payload() -> None:
    """The backend returns 200 with field-less point dicts for metric
    types it doesn't recognise. The library should return them as-is —
    callers can decide how to interpret missing ``field`` entries."""

    async def _run() -> None:
        unknown_payload: list[list[dict[str, Any]]] = [
            [{"seconds": 1777447486}, {"seconds": 1777448147}]
        ]
        response = _FakeResponse(status=200, payload=unknown_payload)
        session = _FakeSession(response)
        auth = _make_auth(session)
        client = AquariteClient(auth)

        result = await client.get_pool_stats("pool-uuid-1", "future_metric", 30)

        assert result == unknown_payload

    asyncio.run(_run())


# ── get_server_date ────────────────────────────────────────────────────


def test_get_server_date_returns_decoded_dict() -> None:
    async def _run() -> None:
        payload = {"date": "260529"}
        response = _FakeResponse(status=200, payload=payload)
        session = _FakeSession(response)
        auth = _make_auth(session, with_token=False)  # endpoint is unauthenticated
        client = AquariteClient(auth)

        result = await client.get_server_date()

        assert result == payload
        assert len(session.get_calls) == 1
        assert session.get_calls[0]["url"] == f"{HAYWARD_REST_API}getServerDate"

    asyncio.run(_run())


def test_get_server_date_raises_on_http_error() -> None:
    async def _run() -> None:
        response = _FakeResponse(status=503, payload=None)
        session = _FakeSession(response)
        auth = _make_auth(session, with_token=False)
        client = AquariteClient(auth)

        with pytest.raises(CommandError, match="503"):
            await client.get_server_date()

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
