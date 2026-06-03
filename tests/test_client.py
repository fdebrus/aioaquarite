"""Tests for AquariteClient.send_command transport-error wrapping."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from aioaquarite.client import AquariteClient
from aioaquarite.exceptions import ConnectionError


def _make_client(post: Any) -> AquariteClient:
    """Build an AquariteClient whose auth is fully stubbed.

    ``get_client`` returns a dummy client and the session exposes the
    supplied ``post`` callable; tokens are pre-populated so the header
    build in ``send_command`` succeeds.
    """
    auth = MagicMock()
    auth.get_client = AsyncMock(return_value=(MagicMock(), False))
    auth.tokens = {"idToken": "id-token"}
    auth._session = MagicMock()
    auth._session.post = post
    return AquariteClient(auth)


def test_send_command_wraps_aiohttp_client_error() -> None:
    def _post(*_args: Any, **_kwargs: Any) -> Any:
        raise aiohttp.ClientError("boom")

    client = _make_client(_post)
    with pytest.raises(ConnectionError):
        asyncio.run(client.send_command({"foo": "bar"}))


def test_send_command_wraps_timeout() -> None:
    def _post(*_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.TimeoutError()

    client = _make_client(_post)
    with pytest.raises(ConnectionError):
        asyncio.run(client.send_command({"foo": "bar"}))


def test_send_command_wraps_auth_transport_error() -> None:
    client = _make_client(MagicMock())
    client._auth.get_client = AsyncMock(
        side_effect=aiohttp.ClientError("refresh failed")
    )
    with pytest.raises(ConnectionError):
        asyncio.run(client.send_command({"foo": "bar"}))


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
