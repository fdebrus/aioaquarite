"""Tests for the 0.14 beta's opt-in experiments.

Each experiment reproduces something the first-party web client has been
observed to send (see PROTOCOL.md) so it can be measured on a real
controller. They are additive: the first tests here pin that the default
write path is byte-for-byte what it was.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aioaquarite import _pending
from aioaquarite.client import AquariteClient
from aioaquarite.const import HAYWARD_REST_API
from aioaquarite.exceptions import CommandError, ConnectionError

POOL_ID = "pool-1"
POOL_DATA: dict[str, Any] = {
    "wifi": "GW-1",
    "main": {"localTime": 1, "temperature": 25.5, "version": 825},
    "light": {"status": 0, "mode": 0},
    "hidro": {"cloration_enabled": 0, "level": 50},
    "relays": {
        "relay1": {"info": {"onoff": 0, "name": "aux"}, "key": 1},
        "relay2": {"info": {"onoff": 1, "name": "other"}, "key": 2},
    },
}


class _Response:
    def __init__(self, status: int = 200, text: str = "") -> None:
        self.status = status
        self._text = text

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def text(self) -> str:
        return self._text


class _Session:
    def __init__(self, response: _Response | None = None, error: Exception | None = None) -> None:
        self.response = response or _Response()
        self.error = error
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, *, json: Any = None, headers: Any = None, timeout: Any = None) -> _Response:
        if self.error is not None:
            raise self.error
        self.posts.append({"url": url, "json": json, "headers": headers})
        return self.response


def _client(session: _Session, **kwargs: Any) -> AquariteClient:
    auth = MagicMock()
    auth._session = session
    auth.ensure_fresh_token = AsyncMock(return_value=0)
    auth.tokens = {"idToken": "id-token"}
    client = AquariteClient(auth, **kwargs)
    client.set_pool_data(POOL_ID, json.loads(json.dumps(POOL_DATA)))
    return client


def _changes(post: dict[str, Any]) -> Any:
    return json.loads(post["json"]["changes"])


# ── the default path is unchanged ─────────────────────────────────────────


def test_default_write_mode_is_branch_and_unchanged() -> None:
    session = _Session()
    client = _client(session)
    assert client.write_mode == "branch"

    asyncio.run(client.set_value(POOL_ID, "light.status", 1))

    (post,) = session.posts
    assert post["url"] == f"{HAYWARD_REST_API}/sendPoolCommand"
    # The historical envelope: exactly these keys, whole branch in changes.
    assert set(post["json"]) == {"gateway", "poolId", "operation", "changes", "source"}
    assert post["json"]["operation"] == "WRP"
    assert _changes(post) == {"light": {"status": 1, "mode": 0}}


def test_unknown_write_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="write_mode"):
        AquariteClient(MagicMock(), write_mode="partial")  # type: ignore[arg-type]


# ── experiment 1: fragment writes through sendEVOCommand ──────────────────


def test_fragment_mode_sends_only_the_changed_leaf_to_sendevocommand() -> None:
    session = _Session()
    client = _client(session, write_mode="fragment")

    asyncio.run(client.set_value(POOL_ID, "light.status", 1))

    (post,) = session.posts
    assert post["url"] == f"{HAYWARD_REST_API}sendEVOCommand"
    assert post["json"] == {
        "gateway": "GW-1",
        "poolId": POOL_ID,
        "operation": "WRP",
        "operationId": None,
        "changes": json.dumps({"light": {"status": 1}}),
        "pool": None,
        "source": "web",
    }
    assert post["headers"] == {"Authorization": "Bearer id-token"}


def test_fragment_mode_sends_deep_values_inside_their_second_level_object() -> None:
    """The web client never sends less than the complete second-level
    object for nested values — siblings ride along, other objects do not."""
    session = _Session()
    client = _client(session, write_mode="fragment")

    asyncio.run(client.set_value(POOL_ID, "relays.relay1.info.onoff", 1))

    assert _changes(session.posts[0]) == {
        "relays": {"relay1": {"info": {"onoff": 1, "name": "aux"}, "key": 1}}
    }


def test_fragment_mode_keeps_several_values_in_one_command() -> None:
    session = _Session()
    client = _client(session, write_mode="fragment")

    asyncio.run(client.set_values(POOL_ID, {"light.mode": 2, "light.status": 1}))

    (post,) = session.posts
    assert _changes(post) == {"light": {"mode": 2, "status": 1}}


def test_fragment_mode_still_derives_the_hidro_fields() -> None:
    session = _Session()
    client = _client(session, write_mode="fragment")

    asyncio.run(client.set_value(POOL_ID, "hidro.cloration_enabled", True))

    assert _changes(session.posts[0]) == {
        "hidro": {"cloration_enabled": 1, "reduction": 1, "disable": 1}
    }


def test_fragment_mode_records_pending_writes_like_branch_mode() -> None:
    """The reconciliation layer is independent of the write route."""

    async def _run() -> None:
        client = _client(_Session(), write_mode="fragment")
        await client.set_value(POOL_ID, "light.status", 1)
        # A stale pre-write read is overlaid with the acknowledged value.
        client._fetch_pool_document = AsyncMock(  # type: ignore[method-assign]
            return_value={"light": {"status": 0}}
        )
        data = await client.fetch_pool_data(POOL_ID)
        assert data["light"]["status"] == 1
        client._release_pending(POOL_ID)

    asyncio.run(_run())


# ── experiment 2: changeLedColor ──────────────────────────────────────────


def test_change_led_color_calls_the_dedicated_function() -> None:
    session = _Session()
    client = _client(session)

    asyncio.run(client.change_led_color(POOL_ID))

    (post,) = session.posts
    assert post["url"] == f"{HAYWARD_REST_API}changeLedColor"
    assert post["json"] == {"gateway": "GW-1", "poolId": POOL_ID, "type": "change"}


def test_change_led_color_needs_pool_data_for_the_gateway() -> None:
    client = AquariteClient(MagicMock())
    with pytest.raises(RuntimeError, match="Pool data not available"):
        asyncio.run(client.change_led_color("unknown-pool"))


# ── experiment 3: the PUP poke ────────────────────────────────────────────


def test_request_updates_sends_the_pup_poke() -> None:
    session = _Session()
    client = _client(session)

    asyncio.run(client.request_updates(POOL_ID))

    (post,) = session.posts
    assert post["url"] == f"{HAYWARD_REST_API}/sendPoolCommand"
    assert post["json"] == {
        "gateway": "GW-1",
        "poolId": POOL_ID,
        "operation": "PUP",
        "operationId": None,
        "changes": "10",
        "pool": None,
        "source": "web",
    }


# ── shared plumbing ───────────────────────────────────────────────────────


def test_failed_function_call_raises_and_logs_the_response_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _Session(_Response(status=400, text="unknown gateway"))
    client = _client(session)

    with caplog.at_level(logging.DEBUG, logger="aioaquarite.client"):
        with pytest.raises(CommandError):
            asyncio.run(client.change_led_color(POOL_ID))

    assert "changeLedColor failed, response body: 'unknown gateway'" in caplog.text


def test_function_call_wraps_transport_errors() -> None:
    client = _client(_Session(error=aiohttp.ClientError("boom")), write_mode="fragment")
    with pytest.raises(ConnectionError):
        asyncio.run(client.set_value(POOL_ID, "light.status", 1))


def test_snapshot_interval_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Each snapshot logs the time since the previous one, so the effect
    of a PUP poke on the controller's cadence can be read from the log."""
    clock = {"now": 100.0}
    delivered: list[dict[str, Any]] = []
    with patch.object(_pending, "monotonic", side_effect=lambda: clock["now"]):
        reconciler = _pending.PendingWriteReconciler(
            "pool test", AsyncMock(), lambda: None, delivered.append
        )
        with caplog.at_level(logging.DEBUG, logger="aioaquarite._pending"):
            reconciler.on_snapshot({"a": 1})
            clock["now"] = 111.2
            reconciler.on_snapshot({"a": 2})

    assert "pool test: snapshot 11.2 s after the previous one" in caplog.text
    assert caplog.text.count("after the previous one") == 1  # not on the first
    assert len(delivered) == 2
