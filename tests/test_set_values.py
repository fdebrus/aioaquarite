"""Tests for AquariteClient.set_values and the post-send cache update."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aioaquarite.client import AquariteClient
from aioaquarite.exceptions import CommandError

POOL_ID = "pool-1"

POOL_DATA: dict[str, Any] = {
    "wifi": "gateway-1",
    "light": {"status": 1, "mode": 1, "from": 79200, "to": 3600},
    "filtration": {"mode": 2, "interval1": {"from": 28800, "to": 36000}},
    "hidro": {"cloration_enabled": 0, "reduction": 0, "disable": 0},
    "relays": {
        "relay1": {"info": {"onoff": 0, "name": "aux1"}},
        "relay2": {"info": {"onoff": 1, "name": "aux2"}},
    },
}


def _make_client() -> AquariteClient:
    """Build a client with stubbed auth, seeded pool data, mocked transport."""
    auth = MagicMock()
    auth.get_client = AsyncMock(return_value=(MagicMock(), False))
    auth.tokens = {"idToken": "id-token"}
    client = AquariteClient(auth)
    client.set_pool_data(POOL_ID, json.loads(json.dumps(POOL_DATA)))
    client.send_command = AsyncMock()  # type: ignore[method-assign]
    return client


def _sent_changes(client: AquariteClient) -> dict[str, Any]:
    """Decode the changes JSON of the last sent command payload."""
    payload = client.send_command.await_args.args[0]  # type: ignore[attr-defined]
    return json.loads(payload["changes"])


def test_set_values_sends_all_fields_in_one_command() -> None:
    client = _make_client()

    asyncio.run(client.set_values(POOL_ID, {"light.mode": 0, "light.status": 0}))

    client.send_command.assert_awaited_once()  # type: ignore[attr-defined]
    changes = _sent_changes(client)
    assert changes["light"]["mode"] == 0
    assert changes["light"]["status"] == 0
    # Untouched fields of the branch ride along unchanged.
    assert changes["light"]["from"] == 79200


def test_set_values_updates_stored_pool_data_on_success() -> None:
    client = _make_client()

    asyncio.run(client.set_values(POOL_ID, {"light.mode": 0, "light.status": 0}))

    data = client.get_pool_data(POOL_ID)
    assert data is not None
    assert data["light"]["mode"] == 0
    assert data["light"]["status"] == 0


def test_sequential_set_value_calls_preserve_each_other() -> None:
    """Regression: the second command must carry the first write's value."""
    client = _make_client()

    asyncio.run(client.set_value(POOL_ID, "filtration.interval1.from", 21600))
    asyncio.run(client.set_value(POOL_ID, "filtration.interval1.to", 43200))

    changes = _sent_changes(client)
    assert changes["filtration"]["interval1"]["from"] == 21600
    assert changes["filtration"]["interval1"]["to"] == 43200


def test_set_values_failure_leaves_stored_pool_data_untouched() -> None:
    client = _make_client()
    client.send_command = AsyncMock(  # type: ignore[method-assign]
        side_effect=CommandError("boom")
    )

    with pytest.raises(CommandError):
        asyncio.run(client.set_values(POOL_ID, {"light.mode": 0}))

    data = client.get_pool_data(POOL_ID)
    assert data is not None
    assert data["light"]["mode"] == 1


def test_set_values_rejects_empty_updates() -> None:
    client = _make_client()
    with pytest.raises(ValueError):
        asyncio.run(client.set_values(POOL_ID, {}))


def test_set_values_rejects_cross_branch_updates() -> None:
    client = _make_client()
    with pytest.raises(ValueError):
        asyncio.run(
            client.set_values(POOL_ID, {"light.mode": 0, "filtration.mode": 1})
        )


def test_set_values_rejects_mixed_depth_same_root() -> None:
    """A deep path sends a two-level sub-branch, so it cannot ride with a shallow one."""
    client = _make_client()
    with pytest.raises(ValueError):
        asyncio.run(
            client.set_values(
                POOL_ID,
                {"relays.relay1.info.onoff": 1, "relays.relay2.info.onoff": 0},
            )
        )


def test_set_values_deep_path_sends_two_level_sub_branch() -> None:
    client = _make_client()

    asyncio.run(
        client.set_values(
            POOL_ID,
            {"relays.relay1.info.onoff": 1, "relays.relay1.info.name": "spa"},
        )
    )

    changes = _sent_changes(client)
    assert changes == {"relays": {"relay1": {"info": {"onoff": 1, "name": "spa"}}}}
    data = client.get_pool_data(POOL_ID)
    assert data is not None
    assert data["relays"]["relay1"]["info"]["onoff"] == 1
    assert data["relays"]["relay2"]["info"]["onoff"] == 1


def test_cloration_enabled_expands_side_fields() -> None:
    client = _make_client()

    asyncio.run(client.set_value(POOL_ID, "hidro.cloration_enabled", True))

    changes = _sent_changes(client)
    assert changes["hidro"]["cloration_enabled"] == 1
    assert changes["hidro"]["reduction"] == 1
    assert changes["hidro"]["disable"] == 1
    data = client.get_pool_data(POOL_ID)
    assert data is not None
    assert data["hidro"]["cloration_enabled"] == 1
    assert data["hidro"]["reduction"] == 1
    assert data["hidro"]["disable"] == 1


def test_set_values_requires_pool_data() -> None:
    client = _make_client()
    with pytest.raises(RuntimeError):
        asyncio.run(client.set_values("unknown-pool", {"light.mode": 0}))


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


def test_concurrent_branch_writes_do_not_revert_each_other() -> None:
    """Two writers on one branch must not each send a pre-other snapshot.

    The payload is built from the stored document, so without
    serialisation both would snapshot before either updated the cache and
    the second command would revert the first field.
    """

    async def _run() -> None:
        client = _make_client()
        sent: list[dict[str, Any]] = []
        first_in_flight = asyncio.Event()
        release_first = asyncio.Event()

        async def _send(payload: dict[str, Any]) -> None:
            changes = json.loads(payload["changes"])
            if not sent:
                sent.append(changes)
                first_in_flight.set()
                await release_first.wait()
                return
            sent.append(changes)

        client.send_command = _send  # type: ignore[method-assign]

        mode = asyncio.create_task(client.set_values(POOL_ID, {"light.mode": 0}))
        await first_in_flight.wait()

        schedule = asyncio.create_task(
            client.set_values(POOL_ID, {"light.from": 21600})
        )
        # The second writer must be waiting, not building a stale payload.
        for _ in range(5):
            await asyncio.sleep(0)
        assert len(sent) == 1

        release_first.set()
        await asyncio.gather(mode, schedule)

        assert len(sent) == 2
        # The second command carries the first writer's field, not the
        # value the branch held before it.
        assert sent[1]["light"]["mode"] == 0
        assert sent[1]["light"]["from"] == 21600

    asyncio.run(_run())
