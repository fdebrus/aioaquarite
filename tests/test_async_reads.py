"""Tests for the natively-async Firestore read path.

``get_pools`` and ``fetch_pool_data`` used to wrap the synchronous
Firestore client in ``asyncio.to_thread``. They now await the async
client directly, so these tests assert both the returned data *and*
that no thread dispatch happens — the latter is what stops the
library silently regressing to a thread-wrapped "async" facade.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aioaquarite.client import AquariteClient

POOL_ID = "pool-1"
LOCAL_ID = "user-1"


class _FakeDoc:
    """Stand-in for a DocumentSnapshot."""

    def __init__(self, payload: dict[str, Any] | None) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any] | None:
        return self._payload


class _FakeDocRef:
    """Async document reference: ``get()`` is a coroutine, as upstream's is."""

    def __init__(self, payload: dict[str, Any] | None) -> None:
        self._payload = payload

    async def get(self) -> _FakeDoc:
        return _FakeDoc(self._payload)


class _FakeCollection:
    def __init__(self, docs: dict[str, dict[str, Any] | None]) -> None:
        self._docs = docs

    def document(self, doc_id: str) -> _FakeDocRef:
        return _FakeDocRef(self._docs.get(doc_id))


class _FakeAsyncClient:
    """Minimal async Firestore client double."""

    def __init__(self, collections: dict[str, dict[str, Any]]) -> None:
        self._collections = collections

    def collection(self, name: str) -> _FakeCollection:
        return _FakeCollection(self._collections.get(name, {}))


def _make_client(collections: dict[str, dict[str, Any]]) -> AquariteClient:
    """Build a client whose auth serves the fake async Firestore client."""
    auth = MagicMock()
    auth.get_async_client = AsyncMock(return_value=_FakeAsyncClient(collections))
    # The sync accessor must not be reached by the read paths.
    auth.get_client = AsyncMock(side_effect=AssertionError("sync client used"))
    auth.tokens = {"idToken": "id-token", "localId": LOCAL_ID}
    return AquariteClient(auth)


# ── fetch_pool_data ────────────────────────────────────────────────────


def test_fetch_pool_data_returns_document() -> None:
    client = _make_client(
        {"pools": {POOL_ID: {"main": {"temperature": 25.5}}}}
    )

    data = asyncio.run(client.fetch_pool_data(POOL_ID))

    assert data == {"main": {"temperature": 25.5}}


def test_fetch_pool_data_populates_the_cache() -> None:
    """The command payload builder reads this cache, so it must be filled."""
    client = _make_client({"pools": {POOL_ID: {"light": {"status": 1}}}})

    asyncio.run(client.fetch_pool_data(POOL_ID))

    assert client.get_pool_data(POOL_ID) == {"light": {"status": 1}}


def test_fetch_pool_data_missing_document_returns_empty() -> None:
    client = _make_client({"pools": {}})

    assert asyncio.run(client.fetch_pool_data("nope")) == {}


def test_fetch_pool_data_does_not_dispatch_to_a_thread() -> None:
    """Regression guard for the async-dependency rule."""
    client = _make_client({"pools": {POOL_ID: {"main": {}}}})

    with patch("asyncio.to_thread", new_callable=AsyncMock) as to_thread:
        asyncio.run(client.fetch_pool_data(POOL_ID))

    to_thread.assert_not_called()


def test_fetch_pool_data_uses_the_async_client() -> None:
    client = _make_client({"pools": {POOL_ID: {"main": {}}}})

    asyncio.run(client.fetch_pool_data(POOL_ID))

    client.auth.get_async_client.assert_awaited_once()


# ── get_pools ──────────────────────────────────────────────────────────


def _pools_fixture(
    pool_ids: list[str], pools: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        "users": {LOCAL_ID: {"pools": pool_ids}},
        "pools": pools,
    }


def test_get_pools_maps_id_to_name() -> None:
    client = _make_client(
        _pools_fixture([POOL_ID], {POOL_ID: {"form": {"name": "Backyard"}}})
    )

    assert asyncio.run(client.get_pools()) == {POOL_ID: "Backyard"}


def test_get_pools_prefers_the_names_list() -> None:
    """The Hayward app writes the display name into form.names[0]."""
    client = _make_client(
        _pools_fixture(
            [POOL_ID],
            {POOL_ID: {"form": {"name": "Backyard", "names": [{"name": "Spa"}]}}},
        )
    )

    assert asyncio.run(client.get_pools()) == {POOL_ID: "Spa"}


def test_get_pools_falls_back_to_unknown() -> None:
    client = _make_client(_pools_fixture([POOL_ID], {POOL_ID: {"form": {}}}))

    assert asyncio.run(client.get_pools()) == {POOL_ID: "Unknown"}


def test_get_pools_skips_missing_pool_documents() -> None:
    client = _make_client(
        _pools_fixture(
            [POOL_ID, "ghost"], {POOL_ID: {"form": {"name": "Backyard"}}}
        )
    )

    assert asyncio.run(client.get_pools()) == {POOL_ID: "Backyard"}


def test_get_pools_with_no_pools_returns_empty() -> None:
    client = _make_client(_pools_fixture([], {}))

    assert asyncio.run(client.get_pools()) == {}


def test_get_pools_does_not_dispatch_to_a_thread() -> None:
    """Regression guard: both the user doc and each pool doc read natively."""
    client = _make_client(
        _pools_fixture(
            [POOL_ID, "pool-2"],
            {
                POOL_ID: {"form": {"name": "Backyard"}},
                "pool-2": {"form": {"name": "Spa"}},
            },
        )
    )

    with patch("asyncio.to_thread", new_callable=AsyncMock) as to_thread:
        result = asyncio.run(client.get_pools())

    assert result == {POOL_ID: "Backyard", "pool-2": "Spa"}
    to_thread.assert_not_called()


def test_get_pools_uses_the_async_client() -> None:
    client = _make_client(_pools_fixture([], {}))

    asyncio.run(client.get_pools())

    client.auth.get_async_client.assert_awaited_once()


# ── the listener still needs the synchronous client ────────────────────


def test_subscribe_pool_still_uses_the_sync_client_in_a_thread() -> None:
    """Documented upstream limitation, asserted so it stays deliberate.

    AsyncDocumentReference.on_snapshot raises NotImplementedError in
    google-cloud-firestore, so the real-time listener has to run the
    synchronous client off-loop.
    """
    watch = MagicMock()
    sync_client = MagicMock()
    sync_client.collection.return_value.document.return_value.on_snapshot = (
        MagicMock(return_value=watch)
    )

    auth = MagicMock()
    auth.get_client = AsyncMock(return_value=(sync_client, False))
    auth.get_async_client = AsyncMock(
        side_effect=AssertionError("async client cannot serve on_snapshot")
    )
    auth.tokens = {"idToken": "id-token", "localId": LOCAL_ID}
    client = AquariteClient(auth)

    async def _run() -> Any:
        return await client.subscribe_pool(POOL_ID, lambda _data: None)

    assert asyncio.run(_run()) is watch
    auth.get_client.assert_awaited_once()


def test_async_document_reference_still_lacks_on_snapshot() -> None:
    """Pin the upstream constraint that justifies the split.

    If google-cloud-firestore ever implements this, the listener can move
    to the async client and this test is the reminder to do it.
    """
    from google.cloud.firestore_v1.async_document import AsyncDocumentReference

    doc_ref = AsyncDocumentReference("pools", POOL_ID)
    with pytest.raises(NotImplementedError):
        doc_ref.on_snapshot(lambda *_a: None)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
