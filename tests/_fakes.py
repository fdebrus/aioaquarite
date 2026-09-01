"""Shared test doubles for the native async watch and its supervisor."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from google.cloud.firestore_v1 import _helpers
from google.cloud.firestore_v1.types import (
    Document,
    DocumentChange,
    ListenRequest,
    ListenResponse,
    Target,
)
from google.cloud.firestore_v1.types.firestore import TargetChange

from aioaquarite._watch import TARGET_ID

# Sentinel script entry: keep the response stream open (until cancelled).
HOLD = object()

_TCT = TargetChange.TargetChangeType


class FakeTaskWatch:
    """Double for AsyncDocumentWatch as the resilient supervisor sees it.

    Backed by a real asyncio task parked on an event, so the supervisor's
    "wake early when the watch task ends" logic is exercised for real.
    ``die()`` completes the task, with or without an exception.
    """

    def __init__(self) -> None:
        self.unsubscribed = False
        self._die = asyncio.Event()
        self._exc: Exception | None = None
        self.task: asyncio.Task[None] = asyncio.create_task(self._park())

    async def _park(self) -> None:
        await self._die.wait()
        if self._exc is not None:
            raise self._exc

    def die(self, exc: Exception | None = None) -> None:
        """Simulate the listen stream ending (with an error, if given)."""
        self._exc = exc
        self._die.set()

    async def aclose(self) -> None:
        self.unsubscribed = True
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self.task


class FakeGapic:
    """Fake ``FirestoreAsyncClient``: its transport replays scripted responses.

    Each script is a list whose items are ``ListenResponse`` protos,
    exceptions (raised mid-stream), or :data:`HOLD` (park until cancelled).
    A stream whose script runs out simply ends, which the watch treats as
    a failure — exactly like the real RPC.

    Only ``transport.listen`` exists — deliberately no ``listen`` method on
    the client itself, so a regression back to the generated wrapper (which
    stamps an empty ``x-goog-request-params`` header the backend rejects)
    fails every protocol test with an ``AttributeError``.
    """

    def __init__(self) -> None:
        self.scripts: list[list[Any]] = []
        self.captured_requests: list[ListenRequest] = []
        self.captured_metadata: list[Any] = []
        self.streams_opened = 0
        self.transport = _FakeTransport(self)


class _FakeTransport:
    """The raw gRPC surface the watch calls: a ``stream_stream``
    multicallable — a plain call returning an async-iterable, no coroutine."""

    def __init__(self, gapic: FakeGapic) -> None:
        self._gapic = gapic

    def listen(self, requests: Any, metadata: Any = ()) -> Any:
        gapic = self._gapic
        gapic.captured_metadata.append(metadata)
        gapic.streams_opened += 1
        script = gapic.scripts.pop(0) if gapic.scripts else []

        async def _gen() -> Any:
            first = await anext(requests)
            gapic.captured_requests.append(first)
            for item in script:
                if item is HOLD:
                    await asyncio.Event().wait()
                elif isinstance(item, Exception):
                    raise item
                else:
                    yield item

        return _gen()


class _FakeDocRef:
    def __init__(self, path: str) -> None:
        self._document_path = path


class _FakeCollection:
    def __init__(self, client: FakeAsyncClient, name: str) -> None:
        self._client = client
        self._name = name

    def document(self, doc_id: str) -> _FakeDocRef:
        self._client.document_calls.append((self._name, doc_id))
        return _FakeDocRef(
            f"{self._client._database_string}/documents/{self._name}/{doc_id}"
        )


class FakeAsyncClient:
    """Fake ``AsyncClient`` exposing exactly what AsyncDocumentWatch uses."""

    def __init__(self, gapic: FakeGapic | None = None) -> None:
        self._firestore_api = gapic or FakeGapic()
        self._database_string = "projects/test-proj/databases/(default)"
        self._rpc_metadata = [
            ("google-cloud-resource-prefix", self._database_string)
        ]
        self.document_calls: list[tuple[str, str]] = []

    def collection(self, name: str) -> _FakeCollection:
        return _FakeCollection(self, name)


def doc_path(client: FakeAsyncClient, collection: str, doc_id: str) -> str:
    return f"{client._database_string}/documents/{collection}/{doc_id}"


def doc_change(
    data: dict[str, Any],
    *,
    name: str = "projects/test-proj/databases/(default)/documents/pools/pool-1",
    target_ids: tuple[int, ...] = (TARGET_ID,),
) -> ListenResponse:
    """A ``document_change`` response carrying ``data`` encoded as protos."""
    return ListenResponse(
        document_change=DocumentChange(
            document=Document(name=name, fields=_helpers.encode_dict(data)),
            target_ids=list(target_ids),
        )
    )


def target_add(*, target_ids: tuple[int, ...] = (TARGET_ID,)) -> ListenResponse:
    return ListenResponse(
        target_change=TargetChange(
            target_change_type=_TCT.ADD, target_ids=list(target_ids)
        )
    )


def target_current() -> ListenResponse:
    return ListenResponse(
        target_change=TargetChange(
            target_change_type=_TCT.CURRENT, target_ids=[TARGET_ID]
        )
    )


def target_no_change(*, resume_token: bytes = b"") -> ListenResponse:
    return ListenResponse(
        target_change=TargetChange(
            target_change_type=_TCT.NO_CHANGE, resume_token=resume_token
        )
    )


def target_reset() -> ListenResponse:
    return ListenResponse(
        target_change=TargetChange(target_change_type=_TCT.RESET)
    )


def target_remove(*, code: int = 7, message: str = "denied") -> ListenResponse:
    return ListenResponse(
        target_change=TargetChange(
            target_change_type=_TCT.REMOVE,
            target_ids=[TARGET_ID],
            cause={"code": code, "message": message},
        )
    )
