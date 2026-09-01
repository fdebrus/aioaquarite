"""Native async single-document Firestore listener.

``google-cloud-firestore`` implements ``on_snapshot`` only on its
synchronous client — ``AsyncDocumentReference.on_snapshot`` raises
``NotImplementedError``. The bidirectional ``Listen`` RPC it is built on,
however, is fully generated in the async GAPIC layer
(``FirestoreAsyncClient.listen``). This module implements the thin
single-document assembly on top of that stream, so the library needs no
thread at all.

Protocol notes (mirroring the synchronous ``Watch`` where it matters):

* ``document_change`` responses are stashed, not emitted; the server marks
  consistent points explicitly.
* ``CURRENT`` means the target has caught up — a stashed document is
  emitted, and the watch reports ready.
* ``NO_CHANGE`` with a ``read_time`` after ``CURRENT`` delimits later
  consistent snapshots — a stashed document is emitted and the attached
  ``resume_token`` is saved so a reconnect does not replay from scratch.
* ``RESET`` drops all stashed state and the resume token. The target stays
  registered on the stream (re-adding it would double-register); the
  server re-sends the world after a reset.
* ``REMOVE`` carries a server-side error and fails the watch.

The data callback is invoked **on the event loop** running the watch task,
never from a thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable, MutableMapping
from typing import Any, cast

from google.cloud.firestore_v1 import AsyncClient, _helpers
from google.cloud.firestore_v1.services.firestore.async_client import (
    FirestoreAsyncClient,
)
from google.cloud.firestore_v1.types import ListenRequest, ListenResponse, Target
from google.cloud.firestore_v1.types.firestore import TargetChange

from .exceptions import ConnectionError

_LOGGER = logging.getLogger(__name__)

# Fixed target id, echoed back by the server. Same value the synchronous
# Watch uses; any constant works for a single-target stream.
TARGET_ID = 20601

# How long start() waits for the server to confirm the target is caught up
# (CURRENT) before failing the subscribe attempt.
DEFAULT_READY_TIMEOUT = 30.0

_TargetChangeType = TargetChange.TargetChangeType


class AsyncDocumentWatch:
    """Single-document listen loop feeding a callback with dict snapshots.

    Owned by an asyncio task; :meth:`start` opens the stream and waits for
    the first consistent snapshot. Stop with the synchronous
    :meth:`unsubscribe` (cancels the task) or ``await`` :meth:`aclose`
    (cancels and waits for full termination).

    ``resume_tokens`` is a mutable mapping shared between successive
    watches of the same document (the :class:`AquariteClient` owns it):
    the token saved at each consistent point is used by the next watch's
    ``add_target``, so a reconnect resumes instead of replaying.
    """

    def __init__(
        self,
        client: AsyncClient,
        document_path: str,
        callback: Callable[[dict[str, Any]], None],
        *,
        resume_tokens: MutableMapping[str, bytes],
        label: str,
    ) -> None:
        self._client = client
        self._document_path = document_path
        self._callback = callback
        self._resume_tokens = resume_tokens
        self._label = label
        self._ready = asyncio.Event()
        self._closed = False
        self._task: asyncio.Task[None] | None = None

    @property
    def task(self) -> asyncio.Task[None]:
        """The task running the listen loop (exists once started)."""
        assert self._task is not None
        return self._task

    @property
    def done(self) -> bool:
        """Whether the listen loop has terminated (for any reason)."""
        return self._task is not None and self._task.done()

    async def start(
        self, *, ready_timeout: float = DEFAULT_READY_TIMEOUT
    ) -> None:
        """Open the stream and wait for the first consistent snapshot.

        Raises whatever brought the stream down if it dies before
        reaching CURRENT, or :class:`ConnectionError` on timeout — so a
        subscribe that cannot actually deliver data fails loudly instead
        of returning a silently dead watch.
        """
        self._task = asyncio.create_task(
            self._run(), name=f"aioaquarite-watch-{self._label}"
        )
        ready = asyncio.ensure_future(self._ready.wait())
        try:
            done, _ = await asyncio.wait(
                {ready, self._task},
                timeout=ready_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            ready.cancel()
        if ready in done:
            return
        if self._task in done:
            # The stream died before the target caught up; surface why.
            exc = self._task.exception()
            if exc is not None:
                raise exc
            raise ConnectionError(f"{self._label}: listen stream ended")
        await self.aclose()
        raise ConnectionError(
            f"{self._label}: no consistent snapshot within {ready_timeout:.0f}s"
        )

    def unsubscribe(self) -> None:
        """Stop listening (synchronous, does not wait for teardown)."""
        self._closed = True
        if self._task is not None:
            self._task.cancel()

    async def aclose(self) -> None:
        """Stop listening and wait for the loop to terminate. Idempotent.

        Never raises: a watch whose stream already failed re-raising that
        error during teardown would turn cleanup into a second failure —
        the supervisor has already seen the original.
        """
        self._closed = True
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 — already surfaced via .task
            _LOGGER.debug(
                "%s: watch task had ended with %r before close", self._label, exc
            )

    # ── listen loop ────────────────────────────────────────────────────

    async def _run(self) -> None:
        gapic = cast(FirestoreAsyncClient, self._client._firestore_api)
        database = cast(str, self._client._database_string)
        metadata = cast(
            "list[tuple[str, str]]", self._client._rpc_metadata
        )

        resume_token = self._resume_tokens.get(self._document_path, b"")
        initial_request = ListenRequest(
            database=database,
            add_target=Target(
                documents=Target.DocumentsTarget(
                    documents=[self._document_path]
                ),
                target_id=TARGET_ID,
                resume_token=resume_token,
            ),
        )

        async def _requests() -> AsyncIterator[ListenRequest]:
            yield initial_request
            # Keep the request side open for the life of the stream; the
            # server drives everything after the target is added.
            await asyncio.Event().wait()

        pending: dict[str, Any] | None = None
        current = False

        if resume_token:
            _LOGGER.debug(
                "%s: opening listen stream, resuming from saved token "
                "(%d bytes)",
                self._label,
                len(resume_token),
            )
        else:
            # Normal on the first connection: a resume token only exists
            # once a previous stream reached a consistency point.
            _LOGGER.debug(
                "%s: opening listen stream (fresh, no resume token)",
                self._label,
            )
        # Call the raw transport multicallable, not the generated
        # FirestoreAsyncClient.listen wrapper: the wrapper stamps an empty
        # ``x-goog-request-params`` routing header onto every call, and the
        # backend rejects the stream as a database mismatch
        # (400 INVALID_ARGUMENT). The synchronous Watch bypasses the
        # wrapper for the same reason; the resource-prefix metadata is all
        # the server needs.
        listen_rpc = cast(
            "Callable[..., AsyncIterable[ListenResponse]]",
            gapic.transport.listen,
        )
        stream = listen_rpc(_requests(), metadata=metadata)
        async for response in stream:
            which = response._pb.WhichOneof("response_type")

            if which == "target_change":
                change = response.target_change
                change_type = change.target_change_type

                if change_type == _TargetChangeType.ADD:
                    if change.target_ids and change.target_ids[0] != TARGET_ID:
                        raise ConnectionError(
                            f"{self._label}: server sent unexpected target id "
                            f"{change.target_ids[0]}"
                        )
                elif change_type == _TargetChangeType.CURRENT:
                    current = True
                    if pending is not None:
                        self._emit(pending)
                        pending = None
                    self._ready.set()
                elif change_type == _TargetChangeType.NO_CHANGE:
                    if not change.target_ids and current:
                        if change.resume_token:
                            self._resume_tokens[self._document_path] = bytes(
                                change.resume_token
                            )
                        if pending is not None:
                            self._emit(pending)
                            pending = None
                elif change_type == _TargetChangeType.RESET:
                    # Everything stashed no longer matters, and the resume
                    # token is no longer valid. The target itself stays
                    # registered; the server re-sends after a reset.
                    _LOGGER.debug("%s: target RESET", self._label)
                    pending = None
                    current = False
                    self._resume_tokens.pop(self._document_path, None)
                elif change_type == _TargetChangeType.REMOVE:
                    code = change.cause.code or 13
                    raise ConnectionError(
                        f"{self._label}: target removed by server "
                        f"({code}: {change.cause.message or 'internal error'})"
                    )
                else:
                    _LOGGER.warning(
                        "%s: unknown target change type %s",
                        self._label,
                        change_type,
                    )

            elif which == "document_change":
                change_doc = response.document_change
                if TARGET_ID in change_doc.target_ids:
                    decoded = _helpers.decode_dict(
                        change_doc.document.fields, self._client
                    )
                    pending = cast("dict[str, Any]", decoded)

            elif which in ("document_delete", "document_remove"):
                # Pool and user documents are not deleted in this API's
                # lifecycle; note it and keep listening.
                _LOGGER.warning(
                    "%s: unexpected %s for %s ignored",
                    self._label,
                    which,
                    self._document_path,
                )

            elif which == "filter":
                # Single explicit document target: no filter mismatch is
                # possible, nothing to reconcile.
                pass

            else:
                _LOGGER.warning(
                    "%s: unknown listen response type %r", self._label, which
                )

        # A listen stream never ends on its own initiative from our side;
        # the server (or a closed channel) ended it. Fail so the
        # supervisor reconnects.
        raise ConnectionError(f"{self._label}: listen stream ended")

    def _emit(self, data: dict[str, Any]) -> None:
        """Deliver one consistent snapshot, guarding against consumer bugs."""
        if self._closed:
            return
        try:
            self._callback(data)
        except Exception:  # noqa: BLE001 — a consumer bug must not kill the stream
            _LOGGER.exception("%s: snapshot callback failed", self._label)
