"""Reading a recorded session for replay.

The reader opens the recording **read-only** and never writes to it.
Replaying a session cannot modify the session: the player emits copies
carrying added replay metadata, and the file on disk is untouched.

Ordering
--------
Messages are yielded in recorded arrival order, which is the order the
lines appear in ``events.jsonl``.  That order is deterministic, is what
actually happened, and is not re-derived from sequence numbers — a
recording containing a genuine sequence reversal must replay that
reversal, not a tidied-up version of it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from engagevr.protocol.messages import MessageSource, MessageType
from engagevr.storage.manifest import SessionManifest, SessionSummary, StoredMessage
from engagevr.storage.session_store import SessionStore


@dataclass(frozen=True, slots=True)
class ReplayFilter:
    """Which recorded messages a replay includes.

    An empty filter includes everything.  Filtering removes messages
    from the replay stream; it never renumbers or re-times the ones that
    remain, so the gaps a filter creates are visible in the sequence
    numbers rather than papered over.
    """

    message_types: frozenset[MessageType] = frozenset()
    sources: frozenset[MessageSource] = frozenset()

    def matches(self, stored: StoredMessage) -> bool:
        if (
            self.message_types
            and stored.envelope.message_type not in self.message_types
        ):
            return False
        if self.sources and stored.envelope.source not in self.sources:
            return False
        return True

    @property
    def is_empty(self) -> bool:
        return not self.message_types and not self.sources

    def describe(self) -> str:
        if self.is_empty:
            return "no filter (all messages)"
        parts: list[str] = []
        if self.message_types:
            parts.append(
                "message_type in {"
                + ", ".join(sorted(t.value for t in self.message_types))
                + "}"
            )
        if self.sources:
            parts.append(
                "source in {" + ", ".join(sorted(s.value for s in self.sources)) + "}"
            )
        return " and ".join(parts)


@dataclass(frozen=True, slots=True)
class RecordedSession:
    """A recording opened for replay."""

    session_id: str
    directory: Path
    manifest: SessionManifest
    summary: SessionSummary
    messages: tuple[StoredMessage, ...]

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def filtered(self, replay_filter: ReplayFilter) -> tuple[StoredMessage, ...]:
        """Messages matching a filter, in recorded arrival order."""
        if replay_filter.is_empty:
            return self.messages
        return tuple(m for m in self.messages if replay_filter.matches(m))

    def gaps_seconds(self, messages: tuple[StoredMessage, ...]) -> tuple[float, ...]:
        """Recorded gaps preceding each message, on the receiver's clock.

        The first message's gap is 0: there is nothing before it to wait
        for.  A negative gap (from a clock adjustment during recording)
        is clamped to 0 rather than rewinding.
        """
        gaps: list[float] = []
        previous = None
        for stored in messages:
            current = stored.ingestion.server_received_at_utc
            if previous is None:
                gaps.append(0.0)
            else:
                gaps.append(max((current - previous).total_seconds(), 0.0))
            previous = current
        return tuple(gaps)


def read_recorded_session(
    store: SessionStore,
    session_id: str,
    *,
    recover_incomplete: bool = True,
) -> RecordedSession:
    """Load a recording for replay, validating every message.

    A session with no ``summary.json`` — an interrupted run — is
    recovered when ``recover_incomplete`` is set, and the recovered
    summary is marked ``recovered=True`` and ``completed=False``.

    Raises
    ------
    SessionStoreError
        If the session, its manifest, or its event stream is missing or
        unreadable.
    JsonlFormatError
        On a malformed line, reporting its 1-based line number.
    """
    manifest = store.read_manifest(session_id)
    messages = tuple(store.iter_messages(session_id))
    summary = store.read_summary(session_id)
    if summary is None:
        if not recover_incomplete:
            from engagevr.storage.session_store import SessionStoreError

            raise SessionStoreError(
                f"session {session_id!r} has no summary.json and recovery was "
                "not requested; the session may still be running"
            )
        summary = store.recover(session_id)
    return RecordedSession(
        session_id=session_id,
        directory=store.directory_for(session_id),
        manifest=manifest,
        summary=summary,
        messages=messages,
    )


def iter_recorded_messages(
    store: SessionStore,
    session_id: str,
    *,
    predicate: Callable[[StoredMessage], bool] | None = None,
) -> Iterator[StoredMessage]:
    """Stream a recording without holding it all in memory."""
    for stored in store.iter_messages(session_id):
        if predicate is None or predicate(stored):
            yield stored
