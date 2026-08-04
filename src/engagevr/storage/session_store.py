"""Local, single-process session storage.

Layout
------
::

    <root>/<session-id>/
        manifest.json     written once, when the session opens
        events.jsonl      append-only, one protocol message per line
        summary.json      written atomically when the session closes

``<root>`` defaults to ``artifacts/sessions``, which is gitignored.

Session-identifier safety
-------------------------
A session id becomes a directory name, so it is validated against a
strict allowlist before ever touching the filesystem.  Anything outside
``[A-Za-z0-9._-]``, anything empty or over-long, anything starting with
a dot, and the reserved names ``.`` and ``..`` are rejected.  The
resolved directory is additionally checked to be inside the resolved
root, so a symlinked root cannot be used to escape it either.

Session ids are pseudonymous labels.  Nothing here associates one with
a real-world identity, and no participant name, email, or device
identifier is stored anywhere in the session directory.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import TracebackType

from engagevr.protocol.envelope import MessageEnvelope
from engagevr.protocol.messages import DisconnectReason
from engagevr.protocol.validation import ProtocolValidationError, decode_stored_message
from engagevr.protocol.version import PROTOCOL_VERSION
from engagevr.storage.jsonl import (
    JsonlFormatError,
    JsonlWriter,
    read_jsonl,
    read_jsonl_tolerant,
    write_json_atomic,
)
from engagevr.storage.manifest import (
    DropRecord,
    IngestionMetadata,
    SessionManifest,
    SessionSummary,
    StoredMessage,
)
from engagevr.utils.timestamps import utc_now

MANIFEST_FILENAME = "manifest.json"
EVENTS_FILENAME = "events.jsonl"
SUMMARY_FILENAME = "summary.json"
DROPS_FILENAME = "dropped.jsonl"

#: The complete set of characters permitted in a session id.
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: Names that are never valid session ids regardless of the pattern.
_RESERVED_SESSION_IDS = frozenset({".", "..", "", "CON", "PRN", "AUX", "NUL"})


class SessionStoreError(RuntimeError):
    """A session directory could not be opened, read, or written."""


class InvalidSessionIdError(ValueError):
    """A session identifier was rejected before touching the filesystem."""


def validate_session_id(session_id: str) -> str:
    """Return ``session_id`` unchanged if it is safe as a directory name.

    Raises
    ------
    InvalidSessionIdError
        If the identifier is empty, over-long, reserved, contains a path
        separator, a parent reference, a null byte, or any character
        outside ``[A-Za-z0-9._-]``.
    """
    if not isinstance(session_id, str):  # pragma: no cover - typed callers
        raise InvalidSessionIdError("session id must be a string")
    if session_id.upper() in _RESERVED_SESSION_IDS:
        raise InvalidSessionIdError(f"session id {session_id!r} is reserved")
    if _SESSION_ID_PATTERN.match(session_id) is None:
        raise InvalidSessionIdError(
            f"session id {session_id!r} is not permitted; it must be 1-128 "
            "characters of [A-Za-z0-9._-], starting with a letter or digit. "
            "Path separators, parent references, and null bytes are rejected "
            "so a session id can never escape the session root."
        )
    return session_id


def session_directory(root: Path, session_id: str) -> Path:
    """Resolve the directory for ``session_id`` under ``root``, safely.

    Raises
    ------
    InvalidSessionIdError
        If the id is invalid, or if the resolved path would fall outside
        the resolved root.
    """
    validate_session_id(session_id)
    resolved_root = root.resolve()
    candidate = (resolved_root / session_id).resolve()
    if candidate != resolved_root / session_id:
        raise InvalidSessionIdError(
            f"session id {session_id!r} resolves outside the session root "
            f"{resolved_root}; refusing to use it"
        )
    if resolved_root not in candidate.parents:
        raise InvalidSessionIdError(
            f"session id {session_id!r} escapes the session root {resolved_root}"
        )
    return candidate


class SessionRecorder:
    """Writes one session's manifest, events, and summary.

    Explicitly **single-process**.  Two processes writing the same
    session directory concurrently is not supported and is not
    defended against; see ``docs/SESSION_FORMAT.md``.
    """

    def __init__(
        self,
        *,
        directory: Path,
        session_id: str,
        configuration: dict[str, object] | None = None,
        engagevr_version: str = "0.1.0",
        flush_every: int = 1,
        created_at_utc: datetime | None = None,
    ) -> None:
        self._directory = directory
        self._session_id = session_id
        self._closed = False

        directory.mkdir(parents=True, exist_ok=True)
        self._manifest = SessionManifest(
            protocol_version=PROTOCOL_VERSION,
            session_id=session_id,
            created_at_utc=created_at_utc if created_at_utc is not None else utc_now(),
            engagevr_version=engagevr_version,
            configuration=configuration or {},
        )
        write_json_atomic(
            directory / MANIFEST_FILENAME, self._manifest.model_dump(mode="json")
        )

        self._events = JsonlWriter(directory / EVENTS_FILENAME, flush_every=flush_every)
        self._drops = JsonlWriter(directory / DROPS_FILENAME, flush_every=1)

        self._arrival_index = 0
        self._message_types: Counter[str] = Counter()
        self._sources: Counter[str] = Counter()
        self._anomalies: Counter[str] = Counter()
        self._dropped_types: Counter[str] = Counter()
        self._dropped_count = 0
        self._synthetic_count = 0
        self._replay_count = 0
        self._first_sent: datetime | None = None
        self._last_sent: datetime | None = None
        self._first_received: datetime | None = None
        self._last_received: datetime | None = None
        self._session_ended = False

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def event_count(self) -> int:
        return self._events.record_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def next_arrival_index(self) -> int:
        """Reserve and return the next arrival index."""
        index = self._arrival_index
        self._arrival_index += 1
        return index

    def append(self, envelope: MessageEnvelope, ingestion: IngestionMetadata) -> None:
        """Append one message in the order it arrived.

        The arrival order written here is authoritative and is never
        re-sorted by sequence number: the two orders are recorded
        separately and reconciled by the reader, not by the writer.
        """
        if self._closed:
            raise SessionStoreError("cannot append to a closed session recorder")
        stored = StoredMessage(envelope=envelope, ingestion=ingestion)
        self._events.write(stored.model_dump(mode="json"))

        self._message_types[envelope.message_type.value] += 1
        self._sources[envelope.source.value] += 1
        for anomaly in ingestion.anomalies:
            self._anomalies[anomaly.value] += 1
        if envelope.provenance.synthetic_label is not None:
            self._synthetic_count += 1
        if envelope.replay is not None:
            self._replay_count += 1

        if self._first_sent is None:
            self._first_sent = envelope.sent_at_utc
        self._last_sent = envelope.sent_at_utc
        if self._first_received is None:
            self._first_received = ingestion.server_received_at_utc
        self._last_received = ingestion.server_received_at_utc

        if envelope.message_type.value == "session_end":
            self._session_ended = True

    def record_drop(self, drop: DropRecord) -> None:
        """Record a message that backpressure discarded.

        A drop is counted and written to ``dropped.jsonl`` so the gap it
        leaves in ``events.jsonl`` is explained rather than invisible.
        """
        if self._closed:
            raise SessionStoreError("cannot record a drop on a closed recorder")
        self._drops.write(drop.model_dump(mode="json"))
        self._dropped_count += 1
        self._dropped_types[drop.message_type.value] += 1

    def build_summary(
        self, *, disconnect_reason: DisconnectReason | None = None
    ) -> SessionSummary:
        """Build the summary for the messages recorded so far."""
        return SessionSummary(
            protocol_version=PROTOCOL_VERSION,
            session_id=self._session_id,
            event_count=self._events.record_count,
            message_type_counts=dict(sorted(self._message_types.items())),
            source_counts=dict(sorted(self._sources.items())),
            anomaly_counts=dict(sorted(self._anomalies.items())),
            dropped_message_count=self._dropped_count,
            dropped_message_types=dict(sorted(self._dropped_types.items())),
            first_message_sent_at_utc=self._first_sent,
            last_message_sent_at_utc=self._last_sent,
            first_received_at_utc=self._first_received,
            last_received_at_utc=self._last_received,
            completed=self._session_ended,
            disconnect_reason=disconnect_reason,
            synthetic_message_count=self._synthetic_count,
            replay_message_count=self._replay_count,
        )

    def close(
        self, *, disconnect_reason: DisconnectReason | None = None
    ) -> SessionSummary:
        """Flush, write ``summary.json`` atomically, and close the files."""
        summary = self.build_summary(disconnect_reason=disconnect_reason)
        self._events.close()
        self._drops.close()
        write_json_atomic(
            self._directory / SUMMARY_FILENAME, summary.model_dump(mode="json")
        )
        self._closed = True
        return summary

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._closed:
            reason = (
                DisconnectReason.ORDERLY
                if exc_type is None
                else DisconnectReason.INTERNAL_FAILURE
            )
            self.close(disconnect_reason=reason)


class SessionStore:
    """Opens, lists, and reads session recordings under one root."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def directory_for(self, session_id: str) -> Path:
        """Resolve one session's directory, validating the identifier."""
        return session_directory(self._root, session_id)

    def exists(self, session_id: str) -> bool:
        """Whether a session directory with a manifest exists."""
        try:
            directory = self.directory_for(session_id)
        except InvalidSessionIdError:
            return False
        return (directory / MANIFEST_FILENAME).is_file()

    def list_sessions(self) -> list[str]:
        """Every readable session id under the root, sorted.

        Directories whose names are not valid session ids, and
        directories with no manifest, are skipped rather than reported
        as sessions.
        """
        if not self._root.is_dir():
            return []
        found: list[str] = []
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue
            try:
                validate_session_id(entry.name)
            except InvalidSessionIdError:
                continue
            if (entry / MANIFEST_FILENAME).is_file():
                found.append(entry.name)
        return found

    def open_recorder(
        self,
        session_id: str,
        *,
        configuration: dict[str, object] | None = None,
        flush_every: int = 1,
        engagevr_version: str = "0.1.0",
        created_at_utc: datetime | None = None,
    ) -> SessionRecorder:
        """Create (or reopen for append) a recorder for ``session_id``."""
        directory = self.directory_for(session_id)
        return SessionRecorder(
            directory=directory,
            session_id=session_id,
            configuration=configuration,
            engagevr_version=engagevr_version,
            flush_every=flush_every,
            created_at_utc=created_at_utc,
        )

    def read_manifest(self, session_id: str) -> SessionManifest:
        """Read and validate ``manifest.json``."""
        path = self.directory_for(session_id) / MANIFEST_FILENAME
        if not path.is_file():
            raise SessionStoreError(f"no manifest for session {session_id!r} at {path}")
        import json

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SessionStoreError(f"{path} is not readable JSON: {exc}") from exc
        return SessionManifest.model_validate(raw)

    def read_summary(self, session_id: str) -> SessionSummary | None:
        """Read ``summary.json``, or None when the session never closed."""
        path = self.directory_for(session_id) / SUMMARY_FILENAME
        if not path.is_file():
            return None
        import json

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SessionStoreError(f"{path} is not readable JSON: {exc}") from exc
        return SessionSummary.model_validate(raw)

    def events_path(self, session_id: str) -> Path:
        """Path of the append-only event stream."""
        return self.directory_for(session_id) / EVENTS_FILENAME

    def iter_messages(self, session_id: str) -> Iterator[StoredMessage]:
        """Yield every stored message in recorded arrival order.

        Each line is re-validated against the protocol on the way out,
        so a recording that was corrupted after the fact cannot be
        replayed as though it were sound.

        Raises
        ------
        JsonlFormatError
            On a line that is not a JSON object, with its line number.
        SessionStoreError
            On a line that parses but is not a valid stored message,
            with its line number.
        """
        path = self.events_path(session_id)
        if not path.is_file():
            raise SessionStoreError(
                f"session {session_id!r} has no event stream at {path}"
            )
        for line in read_jsonl(path):
            record = line.record
            envelope_raw = record.get("envelope")
            ingestion_raw = record.get("ingestion")
            if not isinstance(envelope_raw, dict) or not isinstance(
                ingestion_raw, dict
            ):
                raise SessionStoreError(
                    f"{path}:{line.line_number}: stored record must have object "
                    "'envelope' and 'ingestion' fields"
                )
            try:
                decoded = decode_stored_message(envelope_raw)
            except ProtocolValidationError as exc:
                raise SessionStoreError(
                    f"{path}:{line.line_number}: stored message is not valid "
                    f"under protocol {PROTOCOL_VERSION}: {exc.detail}"
                ) from exc
            try:
                ingestion = IngestionMetadata.model_validate(ingestion_raw)
            except ValueError as exc:
                raise SessionStoreError(
                    f"{path}:{line.line_number}: ingestion metadata is invalid: {exc}"
                ) from exc
            yield StoredMessage(envelope=decoded.envelope, ingestion=ingestion)

    def recover(self, session_id: str) -> SessionSummary:
        """Rebuild a summary from an interrupted session's event stream.

        Every line that parses is counted; lines that do not are listed
        by 1-based line number in ``malformed_line_numbers``.  The
        resulting summary is marked ``recovered=True`` and is only
        ``completed=True`` if a ``session_end`` message was actually
        recorded before the interruption.

        The source recording is not modified.
        """
        path = self.events_path(session_id)
        if not path.is_file():
            raise SessionStoreError(
                f"session {session_id!r} has no event stream at {path}"
            )
        lines, malformed = read_jsonl_tolerant(path)

        message_types: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        anomalies: Counter[str] = Counter()
        synthetic = 0
        replayed = 0
        first_sent: datetime | None = None
        last_sent: datetime | None = None
        first_received: datetime | None = None
        last_received: datetime | None = None
        completed = False
        count = 0

        for line in lines:
            envelope_raw = line.record.get("envelope")
            ingestion_raw = line.record.get("ingestion")
            if not isinstance(envelope_raw, dict) or not isinstance(
                ingestion_raw, dict
            ):
                malformed.append(line.line_number)
                continue
            try:
                envelope = decode_stored_message(envelope_raw).envelope
                ingestion = IngestionMetadata.model_validate(ingestion_raw)
            except (ProtocolValidationError, ValueError):
                malformed.append(line.line_number)
                continue

            count += 1
            message_types[envelope.message_type.value] += 1
            sources[envelope.source.value] += 1
            for anomaly in ingestion.anomalies:
                anomalies[anomaly.value] += 1
            if envelope.provenance.synthetic_label is not None:
                synthetic += 1
            if envelope.replay is not None:
                replayed += 1
            if first_sent is None:
                first_sent = envelope.sent_at_utc
            last_sent = envelope.sent_at_utc
            if first_received is None:
                first_received = ingestion.server_received_at_utc
            last_received = ingestion.server_received_at_utc
            if envelope.message_type.value == "session_end":
                completed = True

        dropped_path = self.directory_for(session_id) / DROPS_FILENAME
        dropped_count = 0
        dropped_types: Counter[str] = Counter()
        if dropped_path.is_file():
            drop_lines, _ = read_jsonl_tolerant(dropped_path)
            for drop_line in drop_lines:
                dropped_count += 1
                message_type = drop_line.record.get("message_type")
                if isinstance(message_type, str):
                    dropped_types[message_type] += 1

        return SessionSummary(
            protocol_version=PROTOCOL_VERSION,
            session_id=session_id,
            event_count=count,
            message_type_counts=dict(sorted(message_types.items())),
            source_counts=dict(sorted(sources.items())),
            anomaly_counts=dict(sorted(anomalies.items())),
            dropped_message_count=dropped_count,
            dropped_message_types=dict(sorted(dropped_types.items())),
            first_message_sent_at_utc=first_sent,
            last_message_sent_at_utc=last_sent,
            first_received_at_utc=first_received,
            last_received_at_utc=last_received,
            completed=completed,
            recovered=True,
            malformed_line_numbers=sorted(set(malformed)),
            synthetic_message_count=synthetic,
            replay_message_count=replayed,
        )


__all__ = [
    "DROPS_FILENAME",
    "EVENTS_FILENAME",
    "MANIFEST_FILENAME",
    "SUMMARY_FILENAME",
    "InvalidSessionIdError",
    "JsonlFormatError",
    "SessionRecorder",
    "SessionStore",
    "SessionStoreError",
    "session_directory",
    "validate_session_id",
]
