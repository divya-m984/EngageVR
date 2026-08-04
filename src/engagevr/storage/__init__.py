"""Local session storage: append-only JSONL plus manifest and summary.

Standard-library JSON Lines only.  No database, no message broker, no
external service.  The store is explicitly single-process and local; see
``docs/SESSION_FORMAT.md`` for what is persisted, what is never
persisted, and how an interrupted session is recovered.
"""

from engagevr.storage.jsonl import (
    JsonlFormatError,
    JsonlLine,
    JsonlWriter,
    read_jsonl,
    read_jsonl_tolerant,
    write_json_atomic,
)
from engagevr.storage.manifest import (
    RECORDING_DISCLAIMER,
    SESSION_FORMAT_VERSION,
    DropRecord,
    IngestionMetadata,
    SessionManifest,
    SessionSummary,
    StoredMessage,
)
from engagevr.storage.session_store import (
    DROPS_FILENAME,
    EVENTS_FILENAME,
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    InvalidSessionIdError,
    SessionRecorder,
    SessionStore,
    SessionStoreError,
    session_directory,
    validate_session_id,
)

__all__ = [
    "DROPS_FILENAME",
    "EVENTS_FILENAME",
    "MANIFEST_FILENAME",
    "RECORDING_DISCLAIMER",
    "SESSION_FORMAT_VERSION",
    "SUMMARY_FILENAME",
    "DropRecord",
    "IngestionMetadata",
    "InvalidSessionIdError",
    "JsonlFormatError",
    "JsonlLine",
    "JsonlWriter",
    "SessionManifest",
    "SessionRecorder",
    "SessionStore",
    "SessionStoreError",
    "SessionSummary",
    "StoredMessage",
    "read_jsonl",
    "read_jsonl_tolerant",
    "session_directory",
    "validate_session_id",
    "write_json_atomic",
]
