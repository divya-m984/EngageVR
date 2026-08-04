"""Deterministic session replay.

Replay re-emits a recorded session in recorded arrival order, at a
chosen pace, over any transport.  It preserves every original message
exactly and adds separate replay metadata rather than pretending a
replayed message is live.  The source recording is opened read-only and
is never modified.
"""

from engagevr.replay.clock import (
    InvalidReplaySpeedError,
    ReplayMode,
    ReplayPacer,
    mode_for_speed,
    validate_speed,
)
from engagevr.replay.player import (
    REPLAY_DISCLAIMER,
    ReplayPlayer,
    ReplayResult,
    parse_message_type_filter,
    parse_source_filter,
)
from engagevr.replay.reader import (
    RecordedSession,
    ReplayFilter,
    iter_recorded_messages,
    read_recorded_session,
)

__all__ = [
    "REPLAY_DISCLAIMER",
    "InvalidReplaySpeedError",
    "RecordedSession",
    "ReplayFilter",
    "ReplayMode",
    "ReplayPacer",
    "ReplayPlayer",
    "ReplayResult",
    "iter_recorded_messages",
    "mode_for_speed",
    "parse_message_type_filter",
    "parse_source_filter",
    "read_recorded_session",
    "validate_speed",
]
