"""UTC timestamp and monotonic clock utilities.

All modalities share a common monotonic session clock.  Wall-clock
(UTC) time is recorded once at session start for external correlation.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current UTC wall-clock time (timezone-aware)."""
    return datetime.now(UTC)


def monotonic_seconds() -> float:
    """Return a monotonic timestamp in fractional seconds.

    Used as the session-local clock for synchronising modality samples.
    The absolute value is arbitrary; only differences are meaningful.
    """
    return time.monotonic()


def format_iso(dt: datetime) -> str:
    """Format a datetime as an ISO-8601 string with UTC offset."""
    return dt.isoformat()


def parse_iso(s: str) -> datetime:
    """Parse an ISO-8601 string into a timezone-aware datetime."""
    return datetime.fromisoformat(s)


def require_utc(value: datetime, field_name: str) -> datetime:
    """Return ``value`` unchanged, rejecting naive datetimes.

    A naive datetime is ambiguous across machines and would silently
    corrupt any cross-client ordering or latency diagnostic, so it is
    rejected rather than assumed to be UTC.

    Raises
    ------
    ValueError
        If ``value`` has no timezone information.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"{field_name} must be timezone-aware; naive datetimes are rejected "
            "because their offset cannot be recovered"
        )
    return value


def to_utc(value: datetime) -> datetime:
    """Convert a timezone-aware datetime into the UTC timezone."""
    return require_utc(value, "value").astimezone(UTC)
