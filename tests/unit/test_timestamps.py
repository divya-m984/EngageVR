"""Tests for UTC timestamp utilities."""

from __future__ import annotations

from datetime import UTC, datetime

from engagevr.utils.timestamps import (
    format_iso,
    monotonic_seconds,
    parse_iso,
    utc_now,
)


class TestUtcNow:
    def test_returns_utc(self):
        now = utc_now()
        assert now.tzinfo == UTC

    def test_returns_datetime(self):
        now = utc_now()
        assert isinstance(now, datetime)


class TestMonotonic:
    def test_monotonically_increasing(self):
        t1 = monotonic_seconds()
        t2 = monotonic_seconds()
        assert t2 >= t1

    def test_returns_float(self):
        assert isinstance(monotonic_seconds(), float)


class TestIsoFormatting:
    def test_roundtrip(self):
        dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)
        iso_str = format_iso(dt)
        parsed = parse_iso(iso_str)
        assert parsed == dt

    def test_format_includes_timezone(self):
        dt = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        s = format_iso(dt)
        assert "+00:00" in s or "Z" in s
