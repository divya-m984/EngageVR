"""Windowing tests: boundaries, overlap, partial policy, and containment.

The central property under test is that a window's evidence comes only
from ``[start, end)`` of its own session. A window that reaches past its
end is future leakage; a window that reaches past its session is evidence
that does not exist.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from engagevr.features.windowing import (
    PartialWindowPolicy,
    SessionInterval,
    WindowingError,
    WindowSpec,
    assert_windows_within_session,
    build_windows,
    select_in_window,
    select_in_window_utc,
    window_id,
)

START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def interval(seconds: float, *, monotonic: bool = True) -> SessionInterval:
    return SessionInterval(
        session_id="session-a",
        start_utc=START,
        end_utc=START + timedelta(seconds=seconds),
        start_monotonic_seconds=100.0 if monotonic else None,
        end_monotonic_seconds=100.0 + seconds if monotonic else None,
    )


@dataclass(frozen=True)
class Sample:
    monotonic_timestamp: float
    utc_timestamp: datetime
    tag: str


class TestWindowBoundaries:
    def test_non_overlapping_windows_tile_the_session(self) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        windows = build_windows(interval(50.0), spec)
        assert len(windows) == 5
        assert [w.index for w in windows] == [0, 1, 2, 3, 4]
        assert windows[0].start_utc == START
        assert windows[-1].end_utc == START + timedelta(seconds=50)
        for earlier, later in itertools.pairwise(windows):
            assert earlier.end_utc == later.start_utc

    def test_overlapping_windows_share_evidence(self) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=5.0)
        assert spec.overlapping is True
        windows = build_windows(interval(30.0), spec)
        assert len(windows) == 5
        assert windows[1].start_utc == START + timedelta(seconds=5)
        assert windows[0].end_utc > windows[1].start_utc

    def test_non_overlapping_spec_reports_no_overlap(self) -> None:
        assert WindowSpec(duration_seconds=10.0, step_seconds=10.0).overlapping is False

    def test_boundaries_are_computed_from_the_session_start(self) -> None:
        spec = WindowSpec(duration_seconds=1.0, step_seconds=0.1)
        windows = build_windows(interval(20.0), spec)
        for window in windows:
            expected = START + timedelta(seconds=window.index * 0.1)
            assert window.start_utc == expected

    def test_monotonic_bounds_track_the_wall_clock(self) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        windows = build_windows(interval(30.0), spec)
        assert windows[0].start_monotonic_seconds == pytest.approx(100.0)
        assert windows[2].start_monotonic_seconds == pytest.approx(120.0)
        assert windows[2].end_monotonic_seconds == pytest.approx(130.0)

    def test_monotonic_bounds_are_optional(self) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        windows = build_windows(interval(20.0, monotonic=False), spec)
        assert all(w.start_monotonic_seconds is None for w in windows)

    def test_window_ids_are_deterministic_and_zero_padded(self) -> None:
        assert window_id("session-a", 0) == "session-a:w000000"
        assert window_id("session-a", 12) == "session-a:w000012"


class TestPartialWindowPolicy:
    def test_drop_discards_the_trailing_fragment(self) -> None:
        spec = WindowSpec(
            duration_seconds=10.0,
            step_seconds=10.0,
            partial_window_policy=PartialWindowPolicy.DROP,
        )
        windows = build_windows(interval(25.0), spec)
        assert len(windows) == 2
        assert windows[-1].end_utc == START + timedelta(seconds=20)

    def test_keep_if_minimum_retains_a_long_enough_fragment(self) -> None:
        spec = WindowSpec(
            duration_seconds=10.0,
            step_seconds=10.0,
            partial_window_policy=PartialWindowPolicy.KEEP_IF_MINIMUM,
            minimum_partial_duration_seconds=4.0,
        )
        windows = build_windows(interval(25.0), spec)
        assert len(windows) == 3
        assert windows[-1].is_partial is True
        assert windows[-1].duration_seconds == pytest.approx(5.0)

    def test_keep_if_minimum_drops_a_fragment_below_the_minimum(self) -> None:
        spec = WindowSpec(
            duration_seconds=10.0,
            step_seconds=10.0,
            partial_window_policy=PartialWindowPolicy.KEEP_IF_MINIMUM,
            minimum_partial_duration_seconds=8.0,
        )
        windows = build_windows(interval(25.0), spec)
        assert len(windows) == 2
        assert all(not w.is_partial for w in windows)

    def test_keep_if_minimum_requires_a_positive_minimum(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            WindowSpec(
                duration_seconds=10.0,
                step_seconds=10.0,
                partial_window_policy=PartialWindowPolicy.KEEP_IF_MINIMUM,
            )

    def test_a_session_shorter_than_one_window_is_refused(self) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        with pytest.raises(WindowingError, match="shorter than"):
            build_windows(interval(4.0), spec)


class TestInvalidInputs:
    def test_reversed_session_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly after"):
            SessionInterval(
                session_id="s",
                start_utc=START,
                end_utc=START - timedelta(seconds=1),
            )

    def test_zero_length_session_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly after"):
            SessionInterval(session_id="s", start_utc=START, end_utc=START)

    def test_naive_timestamps_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            SessionInterval(
                session_id="s",
                start_utc=datetime(2026, 1, 1, 12, 0, 0),
                end_utc=datetime(2026, 1, 1, 12, 1, 0, tzinfo=UTC),
            )

    def test_reversed_monotonic_bounds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="monotonic end"):
            SessionInterval(
                session_id="s",
                start_utc=START,
                end_utc=START + timedelta(seconds=10),
                start_monotonic_seconds=50.0,
                end_monotonic_seconds=40.0,
            )

    def test_half_supplied_monotonic_bounds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="together or not at all"):
            SessionInterval(
                session_id="s",
                start_utc=START,
                end_utc=START + timedelta(seconds=10),
                start_monotonic_seconds=50.0,
            )

    def test_step_larger_than_duration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            WindowSpec(duration_seconds=5.0, step_seconds=10.0)


class TestSessionContainment:
    def test_generated_windows_stay_inside_the_session(self) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=3.0)
        source = interval(40.0)
        windows = build_windows(source, spec)
        for window in windows:
            assert window.start_utc >= source.start_utc
            assert window.end_utc <= source.end_utc

    def test_a_window_past_the_session_end_is_rejected(self) -> None:
        source = interval(20.0)
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        windows = build_windows(source, spec)
        shifted = windows[-1].model_copy(
            update={
                "start_utc": source.end_utc,
                "end_utc": source.end_utc + timedelta(seconds=10),
            }
        )
        with pytest.raises(WindowingError, match="after the session end"):
            assert_windows_within_session([shifted], source)

    def test_a_window_before_the_session_start_is_rejected(self) -> None:
        source = interval(20.0)
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        window = build_windows(source, spec)[0]
        shifted = window.model_copy(
            update={
                "start_utc": source.start_utc - timedelta(seconds=5),
                "end_utc": source.start_utc + timedelta(seconds=5),
            }
        )
        with pytest.raises(WindowingError, match="before the session start"):
            assert_windows_within_session([shifted], source)


class TestSelection:
    @pytest.fixture
    def samples(self) -> list[Sample]:
        return [
            Sample(100.0 + offset, START + timedelta(seconds=offset), f"t{offset}")
            for offset in (0.0, 4.9, 5.0, 9.999, 10.0, 15.0)
        ]

    def test_selection_is_half_open(self, samples: list[Sample]) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        windows = build_windows(interval(20.0), spec)
        first = select_in_window(
            samples, windows[0], timestamp=lambda s: s.monotonic_timestamp
        )
        tags = [s.tag for s in first]
        assert tags == ["t0.0", "t4.9", "t5.0", "t9.999"]
        assert "t10.0" not in tags

    def test_a_boundary_sample_belongs_to_exactly_one_window(
        self, samples: list[Sample]
    ) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        windows = build_windows(interval(20.0), spec)
        selected = [
            sum(
                1
                for s in select_in_window(
                    samples, window, timestamp=lambda s: s.monotonic_timestamp
                )
                if s.tag == "t10.0"
            )
            for window in windows
        ]
        assert sum(selected) == 1

    def test_no_future_event_can_enter_a_window(self, samples: list[Sample]) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        window = build_windows(interval(20.0), spec)[0]
        selected = select_in_window(
            samples, window, timestamp=lambda s: s.monotonic_timestamp
        )
        assert all(
            s.monotonic_timestamp < window.end_monotonic_seconds for s in selected
        )

    def test_utc_selection_matches_monotonic_selection(
        self, samples: list[Sample]
    ) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        window = build_windows(interval(20.0), spec)[0]
        by_monotonic = select_in_window(
            samples, window, timestamp=lambda s: s.monotonic_timestamp
        )
        by_utc = select_in_window_utc(
            samples, window, timestamp=lambda s: s.utc_timestamp
        )
        assert [s.tag for s in by_monotonic] == [s.tag for s in by_utc]

    def test_selection_without_monotonic_bounds_raises(self) -> None:
        spec = WindowSpec(duration_seconds=10.0, step_seconds=10.0)
        window = build_windows(interval(20.0, monotonic=False), spec)[0]
        sample = Sample(0.0, START, "t")
        with pytest.raises(WindowingError, match="no monotonic bounds"):
            select_in_window(
                [sample], window, timestamp=lambda s: s.monotonic_timestamp
            )
