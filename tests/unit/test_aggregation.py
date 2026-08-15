"""Window aggregation tests.

Three properties matter here and are asserted directly:

- a feature whose minimum evidence is unmet is ``None``, not a number;
- a rejected rPPG window contributes nothing to any physiological summary;
- no aggregator produces anything that could be read as a psychological
  construct.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from engagevr.features.aggregation import (
    AggregationConfig,
    RppgWindowSummary,
    TimedTaskEvent,
    aggregate_behavioural,
    aggregate_head_pose,
    aggregate_quality,
    aggregate_rppg,
    aggregate_task,
    combine_aggregates,
    summarize_rppg_result,
)
from engagevr.features.catalog import FEATURE_CATALOG
from engagevr.features.windowing import SessionInterval, WindowSpec, build_windows
from engagevr.schemas.capture import (
    BehaviouralFeatures,
    CaptureQualityReport,
    HeadPoseObservation,
)
from engagevr.schemas.events import EventType, ResponseOutcome, TaskEventDetail
from engagevr.schemas.features import FeatureModality
from engagevr.schemas.rppg import (
    HeartRateEstimate,
    RgbTraceWindow,
    RppgMethod,
    RppgMethodResult,
    RppgQualityComponent,
    RppgQualityReport,
    RppgWaveform,
    UnavailableReason,
)

START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
CONFIG = AggregationConfig()


@pytest.fixture
def window():  # type: ignore[no-untyped-def]
    interval = SessionInterval(
        session_id="s",
        start_utc=START,
        end_utc=START + timedelta(seconds=10),
        start_monotonic_seconds=0.0,
        end_monotonic_seconds=10.0,
    )
    return build_windows(
        interval, WindowSpec(duration_seconds=10.0, step_seconds=10.0)
    )[0]


def behavioural(
    timestamp: float,
    *,
    face: bool = True,
    ear: float | None = 0.3,
    mar: float | None = 0.1,
    blink: bool = False,
    closure: float | None = None,
    stable: bool | None = True,
) -> BehaviouralFeatures:
    return BehaviouralFeatures(
        session_id="s",
        frame_index=int(timestamp * 10),
        monotonic_timestamp=timestamp,
        face_present=face,
        mean_ear=ear if face else None,
        mouth_aspect_ratio=mar if face else None,
        blink_detected=blink if face else None,
        eye_closure_duration_s=closure,
        face_tracking_stable=stable if face else None,
    )


class TestBehaviouralAggregation:
    def test_face_presence_percentage(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [behavioural(float(i), face=i < 6) for i in range(10)]
        result = aggregate_behavioural(frames, window, CONFIG)
        assert result.values["face_presence_pct"] == pytest.approx(60.0)

    def test_eye_openness_summaries(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [behavioural(float(i), ear=0.2 + 0.02 * i) for i in range(8)]
        result = aggregate_behavioural(frames, window, CONFIG)
        assert result.values["eye_openness_proxy_mean"] == pytest.approx(0.27)
        assert result.values["eye_openness_proxy_min"] == pytest.approx(0.2)
        assert result.values["eye_openness_proxy_sd"] is not None

    def test_blink_count_is_zero_when_a_face_was_tracked(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [behavioural(float(i), blink=False) for i in range(8)]
        result = aggregate_behavioural(frames, window, CONFIG)
        assert result.values["blink_proxy_count"] == 0.0

    def test_blink_count_counts_flagged_frames(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [behavioural(float(i), blink=i in (2, 5)) for i in range(8)]
        result = aggregate_behavioural(frames, window, CONFIG)
        assert result.values["blink_proxy_count"] == 2.0
        assert result.values["blink_proxy_rate_per_min"] is not None

    def test_eye_closure_summaries(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [
            behavioural(float(i), closure=0.2 if i in (1, 4) else None)
            for i in range(8)
        ]
        result = aggregate_behavioural(frames, window, CONFIG)
        assert result.values["eye_closure_total_duration_s"] == pytest.approx(0.4)
        assert result.values["eye_closure_max_duration_s"] == pytest.approx(0.2)
        assert result.values["eye_closure_mean_duration_s"] == pytest.approx(0.2)

    def test_no_closure_episode_leaves_mean_unavailable(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [behavioural(float(i)) for i in range(8)]
        result = aggregate_behavioural(frames, window, CONFIG)
        assert result.values["eye_closure_mean_duration_s"] is None
        assert result.values["eye_closure_total_duration_s"] == 0.0

    def test_minimum_evidence_is_enforced(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [behavioural(float(i)) for i in range(3)]
        result = aggregate_behavioural(frames, window, CONFIG)
        assert result.values["eye_openness_proxy_mean"] is None
        assert result.values["blink_proxy_count"] is None
        assert result.values["face_presence_pct"] == pytest.approx(100.0)

    def test_no_frames_makes_the_modality_unavailable(self, window) -> None:  # type: ignore[no-untyped-def]
        result = aggregate_behavioural([], window, CONFIG)
        assert result.available is False
        assert all(value is None for value in result.values.values())

    def test_frames_outside_the_window_are_ignored(self, window) -> None:  # type: ignore[no-untyped-def]
        inside = [behavioural(float(i)) for i in range(8)]
        outside = [behavioural(20.0 + i, ear=0.9) for i in range(8)]
        result = aggregate_behavioural(inside + outside, window, CONFIG)
        assert result.values["eye_openness_proxy_mean"] == pytest.approx(0.3)


class TestHeadPoseAggregation:
    def _pose(self, timestamp: float, yaw: float, available: bool = True):  # type: ignore[no-untyped-def]
        return HeadPoseObservation(
            session_id="s",
            frame_index=int(timestamp),
            monotonic_timestamp=timestamp,
            available=available,
            yaw_deg=yaw if available else None,
            pitch_deg=-yaw if available else None,
            roll_deg=0.0 if available else None,
            angular_velocity_deg_s=abs(yaw) if available else None,
        )

    def test_axis_summaries(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [self._pose(float(i), float(i - 3)) for i in range(8)]
        result = aggregate_head_pose(frames, window, CONFIG)
        assert result.values["head_yaw_mean_deg"] == pytest.approx(0.5)
        assert result.values["head_yaw_range_deg"] == pytest.approx(7.0)
        assert result.values["head_yaw_sd_deg"] is not None
        assert result.values["head_motion_variability_deg_s"] is not None

    def test_availability_percentage(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [self._pose(float(i), 1.0, available=i < 5) for i in range(10)]
        result = aggregate_head_pose(frames, window, CONFIG)
        assert result.values["head_pose_available_pct"] == pytest.approx(50.0)

    def test_too_few_poses_leaves_means_unavailable(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [self._pose(float(i), 1.0) for i in range(3)]
        result = aggregate_head_pose(frames, window, CONFIG)
        assert result.values["head_yaw_mean_deg"] is None
        assert result.values["head_yaw_sd_deg"] is not None


class TestRppgAggregation:
    def _summary(
        self,
        timestamp: float,
        *,
        available: bool,
        bpm: float | None,
        quality: float,
    ) -> RppgWindowSummary:
        return RppgWindowSummary(
            monotonic_timestamp=timestamp,
            method=RppgMethod.POS,
            available=available,
            heart_rate_bpm=bpm,
            quality_score=quality,
            valid_frame_pct=95.0,
            roi_available_pct=90.0,
            timestamp_jitter_s=0.005,
            spectral_peak_ratio=0.5 if available else None,
            peak_prominence=0.4 if available else None,
            illumination_stability=0.8,
            motion_score=10.0,
        )

    def test_accepted_windows_produce_a_heart_rate(self, window) -> None:  # type: ignore[no-untyped-def]
        summaries = [
            self._summary(1.0, available=True, bpm=70.0, quality=0.8),
            self._summary(5.0, available=True, bpm=80.0, quality=0.9),
        ]
        result = aggregate_rppg(summaries, window, CONFIG)
        assert result.available is True
        assert result.values["rppg_heart_rate_bpm"] == pytest.approx(75.0)
        assert result.values["rppg_unavailable_window_pct"] == pytest.approx(0.0)

    def test_rejected_windows_never_enter_the_heart_rate_summary(self, window) -> None:  # type: ignore[no-untyped-def]
        summaries = [
            self._summary(1.0, available=True, bpm=70.0, quality=0.8),
            self._summary(5.0, available=False, bpm=200.0, quality=0.2),
        ]
        result = aggregate_rppg(summaries, window, CONFIG)
        assert result.values["rppg_heart_rate_bpm"] == pytest.approx(70.0)
        assert result.values["rppg_unavailable_window_pct"] == pytest.approx(50.0)

    def test_all_rejected_leaves_the_estimate_unavailable(self, window) -> None:  # type: ignore[no-untyped-def]
        summaries = [
            self._summary(1.0, available=False, bpm=None, quality=0.2),
            self._summary(5.0, available=False, bpm=None, quality=0.1),
        ]
        result = aggregate_rppg(summaries, window, CONFIG)
        assert result.available is False
        assert result.values["rppg_heart_rate_bpm"] is None
        assert result.values["rppg_spectral_peak_ratio"] is None
        assert result.values["rppg_unavailable_window_pct"] == pytest.approx(100.0)
        # Diagnostics still describe what was attempted.
        assert result.values["rppg_quality_score"] is not None

    def test_no_windows_makes_the_modality_unavailable(self, window) -> None:  # type: ignore[no-untyped-def]
        result = aggregate_rppg([], window, CONFIG)
        assert result.available is False
        assert result.categorical_values["rppg_method"] is None

    def test_method_is_recorded_and_mixed_methods_are_named(self, window) -> None:  # type: ignore[no-untyped-def]
        one = self._summary(1.0, available=True, bpm=70.0, quality=0.8)
        other = one.model_copy(
            update={"method": RppgMethod.CHROM, "monotonic_timestamp": 5.0}
        )
        single = aggregate_rppg([one], window, CONFIG)
        mixed = aggregate_rppg([one, other], window, CONFIG)
        assert single.categorical_values["rppg_method"] == "pos"
        assert mixed.categorical_values["rppg_method"] == "mixed"

    def test_summarize_rppg_result_bridges_the_milestone_3_contract(self) -> None:
        result = RppgMethodResult(
            method=RppgMethod.POS,
            waveform=RppgWaveform(method=RppgMethod.POS, available=True),
            quality=RppgQualityReport(
                components=[
                    RppgQualityComponent(
                        name="illumination_stability", score=0.75, value=0.01
                    ),
                    RppgQualityComponent(name="capture_motion", score=0.6, value=12.0),
                ],
                overall_quality=0.7,
                acceptable=True,
                quality_threshold=0.5,
            ),
            heart_rate=HeartRateEstimate(
                available=True,
                bpm=72.0,
                spectral_peak_ratio=0.6,
                peak_prominence=0.5,
            ),
        )
        trace = RgbTraceWindow(n_samples=100, n_valid=90, timestamp_jitter_s=0.004)
        summary = summarize_rppg_result(
            result, monotonic_timestamp=2.0, trace=trace, roi_available_pct=88.0
        )
        assert summary.available is True
        assert summary.heart_rate_bpm == pytest.approx(72.0)
        assert summary.valid_frame_pct == pytest.approx(90.0)
        assert summary.illumination_stability == pytest.approx(0.75)
        assert summary.motion_score == pytest.approx(12.0)

    def test_summarize_follows_the_milestone_3_quality_gate(self) -> None:
        result = RppgMethodResult(
            method=RppgMethod.POS,
            waveform=RppgWaveform(method=RppgMethod.POS, available=False),
            quality=RppgQualityReport(
                overall_quality=0.2, acceptable=False, quality_threshold=0.5
            ),
            heart_rate=HeartRateEstimate(
                available=False,
                bpm=None,
                reason=UnavailableReason.INSUFFICIENT_SIGNAL_QUALITY,
            ),
        )
        summary = summarize_rppg_result(result, monotonic_timestamp=1.0)
        assert summary.available is False
        assert summary.heart_rate_bpm is None


class TestTaskAggregation:
    def _event(
        self,
        timestamp: float,
        event_type: EventType,
        **fields: object,
    ) -> TimedTaskEvent:
        return TimedTaskEvent(
            monotonic_timestamp=timestamp,
            detail=TaskEventDetail(event_type=event_type, **fields),  # type: ignore[arg-type]
        )

    def test_counts_and_proportions(self, window) -> None:  # type: ignore[no-untyped-def]
        events = [
            self._event(0.5, EventType.STIMULUS_PRESENTED, trial_id=0),
            self._event(
                1.0,
                EventType.RESPONSE_REGISTERED,
                trial_id=0,
                response_correct=True,
                response_outcome=ResponseOutcome.CORRECT,
                reaction_time_ms=400.0,
            ),
            self._event(2.0, EventType.STIMULUS_PRESENTED, trial_id=1),
            self._event(
                2.5,
                EventType.RESPONSE_REGISTERED,
                trial_id=1,
                response_correct=False,
                response_outcome=ResponseOutcome.INCORRECT,
                reaction_time_ms=600.0,
            ),
            self._event(4.0, EventType.STIMULUS_PRESENTED, trial_id=2),
            self._event(
                5.0,
                EventType.RESPONSE_TIMEOUT,
                trial_id=2,
                response_outcome=ResponseOutcome.TIMEOUT,
            ),
        ]
        result = aggregate_task(events, window, CONFIG)
        values = result.values
        assert values["task_attempted_trials"] == 3.0
        assert values["task_correct_count"] == 1.0
        assert values["task_incorrect_count"] == 1.0
        assert values["task_timeout_count"] == 1.0
        assert values["task_correct_proportion"] == pytest.approx(1 / 3)
        assert values["task_timeout_proportion"] == pytest.approx(1 / 3)

    def test_a_timeout_contributes_no_reaction_time(self, window) -> None:  # type: ignore[no-untyped-def]
        events = [
            self._event(
                1.0,
                EventType.RESPONSE_REGISTERED,
                response_correct=True,
                reaction_time_ms=500.0,
            ),
            self._event(
                2.0,
                EventType.RESPONSE_TIMEOUT,
                response_outcome=ResponseOutcome.TIMEOUT,
            ),
        ]
        result = aggregate_task(events, window, CONFIG)
        assert result.values["task_reaction_time_mean_ms"] == pytest.approx(500.0)
        assert result.values["task_reaction_time_max_ms"] == pytest.approx(500.0)

    def test_reaction_time_standard_deviation_needs_two_responses(self, window) -> None:  # type: ignore[no-untyped-def]
        events = [
            self._event(
                1.0,
                EventType.RESPONSE_REGISTERED,
                response_correct=True,
                reaction_time_ms=500.0,
            )
        ]
        result = aggregate_task(events, window, CONFIG)
        assert result.values["task_reaction_time_sd_ms"] is None

    def test_difficulty_is_the_last_observed_value(self, window) -> None:  # type: ignore[no-untyped-def]
        events = [
            self._event(1.0, EventType.TRIAL_STARTED, difficulty_level=1),
            self._event(5.0, EventType.TRIAL_STARTED, difficulty_level=3),
        ]
        result = aggregate_task(events, window, CONFIG)
        assert result.values["task_difficulty_level"] == 3.0

    def test_inactivity_is_the_largest_in_window_gap(self, window) -> None:  # type: ignore[no-untyped-def]
        events = [
            self._event(0.0, EventType.TASK_STARTED),
            self._event(1.0, EventType.TRIAL_STARTED),
            self._event(7.0, EventType.TRIAL_STARTED),
        ]
        result = aggregate_task(events, window, CONFIG)
        assert result.values["task_inactivity_seconds"] == pytest.approx(6.0)

    def test_events_after_the_window_never_contribute(self, window) -> None:  # type: ignore[no-untyped-def]
        inside = [self._event(1.0, EventType.STIMULUS_PRESENTED)]
        future = [self._event(50.0, EventType.STIMULUS_PRESENTED) for _ in range(9)]
        result = aggregate_task(inside + future, window, CONFIG)
        assert result.values["task_attempted_trials"] == 1.0

    def test_no_events_makes_the_modality_unavailable(self, window) -> None:  # type: ignore[no-untyped-def]
        result = aggregate_task([], window, CONFIG)
        assert result.available is False
        assert result.values["task_correct_proportion"] is None


class TestQualityAggregation:
    def _report(self, timestamp: float, **fields: object) -> CaptureQualityReport:
        defaults: dict[str, object] = {
            "session_id": "s",
            "frame_index": int(timestamp),
            "monotonic_timestamp": timestamp,
            "webcam_open": True,
            "frame_read_success": True,
            "brightness": 120.0,
            "blur_score": 150.0,
            "motion_score": 10.0,
        }
        defaults.update(fields)
        return CaptureQualityReport(**defaults)  # type: ignore[arg-type]

    def test_means_and_flag_percentages(self, window) -> None:  # type: ignore[no-untyped-def]
        reports = [
            self._report(float(i), is_blurry=i < 2, underexposed=i == 0)
            for i in range(4)
        ]
        result = aggregate_quality(reports, window, CONFIG)
        assert result.values["capture_brightness_mean"] == pytest.approx(120.0)
        assert result.values["capture_blurry_pct"] == pytest.approx(50.0)
        assert result.values["capture_underexposed_pct"] == pytest.approx(25.0)

    def test_dropped_frame_percentage(self, window) -> None:  # type: ignore[no-untyped-def]
        reports = [self._report(float(i), dropped_frames=1) for i in range(3)]
        result = aggregate_quality(reports, window, CONFIG)
        assert result.values["capture_dropped_frame_pct"] == pytest.approx(50.0)

    def test_quality_score_reflects_defect_free_frames(self, window) -> None:  # type: ignore[no-untyped-def]
        reports = [self._report(float(i), is_blurry=i < 2) for i in range(4)]
        result = aggregate_quality(reports, window, CONFIG)
        assert result.quality == pytest.approx(0.5)


class TestCombineAggregates:
    def _full(self, window):  # type: ignore[no-untyped-def]
        frames = [behavioural(float(i)) for i in range(8)]
        return combine_aggregates(
            [
                aggregate_behavioural(frames, window, CONFIG),
                aggregate_head_pose([], window, CONFIG),
                aggregate_rppg([], window, CONFIG),
                aggregate_task([], window, CONFIG),
                aggregate_quality([], window, CONFIG),
            ]
        )

    def test_every_catalog_feature_is_present_exactly_once(self, window) -> None:  # type: ignore[no-untyped-def]
        numeric, categorical, availability, _, _ = self._full(window)
        assert set(numeric) | set(categorical) == set(FEATURE_CATALOG.names())
        assert not set(numeric) & set(categorical)
        assert set(availability) == set(FEATURE_CATALOG.names())

    def test_availability_matches_nullness(self, window) -> None:  # type: ignore[no-untyped-def]
        numeric, categorical, availability, _, _ = self._full(window)
        for name, value in {**numeric, **categorical}.items():
            assert availability[name] == (value is not None), name

    def test_modality_availability_is_reported_per_group(self, window) -> None:  # type: ignore[no-untyped-def]
        _, _, _, modality_available, modality_quality = self._full(window)
        assert modality_available["behavioural"] is True
        assert modality_available["rppg"] is False
        assert modality_quality["rppg"] is None
        assert set(modality_available) == {m.value for m in FeatureModality}

    def test_window_missing_feature_pct_is_always_available(self, window) -> None:  # type: ignore[no-untyped-def]
        numeric, _, availability, _, _ = self._full(window)
        assert availability["window_missing_feature_pct"] is True
        assert numeric["window_missing_feature_pct"] > 0.0

    def test_undeclared_features_are_refused(self, window) -> None:  # type: ignore[no-untyped-def]
        aggregate = aggregate_task([], window, CONFIG)
        polluted = aggregate.model_copy(
            update={"values": {**aggregate.values, "invented_feature": 1.0}}
        )
        with pytest.raises(ValueError, match="not in the catalog"):
            combine_aggregates([polluted])

    def test_omitted_features_are_refused(self, window) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="omitted catalog features"):
            combine_aggregates([aggregate_task([], window, CONFIG)])

    def test_non_finite_values_are_refused(self, window) -> None:  # type: ignore[no-untyped-def]
        frames = [behavioural(float(i)) for i in range(8)]
        behaviour = aggregate_behavioural(frames, window, CONFIG)
        broken = behaviour.model_copy(
            update={"values": {**behaviour.values, "face_presence_pct": float("inf")}}
        )
        with pytest.raises(ValueError, match="non-finite"):
            combine_aggregates(
                [
                    broken,
                    aggregate_head_pose([], window, CONFIG),
                    aggregate_rppg([], window, CONFIG),
                    aggregate_task([], window, CONFIG),
                    aggregate_quality([], window, CONFIG),
                ]
            )
