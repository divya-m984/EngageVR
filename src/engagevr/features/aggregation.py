"""Window-level aggregation of per-frame and per-event observations.

Each aggregator takes the evidence that falls inside one window and
returns the catalog features it can defend, together with an explicit
availability flag for every feature it was asked about.

Three rules are enforced here:

- **Minimum evidence gates the computation, not the interpretation.**  A
  mean over one frame is arithmetically valid and scientifically useless;
  when the gate is unmet the feature is ``None``, not a number nobody
  should trust.
- **Rejected rPPG windows contribute nothing.**  A window whose quality
  gate failed does not enter the heart-rate summary, the spectral
  summaries, or any average.  It is counted in
  ``rppg_unavailable_window_pct`` and nowhere else.
- **No construct is inferred.**  Nothing in this module produces, or is
  permitted to produce, an engagement, cognition, emotion, fatigue,
  stress, or attention value.  These are summaries of observable
  measurements.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Sequence

from pydantic import BaseModel, Field

from engagevr.features.catalog import FEATURE_CATALOG
from engagevr.features.windowing import WindowBounds, select_in_window
from engagevr.schemas.capture import (
    BehaviouralFeatures,
    CaptureQualityReport,
    HeadPoseObservation,
)
from engagevr.schemas.events import EventType, ResponseOutcome, TaskEventDetail
from engagevr.schemas.features import FeatureModality
from engagevr.schemas.rppg import RgbTraceWindow, RppgMethod, RppgMethodResult


class AggregationConfig(BaseModel):
    """Minimum-evidence thresholds shared by every aggregator.

    These are software thresholds, not empirically derived cut-offs.  They
    exist so that a feature is never computed from evidence too thin to
    support it; no claim is made that a window meeting them is
    scientifically adequate.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    min_face_frames: int = Field(
        default=5, ge=1, description="Frames with a tracked face before summarising."
    )
    min_face_seconds: float = Field(
        default=1.0,
        gt=0.0,
        description="Face-present duration before a per-minute rate is meaningful.",
    )
    min_pose_frames: int = Field(default=5, ge=1)
    min_quality_frames: int = Field(default=1, ge=1)
    min_resolved_trials: int = Field(
        default=1,
        ge=1,
        description="Resolved response slots before a proportion is computed.",
    )
    min_reaction_times: int = Field(
        default=1, ge=1, description="Registered responses before RT summaries."
    )
    min_rppg_windows: int = Field(default=1, ge=1)


class TimedTaskEvent(BaseModel):
    """One task event with the timestamp used for window assignment."""

    model_config = {"frozen": True, "extra": "forbid"}

    monotonic_timestamp: float
    detail: TaskEventDetail


class RppgWindowSummary(BaseModel):
    """A compact, window-assignable view of one rPPG pipeline result.

    Built from :class:`~engagevr.schemas.rppg.RppgMethodResult` by
    :func:`summarize_rppg_result` so that the feature layer depends on the
    rPPG *contract* rather than re-deriving rPPG internals.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    monotonic_timestamp: float = Field(
        description="Representative instant used to assign the result to a window."
    )
    method: RppgMethod
    available: bool = Field(
        description="Whether the quality gate passed and a heart rate was produced."
    )
    heart_rate_bpm: float | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    valid_frame_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    roi_available_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    timestamp_jitter_s: float | None = Field(default=None, ge=0.0)
    spectral_peak_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    peak_prominence: float | None = Field(default=None, ge=0.0)
    illumination_stability: float | None = Field(default=None, ge=0.0, le=1.0)
    motion_score: float | None = Field(default=None, ge=0.0)


class ModalityAggregate(BaseModel):
    """One modality's contribution to a window."""

    model_config = {"frozen": True, "extra": "forbid"}

    modality: FeatureModality
    available: bool
    quality: float | None = Field(default=None, ge=0.0, le=1.0)
    values: dict[str, float | None] = Field(default_factory=dict)
    categorical_values: dict[str, str | None] = Field(default_factory=dict)


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _pstdev(values: Sequence[float]) -> float | None:
    return float(statistics.pstdev(values)) if len(values) >= 2 else None


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def aggregate_behavioural(
    observations: Sequence[BehaviouralFeatures],
    window: WindowBounds,
    config: AggregationConfig,
) -> ModalityAggregate:
    """Summarise behavioural proxies for one window.

    Every output is a summary of landmark geometry.  None of them is a
    measure of alertness, attention, or affect.
    """
    frames = select_in_window(
        observations, window, timestamp=lambda o: o.monotonic_timestamp
    )
    names = tuple(
        e.canonical_name
        for e in FEATURE_CATALOG.by_modality(FeatureModality.BEHAVIOURAL)
    )
    values: dict[str, float | None] = dict.fromkeys(names)

    if not frames:
        return ModalityAggregate(
            modality=FeatureModality.BEHAVIOURAL, available=False, values=values
        )

    face_frames = [f for f in frames if f.face_present]
    values["face_presence_pct"] = _pct(len(face_frames), len(frames))

    ear_values = [f.mean_ear for f in face_frames if f.mean_ear is not None]
    if len(ear_values) >= config.min_face_frames:
        values["eye_openness_proxy_mean"] = _mean(ear_values)
        values["eye_openness_proxy_min"] = float(min(ear_values))
    if len(ear_values) >= 2:
        values["eye_openness_proxy_sd"] = _pstdev(ear_values)

    mar_values = [
        f.mouth_aspect_ratio for f in face_frames if f.mouth_aspect_ratio is not None
    ]
    if len(mar_values) >= config.min_face_frames:
        values["mouth_opening_proxy_mean"] = _mean(mar_values)
        values["mouth_opening_proxy_max"] = float(max(mar_values))
    if len(mar_values) >= 2:
        values["mouth_opening_proxy_sd"] = _pstdev(mar_values)

    if len(face_frames) >= config.min_face_frames:
        blinks = sum(1 for f in face_frames if f.blink_detected)
        values["blink_proxy_count"] = float(blinks)

        # Face-present duration is measured from the observed frames, not
        # assumed to be the whole window: a rate normalised by a duration
        # that was never observed would overstate or understate it.
        observed_span = _observed_span(face_frames)
        if observed_span is not None and observed_span >= config.min_face_seconds:
            values["blink_proxy_rate_per_min"] = 60.0 * blinks / observed_span

        closures = [
            f.eye_closure_duration_s
            for f in face_frames
            if f.eye_closure_duration_s is not None and f.eye_closure_duration_s > 0.0
        ]
        values["eye_closure_total_duration_s"] = float(sum(closures))
        if closures:
            values["eye_closure_mean_duration_s"] = _mean(closures)
            values["eye_closure_max_duration_s"] = float(max(closures))

        stable = [f for f in face_frames if f.face_tracking_stable is not None]
        if stable:
            values["face_tracking_stable_pct"] = _pct(
                sum(1 for f in stable if f.face_tracking_stable), len(stable)
            )

    computed = [n for n in names if n != "behavioural_missing_pct"]
    missing = sum(1 for n in computed if values[n] is None)
    values["behavioural_missing_pct"] = _pct(missing, len(computed))

    quality = None
    presence = values["face_presence_pct"]
    if presence is not None:
        quality = presence / 100.0

    return ModalityAggregate(
        modality=FeatureModality.BEHAVIOURAL,
        available=bool(face_frames),
        quality=quality,
        values=values,
    )


def _observed_span(frames: Sequence[BehaviouralFeatures]) -> float | None:
    """Time actually covered by ``frames``, or ``None`` if under two frames."""
    if len(frames) < 2:
        return None
    stamps = sorted(f.monotonic_timestamp for f in frames)
    span = stamps[-1] - stamps[0]
    return span if span > 0.0 else None


def aggregate_head_pose(
    observations: Sequence[HeadPoseObservation],
    window: WindowBounds,
    config: AggregationConfig,
) -> ModalityAggregate:
    """Summarise head-pose geometry for one window."""
    frames = select_in_window(
        observations, window, timestamp=lambda o: o.monotonic_timestamp
    )
    names = tuple(
        e.canonical_name for e in FEATURE_CATALOG.by_modality(FeatureModality.HEAD_POSE)
    )
    values: dict[str, float | None] = dict.fromkeys(names)

    if not frames:
        return ModalityAggregate(
            modality=FeatureModality.HEAD_POSE, available=False, values=values
        )

    available = [f for f in frames if f.available]
    values["head_pose_available_pct"] = _pct(len(available), len(frames))

    axes = (
        ("yaw", "head_yaw_mean_deg", "head_yaw_sd_deg", "head_yaw_range_deg"),
        ("pitch", "head_pitch_mean_deg", "head_pitch_sd_deg", "head_pitch_range_deg"),
        ("roll", "head_roll_mean_deg", "head_roll_sd_deg", None),
    )
    for axis, mean_key, sd_key, range_key in axes:
        samples = [
            value
            for value in (getattr(f, f"{axis}_deg") for f in available)
            if value is not None
        ]
        if len(samples) >= config.min_pose_frames:
            values[mean_key] = _mean(samples)
        if len(samples) >= 2:
            values[sd_key] = _pstdev(samples)
            if range_key is not None:
                values[range_key] = float(max(samples) - min(samples))

    velocities = [
        f.angular_velocity_deg_s
        for f in available
        if f.angular_velocity_deg_s is not None
    ]
    if len(velocities) >= config.min_pose_frames:
        values["head_angular_velocity_mean_deg_s"] = _mean(velocities)
        values["head_angular_velocity_max_deg_s"] = float(max(velocities))
    if len(velocities) >= 2:
        values["head_motion_variability_deg_s"] = _pstdev(velocities)

    quality = None
    pct = values["head_pose_available_pct"]
    if pct is not None:
        quality = pct / 100.0

    return ModalityAggregate(
        modality=FeatureModality.HEAD_POSE,
        available=bool(available),
        quality=quality,
        values=values,
    )


def aggregate_rppg(
    summaries: Sequence[RppgWindowSummary],
    window: WindowBounds,
    config: AggregationConfig,
) -> ModalityAggregate:
    """Summarise rPPG results for one window.

    Only results whose quality gate passed contribute to the heart-rate and
    spectral summaries.  A rejected window is evidence that the *camera
    signal* was unusable; averaging it into a physiological summary would
    manufacture a measurement out of a failure.
    """
    results = select_in_window(
        summaries, window, timestamp=lambda s: s.monotonic_timestamp
    )
    names = tuple(
        e.canonical_name for e in FEATURE_CATALOG.by_modality(FeatureModality.RPPG)
    )
    numeric_names = tuple(n for n in names if n != "rppg_method")
    values: dict[str, float | None] = dict.fromkeys(numeric_names)
    categorical: dict[str, str | None] = {"rppg_method": None}

    if len(results) < config.min_rppg_windows:
        return ModalityAggregate(
            modality=FeatureModality.RPPG,
            available=False,
            values=values,
            categorical_values=categorical,
        )

    accepted = [r for r in results if r.available]
    values["rppg_unavailable_window_pct"] = _pct(
        len(results) - len(accepted), len(results)
    )

    methods = sorted({r.method.value for r in results})
    categorical["rppg_method"] = methods[0] if len(methods) == 1 else "mixed"

    # Diagnostics describe every attempted window, including rejected ones:
    # they are how the rejection is explained.
    for key, attribute in (
        ("rppg_quality_score", "quality_score"),
        ("rppg_valid_frame_pct", "valid_frame_pct"),
        ("rppg_roi_available_pct", "roi_available_pct"),
        ("rppg_timestamp_jitter_s", "timestamp_jitter_s"),
        ("rppg_illumination_stability", "illumination_stability"),
        ("rppg_motion_score", "motion_score"),
    ):
        samples = [
            value
            for value in (getattr(r, attribute) for r in results)
            if value is not None
        ]
        values[key] = _mean(samples)

    # Physiological and spectral summaries come from accepted windows only.
    if accepted:
        bpm = [r.heart_rate_bpm for r in accepted if r.heart_rate_bpm is not None]
        values["rppg_heart_rate_bpm"] = _mean(bpm)
        ratios = [
            r.spectral_peak_ratio for r in accepted if r.spectral_peak_ratio is not None
        ]
        values["rppg_spectral_peak_ratio"] = _mean(ratios)
        prominences = [
            r.peak_prominence for r in accepted if r.peak_prominence is not None
        ]
        values["rppg_peak_prominence"] = _mean(prominences)

    return ModalityAggregate(
        modality=FeatureModality.RPPG,
        available=bool(accepted),
        quality=values["rppg_quality_score"],
        values=values,
        categorical_values=categorical,
    )


def aggregate_task(
    events: Sequence[TimedTaskEvent],
    window: WindowBounds,
    config: AggregationConfig,
) -> ModalityAggregate:
    """Summarise task telemetry for one window.

    Task output is a software measurement of what the task program
    observed.  It is a predictor or an outcome measurement; it is never an
    engagement, attention, cognitive-load, or fatigue measurement.
    """
    in_window = select_in_window(
        events, window, timestamp=lambda e: e.monotonic_timestamp
    )
    names = tuple(
        e.canonical_name for e in FEATURE_CATALOG.by_modality(FeatureModality.TASK)
    )
    values: dict[str, float | None] = dict.fromkeys(names)

    if not in_window:
        return ModalityAggregate(
            modality=FeatureModality.TASK, available=False, values=values
        )

    details = [e.detail for e in in_window]
    counts = {
        "task_attempted_trials": _count(details, EventType.STIMULUS_PRESENTED),
        "task_completed_trials": _count(details, EventType.TRIAL_COMPLETED),
        "task_pause_count": _count(details, EventType.TASK_PAUSED),
        "task_timeout_count": _count(details, EventType.RESPONSE_TIMEOUT),
    }
    registered = [d for d in details if d.event_type is EventType.RESPONSE_REGISTERED]
    counts["task_correct_count"] = sum(1 for d in registered if d.response_correct)
    counts["task_incorrect_count"] = sum(
        1 for d in registered if d.response_correct is False
    )
    for key, count in counts.items():
        values[key] = float(count)

    resolved = (
        counts["task_correct_count"]
        + counts["task_incorrect_count"]
        + counts["task_timeout_count"]
    )
    if resolved >= config.min_resolved_trials:
        values["task_correct_proportion"] = counts["task_correct_count"] / resolved
        values["task_incorrect_proportion"] = counts["task_incorrect_count"] / resolved
        values["task_timeout_proportion"] = counts["task_timeout_count"] / resolved

    reaction_times = [
        d.reaction_time_ms
        for d in registered
        if d.reaction_time_ms is not None
        and d.response_outcome is not ResponseOutcome.TIMEOUT
    ]
    if len(reaction_times) >= config.min_reaction_times:
        values["task_reaction_time_mean_ms"] = _mean(reaction_times)
        values["task_reaction_time_median_ms"] = float(
            statistics.median(reaction_times)
        )
        values["task_reaction_time_min_ms"] = float(min(reaction_times))
        values["task_reaction_time_max_ms"] = float(max(reaction_times))
    if len(reaction_times) >= 2:
        values["task_reaction_time_sd_ms"] = _pstdev(reaction_times)

    difficulties = [
        d.difficulty_level for d in details if d.difficulty_level is not None
    ]
    if difficulties:
        values["task_difficulty_level"] = float(difficulties[-1])

    if len(in_window) >= 2:
        stamps = sorted(e.monotonic_timestamp for e in in_window)
        values["task_inactivity_seconds"] = float(
            max(b - a for a, b in itertools.pairwise(stamps))
        )

    return ModalityAggregate(
        modality=FeatureModality.TASK, available=True, values=values
    )


def _count(details: Sequence[TaskEventDetail], event_type: EventType) -> int:
    return sum(1 for d in details if d.event_type is event_type)


def aggregate_quality(
    reports: Sequence[CaptureQualityReport],
    window: WindowBounds,
    config: AggregationConfig,
) -> ModalityAggregate:
    """Summarise capture-quality diagnostics for one window.

    Capture quality describes the measurement conditions.  It is never a
    state estimate and must never be rendered as low engagement.
    """
    frames = select_in_window(
        reports, window, timestamp=lambda r: r.monotonic_timestamp
    )
    names = tuple(
        e.canonical_name for e in FEATURE_CATALOG.by_modality(FeatureModality.QUALITY)
    )
    values: dict[str, float | None] = dict.fromkeys(names)

    if len(frames) < config.min_quality_frames:
        return ModalityAggregate(
            modality=FeatureModality.QUALITY, available=False, values=values
        )

    for key, attribute in (
        ("capture_brightness_mean", "brightness"),
        ("capture_blur_score_mean", "blur_score"),
        ("capture_motion_score_mean", "motion_score"),
    ):
        samples = [
            value
            for value in (getattr(r, attribute) for r in frames)
            if value is not None
        ]
        values[key] = _mean(samples)

    values["capture_underexposed_pct"] = _pct(
        sum(1 for r in frames if r.underexposed), len(frames)
    )
    values["capture_overexposed_pct"] = _pct(
        sum(1 for r in frames if r.overexposed), len(frames)
    )
    values["capture_blurry_pct"] = _pct(
        sum(1 for r in frames if r.is_blurry), len(frames)
    )
    dropped = sum(r.dropped_frames for r in frames)
    values["capture_dropped_frame_pct"] = _pct(dropped, dropped + len(frames))

    # An interpretable, deliberately simple composite: the fraction of
    # frames free of every flagged capture defect. No weighting is applied
    # because there is no validated basis in this repository for one.
    clean = sum(
        1
        for r in frames
        if not (r.underexposed or r.overexposed or r.is_blurry or r.excessive_motion)
    )
    quality = clean / len(frames)

    return ModalityAggregate(
        modality=FeatureModality.QUALITY,
        available=True,
        quality=quality,
        values=values,
    )


def summarize_rppg_result(
    result: RppgMethodResult,
    *,
    monotonic_timestamp: float,
    trace: RgbTraceWindow | None = None,
    roi_available_pct: float | None = None,
) -> RppgWindowSummary:
    """Build an aggregatable summary from a Milestone 3 rPPG result.

    This is the only bridge between the rPPG pipeline and the feature
    layer, so the two cannot drift apart.  ``available`` follows the
    heart-rate estimate, which is already gated on quality by
    :mod:`engagevr.rppg.window`; this function does not second-guess that
    gate and does not re-derive a heart rate from a rejected window.
    """
    components = {c.name: c for c in result.quality.components}
    illumination = components.get("illumination_stability")
    motion = components.get("capture_motion")
    valid_pct: float | None = None
    jitter: float | None = None
    if trace is not None:
        valid_pct = _pct(trace.n_valid, trace.n_samples)
        jitter = trace.timestamp_jitter_s
    return RppgWindowSummary(
        monotonic_timestamp=monotonic_timestamp,
        method=result.method,
        available=result.heart_rate.available,
        heart_rate_bpm=result.heart_rate.bpm,
        quality_score=result.quality.overall_quality,
        valid_frame_pct=valid_pct,
        roi_available_pct=roi_available_pct,
        timestamp_jitter_s=jitter,
        spectral_peak_ratio=result.heart_rate.spectral_peak_ratio,
        peak_prominence=result.heart_rate.peak_prominence,
        illumination_stability=illumination.score if illumination else None,
        motion_score=motion.value if motion is not None else None,
    )


def combine_aggregates(
    aggregates: Sequence[ModalityAggregate],
) -> tuple[
    dict[str, float | None],
    dict[str, str | None],
    dict[str, bool],
    dict[str, bool],
    dict[str, float | None],
]:
    """Merge modality aggregates into the dictionaries a row needs.

    Returns ``(numeric, categorical, availability, modality_available,
    modality_quality)``.  Every catalog feature appears in exactly one of
    ``numeric`` or ``categorical`` and in ``availability``, so a caller
    cannot accidentally build a row with an undeclared or absent feature.
    """
    numeric: dict[str, float | None] = {}
    categorical: dict[str, str | None] = {}
    modality_available: dict[str, bool] = {}
    modality_quality: dict[str, float | None] = {}

    for aggregate in aggregates:
        modality_available[aggregate.modality.value] = aggregate.available
        modality_quality[aggregate.modality.value] = aggregate.quality
        numeric.update(aggregate.values)
        categorical.update(aggregate.categorical_values)

    known = set(FEATURE_CATALOG.names())
    produced = set(numeric) | set(categorical)
    undeclared = produced - known
    if undeclared:
        raise ValueError(
            "aggregation produced features that are not in the catalog: "
            f"{sorted(undeclared)}"
        )
    absent = known - produced
    if absent:
        raise ValueError(f"aggregation omitted catalog features: {sorted(absent)}")

    for name, value in numeric.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(
                f"feature {name!r} aggregated to a non-finite value {value!r}; "
                "an undefined result must be reported as unavailable, not as "
                "infinity or NaN"
            )

    availability: dict[str, bool] = {
        name: value is not None for name, value in numeric.items()
    }
    availability.update(
        {name: value is not None for name, value in categorical.items()}
    )

    # window_missing_feature_pct describes the whole row, so it is computed
    # last, over every other feature, and is itself always available.
    others = [n for n in availability if n != "window_missing_feature_pct"]
    missing = sum(1 for n in others if not availability[n])
    numeric["window_missing_feature_pct"] = (
        100.0 * missing / len(others) if others else 0.0
    )
    availability["window_missing_feature_pct"] = True

    return numeric, categorical, availability, modality_available, modality_quality
