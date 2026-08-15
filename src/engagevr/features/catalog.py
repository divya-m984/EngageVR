"""The feature catalog: the single declaration of every modelling feature.

Every entry states, for one feature:

- its canonical name and unit;
- the exact aggregation applied to within-window evidence;
- the **minimum evidence** below which the feature is not computed at all;
- what a missing value means;
- which quality gate, if any, must pass first;
- whether the feature is permitted as a model input.

Two rules are enforced by construction rather than by review:

1. *No feature in this catalog is a psychological construct.*  Names such
   as "eye-openness proxy" and "blink proxy" are deliberate: an eye
   aspect ratio is a geometric ratio between landmark distances, and
   calling it "alertness" would smuggle in a conclusion the measurement
   cannot support.
2. *Poor evidence produces null, not a low number.*  Every entry whose
   ``missing_behaviour`` is ``NULL_WHEN_*`` yields ``None`` when its
   minimum evidence is unmet.  ``ZERO_WHEN_NO_EVENTS`` is reserved for
   counts where a zero is a genuine observation — a window in which a
   tracked face produced no blinks really did contain no blinks.
"""

from __future__ import annotations

from engagevr.schemas.features import (
    FeatureCatalog,
    FeatureDtype,
    FeatureModality,
    FeatureSpec,
    MissingBehaviour,
)

#: Version of the catalog. Bump on any change to entries, order, or
#: semantics; it is part of every dataset fingerprint.
FEATURE_CATALOG_VERSION = "1.0"

_NO_QUALITY_GATE = "none"


def _spec(
    name: str,
    modality: FeatureModality,
    unit: str,
    description: str,
    formula: str,
    minimum_evidence: str,
    *,
    missing: MissingBehaviour = MissingBehaviour.NULL_WHEN_EVIDENCE_INSUFFICIENT,
    quality_dependency: str = _NO_QUALITY_GATE,
    permitted_predictor: bool = True,
    dtype: FeatureDtype = FeatureDtype.FLOAT,
    value_minimum: float | None = None,
    value_maximum: float | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        canonical_name=name,
        description=description,
        modality=modality,
        unit=unit,
        aggregation_formula=formula,
        minimum_evidence=minimum_evidence,
        missing_behaviour=missing,
        quality_dependency=quality_dependency,
        permitted_predictor=permitted_predictor,
        dtype=dtype,
        value_minimum=value_minimum,
        value_maximum=value_maximum,
    )


_BEHAVIOURAL: tuple[FeatureSpec, ...] = (
    _spec(
        "face_presence_pct",
        FeatureModality.BEHAVIOURAL,
        "percent",
        "Percentage of frames in the window in which a face was detected.",
        "100 * count(face_present) / count(frames)",
        "at least one frame observation in the window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
    _spec(
        "eye_openness_proxy_mean",
        FeatureModality.BEHAVIOURAL,
        "dimensionless",
        (
            "Mean eye aspect ratio (EAR) over frames with a tracked face. A "
            "geometric landmark ratio, not a measure of alertness."
        ),
        "mean(mean_ear) over frames with mean_ear available",
        "at least `min_face_frames` frames with a usable EAR",
        quality_dependency="face must be present and landmarks available",
        value_minimum=0.0,
    ),
    _spec(
        "eye_openness_proxy_sd",
        FeatureModality.BEHAVIOURAL,
        "dimensionless",
        "Standard deviation of the eye aspect ratio within the window.",
        "population standard deviation of mean_ear over available frames",
        "at least 2 frames with a usable EAR",
        quality_dependency="face must be present and landmarks available",
        value_minimum=0.0,
    ),
    _spec(
        "eye_openness_proxy_min",
        FeatureModality.BEHAVIOURAL,
        "dimensionless",
        "Smallest eye aspect ratio observed within the window.",
        "min(mean_ear) over available frames",
        "at least `min_face_frames` frames with a usable EAR",
        quality_dependency="face must be present and landmarks available",
        value_minimum=0.0,
    ),
    _spec(
        "blink_proxy_count",
        FeatureModality.BEHAVIOURAL,
        "count",
        (
            "Number of blink-proxy events whose onset falls inside the window. "
            "A blink proxy is an EAR excursion below threshold, not a "
            "physiological blink measurement."
        ),
        "count of frames flagged blink_detected with onset inside the window",
        "at least `min_face_frames` frames with a tracked face",
        missing=MissingBehaviour.ZERO_WHEN_NO_EVENTS,
        quality_dependency="face must be present for the counted frames",
        dtype=FeatureDtype.INT,
        value_minimum=0.0,
    ),
    _spec(
        "blink_proxy_rate_per_min",
        FeatureModality.BEHAVIOURAL,
        "events/minute",
        "Blink-proxy count normalised by the observed face-present duration.",
        "60 * blink_proxy_count / face_present_duration_seconds",
        "face present for at least `min_face_seconds` of the window",
        quality_dependency="face must be present for the counted frames",
        value_minimum=0.0,
    ),
    _spec(
        "eye_closure_mean_duration_s",
        FeatureModality.BEHAVIOURAL,
        "seconds",
        "Mean duration of eye-closure proxy episodes ending within the window.",
        "mean(eye_closure_duration_s) over episodes ending in the window",
        "at least one closure episode ending inside the window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        quality_dependency="face must be present and landmarks available",
        value_minimum=0.0,
    ),
    _spec(
        "eye_closure_max_duration_s",
        FeatureModality.BEHAVIOURAL,
        "seconds",
        "Longest eye-closure proxy episode ending within the window.",
        "max(eye_closure_duration_s) over episodes ending in the window",
        "at least one closure episode ending inside the window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        quality_dependency="face must be present and landmarks available",
        value_minimum=0.0,
    ),
    _spec(
        "eye_closure_total_duration_s",
        FeatureModality.BEHAVIOURAL,
        "seconds",
        "Summed duration of eye-closure proxy episodes ending in the window.",
        "sum(eye_closure_duration_s) over episodes ending in the window",
        "at least `min_face_frames` frames with a tracked face",
        missing=MissingBehaviour.ZERO_WHEN_NO_EVENTS,
        quality_dependency="face must be present and landmarks available",
        value_minimum=0.0,
    ),
    _spec(
        "mouth_opening_proxy_mean",
        FeatureModality.BEHAVIOURAL,
        "dimensionless",
        (
            "Mean mouth aspect ratio. A geometric landmark ratio; it does not "
            "identify speech, yawning, or affect."
        ),
        "mean(mouth_aspect_ratio) over available frames",
        "at least `min_face_frames` frames with a usable MAR",
        quality_dependency="face must be present and landmarks available",
        value_minimum=0.0,
    ),
    _spec(
        "mouth_opening_proxy_sd",
        FeatureModality.BEHAVIOURAL,
        "dimensionless",
        "Standard deviation of the mouth aspect ratio within the window.",
        "population standard deviation of mouth_aspect_ratio",
        "at least 2 frames with a usable MAR",
        quality_dependency="face must be present and landmarks available",
        value_minimum=0.0,
    ),
    _spec(
        "mouth_opening_proxy_max",
        FeatureModality.BEHAVIOURAL,
        "dimensionless",
        "Largest mouth aspect ratio observed within the window.",
        "max(mouth_aspect_ratio) over available frames",
        "at least `min_face_frames` frames with a usable MAR",
        quality_dependency="face must be present and landmarks available",
        value_minimum=0.0,
    ),
    _spec(
        "face_tracking_stable_pct",
        FeatureModality.BEHAVIOURAL,
        "percent",
        "Percentage of face-present frames whose tracking was reported stable.",
        "100 * count(face_tracking_stable) / count(face_present frames)",
        "at least `min_face_frames` frames with a tracked face",
        value_minimum=0.0,
        value_maximum=100.0,
    ),
    _spec(
        "behavioural_missing_pct",
        FeatureModality.BEHAVIOURAL,
        "percent",
        (
            "Percentage of behavioural feature slots in this window that could "
            "not be computed. A missingness diagnostic, never a state estimate."
        ),
        "100 * count(null behavioural features) / count(behavioural features)",
        "always computable",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
)


_HEAD_POSE: tuple[FeatureSpec, ...] = (
    _spec(
        "head_yaw_mean_deg",
        FeatureModality.HEAD_POSE,
        "degrees",
        "Mean head yaw. Positive is turning toward image right.",
        "mean(yaw_deg) over frames with an available pose",
        "at least `min_pose_frames` frames with an available pose",
        quality_dependency="PnP solve must have converged for the frame",
    ),
    _spec(
        "head_yaw_sd_deg",
        FeatureModality.HEAD_POSE,
        "degrees",
        "Standard deviation of head yaw within the window.",
        "population standard deviation of yaw_deg",
        "at least 2 frames with an available pose",
        quality_dependency="PnP solve must have converged for the frame",
        value_minimum=0.0,
    ),
    _spec(
        "head_yaw_range_deg",
        FeatureModality.HEAD_POSE,
        "degrees",
        "Peak-to-peak head yaw within the window.",
        "max(yaw_deg) - min(yaw_deg)",
        "at least 2 frames with an available pose",
        quality_dependency="PnP solve must have converged for the frame",
        value_minimum=0.0,
    ),
    _spec(
        "head_pitch_mean_deg",
        FeatureModality.HEAD_POSE,
        "degrees",
        "Mean head pitch. Positive is looking up.",
        "mean(pitch_deg) over frames with an available pose",
        "at least `min_pose_frames` frames with an available pose",
        quality_dependency="PnP solve must have converged for the frame",
    ),
    _spec(
        "head_pitch_sd_deg",
        FeatureModality.HEAD_POSE,
        "degrees",
        "Standard deviation of head pitch within the window.",
        "population standard deviation of pitch_deg",
        "at least 2 frames with an available pose",
        quality_dependency="PnP solve must have converged for the frame",
        value_minimum=0.0,
    ),
    _spec(
        "head_pitch_range_deg",
        FeatureModality.HEAD_POSE,
        "degrees",
        "Peak-to-peak head pitch within the window.",
        "max(pitch_deg) - min(pitch_deg)",
        "at least 2 frames with an available pose",
        quality_dependency="PnP solve must have converged for the frame",
        value_minimum=0.0,
    ),
    _spec(
        "head_roll_mean_deg",
        FeatureModality.HEAD_POSE,
        "degrees",
        "Mean head roll. Positive is clockwise tilt in the image plane.",
        "mean(roll_deg) over frames with an available pose",
        "at least `min_pose_frames` frames with an available pose",
        quality_dependency="PnP solve must have converged for the frame",
    ),
    _spec(
        "head_roll_sd_deg",
        FeatureModality.HEAD_POSE,
        "degrees",
        "Standard deviation of head roll within the window.",
        "population standard deviation of roll_deg",
        "at least 2 frames with an available pose",
        quality_dependency="PnP solve must have converged for the frame",
        value_minimum=0.0,
    ),
    _spec(
        "head_angular_velocity_mean_deg_s",
        FeatureModality.HEAD_POSE,
        "degrees/second",
        "Mean magnitude of head angular velocity within the window.",
        "mean(angular_velocity_deg_s) over frames where it is available",
        "at least `min_pose_frames` frames with an available velocity",
        quality_dependency="PnP solve must have converged for the frame",
        value_minimum=0.0,
    ),
    _spec(
        "head_angular_velocity_max_deg_s",
        FeatureModality.HEAD_POSE,
        "degrees/second",
        "Largest head angular-velocity magnitude within the window.",
        "max(angular_velocity_deg_s) over available frames",
        "at least `min_pose_frames` frames with an available velocity",
        quality_dependency="PnP solve must have converged for the frame",
        value_minimum=0.0,
    ),
    _spec(
        "head_motion_variability_deg_s",
        FeatureModality.HEAD_POSE,
        "degrees/second",
        "Standard deviation of head angular velocity within the window.",
        "population standard deviation of angular_velocity_deg_s",
        "at least 2 frames with an available velocity",
        quality_dependency="PnP solve must have converged for the frame",
        value_minimum=0.0,
    ),
    _spec(
        "head_pose_available_pct",
        FeatureModality.HEAD_POSE,
        "percent",
        "Percentage of frames in the window with an available head pose.",
        "100 * count(pose available) / count(frames)",
        "at least one frame observation in the window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
)


_RPPG: tuple[FeatureSpec, ...] = (
    _spec(
        "rppg_heart_rate_bpm",
        FeatureModality.RPPG,
        "beats/minute",
        (
            "Spectral heart-rate estimate for the window. Present only when "
            "the rPPG quality gate passed. An unvalidated signal-processing "
            "estimate: not a medical measurement, not engagement, not load."
        ),
        "carried through from the accepted rPPG window result",
        "an rPPG window result with available=True",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        quality_dependency=(
            "rPPG quality must be acceptable; rejected windows contribute "
            "nothing, they are not averaged in"
        ),
        value_minimum=0.0,
    ),
    _spec(
        "rppg_quality_score",
        FeatureModality.RPPG,
        "dimensionless",
        (
            "Aggregate rPPG signal-quality score in [0, 1]. Describes the "
            "measurement, never the person."
        ),
        "mean of overall_quality over rPPG windows overlapping this window",
        "at least one rPPG window result overlapping this window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=1.0,
    ),
    _spec(
        "rppg_roi_available_pct",
        FeatureModality.RPPG,
        "percent",
        "Percentage of frames with a usable skin region of interest.",
        "100 * count(roi available) / count(frames)",
        "at least one frame observation in the window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
    _spec(
        "rppg_valid_frame_pct",
        FeatureModality.RPPG,
        "percent",
        "Percentage of RGB trace samples marked valid.",
        "100 * n_valid / n_samples over overlapping rPPG windows",
        "at least one rPPG window result overlapping this window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
    _spec(
        "rppg_timestamp_jitter_s",
        FeatureModality.RPPG,
        "seconds",
        "Standard deviation of inter-sample intervals in the RGB trace.",
        "mean of timestamp_jitter_s over overlapping rPPG windows",
        "at least one rPPG window with a computable jitter",
        quality_dependency="timing diagnostics must be computable",
        value_minimum=0.0,
    ),
    _spec(
        "rppg_spectral_peak_ratio",
        FeatureModality.RPPG,
        "dimensionless",
        "Fraction of in-band power concentrated at the selected pulse peak.",
        "mean of spectral_peak_ratio over accepted rPPG windows",
        "at least one accepted rPPG window with a spectral peak",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        quality_dependency=(
            "computed only from accepted windows; rejected windows are excluded"
        ),
        value_minimum=0.0,
        value_maximum=1.0,
    ),
    _spec(
        "rppg_peak_prominence",
        FeatureModality.RPPG,
        "power units",
        "Prominence of the selected PSD peak, in arbitrary power units.",
        "mean of peak_prominence over accepted rPPG windows",
        "at least one accepted rPPG window with a spectral peak",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        quality_dependency=(
            "computed only from accepted windows; rejected windows are excluded"
        ),
        value_minimum=0.0,
    ),
    _spec(
        "rppg_illumination_stability",
        FeatureModality.RPPG,
        "dimensionless",
        (
            "Illumination-stability quality component in [0, 1]; 1 is a "
            "constant ROI brightness."
        ),
        "mean of the illumination_stability quality component",
        "at least one rPPG window exposing the component",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=1.0,
    ),
    _spec(
        "rppg_motion_score",
        FeatureModality.RPPG,
        "intensity levels",
        "Mean capture motion score associated with the rPPG windows.",
        "mean of mean_capture_motion_score over overlapping rPPG windows",
        "at least one rPPG window carrying a motion score",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
    ),
    _spec(
        "rppg_unavailable_window_pct",
        FeatureModality.RPPG,
        "percent",
        (
            "Percentage of overlapping rPPG windows that abstained. High "
            "values mean the camera signal was unusable, not that the person "
            "was disengaged."
        ),
        "100 * count(unavailable rPPG windows) / count(rPPG windows)",
        "at least one rPPG window overlapping this window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
    _spec(
        "rppg_method",
        FeatureModality.RPPG,
        "method name",
        (
            "rPPG extraction method used for this window (green, chrom, pos). "
            "Recorded for provenance. NOT a permitted predictor: the method "
            "identifier is a property of the pipeline configuration, and a "
            "model that used it would be learning a processing artefact."
        ),
        "the configured method carried through from the rPPG result",
        "at least one rPPG window result overlapping this window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        permitted_predictor=False,
        dtype=FeatureDtype.CATEGORY,
    ),
)


_TASK: tuple[FeatureSpec, ...] = (
    _spec(
        "task_attempted_trials",
        FeatureModality.TASK,
        "count",
        "Trials whose stimulus was presented inside the window.",
        "count of stimulus_presented events inside the window",
        "at least one task event inside the window",
        missing=MissingBehaviour.ZERO_WHEN_NO_EVENTS,
        dtype=FeatureDtype.INT,
        value_minimum=0.0,
    ),
    _spec(
        "task_completed_trials",
        FeatureModality.TASK,
        "count",
        "Trials whose completion event fell inside the window.",
        "count of trial_completed events inside the window",
        "at least one task event inside the window",
        missing=MissingBehaviour.ZERO_WHEN_NO_EVENTS,
        dtype=FeatureDtype.INT,
        value_minimum=0.0,
    ),
    _spec(
        "task_correct_count",
        FeatureModality.TASK,
        "count",
        "Responses registered as correct inside the window.",
        "count of response_registered events with response_correct=True",
        "at least one task event inside the window",
        missing=MissingBehaviour.ZERO_WHEN_NO_EVENTS,
        dtype=FeatureDtype.INT,
        value_minimum=0.0,
    ),
    _spec(
        "task_correct_proportion",
        FeatureModality.TASK,
        "proportion",
        (
            "Correct responses divided by resolved response slots. A software "
            "measurement of task output, never an engagement label."
        ),
        "task_correct_count / (correct + incorrect + timeout)",
        "at least `min_resolved_trials` resolved response slots",
        value_minimum=0.0,
        value_maximum=1.0,
    ),
    _spec(
        "task_incorrect_count",
        FeatureModality.TASK,
        "count",
        "Responses registered as incorrect inside the window.",
        "count of response_registered events with response_correct=False",
        "at least one task event inside the window",
        missing=MissingBehaviour.ZERO_WHEN_NO_EVENTS,
        dtype=FeatureDtype.INT,
        value_minimum=0.0,
    ),
    _spec(
        "task_incorrect_proportion",
        FeatureModality.TASK,
        "proportion",
        "Incorrect responses divided by resolved response slots.",
        "task_incorrect_count / (correct + incorrect + timeout)",
        "at least `min_resolved_trials` resolved response slots",
        value_minimum=0.0,
        value_maximum=1.0,
    ),
    _spec(
        "task_timeout_count",
        FeatureModality.TASK,
        "count",
        (
            "Response slots that expired with no response. Kept distinct from "
            "incorrect: no response is a different observation from a wrong one."
        ),
        "count of response_timeout events inside the window",
        "at least one task event inside the window",
        missing=MissingBehaviour.ZERO_WHEN_NO_EVENTS,
        dtype=FeatureDtype.INT,
        value_minimum=0.0,
    ),
    _spec(
        "task_timeout_proportion",
        FeatureModality.TASK,
        "proportion",
        "Timed-out response slots divided by resolved response slots.",
        "task_timeout_count / (correct + incorrect + timeout)",
        "at least `min_resolved_trials` resolved response slots",
        value_minimum=0.0,
        value_maximum=1.0,
    ),
    _spec(
        "task_reaction_time_mean_ms",
        FeatureModality.TASK,
        "milliseconds",
        "Mean reaction time over registered responses inside the window.",
        "mean(reaction_time_ms) over response_registered events",
        "at least `min_reaction_times` registered responses",
        quality_dependency="timeouts contribute no reaction time and are excluded",
        value_minimum=0.0,
    ),
    _spec(
        "task_reaction_time_median_ms",
        FeatureModality.TASK,
        "milliseconds",
        "Median reaction time over registered responses inside the window.",
        "median(reaction_time_ms) over response_registered events",
        "at least `min_reaction_times` registered responses",
        quality_dependency="timeouts contribute no reaction time and are excluded",
        value_minimum=0.0,
    ),
    _spec(
        "task_reaction_time_sd_ms",
        FeatureModality.TASK,
        "milliseconds",
        "Standard deviation of reaction time within the window.",
        "population standard deviation of reaction_time_ms",
        "at least 2 registered responses",
        quality_dependency="timeouts contribute no reaction time and are excluded",
        value_minimum=0.0,
    ),
    _spec(
        "task_reaction_time_min_ms",
        FeatureModality.TASK,
        "milliseconds",
        "Fastest registered reaction time within the window.",
        "min(reaction_time_ms) over response_registered events",
        "at least `min_reaction_times` registered responses",
        quality_dependency="timeouts contribute no reaction time and are excluded",
        value_minimum=0.0,
    ),
    _spec(
        "task_reaction_time_max_ms",
        FeatureModality.TASK,
        "milliseconds",
        "Slowest registered reaction time within the window.",
        "max(reaction_time_ms) over response_registered events",
        "at least `min_reaction_times` registered responses",
        quality_dependency="timeouts contribute no reaction time and are excluded",
        value_minimum=0.0,
    ),
    _spec(
        "task_difficulty_level",
        FeatureModality.TASK,
        "level",
        (
            "Difficulty setting in force during the window. An experimental "
            "manipulation, NOT a measurement of experienced cognitive load."
        ),
        "the last difficulty_level observed at or before the window end",
        "at least one task event carrying a difficulty level",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        dtype=FeatureDtype.INT,
        value_minimum=0.0,
    ),
    _spec(
        "task_pause_count",
        FeatureModality.TASK,
        "count",
        "Task pauses beginning inside the window.",
        "count of task_paused events inside the window",
        "at least one task event inside the window",
        missing=MissingBehaviour.ZERO_WHEN_NO_EVENTS,
        dtype=FeatureDtype.INT,
        value_minimum=0.0,
    ),
    _spec(
        "task_inactivity_seconds",
        FeatureModality.TASK,
        "seconds",
        (
            "Longest gap between consecutive task events inside the window. "
            "Directly observable from event timing; it is not an attention "
            "measure."
        ),
        "max gap between consecutive in-window task-event timestamps",
        "at least 2 task events inside the window",
        value_minimum=0.0,
    ),
)


_QUALITY: tuple[FeatureSpec, ...] = (
    _spec(
        "capture_brightness_mean",
        FeatureModality.QUALITY,
        "intensity levels",
        "Mean frame brightness across the window, on a 0-255 scale.",
        "mean(brightness) over frames reporting brightness",
        "at least one frame reporting brightness",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=255.0,
    ),
    _spec(
        "capture_blur_score_mean",
        FeatureModality.QUALITY,
        "laplacian variance",
        "Mean Laplacian-variance focus measure; higher is sharper.",
        "mean(blur_score) over frames reporting a blur score",
        "at least one frame reporting a blur score",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
    ),
    _spec(
        "capture_motion_score_mean",
        FeatureModality.QUALITY,
        "intensity levels",
        "Mean absolute inter-frame difference across the window.",
        "mean(motion_score) over frames reporting a motion score",
        "at least one frame reporting a motion score",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
    ),
    _spec(
        "capture_underexposed_pct",
        FeatureModality.QUALITY,
        "percent",
        "Percentage of frames flagged underexposed.",
        "100 * count(underexposed) / count(frames)",
        "at least one frame observation in the window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
    _spec(
        "capture_overexposed_pct",
        FeatureModality.QUALITY,
        "percent",
        "Percentage of frames flagged overexposed.",
        "100 * count(overexposed) / count(frames)",
        "at least one frame observation in the window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
    _spec(
        "capture_blurry_pct",
        FeatureModality.QUALITY,
        "percent",
        "Percentage of frames flagged blurry.",
        "100 * count(is_blurry) / count(frames)",
        "at least one frame observation in the window",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
    _spec(
        "capture_dropped_frame_pct",
        FeatureModality.QUALITY,
        "percent",
        "Dropped frames as a percentage of expected frames in the window.",
        "100 * dropped / (dropped + observed)",
        "at least one frame observation or one recorded drop",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
    _spec(
        "window_missing_feature_pct",
        FeatureModality.QUALITY,
        "percent",
        (
            "Percentage of all catalog features unavailable in this window. A "
            "data-completeness diagnostic. High missingness means the "
            "measurement failed, never that the person disengaged."
        ),
        "100 * count(null features) / count(catalog features)",
        "always computable",
        missing=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
        value_minimum=0.0,
        value_maximum=100.0,
    ),
)


#: The Milestone 5 feature catalog. Order is part of the contract.
FEATURE_CATALOG = FeatureCatalog(
    version=FEATURE_CATALOG_VERSION,
    entries=_BEHAVIOURAL + _HEAD_POSE + _RPPG + _TASK + _QUALITY,
)


def get_catalog(version: str = FEATURE_CATALOG_VERSION) -> FeatureCatalog:
    """Return the catalog for ``version``.

    Raises
    ------
    KeyError
        If the requested catalog version is not implemented.
    """
    if version != FEATURE_CATALOG_VERSION:
        raise KeyError(
            f"feature catalog version {version!r} is not implemented; "
            f"this build provides {FEATURE_CATALOG_VERSION!r}"
        )
    return FEATURE_CATALOG


def feature_names_for_modalities(
    modalities: (
        tuple[FeatureModality, ...] | list[FeatureModality] | frozenset[FeatureModality]
    ),
    *,
    catalog: FeatureCatalog | None = None,
    predictors_only: bool = True,
) -> tuple[str, ...]:
    """Feature names belonging to ``modalities``, in catalog order."""
    active = catalog if catalog is not None else FEATURE_CATALOG
    wanted = set(modalities)
    return tuple(
        entry.canonical_name
        for entry in active.entries
        if entry.modality in wanted
        and (entry.permitted_predictor or not predictors_only)
    )
