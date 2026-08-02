"""rPPG (remote photoplethysmography) data contracts.

All outputs in this module are **signal-processing estimates**, not
clinical measurements.  They are NOT medical, diagnostic, or
psychological conclusions, and they are NOT engagement or
cognitive-load values.

Key invariants
--------------
- Heart rate is expressed in beats per minute (BPM); every internal
  frequency is expressed in hertz (Hz).
- An estimate that cannot be produced is reported as ``available=False``
  with a machine-readable ``reason``.  It is never NaN-filled, never
  zero-filled, and never clamped into a plausible-looking range.
- rPPG signal quality is a property of the *signal*, not of the person.
  A low quality score must never be rendered as low engagement,
  high cognitive load, stress, fatigue, or any affective state.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field

from engagevr.schemas.session import DataSource

#: Permanent marker attached to every generated (non-measured) trace.
SYNTHETIC_LABEL = "SYNTHETIC"

#: Disclaimer persisted with every rPPG artefact.
RPPG_DISCLAIMER = (
    "rPPG heart-rate values are signal-processing estimates from camera "
    "data. They are NOT medical or diagnostic measurements, are NOT "
    "validated against a reference device, and must NOT be interpreted "
    "as engagement, cognitive load, stress, anxiety, or fatigue."
)


class RoiRegion(enum.StrEnum):
    """Face-skin regions used for rPPG sampling."""

    FOREHEAD = "forehead"
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    COMBINED = "combined"


class RppgMethod(enum.StrEnum):
    """Classical rPPG extraction methods implemented in this milestone.

    No method is universally superior; relative performance depends on
    illumination, motion, skin region, and camera characteristics.
    """

    GREEN = "green"
    CHROM = "chrom"
    POS = "pos"


class UnavailableReason(enum.StrEnum):
    """Machine-readable reasons for an unavailable rPPG result."""

    NO_FACE = "no_face"
    NO_LANDMARKS = "no_landmarks"
    ROI_OUT_OF_FRAME = "roi_out_of_frame"
    ROI_TOO_SMALL = "roi_too_small"
    ROI_EMPTY = "roi_empty"
    ROI_INSUFFICIENT_VALID_PIXELS = "roi_insufficient_valid_pixels"
    TOO_FEW_VALID_FRAMES = "too_few_valid_frames"
    WINDOW_TOO_SHORT = "window_too_short"
    NON_MONOTONIC_TIMESTAMPS = "non_monotonic_timestamps"
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    EXCESSIVE_TIMESTAMP_JITTER = "excessive_timestamp_jitter"
    NON_FINITE_VALUES = "non_finite_values"
    CONSTANT_SIGNAL = "constant_signal"
    MISSING_CHANNEL = "missing_channel"
    FILTER_NOT_VIABLE = "filter_not_viable"
    INVALID_FREQUENCY_BAND = "invalid_frequency_band"
    NO_SPECTRAL_PEAK = "no_spectral_peak"
    PEAK_BELOW_MIN_PROMINENCE = "peak_below_min_prominence"
    INSUFFICIENT_SIGNAL_QUALITY = "insufficient_signal_quality"
    SAMPLING_RATE_UNKNOWN = "sampling_rate_unknown"


class RoiObservation(BaseModel):
    """Metadata for one region-of-interest extraction on one frame.

    Raw pixels are never stored in this schema and are never persisted.
    No skin tone, ethnicity, identity, emotion, engagement, or cognitive
    state is inferred from a ROI.
    """

    session_id: str = ""
    frame_index: int = 0
    monotonic_timestamp: float = 0.0

    region: RoiRegion
    available: bool
    reason: UnavailableReason | None = None

    x_min: int | None = Field(default=None, description="Left bound, pixels.")
    y_min: int | None = Field(default=None, description="Top bound, pixels.")
    x_max: int | None = Field(default=None, description="Right bound, exclusive.")
    y_max: int | None = Field(default=None, description="Bottom bound, exclusive.")

    total_pixel_count: int = Field(default=0, ge=0)
    valid_pixel_count: int = Field(default=0, ge=0)
    valid_pixel_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Valid (non-clipped) pixels as a percentage of the ROI.",
    )
    mean_brightness: float | None = Field(
        default=None,
        ge=0.0,
        le=255.0,
        description="Mean luminance of valid ROI pixels [0-255].",
    )
    clipped_pixel_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of ROI pixels at or near sensor saturation.",
    )


class RgbTraceSample(BaseModel):
    """One timestamped spatially-averaged RGB sample.

    A sample with ``valid=False`` carries no channel values.  Missing
    samples are never substituted with zero or an interpolated value at
    this layer.
    """

    frame_index: int
    monotonic_timestamp: float
    utc_timestamp: datetime | None = None

    valid: bool
    reason: UnavailableReason | None = None

    r: float | None = Field(default=None, ge=0.0, le=255.0)
    g: float | None = Field(default=None, ge=0.0, le=255.0)
    b: float | None = Field(default=None, ge=0.0, le=255.0)

    roi_valid_pixel_count: int = Field(default=0, ge=0)
    roi_mean_brightness: float | None = Field(default=None, ge=0.0, le=255.0)


class RgbTraceWindow(BaseModel):
    """A contiguous window of RGB trace samples with timing diagnostics."""

    session_id: str = ""
    data_source: DataSource = DataSource.SYNTHETIC
    synthetic_label: str | None = Field(
        default=None,
        description=(
            "Permanently set to 'SYNTHETIC' for generated traces. "
            "Never set for measured data."
        ),
    )
    window_index: int = 0

    samples: list[RgbTraceSample] = Field(default_factory=list)

    start_monotonic: float | None = None
    end_monotonic: float | None = None
    duration_s: float = Field(default=0.0, ge=0.0)

    n_samples: int = Field(default=0, ge=0)
    n_valid: int = Field(default=0, ge=0)
    n_missing: int = Field(default=0, ge=0)
    missing_pct: float = Field(default=0.0, ge=0.0, le=100.0)

    observed_fps: float | None = Field(
        default=None,
        description="Median-interval sampling rate over valid samples (Hz).",
    )
    timestamp_jitter_s: float | None = Field(
        default=None,
        description="Standard deviation of inter-sample intervals (seconds).",
    )
    timestamps_monotonic: bool = True
    n_duplicate_timestamps: int = Field(default=0, ge=0)
    n_reversed_timestamps: int = Field(default=0, ge=0)

    #: Optional motion score inherited from the capture layer.
    mean_capture_motion_score: float | None = None


class RppgWaveform(BaseModel):
    """A processed rPPG waveform produced by one extraction method.

    Values are dimensionless and arbitrarily scaled; only their temporal
    structure is meaningful.
    """

    method: RppgMethod
    available: bool
    reason: UnavailableReason | None = None

    sampling_rate_hz: float | None = Field(default=None, gt=0.0)
    timestamps: list[float] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    n_samples: int = Field(default=0, ge=0)
    duration_s: float = Field(default=0.0, ge=0.0)

    band_low_hz: float | None = None
    band_high_hz: float | None = None
    filter_order: int | None = None
    method_params: dict[str, float | int | str | bool] = Field(default_factory=dict)


class HeartRateEstimate(BaseModel):
    """Spectral heart-rate estimate for one processed window.

    This is an estimate from camera data. It is not a medical
    measurement and has not been validated against a reference device
    in this repository.  It says nothing about stress, anxiety,
    fatigue, engagement, or cognitive load.
    """

    available: bool
    bpm: float | None = Field(
        default=None,
        description="Estimated pulse rate (beats per minute). None if unavailable.",
    )
    reason: UnavailableReason | None = None

    method: RppgMethod | None = None
    estimator: str = Field(
        default="welch_psd_peak",
        description="Spectral estimator used to locate the pulse peak.",
    )

    peak_frequency_hz: float | None = None
    peak_power: float | None = None
    peak_prominence: float | None = Field(
        default=None,
        description="Prominence of the selected PSD peak (same units as power).",
    )
    in_band_power: float | None = Field(
        default=None,
        description="Integrated PSD within the configured pulse band.",
    )
    out_of_band_power: float | None = Field(
        default=None,
        description="Integrated PSD outside the pulse band, within Nyquist.",
    )
    spectral_peak_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of in-band power concentrated within +/- the peak "
            "bandwidth around the selected peak. 1.0 = fully concentrated."
        ),
    )
    frequency_resolution_hz: float | None = Field(default=None, gt=0.0)

    band_low_hz: float | None = None
    band_high_hz: float | None = None

    disclaimer: str = RPPG_DISCLAIMER


class RppgQualityComponent(BaseModel):
    """One independently-computed, independently-interpretable quality term."""

    name: str
    score: float = Field(ge=0.0, le=1.0, description="0 = unusable, 1 = ideal.")
    value: float | None = Field(
        default=None,
        description="Raw measured value before mapping to a score.",
    )
    passed: bool = True
    is_gate: bool = Field(
        default=False,
        description="If True, failing this component alone forces unavailable.",
    )
    detail: str | None = None


class RppgQualityReport(BaseModel):
    """Interpretable rPPG signal-quality report for one window.

    Signal quality describes the *measurement*, not the *person*.
    It is deliberately kept separate from model confidence, engagement,
    and cognitive load.  Poor quality must never be represented as low
    engagement.

    Aggregation
    -----------
    ``overall_quality`` is the unweighted arithmetic mean of the scores
    of all components that could actually be computed for this window.
    Equal weighting is used deliberately: there is no validated
    empirical basis in this repository for assigning differential
    weights, and inventing them would be an unexplained arbitrary
    choice.  Components that could not be computed are omitted from the
    mean rather than being imputed.

    Independently of the mean, any component with ``is_gate=True`` that
    fails forces ``acceptable=False``.  A gate failure cannot be
    averaged away by other good components.
    """

    session_id: str = ""
    window_index: int = 0
    monotonic_timestamp: float | None = None

    components: list[RppgQualityComponent] = Field(default_factory=list)
    overall_quality: float = Field(ge=0.0, le=1.0)
    aggregation: str = "unweighted_mean_of_available_components"

    acceptable: bool = Field(
        description="True only if overall_quality >= threshold and no gate failed."
    )
    quality_threshold: float = Field(ge=0.0, le=1.0)

    warnings: list[str] = Field(default_factory=list)
    rejection_reasons: list[UnavailableReason] = Field(default_factory=list)

    note: str = (
        "Signal quality is not engagement. Low quality means the camera "
        "signal is unreliable, not that the person is disengaged."
    )


class RppgMethodResult(BaseModel):
    """Complete result of running one rPPG method over one window."""

    method: RppgMethod
    method_params: dict[str, float | int | str | bool] = Field(default_factory=dict)
    waveform: RppgWaveform
    quality: RppgQualityReport
    heart_rate: HeartRateEstimate
    disclaimer: str = RPPG_DISCLAIMER


class DatasetEvaluationRecord(BaseModel):
    """One evaluated window against a real reference physiological signal.

    ``reference_bpm`` is only populated when an actual reference signal
    was read from a documented dataset.  It is never synthesised.
    """

    dataset: str
    subject_id: str
    window_index: int
    method: RppgMethod

    estimated_bpm: float | None = None
    reference_bpm: float | None = None
    absolute_error_bpm: float | None = None

    abstained: bool = False
    reason: UnavailableReason | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)

    start_monotonic: float | None = None
    duration_s: float | None = None


class RppgEvaluationMetrics(BaseModel):
    """Aggregated error metrics over windows with real reference values.

    Metrics are only populated when at least one window has a genuine
    reference BPM.  Synthetic-only runs leave every error metric None.
    """

    dataset: str
    method: RppgMethod
    subject_id: str | None = Field(
        default=None,
        description="None for a pooled across-subject aggregate.",
    )

    n_windows_attempted: int = Field(default=0, ge=0)
    n_windows_valid: int = Field(default=0, ge=0)
    n_windows_abstained: int = Field(default=0, ge=0)
    valid_window_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Valid windows / attempted windows.",
    )

    n_windows_with_reference: int = Field(default=0, ge=0)
    mae_bpm: float | None = None
    rmse_bpm: float | None = None
    bias_bpm: float | None = None
    error_std_bpm: float | None = None

    reference_available: bool = False
    provenance: str = Field(
        default="",
        description="Dataset root, adapter, and configuration used.",
    )


class RppgSessionSummary(BaseModel):
    """Summary of an rPPG processing run over one session or trace."""

    session_id: str = ""
    data_source: DataSource
    synthetic_label: str | None = None

    method: RppgMethod
    n_windows: int = Field(default=0, ge=0)
    n_windows_available: int = Field(default=0, ge=0)
    n_windows_unavailable: int = Field(default=0, ge=0)

    mean_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_bpm: float | None = None

    rejection_reasons: list[UnavailableReason] = Field(default_factory=list)
    disclaimer: str = RPPG_DISCLAIMER
