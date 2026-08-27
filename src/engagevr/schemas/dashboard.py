"""Typed presentation models for the Milestone 9 research dashboard.

Nothing in this module computes a scientific quantity.  Every model here
is a **view** of something a previous milestone already persisted, and
the fields exist so that a page renders a recorded value rather than
passing an untyped dictionary around and hoping the key is spelled the
same on both sides.

Two properties are structural rather than procedural.

Provenance cannot be improved by looking at it.
    :class:`DashboardProvenance` refuses to be constructed with
    ``scientific_evaluation_eligible=True`` when the artifact says the
    data is synthetic.  A page therefore cannot present a self-check as
    an evaluation even by mistake, and no configuration switch, filter,
    or user action anywhere in the dashboard can reach that flag —
    it comes from the artifact and only from the artifact.

A missing number is not zero.
    Every metric crosses into the presentation layer as
    :class:`MetricDisplayValue`, which distinguishes a real ``0.0`` from
    ``None``, from ``NaN``, and from a value the artifact never recorded.
    A model whose folds all failed shows *Unavailable*, not a column of
    zeros that reads like a very bad score.
"""

from __future__ import annotations

import enum
import math
from typing import Self

from pydantic import BaseModel, Field, model_validator

from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER

#: Shown on every result-bearing page, regardless of run family.
DASHBOARD_DISCLAIMER = (
    "EngageVR produces software estimates and controller diagnostics. "
    "Dashboard visualizations do not establish engagement, cognitive load, "
    "psychological state, health status, safety, or adaptation benefit."
)

#: Additional banner shown whenever the displayed run is synthetic.
SYNTHETIC_BANNER = SOFTWARE_SELF_CHECK_BANNER

#: Why a synthetic view stays synthetic no matter what is done to it.
PROVENANCE_PROPAGATION_NOTE = (
    "A view derived from a synthetic or scientifically ineligible artifact "
    "is itself synthetic and ineligible. Selecting, filtering, aggregating, "
    "or plotting a recorded value does not change where it came from."
)

#: The one thing this dashboard is, stated once.
DASHBOARD_PURPOSE = (
    "READ-ONLY RESEARCH OBSERVABILITY. This dashboard inspects artifacts "
    "that previous milestones already wrote. It does not train, calibrate, "
    "re-run, dispatch, or modify anything."
)

#: Rendered wherever a value is absent. Never rendered as a number.
UNAVAILABLE_TEXT = "Unavailable"


class DashboardError(ValueError):
    """A dashboard view model was asked to represent something invalid."""


class DashboardRunFamily(enum.StrEnum):
    """Which milestone's runner produced a run directory.

    Detected from the artifacts a run actually contains, never from the
    directory's name.  ``UNKNOWN`` is a real answer and is preferred to a
    guess: a directory called ``m7-something`` that carries no Milestone 7
    artifact is not a Milestone 7 run.
    """

    BASELINE = "baseline"
    FUSION = "fusion"
    PERSONALIZATION = "personalization"
    UNCERTAINTY = "uncertainty"
    ADAPTATION = "adaptation"
    UNKNOWN = "unknown"


class DashboardRunStatus(enum.StrEnum):
    """What the catalogue could establish about a run directory.

    A directory that exists is not a successful run.  These five failure
    modes are kept apart because the reader's next action differs for
    each: a corrupt document needs investigating, an incomplete run needs
    re-running, and an unsupported one needs a newer dashboard.
    """

    #: A conclusive manifest or summary says the run finished.
    COMPLETED = "completed"
    #: A conclusive manifest says the run failed, and why.
    FAILED = "failed"
    #: No conclusive document, or a required artifact is absent.
    INCOMPLETE = "incomplete"
    #: A document exists but could not be parsed.
    CORRUPT = "corrupt"
    #: The artifact declares a version this dashboard cannot interpret.
    UNSUPPORTED = "unsupported"
    #: Nothing recognisable was found.
    UNKNOWN = "unknown"


class ArtifactIntegrityStatus(enum.StrEnum):
    """Outcome of comparing recorded checksums with the bytes on disk."""

    #: Checksums were not verified (verification is opt-in).
    NOT_CHECKED = "not_checked"
    #: Every recorded checksum matched.
    VALID = "valid"
    #: At least one file's bytes disagree with its recorded checksum.
    MISMATCHED = "mismatched"
    #: The run records no ``checksums.json`` at all.
    CHECKSUM_FILE_UNAVAILABLE = "checksum_file_unavailable"
    #: A checksum names a file that is no longer present.
    REFERENCED_FILE_MISSING = "referenced_file_missing"
    #: ``checksums.json`` exists but could not be parsed.
    CHECKSUM_FILE_CORRUPT = "checksum_file_corrupt"


class DashboardWarningLevel(enum.StrEnum):
    """How loudly a warning must be shown.

    ``ERROR`` is reserved for a statement about the evidence itself —
    a checksum mismatch, a count that does not reconcile, a provenance
    conflict.  It is never used for a merely absent optional artifact.
    """

    INFORMATION = "information"
    WARNING = "warning"
    ERROR = "error"


class MetricKind(enum.StrEnum):
    """What a displayed number means, which fixes how it is formatted.

    A probability and a percentage are different displays of different
    things and are never interchanged.  An interval width carries the
    regression target's own units and is never rendered as a percentage,
    because it is not confined to ``[0, 1]`` and is not a probability.
    """

    #: A bare real number: an error, a coefficient, a score.
    REAL = "real"
    #: A value in [0, 1] that is a probability or a proportion.
    PROBABILITY = "probability"
    #: A value in [0, 1] displayed multiplied by 100 with a % sign.
    PERCENTAGE = "percentage"
    #: A whole count of things. Never shown with decimals.
    COUNT = "count"
    #: A width in the regression target's units. Not a probability.
    INTERVAL_WIDTH = "interval_width"
    #: A free-text label.
    TEXT = "text"


class MetricDisplayValue(BaseModel):
    """One number on its way to a page, with its absence made explicit.

    ``value=None`` and ``value=0.0`` are different states and are stored
    differently.  A non-finite value is refused outright rather than
    formatted: ``NaN`` printed as a number reads as a measurement, and
    ``inf`` printed as a number reads as a very large one.
    """

    model_config = {"extra": "forbid", "frozen": True}

    name: str = Field(min_length=1)
    kind: MetricKind = MetricKind.REAL
    value: float | None = None
    unavailable_reason: str | None = None
    #: Free-text units for an interval width or another dimensioned value.
    units: str | None = None
    #: The artifact this number was read from, for traceability.
    source_artifact: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.value is not None and not math.isfinite(self.value):
            raise DashboardError(
                f"metric {self.name!r} was given the non-finite value "
                f"{self.value!r}. A non-finite metric is displayed as "
                f"{UNAVAILABLE_TEXT!r} with a stated reason; it is never "
                "formatted as a number."
            )
        if self.value is None and not self.unavailable_reason:
            raise DashboardError(
                f"metric {self.name!r} has no value and must state why. An "
                "unexplained blank is indistinguishable from a bug."
            )
        if self.value is not None and self.unavailable_reason:
            raise DashboardError(
                f"metric {self.name!r} carries both a value and an unavailable reason"
            )
        if self.kind is MetricKind.COUNT and self.value is not None:
            if self.value != int(self.value):
                raise DashboardError(
                    f"count metric {self.name!r} has the fractional value "
                    f"{self.value!r}"
                )
            if self.value < 0:
                raise DashboardError(
                    f"count metric {self.name!r} is negative: {self.value!r}"
                )
        return self

    @property
    def available(self) -> bool:
        """Whether this metric has a value to display."""
        return self.value is not None


class DashboardWarning(BaseModel):
    """Something the reader must see before believing a page.

    Warnings are values rather than log lines because a page has to be
    able to render them, and a warning that only reached a log file is a
    warning the reader never got.
    """

    model_config = {"extra": "forbid", "frozen": True}

    level: DashboardWarningLevel
    message: str = Field(min_length=1)
    #: Run directory name, artifact name, or another locator.
    subject: str | None = None


class DashboardArtifactAvailability(BaseModel):
    """Whether one artifact of a run is present, and what depends on it.

    A required artifact's absence makes the run incomplete.  An optional
    artifact's absence degrades one view and says so; it never becomes a
    zero and never hides the rest of the page.
    """

    model_config = {"extra": "forbid", "frozen": True}

    name: str = Field(min_length=1)
    present: bool
    required: bool
    size_bytes: int | None = Field(default=None, ge=0)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.present and not self.unavailable_reason:
            raise DashboardError(
                f"artifact {self.name!r} is absent and must state a reason"
            )
        if self.present and self.size_bytes is None:
            raise DashboardError(
                f"artifact {self.name!r} is present but records no size"
            )
        return self


class DashboardProvenance(BaseModel):
    """Where a displayed result came from, and what it is not.

    This is the model behind the banner that every result-bearing page
    must render.  The eligibility flag is copied from the artifact and is
    checked against the synthetic flag here, so a run cannot be presented
    as scientific evidence merely because its numbers computed cleanly.
    """

    model_config = {"extra": "forbid", "frozen": True}

    run_id: str = Field(min_length=1)
    run_directory: str = Field(min_length=1)
    family: DashboardRunFamily
    status: DashboardRunStatus

    #: ``synthetic``, ``public_dataset``, ``live``, or whatever the
    #: artifact recorded. Never inferred from a directory name.
    data_source: str | None = None
    is_synthetic: bool
    scientific_evaluation_eligible: bool
    evaluation_mode: str | None = None

    target_name: str | None = None
    task_type: str | None = None
    dataset_fingerprint: str | None = None
    split_manifest_fingerprint: str | None = None
    #: Estimator, fusion strategy, or policy mode behind the result.
    model_source: str | None = None
    #: Display metadata only. Never used to establish run identity.
    finished_at_utc: str | None = None

    integrity: ArtifactIntegrityStatus = ArtifactIntegrityStatus.NOT_CHECKED
    failure_reason: str | None = None
    warnings: tuple[DashboardWarning, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise DashboardError(
                f"run {self.run_id!r} is synthetic and cannot be "
                "scientifically eligible. Synthetic output is a software "
                "self-check; plotting it does not make it evidence."
            )
        if self.status is DashboardRunStatus.FAILED and not self.failure_reason:
            raise DashboardError(
                f"run {self.run_id!r} failed and must state a failure reason"
            )
        return self

    @property
    def requires_synthetic_banner(self) -> bool:
        """Whether the page must show the software-self-check banner."""
        return self.is_synthetic

    @property
    def banners(self) -> tuple[str, ...]:
        """Every banner this provenance obliges a page to render."""
        if self.is_synthetic:
            return (SYNTHETIC_BANNER, DASHBOARD_DISCLAIMER)
        return (DASHBOARD_DISCLAIMER,)

    def derive(self, **updates: object) -> DashboardProvenance:
        """Copy this provenance for a derived view.

        Provided so that a page cannot accidentally build a *fresh*
        provenance for a chart and lose the synthetic flag on the way.
        The eligibility and synthetic fields are not accepted here.
        """
        forbidden = {"is_synthetic", "scientific_evaluation_eligible"}
        offered = forbidden & set(updates)
        if offered:
            raise DashboardError(
                f"a derived view may not change {sorted(offered)}. "
                + PROVENANCE_PROPAGATION_NOTE
            )
        return self.model_copy(update=updates)


class DashboardRunSummary(BaseModel):
    """One row of the run catalogue.

    Deliberately cheap: the catalogue reads small documents only, so
    listing a hundred runs never opens a Parquet file.  Detailed loading
    happens after the reader selects a run.
    """

    model_config = {"extra": "forbid", "frozen": True}

    directory_name: str = Field(min_length=1)
    absolute_path: str = Field(min_length=1)
    provenance: DashboardProvenance

    group_field: str | None = None
    group_count: int | None = Field(default=None, ge=0)
    fold_count: int | None = Field(default=None, ge=0)
    #: Windows the run evaluated, when the artifact records one.
    evaluated_window_count: int | None = Field(default=None, ge=0)
    session_count: int | None = Field(default=None, ge=0)

    artifacts: tuple[DashboardArtifactAvailability, ...] = ()
    #: Version string the artifact declared, when it declares one.
    artifact_schema_version: str | None = None
    detection_note: str | None = None

    @property
    def missing_required_artifacts(self) -> tuple[str, ...]:
        """Required artifacts that are absent from the directory."""
        return tuple(a.name for a in self.artifacts if a.required and not a.present)

    @property
    def is_inspectable(self) -> bool:
        """Whether a family page can render anything for this run."""
        return self.provenance.status in (
            DashboardRunStatus.COMPLETED,
            DashboardRunStatus.FAILED,
        )


class DashboardCatalogue(BaseModel):
    """Every candidate run directory under one artifact root.

    A root that does not exist, or that holds no runs, is a state to
    display rather than an exception to raise: a fresh clone has no
    artifacts and the dashboard must still start.
    """

    model_config = {"extra": "forbid", "frozen": True}

    artifact_root: str = Field(min_length=1)
    root_exists: bool
    runs: tuple[DashboardRunSummary, ...] = ()
    warnings: tuple[DashboardWarning, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether the root contains no candidate run directory at all."""
        return not self.runs

    def families(self) -> tuple[DashboardRunFamily, ...]:
        """Families present in this catalogue, in declaration order."""
        present = {run.provenance.family for run in self.runs}
        return tuple(f for f in DashboardRunFamily if f in present)

    def by_family(self, family: DashboardRunFamily) -> tuple[DashboardRunSummary, ...]:
        """Runs of one family, in catalogue order."""
        return tuple(r for r in self.runs if r.provenance.family is family)

    def find(self, directory_name: str) -> DashboardRunSummary | None:
        """The run with this directory name, or ``None``."""
        for run in self.runs:
            if run.directory_name == directory_name:
                return run
        return None


class LabelledTable(BaseModel):
    """A table on its way to a page, with its source named.

    Rows are already-formatted strings.  Formatting happens once, in
    :mod:`engagevr.dashboard.formatting`, so that a missing value renders
    as *Unavailable* everywhere rather than as whatever each page felt
    like doing about ``None``.
    """

    model_config = {"extra": "forbid", "frozen": True}

    title: str = Field(min_length=1)
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...] = ()
    source_artifact: str | None = None
    caption: str | None = None
    #: Rows dropped by a display limit, stated rather than hidden.
    truncated_row_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        width = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise DashboardError(
                    f"table {self.title!r} row {index} has {len(row)} cells "
                    f"but {width} columns"
                )
        return self


class ChartSeries(BaseModel):
    """One named series of a chart, with both axes named.

    A chart whose axis has no name cannot be read, and on this dashboard
    the axis name is what keeps a confidence threshold from being
    mistaken for an interval width.
    """

    model_config = {"extra": "forbid", "frozen": True}

    name: str = Field(min_length=1)
    x_values: tuple[float, ...]
    y_values: tuple[float | None, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.x_values) != len(self.y_values):
            raise DashboardError(
                f"series {self.name!r} has {len(self.x_values)} x values and "
                f"{len(self.y_values)} y values"
            )
        for value in self.x_values:
            if not math.isfinite(value):
                raise DashboardError(
                    f"series {self.name!r} has a non-finite x value {value!r}"
                )
        for entry in self.y_values:
            if entry is not None and not math.isfinite(entry):
                raise DashboardError(
                    f"series {self.name!r} has a non-finite y value {entry!r}"
                )
        return self


class LabelledChart(BaseModel):
    """A chart with an explicit title, axis names, and source artifact."""

    model_config = {"extra": "forbid", "frozen": True}

    title: str = Field(min_length=1)
    x_axis_label: str = Field(min_length=1)
    y_axis_label: str = Field(min_length=1)
    series: tuple[ChartSeries, ...] = ()
    subtitle: str | None = None
    #: Units or semantics of the x-axis, shown beneath the chart.
    x_axis_note: str | None = None
    source_artifact: str | None = None
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.series and not self.unavailable_reason:
            raise DashboardError(
                f"chart {self.title!r} has no series and must state why. An "
                "empty chart is never drawn as a flat line at zero."
            )
        return self

    @property
    def available(self) -> bool:
        """Whether this chart has anything to draw."""
        return bool(self.series)


class ConfusionMatrixView(BaseModel):
    """A confusion matrix with axes named for the provenance it has.

    When the labels are synthetic, the row axis is *observed synthetic
    label*, not *ground truth*: nothing in this repository has
    participant ground truth, and the axis label is where that would
    silently be claimed.
    """

    model_config = {"extra": "forbid", "frozen": True}

    labels: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]
    row_axis_label: str = Field(min_length=1)
    column_axis_label: str = Field(min_length=1)
    source_artifact: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        size = len(self.labels)
        if not size:
            raise DashboardError("a confusion matrix view needs class labels")
        if len(self.counts) != size:
            raise DashboardError(
                f"confusion matrix has {len(self.counts)} rows but {size} labels"
            )
        for index, row in enumerate(self.counts):
            if len(row) != size:
                raise DashboardError(
                    f"confusion matrix row {index} has {len(row)} entries but "
                    f"{size} labels"
                )
            if any(count < 0 for count in row):
                raise DashboardError(
                    f"confusion matrix row {index} has a negative count"
                )
        return self

    @property
    def total(self) -> int:
        """Total count across every cell."""
        return sum(sum(row) for row in self.counts)

    def row_totals(self) -> tuple[int, ...]:
        """Per-label totals along the row axis."""
        return tuple(sum(row) for row in self.counts)


class ClassificationDashboardData(BaseModel):
    """What a classification result page renders for one run."""

    model_config = {"extra": "forbid", "frozen": True}

    provenance: DashboardProvenance
    class_labels: tuple[str, ...] = ()
    model_names: tuple[str, ...] = ()
    aggregate_table: LabelledTable | None = None
    fold_table: LabelledTable | None = None
    per_class_table: LabelledTable | None = None
    confusion_matrices: tuple[ConfusionMatrixView, ...] = ()
    calibration_table: LabelledTable | None = None
    reliability_chart: LabelledChart | None = None
    calibration_method: str | None = None
    warnings: tuple[DashboardWarning, ...] = ()
    unavailable_reason: str | None = None


class RegressionDashboardData(BaseModel):
    """What a regression result page renders for one run."""

    model_config = {"extra": "forbid", "frozen": True}

    provenance: DashboardProvenance
    model_names: tuple[str, ...] = ()
    aggregate_table: LabelledTable | None = None
    fold_table: LabelledTable | None = None
    observed_versus_predicted: LabelledChart | None = None
    residual_histogram: LabelledChart | None = None
    residual_versus_predicted: LabelledChart | None = None
    #: Axis wording for the stored target, e.g. "synthetic target".
    observed_axis_label: str = "observed value"
    warnings: tuple[DashboardWarning, ...] = ()
    unavailable_reason: str | None = None


class FusionDashboardData(BaseModel):
    """What the Milestone 6 fusion page renders for one run.

    Expert disagreement and fusion support weights live in separate
    fields from anything named confidence or uncertainty, because they
    are separate quantities and a shared field would be the first step to
    a shared label.
    """

    model_config = {"extra": "forbid", "frozen": True}

    provenance: DashboardProvenance
    strategies: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    strategy_table: LabelledTable | None = None
    expert_table: LabelledTable | None = None
    fusion_support_weight_table: LabelledTable | None = None
    expert_disagreement_table: LabelledTable | None = None
    modality_availability_table: LabelledTable | None = None
    robustness_table: LabelledTable | None = None
    robustness_chart: LabelledChart | None = None
    warnings: tuple[DashboardWarning, ...] = ()
    unavailable_reason: str | None = None


class PersonalizationDashboardData(BaseModel):
    """What the Milestone 6 personalization page renders for one run.

    ``metric_delta_table`` holds a *difference*, not a benefit.  The
    field name says difference, the column heading says difference, and
    a negative difference is displayed exactly as it was recorded.
    """

    model_config = {"extra": "forbid", "frozen": True}

    provenance: DashboardProvenance
    paired_metric_table: LabelledTable | None = None
    metric_delta_table: LabelledTable | None = None
    coverage_table: LabelledTable | None = None
    fold_table: LabelledTable | None = None
    subject_diagnostic_table: LabelledTable | None = None
    population_evaluation_window_count: int | None = Field(default=None, ge=0)
    personalized_evaluation_window_count: int | None = Field(default=None, ge=0)
    calibration_window_count: int | None = Field(default=None, ge=0)
    cold_start_subject_count: int | None = Field(default=None, ge=0)
    personalized_subject_count: int | None = Field(default=None, ge=0)
    warnings: tuple[DashboardWarning, ...] = ()
    unavailable_reason: str | None = None


class SelectiveAccounting(BaseModel):
    """Accepted, abstained, unavailable, and the total they must reconcile to.

    An abstention is not an error and is not counted as one.  An
    unavailable window is not an abstention: nothing was withheld,
    because nothing was produced.  The three are stored separately and
    the model refuses to hold a set that does not add up.
    """

    model_config = {"extra": "forbid", "frozen": True}

    evaluated_window_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    unavailable_count: int = Field(ge=0)
    reconciles: bool = True
    reconciliation_error: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        total = self.accepted_count + self.abstained_count + self.unavailable_count
        if total == self.evaluated_window_count:
            if not self.reconciles:
                raise DashboardError(
                    "the selective counts reconcile but the view says they do not"
                )
            if self.reconciliation_error:
                raise DashboardError(
                    "the selective counts reconcile but an error is recorded"
                )
            return self
        if self.reconciles:
            raise DashboardError(
                f"accepted + abstained + unavailable = {total} but "
                f"{self.evaluated_window_count} windows were evaluated. A "
                "mismatch is an artifact validation error and is never "
                "normalised away."
            )
        if not self.reconciliation_error:
            raise DashboardError(
                "a non-reconciling selective accounting must state the error"
            )
        return self

    @property
    def coverage(self) -> float | None:
        """Accepted fraction of evaluated windows, or ``None``."""
        if self.evaluated_window_count == 0:
            return None
        return self.accepted_count / self.evaluated_window_count


class UncertaintyDashboardData(BaseModel):
    """What the Milestone 7 page renders for one run.

    The classification and regression fields are disjoint on purpose.  A
    regression run has no calibrated confidence to display and a
    classification run has no interval to widen, so the page hides the
    controls of the other task rather than showing disabled ones.
    """

    model_config = {"extra": "forbid", "frozen": True}

    provenance: DashboardProvenance
    task_type: str
    accounting: SelectiveAccounting | None = None
    abstention_reason_table: LabelledTable | None = None
    threshold_table: LabelledTable | None = None

    # Classification only.
    probability_calibration_status: str | None = None
    calibrated_confidence_histogram: LabelledChart | None = None
    predictive_entropy_histogram: LabelledChart | None = None
    probability_margin_histogram: LabelledChart | None = None
    confidence_coverage_curve: LabelledChart | None = None
    risk_coverage_curve: LabelledChart | None = None

    # Regression only.
    interval_width_histogram: LabelledChart | None = None
    width_coverage_curve: LabelledChart | None = None
    interval_table: LabelledTable | None = None
    empirical_interval_coverage: MetricDisplayValue | None = None
    configured_maximum_interval_width: MetricDisplayValue | None = None

    coverage_axis: str | None = None
    coverage_axis_units: str | None = None
    coverage_monotonicity_rule: str | None = None
    warnings: tuple[DashboardWarning, ...] = ()
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.task_type == "regression":
            forbidden: dict[str, object | None] = {
                "calibrated_confidence_histogram": (
                    self.calibrated_confidence_histogram
                ),
                "probability_margin_histogram": self.probability_margin_histogram,
                "confidence_coverage_curve": self.confidence_coverage_curve,
                "probability_calibration_status": (self.probability_calibration_status),
            }
            present = sorted(name for name, v in forbidden.items() if v is not None)
            if present:
                raise DashboardError(
                    f"a regression uncertainty view carries {present}. A "
                    "regression target has no class probability, so it has no "
                    "calibrated confidence, no probability margin, and no "
                    "confidence threshold. Interval width is not convertible "
                    "into a confidence score."
                )
        if self.task_type == "classification":
            forbidden = {
                "interval_width_histogram": self.interval_width_histogram,
                "width_coverage_curve": self.width_coverage_curve,
                "empirical_interval_coverage": self.empirical_interval_coverage,
                "configured_maximum_interval_width": (
                    self.configured_maximum_interval_width
                ),
            }
            present = sorted(name for name, v in forbidden.items() if v is not None)
            if present:
                raise DashboardError(
                    f"a classification uncertainty view carries {present}. A "
                    "classification target has no prediction interval to widen."
                )
        return self


class AdaptationLifecycleCounts(BaseModel):
    """Proposal, command, dispatch, and acknowledgement, kept apart.

    Collapsing these into one "adaptations" number is the single
    misreading this page exists to prevent.  A proposal is a policy
    intention; a built command is a payload; a dispatched command was
    sent; an acknowledged command was confirmed by the environment.  In
    Milestone 8 nothing is dispatched, so the last two are zero and that
    zero is a true statement about the world.
    """

    model_config = {"extra": "forbid", "frozen": True}

    proposals: int = Field(ge=0)
    commands_built: int = Field(ge=0)
    commands_dispatched: int = Field(ge=0)
    acknowledgements_recorded: int = Field(ge=0)
    applied_confirmed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.commands_built > self.proposals:
            raise DashboardError(
                f"{self.commands_built} commands were built from "
                f"{self.proposals} proposals; a command cannot exist without "
                "a proposal"
            )
        if self.commands_dispatched > self.commands_built:
            raise DashboardError(
                f"{self.commands_dispatched} commands were dispatched but "
                f"only {self.commands_built} were built"
            )
        if self.acknowledgements_recorded > self.commands_dispatched:
            raise DashboardError(
                f"{self.acknowledgements_recorded} acknowledgements were "
                f"recorded but only {self.commands_dispatched} commands were "
                "dispatched. An acknowledgement requires a real environment "
                "reply."
            )
        if self.applied_confirmed > self.acknowledgements_recorded:
            raise DashboardError("an applied adaptation requires an acknowledgement")
        return self


class AdaptationDashboardData(BaseModel):
    """What the Milestone 8 page renders for one run.

    There is deliberately no effectiveness field.  Milestone 8 contains
    no evidence that any adaptation helped anyone, so the page has
    nowhere to put such a claim even if someone wanted to.
    """

    model_config = {"extra": "forbid", "frozen": True}

    provenance: DashboardProvenance
    experiment_mode: str | None = None
    policy_mode: str | None = None
    adaptation_enabled: bool | None = None
    configuration_fingerprint: str | None = None

    evaluated_windows: int | None = Field(default=None, ge=0)
    gate_eligible_windows: int | None = Field(default=None, ge=0)
    gate_blocked_windows: int | None = Field(default=None, ge=0)
    hold_decisions: int | None = Field(default=None, ge=0)
    increases: int | None = Field(default=None, ge=0)
    decreases: int | None = Field(default=None, ge=0)

    lifecycle: AdaptationLifecycleCounts | None = None
    hold_reason_table: LabelledTable | None = None
    guard_table: LabelledTable | None = None
    spacing_table: LabelledTable | None = None
    scenario_table: LabelledTable | None = None
    session_table: LabelledTable | None = None
    difficulty_trace: LabelledChart | None = None
    lifecycle_table: LabelledTable | None = None
    #: Action-frequency comparison only. Never a quality comparison.
    action_frequency_comparison_table: LabelledTable | None = None
    session_ids: tuple[str, ...] = ()
    warnings: tuple[DashboardWarning, ...] = ()
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        counts = (
            self.evaluated_windows,
            self.hold_decisions,
            self.lifecycle.proposals if self.lifecycle else None,
        )
        if all(c is not None for c in counts):
            evaluated, holds, proposals = counts
            assert evaluated is not None and holds is not None
            assert proposals is not None
            if holds + proposals != evaluated:
                raise DashboardError(
                    f"{holds} holds + {proposals} proposals != {evaluated} "
                    "evaluated windows. Every evaluated window is exactly one "
                    "decision."
                )
        eligible, blocked = self.gate_eligible_windows, self.gate_blocked_windows
        if (
            eligible is not None
            and blocked is not None
            and self.evaluated_windows is not None
            and eligible + blocked != self.evaluated_windows
        ):
            raise DashboardError(
                f"{eligible} eligible + {blocked} blocked != "
                f"{self.evaluated_windows} evaluated windows"
            )
        return self


class SignalQualityDashboardData(BaseModel):
    """What the measurement-quality page renders.

    Everything here describes whether a measurement could be taken.  No
    field on this model is an engagement, cognitive-load, or confidence
    quantity, and the page states that in words next to every chart.
    """

    model_config = {"extra": "forbid", "frozen": True}

    provenance: DashboardProvenance
    modality_availability_table: LabelledTable | None = None
    missing_feature_table: LabelledTable | None = None
    missingness_chart: LabelledChart | None = None
    overall_missing_percentage: MetricDisplayValue | None = None
    warnings: tuple[DashboardWarning, ...] = ()
    unavailable_reason: str | None = None


class DatasetProvenanceDashboardData(BaseModel):
    """What the dataset-and-provenance page renders.

    Every field is read from ``dataset.json`` or ``splits.json``.  None
    is derived from a filename: a count this repository did not record
    is *Unavailable*, not a number recovered from a directory name.
    """

    model_config = {"extra": "forbid", "frozen": True}

    provenance: DashboardProvenance
    dataset_table: LabelledTable | None = None
    target_table: LabelledTable | None = None
    split_table: LabelledTable | None = None
    fold_table: LabelledTable | None = None
    data_source_counts: tuple[tuple[str, int], ...] = ()
    split_audit_passed: bool | None = None
    split_audit_notes: tuple[str, ...] = ()
    warnings: tuple[DashboardWarning, ...] = ()
    unavailable_reason: str | None = None


class LimitationRecord(BaseModel):
    """One standing limitation of this repository, as a typed record.

    Kept as data rather than scraped out of Markdown at run time: a page
    that parses prose can silently render nothing when the prose is
    reworded, and this is the page that must never quietly go blank.
    """

    model_config = {"extra": "forbid", "frozen": True}

    identifier: str = Field(min_length=1)
    title: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    #: Document a reader should open for the full statement.
    reference: str | None = None
    resolved: bool = False


__all__ = [
    "DASHBOARD_DISCLAIMER",
    "DASHBOARD_PURPOSE",
    "PROVENANCE_PROPAGATION_NOTE",
    "SYNTHETIC_BANNER",
    "UNAVAILABLE_TEXT",
    "AdaptationDashboardData",
    "AdaptationLifecycleCounts",
    "ArtifactIntegrityStatus",
    "ChartSeries",
    "ClassificationDashboardData",
    "ConfusionMatrixView",
    "DashboardArtifactAvailability",
    "DashboardCatalogue",
    "DashboardError",
    "DashboardProvenance",
    "DashboardRunFamily",
    "DashboardRunStatus",
    "DashboardRunSummary",
    "DashboardWarning",
    "DashboardWarningLevel",
    "DatasetProvenanceDashboardData",
    "FusionDashboardData",
    "LabelledChart",
    "LabelledTable",
    "LimitationRecord",
    "MetricDisplayValue",
    "MetricKind",
    "PersonalizationDashboardData",
    "RegressionDashboardData",
    "SelectiveAccounting",
    "SignalQualityDashboardData",
    "UncertaintyDashboardData",
]
