"""Milestone 10 schemas: configuration, model versions, reproducibility,
distribution-shift diagnostics, tracking summaries, and smoke reports.

These are the persisted form of the operational layer.  They record what
was run, from which inputs, and with which bytes.  **None of them records
a scientific conclusion**, and each one refuses to be turned into one:

- a synthetic document can never carry
  ``scientific_evaluation_eligible=true``;
- no document may carry a status word such as ``production``,
  ``champion``, or ``approved`` (see :data:`FORBIDDEN_STATUS_WORDS`);
- a distribution-shift report is a *diagnostic*, never a diagnosis, and
  the schema has no field an "the model failed" claim could occupy.

Every document is versioned.  A document written by a future, unknown
schema version is refused rather than partially understood: reading half
of a record you do not understand is worse than declining to read it.
"""

from __future__ import annotations

import enum
import math
import re
from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator

from engagevr.schemas.experiments import (
    SOFTWARE_SELF_CHECK_BANNER,
    EvaluationMode,
)

#: Version of every structured document defined in this module.
MLOPS_SCHEMA_VERSION = "1.0"

#: Schema versions this build knows how to read.
SUPPORTED_MLOPS_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})

#: Words that would turn bookkeeping into an endorsement.
#:
#: MLflow, a manifest, and a checksum record that something happened.
#: They do not record that anything was reviewed, approved, or fit to
#: deploy, and no model in this repository has been evaluated against a
#: real participant label.  A record carrying one of these words would
#: read as an approval that nobody granted, so the word is refused at the
#: schema boundary rather than discouraged in a style guide.
FORBIDDEN_STATUS_WORDS: frozenset[str] = frozenset(
    {
        "approved",
        "certified",
        "challenger",
        "champion",
        "clinical",
        "diagnostic",
        "production",
        "staging",
        "validated",
    }
)

#: Attached to every Milestone 10 document produced from synthetic input.
MLOPS_DISCLAIMER = (
    "SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION. This document "
    "records an operational fact: what ran, from which inputs, and with "
    "which bytes. Reproducibility is not validity, tracking is not "
    "validation, registration is not approval, packaging is not "
    "production readiness, and a distribution-shift statistic is an "
    "engineering diagnostic. No model referenced here has been evaluated "
    "against a participant-provided engagement or cognitive-load label."
)

#: Repeated on every model-version manifest.
MODEL_VERSION_LIMITATION = (
    "A model version is an immutable, checksum-linked record of a "
    "serialized estimator. It is NOT an approval, NOT a release, NOT a "
    "deployment target, and NOT a statement that the estimator works. "
    "The estimator was fitted on SYNTHETIC data from a known "
    "data-generating process; its metrics describe whether the pipeline "
    "is wired together correctly. Model files are Python pickles: "
    "loading one executes code in it."
)

#: Repeated on every distribution-shift report.
DRIFT_INTERPRETATION_NOTE = (
    "This is a DISTRIBUTION SHIFT DIAGNOSTIC computed between two named "
    "datasets. A feature distribution shift is not model degradation, is "
    "not concept drift, is not a change in any person's engagement, "
    "attention, cognitive load, or psychological state, and is not "
    "evidence that a model has failed. A threshold crossing means one "
    "statistic exceeded an ENGINEERING DIAGNOSTIC DEFAULT that was chosen "
    "for interpretability, not calibrated against any outcome."
)

#: The one sentence every reader of an MLOps document should leave with.
NO_INFLATION_NOTE = (
    "Reproducibility is not validity. Tracking is not validation. "
    "Registration is not approval. Packaging is not production readiness. "
    "Drift alerts are engineering diagnostics."
)

#: Why a deterministic document carries no wall clock.
#:
#: A document that is a DVC-declared output is part of the pipeline's
#: identity: its bytes are hashed into ``dvc.lock``. A creation timestamp
#: inside one would make every reproduction rewrite the lock, which turns
#: "the lock changed" from a signal into noise. The execution timestamp is
#: not discarded — it is written to a separate ``.execution.json`` sidecar
#: that is never a DVC output.
DETERMINISTIC_DOCUMENT_NOTE = (
    "DETERMINISTIC DOCUMENT. Its bytes are a function of the source, the "
    "locked dependencies, the effective configuration, the synthetic seed, "
    "and the pipeline parameters — and of nothing else. It carries no "
    "wall-clock time, no absolute path, no temporary directory, no process "
    "identifier, and no MLflow run identifier. When it was produced is "
    "recorded beside it, in a .execution.json sidecar that is never a "
    "DVC-declared output."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PYTHON_SERIES = re.compile(r"^\d+\.\d+$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


class UnsupportedMLOpsSchemaError(ValueError):
    """A document declares a schema version this build cannot read."""


def assert_relative_path(value: str, *, field: str) -> str:
    """Reject a path that identifies one machine rather than one artifact.

    An absolute path, a home-directory reference, or a ``..`` escape all
    make a document unreproducible: two correct executions on two machines
    would disagree, and a temporary directory would leak into a record
    whose whole purpose is to be comparable.
    """
    if not value:
        raise ValueError(f"{field} must not be empty")
    if value.startswith(("/", "\\", "~")) or _WINDOWS_ABSOLUTE.match(value):
        raise ValueError(
            f"{field} is {value!r}, which is absolute. A deterministic record "
            "stores paths relative to the pipeline root: an absolute path is a "
            "fact about one machine, and a temporary directory is a fact about "
            "one execution."
        )
    if ".." in value.replace("\\", "/").split("/"):
        raise ValueError(
            f"{field} is {value!r}, which escapes its root with '..'. A record "
            "describes artifacts inside the pipeline, not beside it."
        )
    return value


def assert_python_series(value: str, *, field: str) -> str:
    """Reject anything but a ``major.minor`` Python series.

    The compatibility contract this project states is the *series*.  A
    patch-level version would put an interpreter upgrade into the identity
    of every deterministic document, which is churn rather than
    information; the full version is recorded in the execution sidecar.
    """
    if not _PYTHON_SERIES.match(value):
        raise ValueError(
            f"{field} is {value!r}. A deterministic document records the "
            "Python series as 'major.minor'; the full interpreter version "
            "belongs in the execution sidecar."
        )
    return value


def python_series(version: str) -> str:
    """``'3.12.13'`` to ``'3.12'``."""
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError(f"cannot derive a Python series from {version!r}")
    return f"{parts[0]}.{parts[1]}"


def assert_supported_schema_version(value: str) -> str:
    """Return ``value`` if this build can read it, else refuse.

    Refusing is deliberate.  A forward-compatible reader that ignores
    fields it does not recognise will happily report a document whose
    meaning has changed underneath it.
    """
    if value not in SUPPORTED_MLOPS_SCHEMA_VERSIONS:
        raise UnsupportedMLOpsSchemaError(
            f"MLOps schema version {value!r} is not supported by this build; "
            f"supported: {sorted(SUPPORTED_MLOPS_SCHEMA_VERSIONS)}. The "
            "document is refused rather than partially interpreted."
        )
    return value


def assert_no_status_word(text: str, *, field: str) -> str:
    """Reject a value carrying an endorsement word."""
    lowered = text.lower()
    for word in sorted(FORBIDDEN_STATUS_WORDS):
        if re.search(rf"(?<![a-z]){word}(?![a-z])", lowered):
            raise ValueError(
                f"{field} contains {word!r}. Milestone 10 records bookkeeping, "
                "not endorsement: no model, run, or alias in this repository "
                "may be labelled production, staging, champion, approved, or "
                "validated. Nothing here has been evaluated against a real "
                "participant label."
            )
    return text


def _finite(value: float | None, *, field: str) -> float | None:
    """Reject NaN and infinity, which are not statistics."""
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError(
            f"{field} is {value!r}. A non-finite value is not a statistic; "
            "report the quantity as unavailable with a reason instead of "
            "encoding failure as a number."
        )
    return value


class _VersionedDocument(BaseModel):
    """Base for every Milestone 10 persisted record."""

    model_config = {"extra": "forbid"}

    schema_version: str = MLOPS_SCHEMA_VERSION

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        assert_supported_schema_version(self.schema_version)
        return self


# ---------------------------------------------------------------------------
# Volatile execution metadata, kept strictly outside deterministic documents
# ---------------------------------------------------------------------------


class ExecutionMetadata(_VersionedDocument):
    """When and by what a deterministic document was produced.

    This is the other half of the split.  Every Milestone 10 deterministic
    document has one of these beside it, named ``<name>.execution.json``,
    and **it is never a DVC-declared output**: its contents change on every
    execution by design, which is exactly why it may not participate in any
    identity.

    Nothing here is discarded provenance.  It is provenance kept where it
    cannot make a reproducible pipeline look irreproducible.
    """

    describes: str = Field(
        description="The deterministic document this describes, relative to "
        "the pipeline root."
    )
    produced_by: str = Field(min_length=1)
    created_at_utc: datetime
    engagevr_version: str
    python_version: str = Field(description="Full interpreter version.")
    python_implementation: str
    note: str = (
        "VOLATILE EXECUTION METADATA. Recorded beside a deterministic "
        "document, never inside one, and never declared as a DVC output. "
        "Nothing in this file participates in any fingerprint, checksum, or "
        "identity."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        assert_relative_path(self.describes, field="describes")
        return self


# ---------------------------------------------------------------------------
# Deterministic pipeline records
# ---------------------------------------------------------------------------


class DeterministicArtifact(BaseModel):
    """One file whose bytes are a function of the pipeline's inputs."""

    model_config = {"extra": "forbid"}

    path: str = Field(description="Path relative to the pipeline root.")
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        assert_relative_path(self.path, field="path")
        if not _SHA256.match(self.sha256):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return self


class DeterministicStageRecord(_VersionedDocument):
    """The DVC-declared, byte-stable representation of one pipeline stage.

    Why this exists
    ---------------
    The Milestone 5--8 runners write timestamped provenance into their own
    artifacts — ``manifest.json`` records ``started_at_utc`` and
    ``finished_at_utc``, dataset metadata records ``created_at_utc`` — and
    that is correct: a run *did* happen at a time, and rewriting those
    semantics to please a build tool would be the wrong repair.

    So the run directory is never a DVC output.  This record is.  It names
    the stage, pins its logical identity, and checksums every file the
    stage produced whose bytes are a pure function of the pipeline's
    inputs.  The timestamped documents are listed by path with the reason
    they vary, and **without a checksum**, so their contents cannot enter
    the lock.

    A meaningful change to a run still propagates: alter ``metrics.json``
    and this record's checksum for it changes, so the record's own bytes
    change, so ``dvc.lock`` changes and every downstream stage re-runs.
    What no longer propagates is the clock.
    """

    stage_name: str = Field(min_length=1)
    stage_kind: str = Field(
        description="'dataset', 'experiment_run', 'diagnostic', or 'report'."
    )
    command: str = Field(min_length=1)
    logical_identity: str = Field(
        min_length=1,
        description=(
            "What makes this stage the same stage across executions: a "
            "dataset fingerprint, a run id, or a report fingerprint. Never a "
            "timestamp and never an absolute path."
        ),
    )

    deterministic_artifacts: tuple[DeterministicArtifact, ...] = ()
    volatile_artifacts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Path to the reason its bytes vary between executions. Recorded "
            "for transparency and deliberately WITHOUT a checksum: a volatile "
            "digest in a DVC-declared output is what makes a lock file churn."
        ),
    )

    engagevr_version: str
    python_series: str = Field(description="'major.minor'. Never patch-level.")

    is_synthetic: bool
    scientific_evaluation_eligible: bool
    disclaimers: tuple[str, ...]
    determinism_note: str = DETERMINISTIC_DOCUMENT_NOTE
    note: str = NO_INFLATION_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        allowed = {"dataset", "experiment_run", "diagnostic", "report"}
        if self.stage_kind not in allowed:
            raise ValueError(f"stage_kind must be one of {sorted(allowed)}")
        assert_python_series(self.python_series, field="python_series")
        for path, reason in self.volatile_artifacts.items():
            assert_relative_path(path, field="volatile_artifacts")
            if not reason:
                raise ValueError(
                    f"volatile artifact {path!r} must state why its bytes vary"
                )
        paths = [artifact.path for artifact in self.deterministic_artifacts]
        if len(set(paths)) != len(paths):
            raise ValueError("a deterministic artifact is listed more than once")
        overlap = set(paths) & set(self.volatile_artifacts)
        if overlap:
            raise ValueError(
                f"{sorted(overlap)} are listed as both deterministic and "
                "volatile; a file is one or the other"
            )
        if not self.disclaimers:
            raise ValueError("a stage record must carry at least one disclaimer")
        if self.is_synthetic:
            if self.scientific_evaluation_eligible:
                raise ValueError(
                    "a synthetic stage can never be scientifically eligible"
                )
            if not any(SOFTWARE_SELF_CHECK_BANNER in d for d in self.disclaimers):
                raise ValueError(
                    "a synthetic stage record must carry the banner "
                    f"{SOFTWARE_SELF_CHECK_BANNER!r}"
                )
        return self


# ---------------------------------------------------------------------------
# Configuration versioning
# ---------------------------------------------------------------------------


class ConfigurationVersion(_VersionedDocument):
    """The effective configuration a run was executed under.

    A filename is not a configuration version.  ``configs/defaults.yaml``
    can change between two runs that both name it, so what is recorded
    here is the **normalized effective configuration** — every default
    resolved, every section rendered in JSON mode — together with a
    SHA-256 over its canonical form.
    """

    config_fingerprint: str = Field(min_length=64, max_length=64)
    fingerprint_algorithm: str = "sha256"
    fingerprint_inputs: str = (
        "the normalized effective configuration rendered as canonical JSON "
        "with sorted keys, after removing the environment-specific paths "
        "listed in excluded_paths. Excludes wall-clock values and absolute "
        "filesystem locations."
    )
    excluded_paths: tuple[str, ...] = ()
    exclusion_reasons: dict[str, str] = Field(default_factory=dict)

    engagevr_version: str
    project_config_version: str
    section_snapshots: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not _SHA256.match(self.config_fingerprint):
            raise ValueError(
                "config_fingerprint must be a lowercase hexadecimal SHA-256 digest"
            )
        missing = [p for p in self.excluded_paths if p not in self.exclusion_reasons]
        if missing:
            raise ValueError(
                f"every excluded configuration path must state why it was "
                f"excluded; missing reasons for {missing}"
            )
        return self


# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------


class ModelVersionManifest(_VersionedDocument):
    """An immutable, auditable record of one serialized estimator.

    This is deliberately *not* a model registry entry.  There is no
    stage, no alias, no promotion, and no approval: those concepts
    describe a decision somebody made about a model, and nobody has made
    one about any model in this repository.  What is recorded is what can
    be checked — where the estimator came from, what it was fitted on,
    and whether the bytes on disk are still the bytes that were fitted.

    ``model_version_id`` is a deterministic function of that content, so
    re-deriving the manifest from the same run reproduces the identifier
    rather than minting a new one.
    """

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    model_version_id: str = Field(min_length=1)
    model_version_algorithm: str = "sha256"
    model_version_inputs: str = (
        "source run id, target, task type, estimator type, model name, "
        "dataset fingerprint, split fingerprint, feature-schema "
        "fingerprint, configuration fingerprint, serialization format, "
        "model-artifact SHA-256, and the EngageVR version. Excludes "
        "creation time, absolute paths, and the MLflow run id."
    )

    target_name: str
    task_type: str
    estimator_type: str = Field(
        description="'linear', 'tree', 'dummy', 'rule', or the model kind recorded "
        "by the producing run."
    )
    model_name: str
    estimator_class: str | None = Field(
        default=None,
        description="Class name recorded by the producing run, when it recorded one.",
    )
    fold_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Which outer fold fitted this estimator. A fold-local estimator "
            "is not a model trained on all the data, and the two must not be "
            "confused."
        ),
    )
    is_calibrated: bool = Field(
        default=False,
        description="Whether this artifact is the probability-calibrated wrapper.",
    )
    calibration_method: str | None = None

    source_run_id: str
    source_run_family: str
    source_run_directory: str = Field(
        description="Repository-relative directory of the producing run."
    )

    dataset_fingerprint: str = Field(min_length=64, max_length=64)
    split_fingerprint: str = Field(min_length=64, max_length=64)
    feature_schema_fingerprint: str = Field(min_length=64, max_length=64)
    feature_catalog_version: str
    feature_count: int = Field(ge=0)

    configuration: ConfigurationVersion

    serialization_format: str = "joblib-pickle"
    serialization_library: str = "joblib"
    serialization_library_version: str
    serialization_warning: str = (
        "This artifact is a Python pickle. Loading it executes code "
        "contained in it. Never load a model file from an untrusted "
        "source; every fact needed to audit the producing run is in the "
        "JSON documents beside it."
    )

    model_artifact_path: str = Field(
        description="Path of the estimator file, relative to the run directory."
    )
    model_artifact_sha256: str = Field(min_length=64, max_length=64)
    model_artifact_bytes: int = Field(ge=0)
    referenced_checksums: dict[str, str] = Field(
        default_factory=dict,
        description="Recorded SHA-256 of the run documents this version depends on.",
    )

    engagevr_version: str
    python_series: str = Field(
        description="'major.minor'. The compatibility contract is the series; "
        "the full interpreter version is in the execution sidecar."
    )
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    compatibility_note: str = (
        "The estimator was pickled by the library versions above. "
        "Unpickling under different versions is not guaranteed to work "
        "and is not attempted by any code in this repository."
    )

    evaluation_mode: EvaluationMode
    is_synthetic: bool
    scientific_evaluation_eligible: bool
    data_source_counts: dict[str, int] = Field(default_factory=dict)

    created_by: str = "engagevr model-manifest"
    limitation: str = MODEL_VERSION_LIMITATION
    determinism_note: str = DETERMINISTIC_DOCUMENT_NOTE
    disclaimers: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        for field, value in (
            ("model_version_id", self.model_version_id),
            ("model_name", self.model_name),
            ("estimator_type", self.estimator_type),
        ):
            assert_no_status_word(value, field=field)
        assert_python_series(self.python_series, field="python_series")
        assert_relative_path(self.source_run_directory, field="source_run_directory")
        assert_relative_path(self.model_artifact_path, field="model_artifact_path")
        for name, digest in (
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("split_fingerprint", self.split_fingerprint),
            ("feature_schema_fingerprint", self.feature_schema_fingerprint),
            ("model_artifact_sha256", self.model_artifact_sha256),
        ):
            if not _SHA256.match(digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.disclaimers:
            raise ValueError(
                "a model-version manifest must carry at least one disclaimer"
            )
        if self.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK:
            if self.scientific_evaluation_eligible:
                raise ValueError(
                    "a model version derived from a software self-check can "
                    "never be scientifically eligible"
                )
            if not any(SOFTWARE_SELF_CHECK_BANNER in d for d in self.disclaimers):
                raise ValueError(
                    "a software-self-check model version must carry the banner "
                    f"{SOFTWARE_SELF_CHECK_BANNER!r}"
                )
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                "a model version fitted on synthetic data can never be "
                "scientifically eligible"
            )
        return self


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class ReproducibilityStage(BaseModel):
    """One stage of the deterministic software demo, as recorded."""

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1)
    kind: str = Field(
        description="'dataset', 'experiment_run', 'diagnostic', or 'report'."
    )
    command: str = Field(min_length=1)
    logical_identity: str = Field(
        min_length=1,
        description=(
            "What makes this stage the same stage across executions: a "
            "dataset fingerprint, a run id, or a report fingerprint. Never a "
            "timestamp and never an absolute path."
        ),
    )
    deterministic_artifacts: tuple[DeterministicArtifact, ...] = ()
    volatile_artifacts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Path to the reason its bytes vary. Recorded WITHOUT a checksum: "
            "a volatile digest inside a DVC-declared output is precisely what "
            "makes a lock file churn."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        allowed = {"dataset", "experiment_run", "diagnostic", "report"}
        if self.kind not in allowed:
            raise ValueError(f"stage kind must be one of {sorted(allowed)}")
        for path, reason in self.volatile_artifacts.items():
            assert_relative_path(path, field="volatile_artifacts")
            if not reason:
                raise ValueError(
                    f"volatile artifact {path!r} must state why its bytes vary"
                )
        return self


class ReproducibilityManifest(_VersionedDocument):
    """What it takes to obtain the same demo again.

    ``logical_fingerprint`` covers the stage identities and the checksums
    of every artifact declared deterministic.  Wall-clock timestamps,
    absolute paths, MLflow run identifiers, and the host platform are
    excluded by construction: they differ between two correct executions,
    and folding them in would make a reproducible pipeline look
    irreproducible.

    This document is itself a DVC-declared output, so it obeys the same
    rule it describes — it carries no wall clock at all.  When it was
    built is recorded in ``reproducibility.execution.json`` beside it.
    """

    engagevr_version: str
    python_series: str = Field(description="'major.minor'. Never patch-level.")
    python_implementation: str
    dependency_versions: dict[str, str] = Field(default_factory=dict)

    configuration: ConfigurationVersion
    stages: tuple[ReproducibilityStage, ...] = ()

    logical_fingerprint: str = Field(min_length=64, max_length=64)
    logical_fingerprint_algorithm: str = "sha256"
    logical_fingerprint_inputs: str = (
        "stage names, kinds, commands, logical identities, and the "
        "pipeline-relative path plus SHA-256 of every artifact declared "
        "deterministic."
    )
    excluded_from_identity: tuple[str, ...] = (
        "wall-clock time, which appears nowhere in this document",
        "absolute filesystem paths and temporary directories",
        "the timestamped provenance documents the Milestone 5-8 runners "
        "write, which are listed by path and reason but never checksummed",
        "MLflow run and experiment identifiers",
        "host platform, machine name, and process identifier",
    )

    is_synthetic: bool
    scientific_evaluation_eligible: bool
    disclaimers: tuple[str, ...]
    determinism_note: str = DETERMINISTIC_DOCUMENT_NOTE
    note: str = NO_INFLATION_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not _SHA256.match(self.logical_fingerprint):
            raise ValueError("logical_fingerprint must be a lowercase SHA-256 digest")
        assert_python_series(self.python_series, field="python_series")
        if not self.disclaimers:
            raise ValueError(
                "a reproducibility manifest must carry at least one disclaimer"
            )
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                "a synthetic pipeline can never be scientifically eligible"
            )
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        return self


# ---------------------------------------------------------------------------
# Distribution-shift diagnostics
# ---------------------------------------------------------------------------


class DriftMethod(enum.StrEnum):
    """The minimal, interpretable diagnostic set this milestone computes."""

    MISSINGNESS_RATE_DIFFERENCE = "missingness_rate_difference"
    STANDARDIZED_MEAN_DIFFERENCE = "standardized_mean_difference"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov_statistic"
    POPULATION_STABILITY_INDEX = "population_stability_index"
    CATEGORICAL_TOTAL_VARIATION = "categorical_total_variation_distance"


class DriftStatus(enum.StrEnum):
    """Whether a statistic could be computed, and if not, why."""

    COMPUTED = "computed"
    UNAVAILABLE_MISSING_IN_REFERENCE = "unavailable_missing_in_reference"
    UNAVAILABLE_MISSING_IN_CURRENT = "unavailable_missing_in_current"
    UNAVAILABLE_ALL_VALUES_MISSING = "unavailable_all_values_missing"
    UNAVAILABLE_INSUFFICIENT_SAMPLES = "unavailable_insufficient_samples"
    UNAVAILABLE_ZERO_VARIANCE = "unavailable_zero_variance"
    UNAVAILABLE_TYPE_MISMATCH = "unavailable_type_mismatch"
    UNAVAILABLE_UNSUPPORTED_TYPE = "unavailable_unsupported_type"


class DriftReportKind(enum.StrEnum):
    """What two distributions were compared.

    There is deliberately no ``concept_drift`` member.  Concept drift is a
    change in the relationship between features and labels; establishing
    one requires labels from both periods, and this repository has no
    validated participant-provided label at all.
    """

    FEATURE_DISTRIBUTION_SHIFT = "feature_distribution_shift"
    PREDICTION_DISTRIBUTION_SHIFT = "prediction_distribution_shift"


class DriftStatistic(BaseModel):
    """One statistic for one feature, with its threshold and verdict.

    ``exceeded`` is ``None`` when the statistic could not be computed.  It
    is never ``False`` in that case: "not computable" and "computed and
    within threshold" are different states, and collapsing them would let
    an unavailable feature read as a healthy one.
    """

    model_config = {"extra": "forbid"}

    method: DriftMethod
    method_version: str = "1.0"
    statistic: float | None = None
    threshold: float | None = None
    exceeded: bool | None = None
    status: DriftStatus = DriftStatus.COMPUTED
    unavailable_reason: str | None = None
    interpretation: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        _finite(self.statistic, field=f"{self.method.value}.statistic")
        _finite(self.threshold, field=f"{self.method.value}.threshold")
        if self.status is DriftStatus.COMPUTED:
            if self.statistic is None:
                raise ValueError(
                    f"{self.method.value} is marked computed but carries no statistic"
                )
        else:
            if not self.unavailable_reason:
                raise ValueError(
                    f"{self.method.value} is unavailable and must state a reason"
                )
            if self.statistic is not None or self.exceeded is not None:
                raise ValueError(
                    f"{self.method.value} is unavailable and must not report a "
                    "statistic or a verdict; an unavailable diagnostic is not "
                    "a passing one"
                )
        return self


class FeatureDriftResult(BaseModel):
    """Every statistic computed for one column."""

    model_config = {"extra": "forbid"}

    feature_name: str
    value_kind: str = Field(description="'numeric' or 'categorical'.")

    reference_row_count: int = Field(ge=0)
    current_row_count: int = Field(ge=0)
    reference_present_count: int = Field(ge=0)
    current_present_count: int = Field(ge=0)
    reference_missing_rate: float | None = None
    current_missing_rate: float | None = None

    statistics: tuple[DriftStatistic, ...] = ()
    status: DriftStatus = DriftStatus.COMPUTED
    unavailable_reason: str | None = None
    exceeded_methods: tuple[DriftMethod, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.value_kind not in {"numeric", "categorical"}:
            raise ValueError("value_kind must be 'numeric' or 'categorical'")
        _finite(self.reference_missing_rate, field="reference_missing_rate")
        _finite(self.current_missing_rate, field="current_missing_rate")
        if self.status is not DriftStatus.COMPUTED and not self.unavailable_reason:
            raise ValueError(
                f"feature {self.feature_name!r} is unavailable and must state a reason"
            )
        declared = {s.method for s in self.statistics if s.exceeded}
        if set(self.exceeded_methods) != declared:
            raise ValueError(
                f"feature {self.feature_name!r} lists exceeded_methods "
                f"{sorted(m.value for m in self.exceeded_methods)} but its "
                f"statistics report {sorted(m.value for m in declared)}"
            )
        return self


class DriftDatasetReference(BaseModel):
    """One side of the comparison, named explicitly.

    Neither side is inferred.  A diagnostic that silently compared
    whichever two directories it found would produce a number nobody
    could interpret.
    """

    model_config = {"extra": "forbid"}

    role: str = Field(description="'reference' or 'current'.")
    path: str = Field(description="Repository-relative path of the dataset.")
    dataset_fingerprint: str | None = None
    row_count: int = Field(ge=0)
    subject_count: int | None = Field(default=None, ge=0)
    data_source_counts: dict[str, int] = Field(default_factory=dict)
    is_synthetic: bool
    scientific_evaluation_eligible: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.role not in {"reference", "current"}:
            raise ValueError("role must be 'reference' or 'current'")
        assert_relative_path(self.path, field="path")
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError("a synthetic dataset can never be scientifically eligible")
        return self


class DriftReport(_VersionedDocument):
    """A distribution-shift diagnostic between two named datasets.

    There is no overall pass/fail field, and there is no place to record
    that a model failed.  A threshold crossing is reported per feature,
    per method, with the statistic and the threshold beside it, because
    the only defensible reading of these numbers is "this feature's
    distribution moved by this much, judged against a default somebody
    chose for interpretability".
    """

    report_kind: DriftReportKind
    terminology_note: str = (
        "This report describes a DISTRIBUTION SHIFT between two datasets. "
        "It is not concept drift: establishing concept drift requires "
        "labels from both periods, and no validated participant-provided "
        "engagement or cognitive-load label exists in this repository."
    )

    reference: DriftDatasetReference
    current: DriftDatasetReference

    compared_features: tuple[str, ...] = ()
    excluded_features: dict[str, str] = Field(
        default_factory=dict,
        description="Column name to the reason it took no part in the comparison.",
    )
    unavailable_features: tuple[str, ...] = ()

    thresholds: dict[str, float] = Field(default_factory=dict)
    threshold_policy: str = (
        "ENGINEERING DIAGNOSTIC DEFAULTS. Every threshold below was chosen "
        "for interpretability and conventional use. None was calibrated "
        "against an outcome, a participant, or a failure, and crossing one "
        "is an invitation to look, not a verdict."
    )
    minimum_samples: int = Field(ge=1)
    histogram_bin_count: int = Field(ge=2)

    results: tuple[FeatureDriftResult, ...] = ()
    features_compared_count: int = Field(ge=0)
    features_exceeding_count: int = Field(ge=0)
    features_unavailable_count: int = Field(ge=0)

    report_fingerprint: str = Field(min_length=64, max_length=64)
    report_fingerprint_inputs: str = (
        "report kind, both dataset fingerprints and row counts, the "
        "compared feature list, the thresholds, and every computed "
        "statistic. Excludes wall-clock time and absolute paths, neither of "
        "which appears anywhere in this document."
    )

    is_synthetic: bool
    scientific_evaluation_eligible: bool
    interpretation: str = DRIFT_INTERPRETATION_NOTE
    determinism_note: str = DETERMINISTIC_DOCUMENT_NOTE
    disclaimers: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not _SHA256.match(self.report_fingerprint):
            raise ValueError("report_fingerprint must be a lowercase SHA-256 digest")
        if not self.disclaimers:
            raise ValueError("a drift report must carry at least one disclaimer")
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                "a synthetic distribution-shift report can never be "
                "scientifically eligible"
            )
        for value in self.thresholds.values():
            _finite(value, field="thresholds")
        computed = tuple(
            r.feature_name for r in self.results if r.status is DriftStatus.COMPUTED
        )
        unavailable = tuple(
            r.feature_name for r in self.results if r.status is not DriftStatus.COMPUTED
        )
        if self.features_compared_count != len(computed):
            raise ValueError(
                "features_compared_count disagrees with the per-feature results"
            )
        if tuple(self.unavailable_features) != unavailable:
            raise ValueError(
                "unavailable_features disagrees with the per-feature results"
            )
        if self.features_unavailable_count != len(unavailable):
            raise ValueError(
                "features_unavailable_count disagrees with the per-feature results"
            )
        exceeding = sum(1 for r in self.results if r.exceeded_methods)
        if self.features_exceeding_count != exceeding:
            raise ValueError(
                "features_exceeding_count disagrees with the per-feature results"
            )
        return self


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------

#: Tags every tracked run must carry before it is considered logged.
REQUIRED_TRACKING_TAGS: tuple[str, ...] = (
    "engagevr.data_source",
    "engagevr.is_synthetic",
    "engagevr.scientific_evaluation_eligible",
    "engagevr.evaluation_mode",
    "engagevr.disclaimer",
    "engagevr.run_family",
    "engagevr.run_id",
    "engagevr.version",
)


class MLOpsRunSummary(_VersionedDocument):
    """What one tracking call actually wrote.

    Returned so a caller can assert on it, and persisted so a reviewer can
    check that a synthetic run entered the tracking store labelled as
    synthetic and ineligible.  A run appearing in a tracking store is not
    a validated run; this document exists partly to say so in the same
    place the run id is recorded.
    """

    tracking_uri: str
    experiment_name: str
    experiment_id: str
    mlflow_run_id: str
    mlflow_run_name: str
    mlflow_version: str

    source_run_directory: str
    source_run_id: str
    run_family: str

    tags: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    logged_artifacts: tuple[str, ...] = ()
    skipped_metrics: dict[str, str] = Field(
        default_factory=dict,
        description="Metric name to the reason it was not logged. Never a zero.",
    )
    model_versions: tuple[str, ...] = ()

    is_synthetic: bool
    scientific_evaluation_eligible: bool
    registered_model: None = Field(
        default=None,
        description=(
            "Always null. Milestone 10 does not register models: a registry "
            "entry with a stage would read as a promotion decision that "
            "nobody made."
        ),
    )
    created_at_utc: datetime
    disclaimers: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        missing = [tag for tag in REQUIRED_TRACKING_TAGS if tag not in self.tags]
        if missing:
            raise ValueError(
                f"a tracked run must carry every provenance tag; missing {missing}"
            )
        assert_no_status_word(self.mlflow_run_name, field="mlflow_run_name")
        assert_no_status_word(self.experiment_name, field="experiment_name")
        for key, value in self.tags.items():
            if key.endswith(("disclaimer", "note", "limitation")):
                continue
            assert_no_status_word(value, field=f"tags[{key!r}]")
        for metric in self.metrics.values():
            _finite(metric, field="metrics")
        if self.is_synthetic:
            if self.scientific_evaluation_eligible:
                raise ValueError(
                    "a synthetic tracked run can never be scientifically eligible"
                )
            if self.tags.get("engagevr.is_synthetic") != "true":
                raise ValueError(
                    "a synthetic tracked run must carry engagevr.is_synthetic='true'"
                )
            if self.tags.get("engagevr.scientific_evaluation_eligible") != "false":
                raise ValueError(
                    "a synthetic tracked run must carry "
                    "engagevr.scientific_evaluation_eligible='false'"
                )
            if SOFTWARE_SELF_CHECK_BANNER not in self.tags.get(
                "engagevr.disclaimer", ""
            ):
                raise ValueError(
                    "a synthetic tracked run must carry the banner "
                    f"{SOFTWARE_SELF_CHECK_BANNER!r} in engagevr.disclaimer"
                )
        if not self.disclaimers:
            raise ValueError("a tracking summary must carry at least one disclaimer")
        return self


# ---------------------------------------------------------------------------
# System smoke
# ---------------------------------------------------------------------------


class SmokeCheckStatus(enum.StrEnum):
    """Outcome of one smoke check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SmokeCheckResult(BaseModel):
    """One integrated-software check.

    No timing is recorded.  A duration is wall-clock, would differ between
    two identical executions, and would make an otherwise deterministic
    report impossible to compare.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1)
    status: SmokeCheckStatus
    detail: str = ""
    failure_reason: str | None = None
    skip_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.status is SmokeCheckStatus.FAILED and not self.failure_reason:
            raise ValueError(f"failed check {self.name!r} must state a reason")
        if self.status is SmokeCheckStatus.SKIPPED and not self.skip_reason:
            raise ValueError(f"skipped check {self.name!r} must state a reason")
        if self.status is SmokeCheckStatus.PASSED and self.failure_reason:
            raise ValueError(f"passed check {self.name!r} must not record a failure")
        return self


class SmokeReport(_VersionedDocument):
    """The structured result of the integrated software self-check.

    A passing report means the components interoperate.  It does not mean
    a model is accurate, calibrated, useful, or validated, and the banner
    is a required field so that a report cannot be quoted without it.
    """

    banner: str = SOFTWARE_SELF_CHECK_BANNER
    engagevr_version: str
    python_version: str

    checks: tuple[SmokeCheckResult, ...] = ()
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    status: SmokeCheckStatus

    is_synthetic: bool = True
    scientific_evaluation_eligible: bool = False
    created_at_utc: datetime
    disclaimers: tuple[str, ...]
    note: str = NO_INFLATION_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.banner != SOFTWARE_SELF_CHECK_BANNER:
            raise ValueError(
                "a smoke report must carry the banner "
                f"{SOFTWARE_SELF_CHECK_BANNER!r} verbatim"
            )
        if self.scientific_evaluation_eligible:
            raise ValueError(
                "a smoke check is a software self-check and can never be "
                "scientifically eligible"
            )
        if self.status is SmokeCheckStatus.SKIPPED:
            raise ValueError(
                "the overall smoke status is 'passed' or 'failed'; individual "
                "checks may be skipped, the report may not"
            )
        counts = {
            SmokeCheckStatus.PASSED: 0,
            SmokeCheckStatus.FAILED: 0,
            SmokeCheckStatus.SKIPPED: 0,
        }
        for check in self.checks:
            counts[check.status] += 1
        if (
            counts[SmokeCheckStatus.PASSED] != self.passed_count
            or counts[SmokeCheckStatus.FAILED] != self.failed_count
            or counts[SmokeCheckStatus.SKIPPED] != self.skipped_count
        ):
            raise ValueError("smoke counts disagree with the recorded checks")
        expected = (
            SmokeCheckStatus.FAILED if self.failed_count else SmokeCheckStatus.PASSED
        )
        if self.status is not expected:
            raise ValueError(
                f"a report with {self.failed_count} failed check(s) must have "
                f"status {expected.value!r}"
            )
        if not self.disclaimers:
            raise ValueError("a smoke report must carry at least one disclaimer")
        return self


__all__ = [
    "DETERMINISTIC_DOCUMENT_NOTE",
    "DRIFT_INTERPRETATION_NOTE",
    "FORBIDDEN_STATUS_WORDS",
    "MLOPS_DISCLAIMER",
    "MLOPS_SCHEMA_VERSION",
    "MODEL_VERSION_LIMITATION",
    "NO_INFLATION_NOTE",
    "REQUIRED_TRACKING_TAGS",
    "SUPPORTED_MLOPS_SCHEMA_VERSIONS",
    "ConfigurationVersion",
    "DeterministicArtifact",
    "DeterministicStageRecord",
    "DriftDatasetReference",
    "DriftMethod",
    "DriftReport",
    "DriftReportKind",
    "DriftStatistic",
    "DriftStatus",
    "ExecutionMetadata",
    "FeatureDriftResult",
    "MLOpsRunSummary",
    "ModelVersionManifest",
    "ReproducibilityManifest",
    "ReproducibilityStage",
    "SmokeCheckResult",
    "SmokeCheckStatus",
    "SmokeReport",
    "UnsupportedMLOpsSchemaError",
    "assert_no_status_word",
    "assert_python_series",
    "assert_relative_path",
    "assert_supported_schema_version",
    "python_series",
]
