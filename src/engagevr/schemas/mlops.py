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
#:
#: 1.1 adds the ``cpu_dependent_numeric`` classification (DEC-106). A 1.0
#: document is still readable: the new field defaults to empty, which is
#: the truthful reading of a record written before the class existed.
MLOPS_SCHEMA_VERSION = "1.1"

#: Schema versions this build knows how to read.
SUPPORTED_MLOPS_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0", "1.1"})

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
    "identifier, no MLflow run identifier, and no checksum of a serialized "
    "estimator. When it was produced is recorded beside it, in a "
    ".execution.json sidecar that is never a DVC-declared output."
)

#: Why a serialized estimator's checksum is not a portable identity.
#:
#: Measured, not assumed.  ``sklearn.tree._tree.Tree.__getstate__`` returns
#: the raw ``nodes`` buffer, whose C struct carries seven bytes of padding
#: per node that nothing ever initialises, and ``joblib.dump`` writes that
#: buffer verbatim.  In this repository's own baseline run each
#: random-forest artifact contains 112,826 such bytes, about ten thousand
#: of them non-zero heap residue — and 191 of the 200 trees that are
#: byte-identical in their *declared fields* between the plain and the
#: calibrated artifact disagree in that padding.  Two serializations of one
#: model, in one process, already differ.
#:
#: So a ``.joblib`` digest is a fact about one execution's heap, not about
#: the experiment.  It is still recorded — tamper detection needs it — but
#: in an execution-specific integrity record, never in a portable identity.
EXECUTION_SPECIFIC_NOTE = (
    "EXECUTION-SPECIFIC ARTIFACT. Its bytes are not a pure function of the "
    "pipeline's inputs, so its checksum is not part of any portable "
    "identity. A serialized Python estimator (.joblib, .pkl) embeds "
    "uninitialised C struct padding and reflects the interpreter, the "
    "library build, and the CPU that produced it. The artifact is still "
    "created, still checksummed, and still tamper-checked — in an "
    "<name>.artifact-integrity.execution.json record beside the "
    "deterministic document, which is never a DVC-declared output."
)

#: Why a model-derived numerical artifact's *bytes* are not portable.
#:
#: Measured, not assumed, and predicted in advance: DEC-105 recorded that
#: changing only the OpenBLAS kernel changed ``metrics.json``,
#: ``predictions.parquet``, and ``feature_importance.parquet``, and said
#: that if a runner's CPU ever produced different numbers the lock check
#: would fail loudly.  On PR #10 it did.  numpy and scipy ship OpenBLAS
#: built ``DYNAMIC_ARCH``, which picks a kernel from the CPU it finds at
#: run time; a different kernel accumulates a dot product in a different
#: order, and the last bits of every derived number follow.
#:
#: What differs is *only the floating-point values*.  The schema, the
#: column and row order, the dtypes, the predicted labels, the integer
#: counts, and the positions of missing and non-finite values are
#: identical across CPUs, and are still pinned exactly by a structure
#: digest.  See :mod:`engagevr.mlops.numeric_contract` and DEC-106.
CPU_DEPENDENT_NUMERIC_NOTE = (
    "CPU-DEPENDENT NUMERICAL ARTIFACT. Its structure is portable and is "
    "pinned exactly by structure_sha256; its floating-point values follow "
    "the CPU that produced them, because numpy and scipy ship OpenBLAS "
    "built DYNAMIC_ARCH and a different kernel sums in a different order. "
    "The raw SHA-256 of the real bytes is still recorded, in the "
    "<name>.artifact-integrity.execution.json record beside the "
    "deterministic document, where it still detects corruption and "
    "tampering. Numerical agreement is checked by comparison against a "
    "reference under a declared engineering portability tolerance — never "
    "by a digest, because a hash has no notion of 'close'."
)

#: Repeated on every artifact-integrity record.
ARTIFACT_INTEGRITY_NOTE = (
    "ARTIFACT INTEGRITY IS NOT SCIENTIFIC VALIDITY. A matching SHA-256 "
    "means the bytes on disk are the bytes that were written. It says "
    "nothing about whether the estimator is correct, useful, or evaluated "
    "against any participant-provided label, and a mismatch between two "
    "execution environments is expected rather than alarming."
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


#: Suffixes of a serialized Python estimator, whose bytes are execution-specific.
#:
#: Held here rather than in the producing module because this is the layer
#: that *refuses* them: a schema that can reject the mistake is worth more
#: than a convention that documents it.
SERIALIZED_ESTIMATOR_SUFFIXES: tuple[str, ...] = (".joblib", ".pkl", ".pickle")


def is_serialized_estimator(path: str) -> bool:
    """Whether a path names a pickled estimator.

    See :data:`EXECUTION_SPECIFIC_NOTE` for why the answer matters: such a
    file may be produced, kept, and checksummed, but its digest may never
    be part of a portable deterministic identity.
    """
    return path.lower().endswith(SERIALIZED_ESTIMATOR_SUFFIXES)


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


class ArtifactIntegrityEntry(BaseModel):
    """One concrete generated file, and the bytes it actually has."""

    model_config = {"extra": "forbid"}

    path: str = Field(
        description="Relative path, resolved against the record's "
        "`paths_relative_to` field. Never absolute."
    )
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    excluded_from_portable_identity: str = Field(
        min_length=1,
        description="Why this digest is not part of any portable identity.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        assert_relative_path(self.path, field="path")
        if not _SHA256.match(self.sha256):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return self


class ArtifactIntegrityRecord(_VersionedDocument):
    """The real checksums of the files a portable record cannot carry.

    The other half of the correction in DEC-105.  A serialized estimator's
    digest is genuine and worth keeping — it is how anybody detects that a
    model file changed after it was written — but it describes *one
    execution on one machine*, so it cannot live inside a DVC-declared
    document without making a correct reproduction elsewhere look like a
    changed pipeline.

    So it lives here, in ``<name>.artifact-integrity.execution.json``
    beside the deterministic document, which is **never a DVC-declared
    output**.  Nothing is weakened: every generated model file still has a
    recorded SHA-256, and changing one still changes this record.

    The environment fields exist because they are the explanation.  Two of
    these records disagreeing is expected when the Python build, the
    library build, or the CPU differs; that is a fact about pickles, not a
    finding about the experiment.
    """

    describes: str = Field(
        description="The deterministic document this belongs to, relative to "
        "the pipeline root."
    )
    produced_by: str = Field(min_length=1)
    created_at_utc: datetime

    paths_relative_to: str = Field(
        description=(
            "What every artifact path is resolved against — the pipeline root "
            "for a stage record, the producing run directory for a model "
            "version. Stated rather than assumed: two records with different "
            "bases and no field saying so is a trap for whoever verifies them."
        )
    )
    artifacts: tuple[ArtifactIntegrityEntry, ...] = ()

    engagevr_version: str
    python_version: str = Field(description="Full interpreter version.")
    python_implementation: str
    platform: str = Field(description="Host platform string of the execution.")
    dependency_versions: dict[str, str] = Field(default_factory=dict)

    is_synthetic: bool = True
    scientific_evaluation_eligible: bool = False
    integrity_note: str = ARTIFACT_INTEGRITY_NOTE
    execution_specific_note: str = EXECUTION_SPECIFIC_NOTE
    note: str = (
        "VOLATILE EXECUTION RECORD. Never a DVC-declared output and never "
        "part of a portable identity, fingerprint, or logical model version."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        assert_relative_path(self.describes, field="describes")
        if not self.paths_relative_to:
            raise ValueError(
                "paths_relative_to must name what the artifact paths resolve "
                "against; a checksum nobody can locate verifies nothing"
            )
        paths = [artifact.path for artifact in self.artifacts]
        if len(set(paths)) != len(paths):
            raise ValueError("an artifact is listed more than once")
        if self.scientific_evaluation_eligible:
            raise ValueError(
                "an artifact-integrity record can never be scientifically "
                "eligible: a checksum is not an evaluation"
            )
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


class CpuDependentNumericArtifact(BaseModel):
    """One file whose structure is portable but whose floats follow the CPU.

    The fourth classification (DEC-106).  It exists so that an artifact
    whose numbers move in the last bits between two CPUs is neither
    falsely pinned by a raw checksum nor quietly dropped from the record.

    ``structure_sha256`` covers the artifact's exact non-floating
    content — schema, ordering, dtypes, every non-float value, and the
    positions of nulls and non-finite floats — and is derived
    **separately from the raw bytes**.  It does not replace the raw
    digest, which stays in the artifact-integrity sidecar.

    There is deliberately **no size field**.  A one-ULP change in a float
    changes its decimal rendering length, so the file size of one of
    these artifacts is as CPU-dependent as its bytes: the observed
    ``metrics.json`` was 198,598 bytes on one machine and 198,605 on
    another.  Recording it here would put a machine fact back into a
    portable identity, which is the defect this class removes.
    """

    model_config = {"extra": "forbid"}

    path: str = Field(description="Path relative to the pipeline root.")
    structure_sha256: str = Field(
        min_length=64,
        max_length=64,
        description=(
            "SHA-256 over the artifact's exact CPU-independent content. "
            "Never the digest of the raw bytes, and never a digest of "
            "rounded values: rounding puts values on a grid, and two "
            "values one ULP apart can straddle a grid boundary."
        ),
    )
    excluded_from_portable_identity: str = Field(
        min_length=1,
        description="Why the raw byte digest is not part of portable identity.",
    )
    atol: float = Field(
        gt=0.0,
        description=(
            "Absolute half of the portability tolerance, structurally. "
            "Recorded as a number rather than only as prose so that a "
            "mismatch between this record and the accepted reference set "
            "is a comparison a test can make, not a sentence a reader has "
            "to notice."
        ),
    )
    rtol: float = Field(gt=0.0, description="Relative half of the tolerance.")
    numeric_tolerance: str = Field(
        min_length=1,
        description=(
            "Human-readable rendering of the same tolerance. Not a "
            "scientific uncertainty interval and not evidence about model "
            "validity."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        assert_relative_path(self.path, field="path")
        if not _SHA256.match(self.structure_sha256):
            raise ValueError("structure_sha256 must be a lowercase SHA-256 digest")
        if not (math.isfinite(self.atol) and math.isfinite(self.rtol)):
            raise ValueError("atol and rtol must be finite")
        if is_serialized_estimator(self.path):
            raise ValueError(
                f"{self.path!r} is a serialized Python estimator. Its bytes "
                "are execution-specific (DEC-105), not CPU-dependent "
                "numerical content: it has no readable structure to digest "
                "and belongs in execution_specific_artifacts."
            )
        return self


class NumericReference(_VersionedDocument):
    """The accepted numerical result of one CPU-dependent artifact.

    Why this exists
    ---------------
    The structure digest deliberately excludes floating-point *values*, so
    two executions whose numbers differ grossly still agree on it.  That
    is correct for identity and useless for detection: if the only
    numerical check compares two executions of the *same* machine, both
    can be wrong together and every gate passes.  Measured on this
    repository: mutating 1,463 floats in ``metrics.json`` left
    ``dvc.lock`` byte-identical and a same-runner comparison exited zero.

    So the accepted numbers are committed to the repository, and a fresh
    execution anywhere is compared against **them** rather than against
    itself.  This document is that record: version-controlled, checksummed
    in ``MANIFEST.json``, and therefore tied to a repository revision
    rather than to whatever a runner happened to produce ten minutes ago.

    What it holds
    -------------
    ``structure_sha256`` pins everything exact — schema, ordering, dtypes,
    non-float values, labels, and the positions of missing and non-finite
    entries.  ``values`` holds the finite floats in the canonical
    traversal order that the structure digest also uses, so a matching
    structure guarantees the two sequences correspond element for element.

    Non-finite values are absent on purpose: a NaN or an infinity is
    already pinned exactly by the structure digest, and JSON has no
    portable spelling for either.

    This is an accepted engineering baseline for a SYNTHETIC software
    self-check.  It is not a validated result, not a benchmark, and not
    evidence about any model.
    """

    artifact_path: str = Field(
        description="Path of the artifact, relative to the pipeline root."
    )
    structure_sha256: str = Field(min_length=64, max_length=64)
    value_count: int = Field(ge=0)
    values: tuple[float, ...] = Field(
        default=(),
        description=(
            "Every FINITE float, in the canonical traversal order: sorted "
            "keys depth-first for JSON, column-then-row for parquet. "
            "Full precision — nothing here is rounded or quantised."
        ),
    )
    value_order: str = Field(
        default=(
            "JSON: depth-first with dictionary keys sorted. Parquet: column "
            "order, then row order, descending into list-valued cells. The "
            "same traversal the structure digest uses, so a matching "
            "structure digest makes these values positionally comparable."
        ),
    )
    atol: float = Field(
        gt=0.0,
        description=(
            "Absolute half of the tolerance this reference was accepted "
            "under. THE ACCEPTED REFERENCE SET IS AUTHORITATIVE: a "
            "comparison uses this value, not whatever the running code "
            "happens to default to, so the contract cannot be widened "
            "without editing the committed reference."
        ),
    )
    rtol: float = Field(gt=0.0, description="Relative half of the tolerance.")
    numeric_tolerance: str = Field(min_length=1)

    is_synthetic: bool = True
    scientific_evaluation_eligible: bool = False
    disclaimers: tuple[str, ...] = ()
    note: str = (
        "ACCEPTED NUMERICAL REFERENCE for a SYNTHETIC software self-check. "
        "It records the numbers this repository accepts as correct for "
        "cross-environment comparison. It is NOT a validated result, NOT a "
        "benchmark, and NOT evidence that any model is accurate or "
        "calibrated. Reproducing it means the software is portable, not "
        "that anything it computes is true."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        assert_relative_path(self.artifact_path, field="artifact_path")
        if not _SHA256.match(self.structure_sha256):
            raise ValueError("structure_sha256 must be a lowercase SHA-256 digest")
        if self.value_count != len(self.values):
            raise ValueError(
                f"value_count is {self.value_count} but {len(self.values)} "
                "values are recorded; a truncated reference would silently "
                "check fewer numbers than it claims"
            )
        if not (math.isfinite(self.atol) and math.isfinite(self.rtol)):
            raise ValueError("atol and rtol must be finite")
        for value in self.values:
            if not math.isfinite(value):
                raise ValueError(
                    "a numeric reference holds finite values only; a NaN or "
                    "an infinity is pinned exactly by the structure digest"
                )
        if self.scientific_evaluation_eligible:
            raise ValueError(
                "an accepted numerical reference can never be scientifically "
                "eligible: reproducing a number is not validating it"
            )
        return self


class NumericReferenceManifest(_VersionedDocument):
    """The exact checksums of the accepted numerical references.

    The references are the thing CI trusts, so they need their own
    tamper-evidence.  Each entry is the SHA-256 of one reference document
    exactly as committed; a reference edited without updating this file —
    or listed here and missing from the tree — fails the check rather than
    being read anyway.

    Unlike the artifacts it describes, this document and every reference
    it names **are** byte-deterministic: they are committed text, not
    something a CPU recomputes.
    """

    target: str = Field(min_length=1)
    references: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Reference file name to the SHA-256 of its bytes. Every "
            "cpu-dependent numerical artifact the pipeline declares must "
            "appear; one that does not fails the check closed."
        ),
    )
    atol: float = Field(
        gt=0.0,
        description=(
            "Absolute half of the accepted portability tolerance. This "
            "manifest is the single authority: every reference it names, "
            "every stage record, and the running code must agree with it, "
            "and changing it is a deliberate repository contract change."
        ),
    )
    rtol: float = Field(gt=0.0, description="Relative half of the tolerance.")
    numeric_tolerance: str = Field(min_length=1)
    note: str = (
        "Accepted numerical references for a SYNTHETIC software self-check. "
        "Reproducing them demonstrates cross-environment numerical "
        "portability of the software. It is not evidence of model accuracy, "
        "calibration, or validity."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not (math.isfinite(self.atol) and math.isfinite(self.rtol)):
            raise ValueError("atol and rtol must be finite")
        for name, digest in self.references.items():
            if not _SHA256.match(digest):
                raise ValueError(
                    f"references[{name!r}] must be a lowercase SHA-256 digest"
                )
            if "/" in name or "\\" in name:
                raise ValueError(
                    f"references[{name!r}] must be a bare file name; the "
                    "manifest sits beside the references it names"
                )
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

    deterministic_artifacts: tuple[DeterministicArtifact, ...] = Field(
        default=(),
        description=(
            "Files whose bytes are a pure function of the pipeline's inputs. "
            "Only these are checksummed here, and only these reach dvc.lock."
        ),
    )
    cpu_dependent_numeric_artifacts: tuple[CpuDependentNumericArtifact, ...] = Field(
        default=(),
        description=(
            "Files whose structure is a pure function of the pipeline's "
            "inputs but whose floating-point values follow the CPU. Pinned "
            "here by structure_sha256 — which is exact and portable — while "
            "the raw byte digest goes to the artifact-integrity sidecar and "
            "the values are held to a declared tolerance by comparison. "
            "See DEC-106."
        ),
    )
    execution_specific_artifacts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Path to the reason its bytes belong to one execution rather than "
            "to the experiment — a serialized estimator, above all. Recorded "
            "WITHOUT a checksum: the digest is real and is kept, but in the "
            "<name>.artifact-integrity.execution.json record beside this "
            "document, because a platform-sensitive digest inside a "
            "DVC-declared output makes a fresh reproduction on a different "
            "machine look like a changed pipeline."
        ),
    )
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
    execution_specific_note: str = EXECUTION_SPECIFIC_NOTE
    cpu_dependent_numeric_note: str = CPU_DEPENDENT_NUMERIC_NOTE
    note: str = NO_INFLATION_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        allowed = {"dataset", "experiment_run", "diagnostic", "report"}
        if self.stage_kind not in allowed:
            raise ValueError(f"stage_kind must be one of {sorted(allowed)}")
        assert_python_series(self.python_series, field="python_series")
        for field, mapping in (
            ("volatile_artifacts", self.volatile_artifacts),
            ("execution_specific_artifacts", self.execution_specific_artifacts),
        ):
            for path, reason in mapping.items():
                assert_relative_path(path, field=field)
                if not reason:
                    raise ValueError(
                        f"{field} entry {path!r} must state why it is excluded "
                        "from the portable deterministic identity"
                    )
        paths = [artifact.path for artifact in self.deterministic_artifacts]
        if len(set(paths)) != len(paths):
            raise ValueError("a deterministic artifact is listed more than once")
        numeric_paths = [
            artifact.path for artifact in self.cpu_dependent_numeric_artifacts
        ]
        if len(set(numeric_paths)) != len(numeric_paths):
            raise ValueError(
                "a cpu-dependent numeric artifact is listed more than once"
            )
        # Four classifications, and a file has exactly one. Checked
        # pairwise rather than by convention: a file in two classes would
        # be pinned by one rule and excused by another, which is the kind
        # of ambiguity that lets an unstable output hide.
        classes: tuple[tuple[str, set[str]], ...] = (
            ("portable deterministic", set(paths)),
            ("cpu-dependent numeric", set(numeric_paths)),
            ("execution-specific", set(self.execution_specific_artifacts)),
            ("volatile", set(self.volatile_artifacts)),
        )
        for index, (label, members) in enumerate(classes):
            for other_label, others in classes[index + 1 :]:
                overlap = members & others
                if overlap:
                    raise ValueError(
                        f"{sorted(overlap)} are listed as both {label} and "
                        f"{other_label}; a file has exactly one classification"
                    )
        for artifact in self.deterministic_artifacts:
            if is_serialized_estimator(artifact.path):
                raise ValueError(
                    f"{artifact.path!r} is a serialized Python estimator and "
                    "was checksummed as portable deterministic identity. Its "
                    "bytes embed uninitialised struct padding and the build "
                    "that produced them, so a fresh reproduction on another "
                    "machine would look like a changed pipeline. Classify it "
                    "execution_specific and record its digest in the "
                    "artifact-integrity sidecar."
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

    Logical identity, not serialized bytes
    --------------------------------------
    The identifier is **scientific and software provenance only**: the
    source run, the target, the estimator and its hyperparameters, the
    dataset, split, feature-schema and configuration fingerprints, and the
    serializer kind.  It deliberately excludes the SHA-256 of the
    ``.joblib``, because that digest is a fact about one execution's heap
    and one machine's libraries (see :data:`EXECUTION_SPECIFIC_NOTE`), and
    an identifier built on it renames the same model on every machine.

    The relation is therefore one logical model version to N serialized
    artifact instances.  Each instance's real digest is recorded in an
    ``<name>.artifact-integrity.execution.json`` record beside this
    directory — see :class:`ArtifactIntegrityRecord`.  Tamper detection is
    unchanged; only its location is.
    """

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    model_version_id: str = Field(min_length=1)
    model_version_algorithm: str = "sha256"
    model_version_inputs: str = (
        "source run id, target, task type, estimator type, estimator class, "
        "estimator hyperparameter fingerprint, model name, dataset "
        "fingerprint, split fingerprint, feature-schema fingerprint, "
        "configuration fingerprint, serialization format, and the EngageVR "
        "version. Excludes creation time, absolute paths, the MLflow run id, "
        "and the serialized artifact's SHA-256 — that digest belongs to one "
        "execution, not to the model this identifier names."
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
    estimator_parameters_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        description=(
            "SHA-256 over the producing run's recorded hyperparameters for "
            "this estimator. Changing an estimator's configuration changes "
            "the logical model version even when nothing else moves."
        ),
    )
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
    model_artifact_integrity_document: str = Field(
        default="model_versions.artifact-integrity.execution.json",
        description=(
            "File name of the execution-specific record holding this "
            "artifact's actual SHA-256 and size, written beside the model "
            "version directory and never DVC-declared."
        ),
    )
    referenced_checksums: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Recorded SHA-256 of the byte-stable run documents this version "
            "depends on. The model file is deliberately absent: its digest is "
            "execution-specific and lives in the integrity record. A "
            "CPU-dependent numerical document is absent too, for the same "
            "class of reason; it appears in "
            "referenced_structure_digests instead."
        ),
    )
    numerical_portability_note: str = Field(
        default=(
            "MODEL VERSION IDENTITY DOES NOT CERTIFY NUMERICAL PORTABILITY. "
            "This identifier covers provenance, configuration, data, and "
            "code. referenced_structure_digests covers structure — schema, "
            "ordering, dtypes, non-float values, labels, and missing-value "
            "positions. NEITHER covers floating-point VALUES: a version is "
            "derived from one run, so it has nothing to compare a number "
            "against. Two executions whose scores differ far beyond the "
            "declared tolerance can carry the same model_version_id and the "
            "same structure digests. Cross-environment numerical "
            "portability is established only by comparing against the "
            "committed accepted references — 'engagevr numeric-check "
            "--accepted' — and never by this record alone. See DEC-107."
        ),
        description="What this record does NOT establish about numbers.",
    )
    referenced_structure_digests: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Structure SHA-256 of the CPU-dependent numerical run documents "
            "this version depends on — metrics.json above all. Exact and "
            "portable: it covers schema, ordering, dtypes, every non-float "
            "value, and every null position, and excludes only the "
            "floating-point values, which follow the CPU's BLAS kernel. This "
            "is what keeps one fitted model's identity stable across "
            "machines while still changing when its structure does. "
            "See DEC-106."
        ),
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
    execution_specific_note: str = EXECUTION_SPECIFIC_NOTE
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
            ("estimator_parameters_fingerprint", self.estimator_parameters_fingerprint),
        ):
            if not _SHA256.match(digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        offending = sorted(
            name
            for name in (*self.referenced_checksums, *self.referenced_structure_digests)
            if is_serialized_estimator(name)
        )
        if offending:
            raise ValueError(
                f"{offending} are serialized estimators and were referenced by "
                "checksum from a DVC-declared model version. Their digests are "
                "execution-specific; record them in the artifact-integrity "
                "sidecar instead."
            )
        both = sorted(
            set(self.referenced_checksums) & set(self.referenced_structure_digests)
        )
        if both:
            raise ValueError(
                f"{both} are referenced both by exact checksum and by "
                "structure digest. A document is one or the other: pinning a "
                "CPU-dependent numerical document by its raw bytes is the "
                "defect DEC-106 removes."
            )
        for name, digest in self.referenced_structure_digests.items():
            if not _SHA256.match(digest):
                raise ValueError(
                    f"referenced_structure_digests[{name!r}] must be a "
                    "lowercase SHA-256 digest"
                )
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
    cpu_dependent_numeric_artifacts: tuple[CpuDependentNumericArtifact, ...] = Field(
        default=(),
        description=(
            "Files pinned by an exact structure digest because their "
            "floating-point values follow the CPU. Their raw digests are in "
            "the stage's artifact-integrity record. See DEC-106."
        ),
    )
    execution_specific_artifacts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Path to the reason its bytes belong to one execution. Recorded "
            "WITHOUT a checksum; the real digest is in the stage's "
            "artifact-integrity record, which is never DVC-declared."
        ),
    )
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
        for field, mapping in (
            ("volatile_artifacts", self.volatile_artifacts),
            ("execution_specific_artifacts", self.execution_specific_artifacts),
        ):
            for path, reason in mapping.items():
                assert_relative_path(path, field=field)
                if not reason:
                    raise ValueError(
                        f"{field} entry {path!r} must state why it is excluded "
                        "from the portable deterministic identity"
                    )
        for artifact in self.deterministic_artifacts:
            if is_serialized_estimator(artifact.path):
                raise ValueError(
                    f"{artifact.path!r} is a serialized Python estimator and "
                    "cannot be part of a portable deterministic identity"
                )
        exact = {artifact.path for artifact in self.deterministic_artifacts}
        numeric = {artifact.path for artifact in self.cpu_dependent_numeric_artifacts}
        classes: tuple[tuple[str, set[str]], ...] = (
            ("portable deterministic", exact),
            ("cpu-dependent numeric", numeric),
            ("execution-specific", set(self.execution_specific_artifacts)),
            ("volatile", set(self.volatile_artifacts)),
        )
        for index, (label, members) in enumerate(classes):
            for other_label, others in classes[index + 1 :]:
                overlap = members & others
                if overlap:
                    raise ValueError(
                        f"{sorted(overlap)} are listed as both {label} and "
                        f"{other_label}; a file has exactly one classification"
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
        "stage names, kinds, commands, logical identities, the "
        "pipeline-relative path plus SHA-256 of every artifact declared "
        "byte-deterministic, and the pipeline-relative path plus structure "
        "SHA-256 of every artifact declared cpu-dependent numeric."
    )
    exact_byte_reproducibility: str = Field(
        default=(
            "Artifacts listed under deterministic_artifacts reproduce "
            "BYTE FOR BYTE from the same source, locked dependencies, "
            "configuration, seed, and parameters — on any machine. Their "
            "SHA-256 is the digest of the real bytes."
        ),
        description="What byte reproducibility means here, stated separately.",
    )
    numerical_portability: str = Field(
        default=(
            "Artifacts listed under cpu_dependent_numeric_artifacts do NOT "
            "reproduce byte for byte across CPUs, and this document does not "
            "claim they do. What reproduces exactly is their STRUCTURE — "
            "schema, ordering, dtypes, non-float values, predicted labels, "
            "and the positions of missing and non-finite values — pinned by "
            "structure_sha256. Their floating-point values are held to a "
            "declared engineering portability tolerance by COMPARISON "
            "against a reference, not by any digest. This is numerical "
            "portability; it is not byte reproducibility, and the two are "
            "reported separately on purpose. See DEC-106."
        ),
        description="What numerical portability means here, and what it is not.",
    )
    excluded_from_identity: tuple[str, ...] = (
        "wall-clock time, which appears nowhere in this document",
        "absolute filesystem paths and temporary directories",
        "the timestamped provenance documents the Milestone 5-8 runners "
        "write, which are listed by path and reason but never checksummed",
        "the SHA-256 of every serialized estimator (.joblib, .pkl), which "
        "describes one execution's heap and libraries rather than the "
        "experiment, and which is recorded in an artifact-integrity record "
        "beside the pipeline instead",
        "the RAW SHA-256 of every cpu-dependent numerical artifact, which "
        "follows the CPU's BLAS kernel rather than the experiment; its "
        "exact structure digest participates instead, and the raw digest is "
        "recorded in an artifact-integrity record",
        "the size in bytes of every cpu-dependent numerical artifact, "
        "because a one-ULP change alters a float's decimal rendering length",
        "MLflow run and experiment identifiers",
        "host platform, machine name, and process identifier",
    )

    is_synthetic: bool
    scientific_evaluation_eligible: bool
    disclaimers: tuple[str, ...]
    determinism_note: str = DETERMINISTIC_DOCUMENT_NOTE
    execution_specific_note: str = EXECUTION_SPECIFIC_NOTE
    cpu_dependent_numeric_note: str = CPU_DEPENDENT_NUMERIC_NOTE
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
    "CPU_DEPENDENT_NUMERIC_NOTE",
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
    "CpuDependentNumericArtifact",
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
    "NumericReference",
    "NumericReferenceManifest",
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
