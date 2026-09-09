"""The deterministic, DVC-declared representation of one pipeline stage.

The problem this solves
-----------------------
The Milestone 5--8 runners write timestamped provenance into their own
artifacts.  ``manifest.json`` records ``started_at_utc`` and
``finished_at_utc``; dataset metadata records ``created_at_utc``;
``checksums.json`` digests those documents and inherits their
instability.  That is correct behaviour — a run *did* happen at a time —
and rewriting it to please a build tool would be the wrong repair.

But a run directory declared as a DVC output puts those bytes into
``dvc.lock``, so every reproduction rewrites the lock, and a tracked file
that changes on every run stops carrying information.

The boundary
------------
::

    existing runner output          (timestamped, intact, NOT DVC-declared)
              |
    deterministic stage record      (this module)
              |
    DVC-declared output             (byte-stable, hashed into dvc.lock)

The record pins the stage's logical identity and checksums every file the
stage produced whose bytes are a pure function of the pipeline's inputs.

Three classifications, not two
-------------------------------
``portable_deterministic``
    Checksummed here.  Its bytes follow from the source, the locked
    dependencies, the configuration, the seed, and the parameters.

``execution_specific``
    Listed by path and reason, **without a checksum**.  A serialized
    estimator is the case: ``joblib.dump`` writes
    ``sklearn.tree._tree.Tree``'s raw ``nodes`` buffer, whose C struct has
    seven never-initialised padding bytes per node.  This repository's own
    baseline run puts 112,826 such bytes in each random-forest artifact,
    about ten thousand of them non-zero heap residue, and 191 of the 200
    trees that agree field-for-field between the plain and the calibrated
    artifact disagree in that padding.  Hashing those bytes hashes memory
    that no pipeline input determines.  The real digest is not lost: it is
    written to ``<name>.artifact-integrity.execution.json`` beside the
    record, which is never DVC-declared.  See DEC-105.

``volatile_provenance``
    Listed by path and reason, without a checksum.  The timestamped
    Milestone 5--8 documents: a run *did* happen at a time, and rewriting
    that to please a build tool would be the wrong repair.

An unclassified file is treated as portable deterministic, so a genuinely
unstable new output fails the reproduction test loudly rather than
disappearing into an exclusion list.

A meaningful change still propagates.  Alter ``metrics.json`` and the
record's checksum for it changes, so the record's own bytes change, so
``dvc.lock`` changes and every downstream stage re-runs.  What no longer
propagates is the clock, or one machine's heap.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from engagevr.mlops.fingerprints import sha256_payload
from engagevr.schemas.experiments import SELF_CHECK_DISCLAIMER
from engagevr.schemas.mlops import (
    MLOPS_DISCLAIMER,
    SERIALIZED_ESTIMATOR_SUFFIXES,
    ArtifactIntegrityEntry,
    DeterministicArtifact,
    DeterministicStageRecord,
    is_serialized_estimator,
    python_series,
)
from engagevr.training.artifacts import engagevr_version, sha256_file

#: Documents whose bytes vary between two correct executions, and why.
#:
#: Every entry is a Milestone 5--8 artifact that records when it was
#: written. None of them is wrong; none of them may be DVC-declared.
VOLATILE_ARTIFACT_REASONS: dict[str, str] = {
    "manifest.json": (
        "records started_at_utc and finished_at_utc. The run's identity is "
        "its run_id, a hash of the run's inputs that carries no wall clock, "
        "and that identity is this stage's logical_identity."
    ),
    "dataset.json": (
        "copies the dataset metadata, which records created_at_utc. The "
        "dataset's identity is its dataset_fingerprint, from which the wall "
        "clock is excluded by construction."
    ),
    "checksums.json": (
        "digests dataset.json and therefore inherits its creation time. Every "
        "checksum it holds for a byte-stable file is recorded here instead."
    ),
}

#: Suffixes of dataset provenance documents that record a creation time.
VOLATILE_DATASET_SUFFIX = ".metadata.json"

#: Reason recorded for a dataset metadata document.
VOLATILE_DATASET_REASON = (
    "records created_at_utc. The dataset's identity is its "
    "dataset_fingerprint, which excludes the wall clock and which this "
    "stage records as its logical_identity."
)

#: Why a serialized estimator's checksum is not portable identity.
#:
#: Measured on this repository's own baseline run rather than assumed. See
#: :data:`engagevr.schemas.mlops.EXECUTION_SPECIFIC_NOTE` and DEC-105.
EXECUTION_SPECIFIC_MODEL_REASON = (
    "is a serialized Python estimator. joblib writes scikit-learn's raw "
    "tree node buffer verbatim, and that C struct carries seven bytes of "
    "never-initialised padding per node. In this repository's own baseline "
    "run, 191 of 200 trees that are identical field-for-field between the "
    "plain and the calibrated artifact disagree in those bytes. The digest "
    "is therefore a fact about one execution's heap and one machine's "
    "library build, not about the experiment, so it is recorded in the "
    "artifact-integrity execution sidecar instead of here. The file itself "
    "is still written, still checksummed, and still tamper-checked."
)


class StageRecordError(ValueError):
    """A deterministic stage record could not be built."""


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None


def is_volatile(relative: str) -> bool:
    """Whether a pipeline-relative path names a timestamped document."""
    name = relative.rsplit("/", 1)[-1]
    if name in VOLATILE_ARTIFACT_REASONS:
        return True
    return name.endswith(VOLATILE_DATASET_SUFFIX)


def volatile_reason(relative: str) -> str:
    """Why a timestamped document's bytes vary."""
    name = relative.rsplit("/", 1)[-1]
    if name in VOLATILE_ARTIFACT_REASONS:
        return VOLATILE_ARTIFACT_REASONS[name]
    if name.endswith(VOLATILE_DATASET_SUFFIX):
        return VOLATILE_DATASET_REASON
    raise StageRecordError(f"{relative} is not a known volatile artifact")


def is_execution_specific(relative: str) -> bool:
    """Whether a path names a file whose bytes belong to one execution."""
    return is_serialized_estimator(relative)


def execution_specific_reason(relative: str) -> str:
    """Why an execution-specific file's digest is not portable identity."""
    if is_serialized_estimator(relative):
        return EXECUTION_SPECIFIC_MODEL_REASON
    raise StageRecordError(f"{relative} is not a known execution-specific artifact")


def normalize_command(command: str, root: Path) -> str:
    """Render a stage command with the pipeline root made relative.

    A command is part of a stage's identity — it carries the seed, the
    fold count, and the target — so it must not also carry a machine.
    ``engagevr mlops-demo --pipeline-root /tmp/scratch`` and the same
    demo under ``artifacts/pipeline`` are the same stage, and a record
    that disagreed would make two correct executions look different.

    Both the resolved and unresolved spellings are replaced, because a
    caller may pass either.
    """
    rendered = command
    for spelling in {str(root.resolve()), str(root)}:
        if not spelling:
            continue
        rendered = rendered.replace(f"{spelling}/", "").replace(spelling, ".")
    return rendered


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:  # pragma: no cover - callers pass inside paths
        raise StageRecordError(
            f"{path} lies outside the pipeline root {root}; a stage record "
            "stores pipeline-relative paths only"
        ) from exc


def _files_under(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        return []
    return sorted(
        path
        for path in target.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )


class StageClassification(NamedTuple):
    """A stage's produced files, sorted into the three categories."""

    #: Checksummed into the record, and therefore into ``dvc.lock``.
    deterministic: tuple[DeterministicArtifact, ...]
    #: Path to reason. Real digests live in :attr:`integrity`.
    execution_specific: dict[str, str]
    #: Path to reason. Timestamped provenance; never checksummed anywhere.
    volatile: dict[str, str]
    #: The actual SHA-256 of every execution-specific file.
    integrity: tuple[ArtifactIntegrityEntry, ...]


def classify(targets: list[Path], root: Path) -> StageClassification:
    """Sort a stage's produced files into the three categories.

    Membership is decided by :func:`is_volatile` and
    :func:`is_execution_specific`, which name their members explicitly
    rather than guessing.  A file this repository has not classified is
    treated as **deterministic** and will fail the two-execution test
    loudly if it is not, which is the failure mode to prefer: an unknown
    output that quietly landed in an exclusion list would weaken the
    guarantee in silence.
    """
    deterministic: list[DeterministicArtifact] = []
    execution_specific: dict[str, str] = {}
    integrity: list[ArtifactIntegrityEntry] = []
    volatile: dict[str, str] = {}
    seen: set[str] = set()
    for target in targets:
        for path in _files_under(target):
            relative = _relative(path, root)
            if relative in seen:
                continue
            seen.add(relative)
            if is_volatile(relative):
                volatile[relative] = volatile_reason(relative)
                continue
            if is_execution_specific(relative):
                reason = execution_specific_reason(relative)
                execution_specific[relative] = reason
                integrity.append(
                    ArtifactIntegrityEntry(
                        path=relative,
                        sha256=sha256_file(path),
                        size_bytes=path.stat().st_size,
                        excluded_from_portable_identity=reason,
                    )
                )
                continue
            deterministic.append(
                DeterministicArtifact(
                    path=relative,
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
    deterministic.sort(key=lambda artifact: artifact.path)
    integrity.sort(key=lambda entry: entry.path)
    return StageClassification(
        deterministic=tuple(deterministic),
        execution_specific=dict(sorted(execution_specific.items())),
        volatile=dict(sorted(volatile.items())),
        integrity=tuple(integrity),
    )


def dataset_identity(dataset: Path) -> str:
    """A dataset stage's identity: the fingerprint its metadata records."""
    metadata = _read_json(dataset.with_name(f"{dataset.stem}{VOLATILE_DATASET_SUFFIX}"))
    if metadata is None:
        raise StageRecordError(
            f"no metadata document beside {dataset}; the dataset stage did not "
            "complete, so it has no fingerprint to record"
        )
    fingerprint = metadata.get("dataset_fingerprint")
    if not isinstance(fingerprint, str):
        raise StageRecordError(f"{dataset} has no recorded dataset_fingerprint")
    return f"dataset_fingerprint:{fingerprint}"


def run_identity(run_directory: Path) -> str:
    """An experiment stage's identity: the run id its manifest records."""
    manifest = _read_json(run_directory / "manifest.json")
    if manifest is None:
        raise StageRecordError(
            f"{run_directory} holds no manifest.json, so the run reached no "
            "conclusion and has no identity to record"
        )
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str):
        raise StageRecordError(f"{run_directory} records no run_id")
    return f"run_id:{run_id}"


def build_stage_record(
    *,
    stage_name: str,
    stage_kind: str,
    command: str,
    logical_identity: str,
    targets: list[Path],
    root: Path,
) -> tuple[DeterministicStageRecord, tuple[ArtifactIntegrityEntry, ...]]:
    """Build one stage's portable record and its execution-specific digests.

    Returned together and never merged.  The record is DVC-declared; the
    integrity entries are what the record may not carry, and the caller
    writes them to the sidecar beside it.
    """
    classification = classify(targets, root)
    if not any(
        (
            classification.deterministic,
            classification.execution_specific,
            classification.volatile,
        )
    ):
        raise StageRecordError(
            f"stage {stage_name!r} produced no file under {root}; there is "
            "nothing to record"
        )
    record = DeterministicStageRecord(
        stage_name=stage_name,
        stage_kind=stage_kind,
        command=normalize_command(command, root),
        logical_identity=logical_identity,
        deterministic_artifacts=classification.deterministic,
        execution_specific_artifacts=classification.execution_specific,
        volatile_artifacts=classification.volatile,
        engagevr_version=engagevr_version(),
        python_series=python_series(platform.python_version()),
        is_synthetic=True,
        scientific_evaluation_eligible=False,
        disclaimers=(SELF_CHECK_DISCLAIMER, MLOPS_DISCLAIMER),
    )
    return record, classification.integrity


def write_stage_record(record: DeterministicStageRecord, path: Path) -> Path:
    """Write a stage record atomically."""
    from engagevr.training.artifacts import write_json_atomic

    return write_json_atomic(path, record.model_dump(mode="json"))


def read_stage_record(path: Path) -> DeterministicStageRecord:
    """Read and validate a persisted stage record."""
    return DeterministicStageRecord.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def record_digest(record: DeterministicStageRecord) -> str:
    """SHA-256 over a record's identity-bearing content.

    Used by the reproducibility manifest so that one stage's identity is a
    single value rather than a nested structure repeated twice.
    """
    return sha256_payload(
        {
            "stage_name": record.stage_name,
            "stage_kind": record.stage_kind,
            "command": record.command,
            "logical_identity": record.logical_identity,
            "deterministic_artifacts": [
                (artifact.path, artifact.sha256)
                for artifact in record.deterministic_artifacts
            ],
        }
    )


def artifact_map(record: DeterministicStageRecord) -> Mapping[str, str]:
    """Pipeline-relative path to SHA-256, for the deterministic artifacts."""
    return {a.path: a.sha256 for a in record.deterministic_artifacts}


__all__ = [
    "EXECUTION_SPECIFIC_MODEL_REASON",
    "SERIALIZED_ESTIMATOR_SUFFIXES",
    "VOLATILE_ARTIFACT_REASONS",
    "VOLATILE_DATASET_REASON",
    "VOLATILE_DATASET_SUFFIX",
    "StageClassification",
    "StageRecordError",
    "artifact_map",
    "build_stage_record",
    "classify",
    "dataset_identity",
    "execution_specific_reason",
    "is_execution_specific",
    "is_volatile",
    "normalize_command",
    "read_stage_record",
    "record_digest",
    "run_identity",
    "volatile_reason",
    "write_stage_record",
]
