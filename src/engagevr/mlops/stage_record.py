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
The timestamped documents are listed by path with the reason they vary,
**without a checksum**, so their contents cannot enter the lock.

A meaningful change still propagates.  Alter ``metrics.json`` and the
record's checksum for it changes, so the record's own bytes change, so
``dvc.lock`` changes and every downstream stage re-runs.  What no longer
propagates is the clock.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from engagevr.mlops.fingerprints import sha256_payload
from engagevr.schemas.experiments import SELF_CHECK_DISCLAIMER
from engagevr.schemas.mlops import (
    MLOPS_DISCLAIMER,
    DeterministicArtifact,
    DeterministicStageRecord,
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


def classify(
    targets: list[Path], root: Path
) -> tuple[tuple[DeterministicArtifact, ...], dict[str, str]]:
    """Split a stage's produced files into deterministic and volatile.

    Membership is decided by :func:`is_volatile`, which names documents
    explicitly rather than guessing: a file this repository has not
    classified is treated as deterministic and will fail the two-execution
    test loudly if it is not, which is the failure mode to prefer.
    """
    deterministic: list[DeterministicArtifact] = []
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
            deterministic.append(
                DeterministicArtifact(
                    path=relative,
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
    deterministic.sort(key=lambda artifact: artifact.path)
    return tuple(deterministic), dict(sorted(volatile.items()))


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
) -> DeterministicStageRecord:
    """Build the deterministic record for one executed stage."""
    deterministic, volatile = classify(targets, root)
    if not deterministic and not volatile:
        raise StageRecordError(
            f"stage {stage_name!r} produced no file under {root}; there is "
            "nothing to record"
        )
    return DeterministicStageRecord(
        stage_name=stage_name,
        stage_kind=stage_kind,
        command=normalize_command(command, root),
        logical_identity=logical_identity,
        deterministic_artifacts=deterministic,
        volatile_artifacts=volatile,
        engagevr_version=engagevr_version(),
        python_series=python_series(platform.python_version()),
        is_synthetic=True,
        scientific_evaluation_eligible=False,
        disclaimers=(SELF_CHECK_DISCLAIMER, MLOPS_DISCLAIMER),
    )


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
    "VOLATILE_ARTIFACT_REASONS",
    "VOLATILE_DATASET_REASON",
    "VOLATILE_DATASET_SUFFIX",
    "StageRecordError",
    "artifact_map",
    "build_stage_record",
    "classify",
    "dataset_identity",
    "is_volatile",
    "normalize_command",
    "read_stage_record",
    "record_digest",
    "run_identity",
    "volatile_reason",
    "write_stage_record",
]
