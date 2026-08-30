"""What it takes to obtain the same demo again.

The manifest is assembled from the deterministic stage records, never by
walking the filesystem.  That is the whole repair: a stage record has
already separated the byte-stable artifacts from the timestamped
provenance the Milestone 5--8 runners write, so nothing volatile can leak
into this document — and this document is itself a DVC-declared output,
so it has to obey the rule it describes.

What is in the identity
-----------------------
Stage names, kinds, commands, logical identities (a dataset fingerprint,
a run id, a report fingerprint), and the pipeline-relative path plus
SHA-256 of every artifact declared deterministic.

What is not, by construction
----------------------------
Wall-clock time — which appears nowhere in this document at all — absolute
paths, temporary directories, the contents of timestamped run manifests,
MLflow run and experiment identifiers, the host platform, and the process
identifier.

When the manifest was built is recorded in
``reproducibility.execution.json`` beside it, which is never a DVC output.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from engagevr.config import EngageVRConfig
from engagevr.mlops.fingerprints import sha256_payload
from engagevr.mlops.pipeline import PipelineLayout, StageSpec
from engagevr.mlops.stage_record import (
    StageRecordError,
    classify,
    read_stage_record,
)
from engagevr.schemas.experiments import SELF_CHECK_DISCLAIMER
from engagevr.schemas.mlops import (
    MLOPS_DISCLAIMER,
    ConfigurationVersion,
    DeterministicArtifact,
    ReproducibilityManifest,
    ReproducibilityStage,
    python_series,
)
from engagevr.training.artifacts import (
    dependency_versions,
    engagevr_version,
    runtime_environment,
)


class ReproducibilityError(ValueError):
    """The manifest could not be built from the given pipeline root."""


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None


def _direct_identity(stage: StageSpec, layout: PipelineLayout) -> str:
    """Identity for a stage that produces its own deterministic document."""
    if stage.name == "drift":
        report = _read_json(layout.drift_report)
        if report is None:
            return "drift_report:unavailable"
        return f"report_fingerprint:{report.get('report_fingerprint', 'unknown')}"
    if stage.name == "model-version":
        identifiers = sorted(
            document["model_version_id"]
            for document in (
                _read_json(path)
                for path in sorted(layout.model_versions.glob("*.json"))
            )
            if document is not None and "model_version_id" in document
        )
        if not identifiers:
            return "model_versions:unavailable"
        return "model_version_ids:" + ",".join(identifiers)
    if stage.name == "integrity":
        catalogue = _read_json(layout.catalogue)
        if catalogue is None:
            return "catalogue:unavailable"
        runs = catalogue.get("runs", ())
        rows = (
            sorted(
                (
                    str(run.get("run_id")),
                    str(run.get("family")),
                    str(run.get("status")),
                    str(run.get("integrity")),
                    str(run.get("scientific_evaluation_eligible")),
                )
                for run in runs
            )
            if isinstance(runs, list)
            else []
        )
        return "catalogue:" + sha256_payload(rows)
    return f"stage:{stage.name}"


def build_stage_entries(
    stages: Sequence[StageSpec], layout: PipelineLayout
) -> tuple[ReproducibilityStage, ...]:
    """One manifest entry per executed stage.

    A stage that wrote a deterministic record is read from that record —
    the classification of stable versus timestamped has already been made
    and must not be made twice.  A stage that produces its own
    deterministic document is classified here, and everything it produces
    is byte-stable by construction.
    """
    entries: list[ReproducibilityStage] = []
    for stage in stages:
        if stage.name == "reproducibility":
            # This document is the stage's own output. Recording its
            # checksum inside itself is impossible, and a placeholder
            # would be worse.
            continue
        if stage.record is not None:
            try:
                record = read_stage_record(stage.record)
            except (OSError, ValueError) as exc:
                raise ReproducibilityError(
                    f"stage {stage.name!r} has no readable deterministic record "
                    f"at {stage.record}: {exc}. Run the pipeline before asking "
                    "for its reproducibility manifest."
                ) from exc
            entries.append(
                ReproducibilityStage(
                    name=record.stage_name,
                    kind=record.stage_kind,
                    command=record.command,
                    logical_identity=record.logical_identity,
                    deterministic_artifacts=record.deterministic_artifacts,
                    volatile_artifacts=record.volatile_artifacts,
                )
            )
            continue
        deterministic, volatile = classify(list(stage.outputs), layout.root)
        entries.append(
            ReproducibilityStage(
                name=stage.name,
                kind=stage.kind,
                command=stage.command,
                logical_identity=_direct_identity(stage, layout),
                deterministic_artifacts=deterministic,
                volatile_artifacts=volatile,
            )
        )
    return tuple(entries)


def logical_fingerprint(stages: Sequence[ReproducibilityStage]) -> str:
    """SHA-256 over stage identities and deterministic checksums."""
    payload = [
        {
            "name": stage.name,
            "kind": stage.kind,
            "command": stage.command,
            "logical_identity": stage.logical_identity,
            "deterministic_artifacts": [
                (artifact.path, artifact.sha256)
                for artifact in stage.deterministic_artifacts
            ],
        }
        for stage in stages
    ]
    return sha256_payload(payload)


def build_manifest(
    stages: Sequence[StageSpec],
    layout: PipelineLayout,
    *,
    config: EngageVRConfig,
    configuration_version: ConfigurationVersion | None = None,
) -> ReproducibilityManifest:
    """Build the reproducibility manifest for one executed pipeline."""
    from engagevr.mlops.fingerprints import build_configuration_version

    entries = build_stage_entries(stages, layout)
    if not entries:
        raise ReproducibilityError(
            "no pipeline stage produced a record. Run the pipeline before "
            "asking for its reproducibility manifest."
        )
    environment = runtime_environment()
    return ReproducibilityManifest(
        engagevr_version=engagevr_version(),
        python_series=python_series(environment["python_version"]),
        python_implementation=environment["python_implementation"],
        dependency_versions=dependency_versions(),
        configuration=configuration_version or build_configuration_version(config),
        stages=entries,
        logical_fingerprint=logical_fingerprint(entries),
        is_synthetic=True,
        scientific_evaluation_eligible=False,
        disclaimers=(SELF_CHECK_DISCLAIMER, MLOPS_DISCLAIMER),
    )


def read_manifest(path: Path) -> ReproducibilityManifest:
    """Read and validate a persisted reproducibility manifest."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReproducibilityManifest.model_validate(document)


def compare(
    first: ReproducibilityManifest, second: ReproducibilityManifest
) -> tuple[str, ...]:
    """Differences in *logical* identity between two executions.

    An empty result means the two executions are the same run in every
    sense this repository claims to control.  There is nothing to exclude
    from the comparison: neither manifest contains a timestamp, an
    absolute path, or an MLflow identifier in the first place.
    """
    differences: list[str] = []
    if first.logical_fingerprint != second.logical_fingerprint:
        differences.append(
            f"logical_fingerprint differs: {first.logical_fingerprint} != "
            f"{second.logical_fingerprint}"
        )
    if (
        first.configuration.config_fingerprint
        != second.configuration.config_fingerprint
    ):
        differences.append("config_fingerprint differs")
    first_stages: Mapping[str, ReproducibilityStage] = {
        stage.name: stage for stage in first.stages
    }
    second_stages: Mapping[str, ReproducibilityStage] = {
        stage.name: stage for stage in second.stages
    }
    for name in sorted(set(first_stages) | set(second_stages)):
        left = first_stages.get(name)
        right = second_stages.get(name)
        if left is None or right is None:
            differences.append(f"stage {name!r} is present in only one manifest")
            continue
        if left.logical_identity != right.logical_identity:
            differences.append(
                f"stage {name!r} logical identity differs: "
                f"{left.logical_identity} != {right.logical_identity}"
            )
        differences.extend(_artifact_differences(name, left, right))
    return tuple(differences)


def _artifact_differences(
    name: str, left: ReproducibilityStage, right: ReproducibilityStage
) -> list[str]:
    def by_path(
        stage: ReproducibilityStage,
    ) -> dict[str, DeterministicArtifact]:
        return {artifact.path: artifact for artifact in stage.deterministic_artifacts}

    differences: list[str] = []
    left_artifacts = by_path(left)
    right_artifacts = by_path(right)
    for path in sorted(set(left_artifacts) | set(right_artifacts)):
        first = left_artifacts.get(path)
        second = right_artifacts.get(path)
        if first is None or second is None:
            differences.append(
                f"stage {name!r}: {path!r} is present in only one manifest"
            )
        elif first.sha256 != second.sha256:
            differences.append(
                f"stage {name!r}: deterministic artifact {path!r} has different bytes"
            )
    return differences


def python_environment_series() -> str:
    """The Python series this build records in deterministic documents."""
    return python_series(platform.python_version())


__all__ = [
    "ReproducibilityError",
    "StageRecordError",
    "build_manifest",
    "build_stage_entries",
    "compare",
    "logical_fingerprint",
    "python_environment_series",
    "read_manifest",
]
