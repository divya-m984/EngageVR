"""Immutable, checksum-linked model/artifact versions.

What this is
------------
A *model version* here is a record, not a registry entry.  It answers
three questions and refuses to answer a fourth:

- **Where did this estimator come from?**  Source run id, estimator class
  and hyperparameters, dataset fingerprint, split fingerprint,
  feature-schema fingerprint, configuration fingerprint.
- **Which bytes are it?**  Recorded — but not here.  See below.
- **What may be said about it?**  That it was fitted on SYNTHETIC data as
  a software self-check, and nothing more.

Logical version, serialized instance
------------------------------------
These are two different things and this module keeps them apart.

``ModelVersionManifest`` is the **portable logical** record and is
DVC-declared.  Its identifier is built from scientific and software
provenance only.

The ``.joblib``'s actual SHA-256 is an **execution-specific** fact and
lives in an :class:`~engagevr.schemas.mlops.ArtifactIntegrityRecord`
written beside the version directory, which is never DVC-declared.  It
must: ``joblib.dump`` writes scikit-learn's raw tree-node buffer, whose C
struct has seven never-initialised padding bytes per node, so two
serializations of one model disagree — measurably, in this repository's
own baseline run, for 191 of 200 trees.  An identifier built on that
digest renames the same model on every machine.

The relation is therefore::

    one logical model version  ->  N serialized artifact instances

Tamper detection is unchanged and is not weakened: every generated model
file still has a recorded SHA-256, the run's own ``checksums.json`` still
records it, ``--verify`` still checks it, and altering a model file still
changes its integrity record.  See DEC-105.

The fourth question — *should it be used?* — has no field.  There is no
stage, no alias, no promotion, and no approval, because nobody has made
that decision about any model in this repository and a schema that
offered a place to record one would invite somebody to invent it.  See
:data:`engagevr.schemas.mlops.FORBIDDEN_STATUS_WORDS`.

Immutability
------------
Nothing in this module writes to the run directory it reads.  Building a
version leaves the producing run byte-identical, which is what makes the
recorded checksums meaningful: they describe a run that was not touched
in order to be described.

The identifier
--------------
``model_version_id`` is a deterministic function of the provenance above,
so re-deriving a version from the same run reproduces the identifier
rather than minting a new one.  No wall clock, no random component, and
no platform-sensitive binary checksum participates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from engagevr.config import EngageVRConfig
from engagevr.dashboard.catalogue import detect_family
from engagevr.mlops.fingerprints import (
    build_configuration_version,
    feature_schema_fingerprint,
    repository_relative,
    sha256_payload,
    split_fingerprint,
)
from engagevr.mlops.stage_record import EXECUTION_SPECIFIC_MODEL_REASON
from engagevr.schemas.experiments import (
    SELF_CHECK_DISCLAIMER,
    SOFTWARE_SELF_CHECK_BANNER,
    EvaluationMode,
)
from engagevr.schemas.mlops import (
    MLOPS_DISCLAIMER,
    MODEL_VERSION_LIMITATION,
    ArtifactIntegrityEntry,
    ArtifactIntegrityRecord,
    ConfigurationVersion,
    ModelVersionManifest,
    python_series,
)
from engagevr.training.artifacts import (
    ArtifactError,
    dependency_versions,
    engagevr_version,
    sha256_file,
)

#: File name a written model version takes inside the output directory.
MANIFEST_SUFFIX = ".model-version.json"

#: Run documents a version record checksum-links, in recorded order.
#:
#: ``dataset.json`` is deliberately absent. It copies the dataset metadata
#: verbatim, including ``created_at_utc``, so its digest changes on every
#: execution — and a volatile digest inside a version record would make
#: the record volatile, which would make it unusable as a DVC-declared
#: output. The dataset is still pinned: ``dataset_fingerprint`` is a
#: separate field on this manifest, computed over canonical row content
#: with the wall clock excluded by construction.
REFERENCED_DOCUMENTS: tuple[str, ...] = (
    "feature_catalog.json",
    "splits.json",
    "metrics.json",
)


class ModelVersionError(ValueError):
    """A model version could not be built from the given run directory."""


def _read_json(directory: Path, name: str) -> dict[str, Any]:
    path = directory / name
    if not path.is_file():
        raise ModelVersionError(
            f"{directory} contains no {name}. A model version is derived from "
            "a completed run's own documents; an incomplete run has nothing "
            "to version."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ModelVersionError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ModelVersionError(f"{path} does not hold a JSON object")
    return document


def _parse_model_file(stem: str) -> tuple[str, int | None, bool]:
    """Split ``<model>-fold<N>[-calibrated]`` into its parts."""
    calibrated = stem.endswith("-calibrated")
    core = stem[: -len("-calibrated")] if calibrated else stem
    base, separator, tail = core.rpartition("-fold")
    if separator and tail.isdigit():
        return base, int(tail), calibrated
    return core, None, calibrated


def _estimator_kinds(metrics: Mapping[str, Any]) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for result in metrics.get("results", ()):
        name = result.get("model_name")
        kind = result.get("model_kind")
        if isinstance(name, str) and isinstance(kind, str):
            kinds[name] = kind
    return kinds


def estimator_parameters_fingerprint(parameters: Mapping[str, Any]) -> str:
    """SHA-256 over one estimator's recorded configuration.

    What the producing run wrote for this estimator: its class, its
    hyperparameters, its imputation and standardisation treatment, and its
    search grid.  Changing any of them is a different model, so the
    fingerprint participates in the logical identity; nothing in it varies
    between two machines.
    """
    return sha256_payload(parameters)


def build_model_version_id(
    *,
    source_run_id: str,
    target_name: str,
    task_type: str,
    estimator_type: str,
    estimator_class: str | None,
    parameters_digest: str,
    model_name: str,
    dataset_fingerprint: str,
    split_digest: str,
    feature_digest: str,
    config_digest: str,
    serialization_format: str,
    version: str,
) -> str:
    """Portable logical identifier for one fitted estimator.

    Provenance only.  The serialized artifact's SHA-256 is deliberately
    absent: it is a fact about one execution's heap and one machine's
    library build, and folding it in would give the same model a different
    identifier on every machine — which is exactly the defect DEC-105
    corrects.  The digest is still recorded, in the artifact-integrity
    execution record.
    """
    digest = sha256_payload(
        {
            "source_run_id": source_run_id,
            "target_name": target_name,
            "task_type": task_type,
            "estimator_type": estimator_type,
            "estimator_class": estimator_class,
            "estimator_parameters_fingerprint": parameters_digest,
            "model_name": model_name,
            "dataset_fingerprint": dataset_fingerprint,
            "split_fingerprint": split_digest,
            "feature_schema_fingerprint": feature_digest,
            "config_fingerprint": config_digest,
            "serialization_format": serialization_format,
            "engagevr_version": version,
        }
    )[:12]
    return f"mv-{target_name}-{model_name}-{digest}"


def build_model_versions(
    run_directory: Path,
    *,
    config: EngageVRConfig,
    configuration_version: ConfigurationVersion | None = None,
    model_names: Sequence[str] | None = None,
) -> tuple[ModelVersionManifest, ...]:
    """Derive one version record per persisted estimator in ``run_directory``.

    Reads ``manifest.json``, ``metrics.json``, ``splits.json``, and
    ``checksums.json``.  Never opens a ``.joblib``: a model file is a
    pickle, and hashing its bytes tells us everything a version record
    needs without executing anything in it.
    """
    directory = Path(run_directory)
    manifest = _read_json(directory, "manifest.json")
    metrics = _read_json(directory, "metrics.json")
    splits = _read_json(directory, "splits.json")

    if manifest.get("status") != "completed":
        raise ModelVersionError(
            f"{directory} records status "
            f"{manifest.get('status', 'unknown')!r}. Only a completed run may "
            "be versioned: versioning a failed or interrupted run would "
            "publish an identifier for an estimator nobody finished fitting."
        )

    recorded_checksums = _recorded_checksums(directory)

    models_directory = directory / "models"
    if not models_directory.is_dir():
        raise ModelVersionError(
            f"{directory} has no models/ directory, so it persisted no "
            "estimator to version."
        )
    files = sorted(p for p in models_directory.glob("*.joblib") if p.is_file())
    if not files:
        raise ModelVersionError(
            f"{models_directory} contains no .joblib file. The run completed "
            "without persisting an estimator; there is nothing to version."
        )

    evaluation_mode = EvaluationMode(manifest["evaluation_mode"])
    eligible = bool(manifest.get("scientific_evaluation_eligible", False))
    feature_set = tuple(str(name) for name in manifest.get("feature_set", ()))
    catalog_version = str(manifest.get("feature_catalog_version", "unknown"))
    split_digest = split_fingerprint(splits)
    feature_digest = feature_schema_fingerprint(
        feature_set, catalog_version=catalog_version
    )
    configuration = configuration_version or build_configuration_version(config)
    family, _evidence = detect_family(directory)
    kinds = _estimator_kinds(metrics)
    parameters = manifest.get("model_parameters", {})
    versions = manifest.get("dependency_versions", {})
    joblib_version = str(versions.get("joblib", "unknown"))
    data_source_counts = _data_source_counts(directory)
    is_synthetic = _is_synthetic(data_source_counts, eligible=eligible)

    wanted = set(model_names) if model_names else None
    manifests: list[ModelVersionManifest] = []
    for path in files:
        base, fold_index, calibrated = _parse_model_file(path.stem)
        if wanted is not None and base not in wanted:
            continue
        relative = f"models/{path.name}"
        digest = sha256_file(path)
        recorded = recorded_checksums.get(relative)
        if recorded is not None and recorded != digest:
            raise ModelVersionError(
                f"{relative} in {directory} does not match the SHA-256 the run "
                "recorded for it. The bytes changed after the run finished; a "
                "version built now would certify content the run never "
                "produced. Investigate before versioning."
            )
        estimator_parameters = parameters.get(base, {})
        estimator_class = estimator_parameters.get("estimator_class")
        parameters_digest = estimator_parameters_fingerprint(estimator_parameters)
        manifests.append(
            ModelVersionManifest(
                model_version_id=build_model_version_id(
                    source_run_id=str(manifest["run_id"]),
                    target_name=str(manifest["target_name"]),
                    task_type=str(manifest["task_type"]),
                    estimator_type=kinds.get(base, "unknown"),
                    estimator_class=(
                        str(estimator_class)
                        if isinstance(estimator_class, str)
                        else None
                    ),
                    parameters_digest=parameters_digest,
                    model_name=path.stem,
                    dataset_fingerprint=str(manifest["dataset_fingerprint"]),
                    split_digest=split_digest,
                    feature_digest=feature_digest,
                    config_digest=configuration.config_fingerprint,
                    serialization_format="joblib-pickle",
                    version=engagevr_version(),
                ),
                target_name=str(manifest["target_name"]),
                task_type=str(manifest["task_type"]),
                estimator_type=kinds.get(base, "unknown"),
                model_name=path.stem,
                estimator_class=(
                    str(estimator_class) if isinstance(estimator_class, str) else None
                ),
                fold_index=fold_index,
                is_calibrated=calibrated,
                calibration_method=(
                    str(manifest["calibration_method"])
                    if manifest.get("calibration_method") is not None
                    else None
                ),
                source_run_id=str(manifest["run_id"]),
                source_run_family=family.value,
                source_run_directory=repository_relative(directory),
                dataset_fingerprint=str(manifest["dataset_fingerprint"]),
                split_fingerprint=split_digest,
                feature_schema_fingerprint=feature_digest,
                estimator_parameters_fingerprint=parameters_digest,
                feature_catalog_version=catalog_version,
                feature_count=len(feature_set),
                configuration=configuration,
                serialization_library_version=joblib_version,
                model_artifact_path=relative,
                # The model file's own digest is deliberately absent: see
                # DEC-105. It is in the artifact-integrity record, which
                # build_model_artifact_integrity assembles from the same run.
                referenced_checksums={
                    name: recorded_checksums[name]
                    for name in REFERENCED_DOCUMENTS
                    if name in recorded_checksums
                },
                engagevr_version=str(manifest.get("engagevr_version", "unknown")),
                python_series=python_series(str(manifest.get("python_version", "0.0"))),
                dependency_versions={str(k): str(v) for k, v in dict(versions).items()}
                or dependency_versions(),
                evaluation_mode=evaluation_mode,
                is_synthetic=is_synthetic,
                scientific_evaluation_eligible=eligible,
                data_source_counts=data_source_counts,
                limitation=MODEL_VERSION_LIMITATION,
                disclaimers=_disclaimers(evaluation_mode),
            )
        )
    if not manifests:
        raise ModelVersionError(
            f"no persisted estimator in {models_directory} matched "
            f"{sorted(wanted or ())}"
        )
    return tuple(manifests)


def _disclaimers(mode: EvaluationMode) -> tuple[str, ...]:
    if mode is EvaluationMode.SOFTWARE_SELF_CHECK:
        return (SELF_CHECK_DISCLAIMER, MLOPS_DISCLAIMER)
    return (
        "This model version was derived from a run declared scientific. "
        "The version record still certifies only provenance and bytes; it "
        "is not an evaluation, an approval, or a release.",
        MLOPS_DISCLAIMER,
    )


def _data_source_counts(directory: Path) -> dict[str, int]:
    path = directory / "dataset.json"
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - caught upstream
        return {}
    counts = (
        document.get("data_source_counts", {}) if isinstance(document, dict) else {}
    )
    if not isinstance(counts, dict):
        return {}
    return {str(key): int(value) for key, value in counts.items()}


def _is_synthetic(counts: Mapping[str, int], *, eligible: bool) -> bool:
    """Whether every row the run saw was synthetic.

    Absent counts default to synthetic when the run is ineligible: every
    ineligible run in this repository is ineligible *because* its data was
    synthetic, and defaulting the other way would silently drop the
    provenance tag that the tracking layer requires.
    """
    if counts:
        return set(counts) == {"synthetic"}
    return not eligible


def build_model_artifact_integrity(
    manifests: Sequence[ModelVersionManifest], run_directory: Path
) -> tuple[ArtifactIntegrityEntry, ...]:
    """The real SHA-256 of every serialized estimator a version names.

    This is where the model checksums went, and nothing about tamper
    detection is softer for the move: every file is hashed, a changed
    model file changes this record, and :func:`verify_model_version`
    checks against it.  What changed is only that these digests no longer
    sit inside a DVC-declared document, where a correct reproduction on
    another machine would have looked like a changed pipeline.

    The model file is hashed, never loaded: it is a pickle.
    """
    directory = Path(run_directory)
    entries: list[ArtifactIntegrityEntry] = []
    seen: set[str] = set()
    for manifest in manifests:
        relative = manifest.model_artifact_path
        if relative in seen:
            continue
        target = directory / relative
        if not target.is_file():
            raise ModelVersionError(
                f"{target} is named by model version "
                f"{manifest.model_version_id} but does not exist; its "
                "integrity cannot be recorded."
            )
        seen.add(relative)
        entries.append(
            ArtifactIntegrityEntry(
                path=relative,
                sha256=sha256_file(target),
                size_bytes=target.stat().st_size,
                excluded_from_portable_identity=EXECUTION_SPECIFIC_MODEL_REASON,
            )
        )
    return tuple(sorted(entries, key=lambda entry: entry.path))


def write_model_versions(
    manifests: Sequence[ModelVersionManifest],
    output_directory: Path,
    *,
    integrity: Sequence[ArtifactIntegrityEntry] = (),
) -> tuple[Path, ...]:
    """Write one JSON document per version, atomically.

    Every written document is byte-stable: a version record carries no
    creation timestamp and no serialized-artifact checksum, because it is
    a DVC-declared output and either one would rewrite ``dvc.lock`` — the
    timestamp on every reproduction, the checksum on every machine.

    Two sidecars are written beside the directory, never inside it, and
    neither is ever DVC-declared: ``<directory>.execution.json`` records
    when the versions were built, and
    ``<directory>.artifact-integrity.execution.json`` records the actual
    SHA-256 of every model file they name.
    """
    from engagevr.mlops.execution import (
        write_execution_sidecar,
        write_integrity_sidecar,
    )
    from engagevr.training.artifacts import write_json_atomic

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for manifest in manifests:
        path = directory / f"{manifest.model_version_id}{MANIFEST_SUFFIX}"
        write_json_atomic(path, manifest.model_dump(mode="json"))
        written.append(path)
    write_execution_sidecar(
        directory,
        describes=repository_relative(directory),
        produced_by="engagevr model-manifest",
    )
    write_integrity_sidecar(
        directory,
        integrity,
        describes=repository_relative(directory),
        produced_by="engagevr model-manifest",
        # Run-relative, matching ModelVersionManifest.model_artifact_path,
        # so a reader holding a version record can resolve its artifact
        # without knowing where the pipeline root is.
        paths_relative_to=(
            "the producing run directory named by each version's source_run_directory"
        ),
    )
    return tuple(written)


def read_artifact_integrity(path: Path) -> ArtifactIntegrityRecord:
    """Read and validate a persisted artifact-integrity record."""
    return ArtifactIntegrityRecord.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def read_model_version(path: Path) -> ModelVersionManifest:
    """Read and validate one persisted model version."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelVersionManifest.model_validate(document)


def verify_model_version(
    manifest: ModelVersionManifest,
    *,
    run_directory: Path | None = None,
    integrity: ArtifactIntegrityRecord | Sequence[ArtifactIntegrityEntry] | None = None,
) -> tuple[str, ...]:
    """Artifacts whose current bytes disagree with the recorded checksum.

    The byte-stable run documents are checked against the version's own
    ``referenced_checksums``.  The model file is checked against the
    execution-specific integrity record, which is where its digest lives
    after DEC-105; when no record is supplied the run's own
    ``checksums.json`` is used, and when neither is available the model
    file is reported as unverifiable rather than silently passed.

    The model file is checked by hashing it, never by loading it: it is a
    pickle, and loading one executes code in it.
    """
    directory = (
        Path(run_directory)
        if run_directory is not None
        else Path(manifest.source_run_directory)
    )
    mismatched: list[str] = []
    relative = manifest.model_artifact_path
    expected = _expected_model_digest(manifest, directory, integrity)
    target = directory / relative
    if not target.is_file() or expected is None:
        mismatched.append(relative)
    elif sha256_file(target) != expected:
        mismatched.append(relative)
    for name, digest in manifest.referenced_checksums.items():
        referenced = directory / name
        if not referenced.is_file() or sha256_file(referenced) != digest:
            mismatched.append(name)
    return tuple(sorted(set(mismatched)))


def _expected_model_digest(
    manifest: ModelVersionManifest,
    run_directory: Path,
    integrity: ArtifactIntegrityRecord | Sequence[ArtifactIntegrityEntry] | None,
) -> str | None:
    """The SHA-256 this model file is supposed to have, or ``None``."""
    entries: Sequence[ArtifactIntegrityEntry] = ()
    if isinstance(integrity, ArtifactIntegrityRecord):
        entries = integrity.artifacts
    elif integrity is not None:
        entries = tuple(integrity)
    for entry in entries:
        if entry.path == manifest.model_artifact_path:
            return entry.sha256
    recorded = _recorded_checksums(run_directory)
    return recorded.get(manifest.model_artifact_path)


def _recorded_checksums(run_directory: Path) -> dict[str, str]:
    """The producing run's own ``checksums.json``, or an empty mapping."""
    path = Path(run_directory) / "checksums.json"
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - caught upstream
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(key): str(value) for key, value in loaded.items()}


def summarise(manifest: ModelVersionManifest) -> str:
    """One-line rendering used by the CLI."""
    fold = "all" if manifest.fold_index is None else str(manifest.fold_index)
    return (
        f"{manifest.model_version_id}  target={manifest.target_name} "
        f"kind={manifest.estimator_type} fold={fold} "
        f"calibrated={str(manifest.is_calibrated).lower()} "
        f"eligible={str(manifest.scientific_evaluation_eligible).lower()}"
    )


__all__ = [
    "MANIFEST_SUFFIX",
    "REFERENCED_DOCUMENTS",
    "SOFTWARE_SELF_CHECK_BANNER",
    "ArtifactError",
    "ModelVersionError",
    "build_model_artifact_integrity",
    "build_model_version_id",
    "build_model_versions",
    "estimator_parameters_fingerprint",
    "read_artifact_integrity",
    "read_model_version",
    "summarise",
    "verify_model_version",
    "write_model_versions",
]
