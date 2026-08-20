"""Lightweight local experiment tracking.

Every run writes a self-describing directory::

    artifacts/experiments/<run-name>/
        manifest.json            written atomically, last
        dataset.json             dataset provenance and fingerprint
        feature_catalog.json     the catalog the run was audited against
        splits.json              fold assignments and the split audit
        metrics.json             fold and aggregate metrics
        calibration.json         calibration design and per-fold outcomes
        ablations.json           feature-subset comparisons on shared folds
        predictions.parquet      per-row predictions and probabilities
        feature_importance.parquet  fold-level interpretation, unaggregated
        models/                  persisted estimators
        checksums.json           SHA-256 of every artifact above

``artifacts/`` is gitignored.  Datasets, predictions, and model files are
never committed.

Why not MLflow
--------------
MLflow is a Milestone 10 deliverable.  Adding a tracking server, a
database, and a UI now would introduce infrastructure before there is
anything to track: this milestone produces synthetic self-check runs only.
A directory of JSON and Parquet is inspectable with no service running,
diffable, and trivially reproducible, which is what a reviewer needs.

Model files are executable content
----------------------------------
``models/*.joblib`` are pickles.  Loading a pickle executes code in it.
**Never load a model file from an untrusted source.**  Everything needed
to judge a run — its provenance, its metrics, its splits, its
disclaimers — is in the JSON documents, so auditing a run never requires
unpickling anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Iterable, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import pyarrow as pa

from engagevr.features.assembly import write_parquet_atomic
from engagevr.schemas.experiments import RunManifest

#: Packages whose versions are recorded in every manifest.
TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "joblib",
    "pydantic",
)

#: Warning persisted next to every saved model.
MODEL_FILE_WARNING = (
    "Model files in this directory are Python pickles. Loading a pickle "
    "executes code contained in it. Never load a model file from an "
    "untrusted source. Auditing this run does not require loading any of "
    "them: the JSON documents in the run directory carry the provenance, "
    "the metrics, and the disclaimers."
)


class ArtifactError(OSError):
    """A run artifact could not be written or is incomplete."""


def dependency_versions() -> dict[str, str]:
    """Installed versions of the packages this pipeline depends on."""
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:  # pragma: no cover - all are required
            versions[package] = "not installed"
    return versions


def engagevr_version() -> str:
    """Installed EngageVR version."""
    try:
        return metadata.version("engagevr")
    except metadata.PackageNotFoundError:  # pragma: no cover - editable install
        return "unknown"


def build_run_id(
    *,
    target_name: str,
    evaluation_mode: str,
    dataset_fingerprint: str,
    random_seed: int,
    fold_count: int,
    model_names: Sequence[str],
    feature_set: Sequence[str],
    calibration_method: str | None,
) -> str:
    """Deterministic, collision-resistant identifier for a run.

    The identifier is a hash of everything that defines the run, so two
    runs differing in any of those inputs get different identifiers, and
    re-running an identical configuration reproduces the same identifier
    rather than accumulating near-duplicate directories.  No wall clock and
    no random component participates.
    """
    payload = {
        "target_name": target_name,
        "evaluation_mode": evaluation_mode,
        "dataset_fingerprint": dataset_fingerprint,
        "random_seed": random_seed,
        "fold_count": fold_count,
        "model_names": sorted(model_names),
        "feature_set": list(feature_set),
        "calibration_method": calibration_method,
        "engagevr_version": engagevr_version(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    short_mode = "sci" if evaluation_mode == "scientific" else "selfcheck"
    return f"{target_name}-{short_mode}-{digest}"


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExperimentRun:
    """Writer for one run directory.

    ``manifest.json`` is written **last** and atomically.  A directory
    without a manifest is an interrupted run, and a manifest with
    ``status=failed`` is a run that reached a conclusion and that
    conclusion was failure.  Neither is mistakable for success.
    """

    #: Artifacts a completed run must contain.
    REQUIRED_ARTIFACTS: tuple[str, ...] = (
        "dataset.json",
        "feature_catalog.json",
        "splits.json",
        "metrics.json",
    )

    def __init__(
        self,
        directory: Path,
        run_id: str,
        *,
        required_artifacts: Sequence[str] | None = None,
    ) -> None:
        self.directory = directory
        self.run_id = run_id
        # A run may declare a larger required set than the baseline one — a
        # fusion run must also produce its fusion documents before it may
        # claim completion — but never a smaller one.
        self.required_artifacts: tuple[str, ...] = (
            self.REQUIRED_ARTIFACTS
            if required_artifacts is None
            else tuple(required_artifacts)
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "models").mkdir(parents=True, exist_ok=True)
        self._written: list[str] = []

    @property
    def written_artifacts(self) -> tuple[str, ...]:
        """Relative paths written so far, in write order."""
        return tuple(self._written)

    def write_json(self, name: str, data: dict[str, Any]) -> Path:
        """Write a JSON artifact atomically."""
        path = self.directory / name
        write_json_atomic(path, data)
        self._record(name)
        return path

    def write_table(self, name: str, table: pa.Table) -> Path:
        """Write a Parquet artifact atomically."""
        path = self.directory / name
        write_parquet_atomic(table, path)
        self._record(name)
        return path

    def save_model(self, name: str, estimator: object) -> Path:
        """Persist a fitted estimator with joblib.

        See :data:`MODEL_FILE_WARNING`: the resulting file is executable
        content and must never be loaded from an untrusted source.
        """
        path = self.directory / "models" / f"{name}.joblib"
        temporary = path.with_name(f".{path.name}.tmp")
        joblib.dump(estimator, temporary)
        os.replace(temporary, path)
        self._record(f"models/{name}.joblib")
        return path

    def write_model_warning(self) -> Path:
        """Write the untrusted-pickle warning beside the saved models."""
        path = self.directory / "models" / "README.txt"
        path.write_text(MODEL_FILE_WARNING + "\n", encoding="utf-8")
        self._record("models/README.txt")
        return path

    def _record(self, name: str) -> None:
        if name not in self._written:
            self._written.append(name)

    def compute_checksums(self) -> dict[str, str]:
        """SHA-256 of every artifact written so far."""
        checksums: dict[str, str] = {}
        for name in self._written:
            path = self.directory / name
            if path.exists():
                checksums[name] = sha256_file(path)
        return checksums

    def missing_required_artifacts(self) -> tuple[str, ...]:
        """Required artifacts that are absent from the run directory."""
        return tuple(
            name
            for name in self.required_artifacts
            if not (self.directory / name).exists()
        )

    def finalize(self, manifest: RunManifest) -> Path:
        """Write ``checksums.json`` then ``manifest.json``, both atomically.

        Raises
        ------
        ArtifactError
            If a run claiming completion is missing a required artifact.
        """
        if manifest.status.value == "completed":
            missing = self.missing_required_artifacts()
            if missing:
                raise ArtifactError(
                    f"run {self.run_id!r} claims to have completed but is "
                    f"missing required artifact(s): {list(missing)}"
                )
        checksums = self.compute_checksums()
        write_json_atomic(self.directory / "checksums.json", checksums)
        final = manifest.model_copy(update={"artifact_checksums": checksums})
        path = self.directory / "manifest.json"
        write_json_atomic(path, final.model_dump(mode="json"))
        self._record("checksums.json")
        self._record("manifest.json")
        return path


def write_json_atomic(path: Path, data: dict[str, Any]) -> Path:
    """Write ``data`` to ``path`` via a temporary file and an atomic rename.

    Shared so that every run directory in the repository is written the
    same way: a reader never sees a half-written document, and an
    interrupted run leaves no file rather than a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(data, indent=2, default=str, sort_keys=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def read_manifest(directory: Path) -> RunManifest:
    """Read a run manifest.

    Raises
    ------
    ArtifactError
        If the directory holds no manifest, which means the run was
        interrupted before it reached a conclusion.
    """
    path = directory / "manifest.json"
    if not path.exists():
        raise ArtifactError(
            f"{directory} contains no manifest.json. The run was interrupted "
            "before it reached a conclusion; it is not a successful run."
        )
    with path.open("r", encoding="utf-8") as handle:
        return RunManifest.model_validate(json.load(handle))


def verify_checksums(directory: Path) -> tuple[str, ...]:
    """Artifacts whose current bytes disagree with the recorded checksum."""
    path = directory / "checksums.json"
    if not path.exists():
        raise ArtifactError(f"{directory} contains no checksums.json")
    with path.open("r", encoding="utf-8") as handle:
        recorded: dict[str, str] = json.load(handle)
    mismatched: list[str] = []
    for name, digest in recorded.items():
        artifact = directory / name
        if not artifact.exists() or sha256_file(artifact) != digest:
            mismatched.append(name)
    return tuple(mismatched)


def runtime_environment() -> dict[str, str]:
    """Python and platform details recorded in every manifest."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def predictions_table(
    *,
    window_ids: Sequence[str],
    subject_ids: Sequence[str],
    session_ids: Sequence[str],
    fold_indices: Sequence[int],
    model_names: Sequence[str],
    true_values: Sequence[str | float],
    predicted_values: Sequence[str | float],
    probability_labels: Sequence[str],
    uncalibrated: Sequence[Sequence[float] | None],
    calibrated: Sequence[Sequence[float] | None],
) -> pa.Table:
    """Build the per-row predictions table.

    Uncalibrated and calibrated probabilities are stored in separate
    columns so a reader can compare them without re-running anything.
    """
    columns: dict[str, list[Any]] = {
        "window_id": list(window_ids),
        "subject_id": list(subject_ids),
        "session_id": list(session_ids),
        "fold_index": [int(v) for v in fold_indices],
        "model_name": list(model_names),
        "true_value": [str(v) if isinstance(v, str) else v for v in true_values],
        "predicted_value": [
            str(v) if isinstance(v, str) else v for v in predicted_values
        ],
    }
    for index, label in enumerate(probability_labels):
        columns[f"probability_uncalibrated__{label}"] = [
            None if row is None else float(row[index]) for row in uncalibrated
        ]
        columns[f"probability_calibrated__{label}"] = [
            None if row is None else float(row[index]) for row in calibrated
        ]
    return pa.table(columns)


def importance_table(records: Iterable[dict[str, Any]]) -> pa.Table:
    """Build the fold-level feature-importance table from record dicts."""
    rows = list(records)
    if not rows:
        return pa.table(
            {
                "model_name": pa.array([], type=pa.string()),
                "fold_index": pa.array([], type=pa.int64()),
                "feature_name": pa.array([], type=pa.string()),
                "kind": pa.array([], type=pa.string()),
                "value": pa.array([], type=pa.float64()),
                "absolute_value": pa.array([], type=pa.float64()),
                "sign": pa.array([], type=pa.int64()),
                "standard_deviation": pa.array([], type=pa.float64()),
                "repeat_count": pa.array([], type=pa.int64()),
                "scaling_context": pa.array([], type=pa.string()),
                "class_label": pa.array([], type=pa.string()),
                "warning": pa.array([], type=pa.string()),
            }
        )
    keys = list(rows[0])
    return pa.table({key: [row.get(key) for row in rows] for key in keys})


def utc_stamp(moment: datetime) -> str:
    """ISO-8601 rendering used in artifact documents."""
    return moment.isoformat()
