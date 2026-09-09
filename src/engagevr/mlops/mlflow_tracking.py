"""A thin, opt-in adapter between run directories and a local MLflow store.

Design
------
There is exactly one module in this repository that knows MLflow exists.
No Milestone 5--8 runner imports it, no runner calls it, and importing
this module does not import ``mlflow``: the client is imported inside the
functions that need it.  A tracking integration scattered through five
runners would make "did this command write to a tracking store?" a
question you answer by reading five files.

Direction of dependency
-----------------------
This module **reads** finished artifacts and **writes** to a tracking
store.  It never writes into the run directory, never recomputes a
metric, and never re-derives a conclusion.  Every number logged here was
already computed and already persisted by the runner that produced it,
which is what keeps the tracked value and the artifact value the same
number.

Local by default
----------------
``mlops.mlflow.tracking_uri`` is a repository-relative directory, which
MLflow reads as a file store.  No server, no database, no account, no API
key.  Remote and database schemes are refused at configuration load, not
here, so a misconfiguration fails before anything is written.

Why the file store, and why the opt-out
---------------------------------------
MLflow 3.15 puts the filesystem tracking backend in maintenance mode and
raises unless ``MLFLOW_ALLOW_FILE_STORE`` is set.  The alternative it
points at — a SQLAlchemy backend — needs the full ``mlflow``
distribution, and that distribution pins ``pandas<3``.  Adopting it would
downgrade this project's pandas across a major version to satisfy a
bookkeeping layer, which is the wrong trade: the modelling code is what
matters, and it is written against pandas 3.

So this project depends on ``mlflow-skinny`` (the tracking client
without the server stack) and sets the documented opt-out for the
duration of one client call, restoring whatever was there before.  The
flag is scoped by :func:`_file_store_opt_out` rather than exported into
the process, so nothing else in the program sees a changed environment.
The upper bound ``mlflow-skinny<4`` is the guard: if MLflow 4 removes the
file store, this project does not silently follow it.

What a tracked run is not
-------------------------
A run in the store is bookkeeping.  It is not validated, not approved,
not a release, and not a candidate for one.  No model is registered: a
registry entry carries a stage, and a stage is a decision somebody made
about a model.  Nobody has made one here.  The word list in
:data:`engagevr.schemas.mlops.FORBIDDEN_STATUS_WORDS` is enforced on the
experiment name, the run name, and every tag value.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engagevr.config import EngageVRConfig
from engagevr.dashboard.catalogue import detect_family
from engagevr.mlops.fingerprints import (
    build_configuration_version,
    repository_relative,
    split_fingerprint,
)
from engagevr.schemas.dashboard import DashboardRunFamily
from engagevr.schemas.experiments import (
    SELF_CHECK_DISCLAIMER,
    SOFTWARE_SELF_CHECK_BANNER,
)
from engagevr.schemas.mlops import (
    MLOPS_DISCLAIMER,
    ConfigurationVersion,
    MLOpsRunSummary,
    ModelVersionManifest,
)

#: Documents copied into the tracking store when artifact logging is on.
#:
#: JSON only, plus the checksum file. These are the documents a reviewer
#: needs to judge a run, and none of them contains media, a credential,
#: or anything personal.
LOGGED_JSON_ARTIFACTS: tuple[str, ...] = (
    "manifest.json",
    "dataset.json",
    "feature_catalog.json",
    "splits.json",
    "metrics.json",
    "calibration.json",
    "ablations.json",
    "fusion_config.json",
    "fusion_metrics.json",
    "experts.json",
    "robustness.json",
    "personalization_config.json",
    "personalization.json",
    "personal_baselines.json",
    "uncertainty_config.json",
    "uncertainty.json",
    "thresholds.json",
    "selective_metrics.json",
    "coverage_curve.json",
    "adaptation_policy_config.json",
    "adaptation_summary.json",
    "scenarios.json",
    "checksums.json",
)

#: Parquet tables eligible for logging, subject to :data:`MAX_PARQUET_BYTES`.
LOGGED_PARQUET_ARTIFACTS: tuple[str, ...] = ("predictions.parquet",)

#: Largest Parquet artifact copied into the tracking store.
#:
#: A tracking store is metadata plus small evidence. Copying an
#: arbitrarily large table into it turns a browsable record into a second
#: copy of the dataset, so anything above this is skipped with a stated
#: reason rather than silently.
MAX_PARQUET_BYTES = 8 * 1024 * 1024

#: Never copied into a tracking store, under any setting.
#:
#: Model files are pickles and are executable content; frames, crops, and
#: recordings are raw media; ``.env`` and ``secrets/`` are credentials.
#: None of them belongs in a shareable metadata store.
NEVER_LOGGED: tuple[str, ...] = (
    "models/",
    "*.joblib",
    "*.pkl",
    "*.env",
    ".env",
    "secrets/",
    "*.mp4",
    "*.avi",
    "*.png",
    "*.jpg",
    "*.npy",
    "events.jsonl",
)

#: Longest tag or parameter value written. Longer values are truncated
#: with a visible marker rather than silently cut.
MAX_VALUE_LENGTH = 480


#: MLflow's documented opt-out for the maintenance-mode file store.
FILE_STORE_OPT_OUT_ENV = "MLFLOW_ALLOW_FILE_STORE"

#: MLflow's telemetry opt-out.
#:
#: This milestone is local-first and must work with no network. MLflow's
#: usage telemetry is disabled for the duration of every call this module
#: makes, so tracking a synthetic run cannot become an outbound request.
TELEMETRY_OPT_OUT_ENV = "MLFLOW_DISABLE_TELEMETRY"

#: A path component MLflow's file store refuses to hold a run under.
#:
#: ``FileStore._is_valid_run_directory`` rejects any run directory with a
#: component named ``artifacts`` anywhere above it — a path-traversal
#: defence added in MLflow 3. A tracking store placed inside this
#: project's own ``artifacts/`` tree would therefore create runs it then
#: reports as "not found", so the store lives at ``mlruns/`` instead.
FORBIDDEN_STORE_COMPONENT = "artifacts"


class TrackingError(RuntimeError):
    """Tracking was requested but could not be performed."""


@contextmanager
def _file_store_opt_out() -> Iterator[None]:
    """Enable the local file store and disable telemetry, for one call.

    Scoped rather than exported: a library that permanently rewrites the
    process environment surprises everything else running in it, and the
    previous values — including their absence — are restored on the way
    out.
    """
    previous = {
        name: os.environ.get(name)
        for name in (FILE_STORE_OPT_OUT_ENV, TELEMETRY_OPT_OUT_ENV)
    }
    os.environ[FILE_STORE_OPT_OUT_ENV] = "true"
    os.environ[TELEMETRY_OPT_OUT_ENV] = "true"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def assert_usable_store(directory: Path) -> Path:
    """Refuse a store location MLflow's file store cannot serve.

    Raised here rather than left to MLflow, whose own error is
    ``Run '<id>' not found`` after it has already written the run to
    disk — a message that sends a reader looking for a missing file
    rather than at the path they configured.
    """
    parts = {part.lower() for part in directory.resolve().parts}
    if FORBIDDEN_STORE_COMPONENT in parts:
        raise TrackingError(
            f"the MLflow store {directory} sits under a directory named "
            f"{FORBIDDEN_STORE_COMPONENT!r}. MLflow's file store treats any "
            "run path containing that component as a path-traversal attempt "
            "and reports every run it writes there as missing. Configure "
            "mlops.mlflow.tracking_uri to a path without it; the default is "
            "'mlruns'."
        )
    return directory


def resolve_tracking_uri(config: EngageVRConfig, *, base: Path | None = None) -> str:
    """The absolute, local tracking URI this configuration describes."""
    from engagevr.mlops.fingerprints import REPOSITORY_ROOT

    raw = config.mlops.mlflow.tracking_uri.strip()
    if "://" in raw:
        return raw
    root = Path(base) if base is not None else REPOSITORY_ROOT
    directory = assert_usable_store((root / raw).resolve())
    directory.mkdir(parents=True, exist_ok=True)
    return directory.as_uri()


def _truncate(value: str) -> str:
    if len(value) <= MAX_VALUE_LENGTH:
        return value
    return value[: MAX_VALUE_LENGTH - 15] + "...(truncated)"


def _read_json(directory: Path, name: str) -> dict[str, Any] | None:
    path = directory / name
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None


def _aggregate_metrics(
    metrics: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, str]]:
    """Fold-aggregated metrics exactly as the run recorded them.

    Nothing is recomputed and nothing is defaulted.  An aggregate the run
    could not compute is recorded in the skipped map with the run's own
    reason; it never becomes a zero, because zero is a legitimate score.
    """
    logged: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for result in metrics.get("results", ()):
        model = str(result.get("model_name", "unknown"))
        for aggregate in result.get("aggregate", ()):
            name = str(aggregate.get("name", "unknown"))
            key = f"{model}/{name}"
            mean = aggregate.get("mean")
            if mean is None:
                skipped[key] = str(
                    aggregate.get("unavailable_reason")
                    or "the run recorded no value for this aggregate"
                )
                continue
            logged[key] = float(mean)
            deviation = aggregate.get("standard_deviation")
            if deviation is not None:
                logged[f"{key}.standard_deviation"] = float(deviation)
            valid = aggregate.get("valid_fold_count")
            if valid is not None:
                logged[f"{key}.valid_fold_count"] = float(valid)
    return logged, skipped


def _selective_metrics(document: Mapping[str, Any]) -> dict[str, float]:
    """Selective-prediction accounting, kept in its own namespace.

    Coverage and abstention describe *which windows were answered*.  They
    are not predictive performance and must never be read beside accuracy
    as though they were, so they never share a metric prefix with it.
    """
    logged: dict[str, float] = {}
    for name in (
        "coverage",
        "abstention_rate",
        "total_window_count",
        "accepted_count",
        "abstained_count",
        "unavailable_count",
    ):
        value = document.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool):
            logged[f"selective/{name}"] = float(value)
    return logged


def _personalization_metrics(document: Mapping[str, Any]) -> dict[str, float]:
    logged: dict[str, float] = {}
    for name in (
        "personalization_coverage",
        "personalized_subject_count",
        "cold_start_subject_count",
        "unavailable_personalization_count",
        "total_calibration_window_count",
        "total_evaluation_window_count",
    ):
        value = document.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool):
            logged[f"personalization/{name}"] = float(value)
    for prefix, key in (
        ("population", "population_aggregate"),
        ("personalized", "personalized_aggregate"),
    ):
        for aggregate in document.get(key, ()):
            mean = aggregate.get("mean")
            if mean is not None:
                logged[f"{prefix}/{aggregate.get('name', 'unknown')}"] = float(mean)
    return logged


def _controller_metrics(document: Mapping[str, Any]) -> dict[str, float]:
    """Adaptation-controller diagnostics.

    Namespaced ``controller/`` and tagged as a non-training run.  These
    count what a deterministic rule did.  Adaptation activity is not
    adaptation benefit, and no field here says otherwise.
    """
    logged: dict[str, float] = {}
    for name, value in dict(document.get("metrics", {})).items():
        if isinstance(value, int | float) and not isinstance(value, bool):
            logged[f"controller/{name}"] = float(value)
    return logged


def _parameters(
    directory: Path,
    manifest: Mapping[str, Any] | None,
    family: DashboardRunFamily,
    configuration: ConfigurationVersion,
) -> dict[str, str]:
    parameters: dict[str, str] = {
        "engagevr.run_family": family.value,
        "engagevr.config_fingerprint": configuration.config_fingerprint,
    }
    if manifest is None:
        summary = _read_json(directory, "adaptation_summary.json") or {}
        parameters.update(
            {
                "target": "none",
                "task_type": "controller_diagnostics",
                "estimator": "deterministic_rule",
                "seed": str(
                    summary.get("configuration", {}).get("random_seed", "none")
                ),
                "scenario_count": str(len(summary.get("scenario_names", ()))),
                "policy_mode": str(
                    summary.get("configuration", {}).get("mode", "unknown")
                ),
            }
        )
        return {key: _truncate(str(value)) for key, value in parameters.items()}

    splits = _read_json(directory, "splits.json") or {}
    parameters.update(
        {
            "target": str(manifest.get("target_name", "unknown")),
            "task_type": str(manifest.get("task_type", "unknown")),
            "estimator": ", ".join(str(n) for n in manifest.get("model_names", ())),
            "feature_subset": (
                f"{len(manifest.get('feature_set', ()))} predictor columns from "
                f"feature catalog {manifest.get('feature_catalog_version', '?')}"
            ),
            "feature_count": str(len(manifest.get("feature_set", ()))),
            "seed": str(manifest.get("random_seed", "unknown")),
            "split_strategy": str(manifest.get("split_strategy", "unknown")),
            "group_field": str(manifest.get("group_field", "unknown")),
            "group_count": str(manifest.get("group_count", "unknown")),
            "fold_count": str(manifest.get("fold_count", "unknown")),
            "calibration_method": str(manifest.get("calibration_method", "none")),
            "evaluation_mode": str(manifest.get("evaluation_mode", "unknown")),
            "scientific_evaluation_eligible": str(
                bool(manifest.get("scientific_evaluation_eligible", False))
            ).lower(),
            "dataset_fingerprint": str(manifest.get("dataset_fingerprint", "unknown")),
            "split_fingerprint": split_fingerprint(splits) if splits else "unavailable",
        }
    )
    for name, key in (
        ("fusion_strategy", "fusion_config.json"),
        ("personalization_configuration", "personalization_config.json"),
        ("uncertainty_configuration", "uncertainty_config.json"),
    ):
        document = _read_json(directory, key)
        if document is not None:
            parameters[name] = json.dumps(
                document.get("configuration", document), sort_keys=True
            )
    return {key: _truncate(str(value)) for key, value in parameters.items()}


def _tags(
    directory: Path,
    manifest: Mapping[str, Any] | None,
    family: DashboardRunFamily,
    *,
    run_id: str,
    is_synthetic: bool,
    eligible: bool,
    evaluation_mode: str,
    data_source: str,
) -> dict[str, str]:
    from engagevr.training.artifacts import engagevr_version

    tags = {
        "engagevr.data_source": data_source,
        "engagevr.is_synthetic": str(is_synthetic).lower(),
        "engagevr.scientific_evaluation_eligible": str(eligible).lower(),
        "engagevr.evaluation_mode": evaluation_mode,
        "engagevr.disclaimer": SOFTWARE_SELF_CHECK_BANNER
        if not eligible
        else "Declared scientific by the producing run.",
        "engagevr.run_family": family.value,
        "engagevr.run_id": run_id,
        "engagevr.version": engagevr_version(),
        "engagevr.limitation": _truncate(MLOPS_DISCLAIMER),
        "engagevr.source_directory": repository_relative(directory),
        "engagevr.is_model_training_run": str(
            family is not DashboardRunFamily.ADAPTATION
        ).lower(),
        "engagevr.registered_model": "none",
    }
    if manifest is not None:
        tags["engagevr.target"] = str(manifest.get("target_name", "unknown"))
        tags["engagevr.task_type"] = str(manifest.get("task_type", "unknown"))
    return tags


def _artifact_paths(
    directory: Path, *, log_artifacts: bool
) -> tuple[list[Path], dict[str, str]]:
    if not log_artifacts:
        return [], {}
    paths: list[Path] = []
    skipped: dict[str, str] = {}
    for name in LOGGED_JSON_ARTIFACTS:
        path = directory / name
        if path.is_file():
            paths.append(path)
    for name in LOGGED_PARQUET_ARTIFACTS:
        path = directory / name
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_PARQUET_BYTES:
            skipped[name] = (
                f"{size} bytes exceeds MAX_PARQUET_BYTES ({MAX_PARQUET_BYTES}); "
                "a tracking store holds metadata and small evidence, not a "
                "second copy of the dataset"
            )
            continue
        paths.append(path)
    return paths, skipped


def log_run_directory(
    run_directory: Path,
    *,
    config: EngageVRConfig,
    configuration_version: ConfigurationVersion | None = None,
    model_versions: Sequence[ModelVersionManifest] = (),
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> MLOpsRunSummary:
    """Log one finished run directory to a local MLflow store.

    Raises
    ------
    TrackingError
        If MLflow is not installed, or the directory holds no conclusive
        document.  An interrupted run is not tracked: recording it would
        put a run in the store that never reached a conclusion.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - mlflow is a declared dependency
        raise TrackingError(
            "MLflow is not installed. It is a declared dependency of this "
            "project; run `uv sync` and try again."
        ) from exc

    directory = Path(run_directory)
    manifest = _read_json(directory, "manifest.json")
    adaptation = _read_json(directory, "adaptation_summary.json")
    if manifest is None and adaptation is None:
        raise TrackingError(
            f"{directory} holds neither manifest.json nor "
            "adaptation_summary.json, so no run in it reached a conclusion. "
            "An interrupted run is not tracked."
        )
    family, _evidence = detect_family(directory)
    conclusion: Mapping[str, Any] = manifest if manifest is not None else adaptation  # type: ignore[assignment]
    run_id = str(conclusion.get("run_id", directory.name))
    eligible = bool(conclusion.get("scientific_evaluation_eligible", False))
    evaluation_mode = str(conclusion.get("evaluation_mode", "software_self_check"))
    dataset = _read_json(directory, "dataset.json") or {}
    counts = {str(k): int(v) for k, v in dataset.get("data_source_counts", {}).items()}
    if counts:
        is_synthetic = set(counts) == {"synthetic"}
        data_source = "synthetic" if is_synthetic else "+".join(sorted(counts))
    else:
        is_synthetic = bool(conclusion.get("is_synthetic", not eligible))
        data_source = str(conclusion.get("data_source", "synthetic"))

    configuration = configuration_version or build_configuration_version(config)
    uri = tracking_uri or resolve_tracking_uri(config)
    experiment = experiment_name or config.mlops.mlflow.experiment_name
    name = run_name or f"{family.value}-{run_id}"

    if uri.startswith("file:"):
        assert_usable_store(Path(uri.removeprefix("file://")))
    with _file_store_opt_out():
        client = MlflowClient(tracking_uri=uri)
        existing = client.get_experiment_by_name(experiment)
        experiment_id = (
            existing.experiment_id
            if existing is not None
            else client.create_experiment(experiment)
        )

    tags = _tags(
        directory,
        manifest,
        family,
        run_id=run_id,
        is_synthetic=is_synthetic,
        eligible=eligible,
        evaluation_mode=evaluation_mode,
        data_source=data_source,
    )
    if model_versions:
        tags["engagevr.model_versions"] = _truncate(
            ", ".join(v.model_version_id for v in model_versions)
        )
    parameters = _parameters(directory, manifest, family, configuration)

    metrics: dict[str, float] = {}
    skipped: dict[str, str] = {}
    metrics_document = _read_json(directory, "metrics.json")
    if metrics_document is not None:
        metrics, skipped = _aggregate_metrics(metrics_document)
    selective = _read_json(directory, "selective_metrics.json")
    if selective is not None:
        metrics.update(_selective_metrics(selective))
    personalization = _read_json(directory, "personalization.json")
    if personalization is not None:
        metrics.update(_personalization_metrics(personalization))
    if adaptation is not None:
        metrics.update(_controller_metrics(adaptation))

    artifacts, artifact_skips = _artifact_paths(
        directory, log_artifacts=config.mlops.mlflow.log_artifacts
    )
    skipped.update(artifact_skips)

    with _file_store_opt_out():
        run = client.create_run(experiment_id, run_name=name, tags=tags)
        mlflow_run_id = run.info.run_id
        try:
            for key, parameter in sorted(parameters.items()):
                client.log_param(mlflow_run_id, key, parameter)
            for key, metric in sorted(metrics.items()):
                client.log_metric(mlflow_run_id, key, metric)
            for path in artifacts:
                client.log_artifact(mlflow_run_id, str(path), artifact_path="run")
        except Exception:
            client.set_terminated(mlflow_run_id, "FAILED")
            raise
        client.set_terminated(mlflow_run_id, "FINISHED")

    return MLOpsRunSummary(
        tracking_uri=uri,
        experiment_name=experiment,
        experiment_id=str(experiment_id),
        mlflow_run_id=mlflow_run_id,
        mlflow_run_name=name,
        mlflow_version=str(mlflow.__version__),
        source_run_directory=repository_relative(directory),
        source_run_id=run_id,
        run_family=family.value,
        tags=tags,
        parameters=parameters,
        metrics=metrics,
        logged_artifacts=tuple(sorted(f"run/{p.name}" for p in artifacts)),
        skipped_metrics=skipped,
        model_versions=tuple(v.model_version_id for v in model_versions),
        is_synthetic=is_synthetic,
        scientific_evaluation_eligible=eligible,
        created_at_utc=datetime.now(UTC),
        disclaimers=(
            SELF_CHECK_DISCLAIMER if is_synthetic else MLOPS_DISCLAIMER,
            MLOPS_DISCLAIMER,
        ),
    )


def log_run_directories(
    directories: Iterable[Path],
    *,
    config: EngageVRConfig,
    configuration_version: ConfigurationVersion | None = None,
) -> tuple[MLOpsRunSummary, ...]:
    """Log several finished runs into the same local experiment."""
    return tuple(
        log_run_directory(
            directory, config=config, configuration_version=configuration_version
        )
        for directory in directories
    )


__all__ = [
    "FILE_STORE_OPT_OUT_ENV",
    "LOGGED_JSON_ARTIFACTS",
    "LOGGED_PARQUET_ARTIFACTS",
    "MAX_PARQUET_BYTES",
    "MAX_VALUE_LENGTH",
    "NEVER_LOGGED",
    "TELEMETRY_OPT_OUT_ENV",
    "TrackingError",
    "assert_usable_store",
    "log_run_directories",
    "log_run_directory",
    "resolve_tracking_uri",
]
