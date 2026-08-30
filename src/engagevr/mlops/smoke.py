"""The integrated system smoke check.

What it proves
--------------
That the pieces fit together: the package imports, the configuration
parses, a synthetic dataset can be generated and validated, the
Milestone 5 pipeline runs end to end on it, its artifacts validate
against their own checksums, a model version can be derived, local MLflow
tracking works, a distribution-shift diagnostic runs, the Milestone 9
catalogue discovers the resulting run, the Milestone 4 backend
application can be constructed, and the Milestone 9 dashboard module
imports.

What it does not prove
----------------------
Anything about a model.  A passing smoke check is
``SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION``, and the report
carries ``scientific_evaluation_eligible=false`` as a required field so
it cannot be quoted without that.

What it never needs
-------------------
A webcam, a network, Unity, a browser, a display server, an external
dataset, participant data, or a running server.  The backend check
*constructs* the FastAPI application and inspects its routes; it never
binds a socket.  The dashboard check imports the module and builds the
launch argv; it never starts Streamlit.  Every check runs against a
caller-supplied directory, so nothing is written outside it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from engagevr.config import EngageVRConfig
from engagevr.schemas.experiments import SELF_CHECK_DISCLAIMER
from engagevr.schemas.mlops import (
    MLOPS_DISCLAIMER,
    MLOpsRunSummary,
    SmokeCheckResult,
    SmokeCheckStatus,
    SmokeReport,
)

#: Dataset size used by the smoke check.
#:
#: Small on purpose: this layer exists to be run on every commit, and a
#: check nobody waits for is a check nobody runs. Statistical adequacy is
#: irrelevant here — the question is whether the wiring holds.
SMOKE_SUBJECTS = 12
SMOKE_SESSIONS_PER_SUBJECT = 2
SMOKE_WINDOWS_PER_SESSION = 5
SMOKE_FOLDS = 3
SMOKE_SEED = 42
SMOKE_DRIFT_SEED = 7

#: Estimators the smoke run fits.
#:
#: Two, not the whole registry. One linear model and one software-check
#: baseline are enough to exercise fitting, calibration, persistence, and
#: the interpretation path; fitting three more would only make the check
#: slower. The question here is whether the wiring holds, never how well
#: anything predicts.
SMOKE_MODELS = "dummy,logistic_regression"

#: Permutation repeats for the smoke run's interpretation step.
#:
#: One. A permutation importance from a single repeat has no standard
#: error worth reading, which is fine: this check asserts that the step
#: runs and writes its table, not what the table says.
SMOKE_PERMUTATION_REPEATS = 1


@dataclass
class _Context:
    """Mutable state shared between checks, in declaration order."""

    directory: Path
    config: EngageVRConfig
    reference_dataset: Path
    current_dataset: Path
    run_directory: Path


@contextmanager
def _quiet() -> Iterator[StringIO]:
    """Swallow a subcommand's stdout, keeping the smoke output readable."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        yield buffer


def _passed(name: str, detail: str) -> SmokeCheckResult:
    return SmokeCheckResult(name=name, status=SmokeCheckStatus.PASSED, detail=detail)


def _failed(name: str, reason: str) -> SmokeCheckResult:
    return SmokeCheckResult(
        name=name, status=SmokeCheckStatus.FAILED, failure_reason=reason
    )


def _skipped(name: str, reason: str) -> SmokeCheckResult:
    return SmokeCheckResult(
        name=name, status=SmokeCheckStatus.SKIPPED, skip_reason=reason
    )


def _check_package_imports(_: _Context) -> SmokeCheckResult:
    import importlib

    modules = (
        "engagevr",
        "engagevr.config",
        "engagevr.features.synthetic",
        "engagevr.training.runner",
        "engagevr.training.uncertainty_runner",
        "engagevr.adaptation.runner",
        "engagevr.dashboard.catalogue",
        "engagevr.api.app",
        "engagevr.mlops.pipeline",
    )
    for name in modules:
        importlib.import_module(name)
    return _passed(
        "package_imports", f"imported {len(modules)} modules across every milestone"
    )


def _check_configuration_loads(context: _Context) -> SmokeCheckResult:
    config = context.config
    return _passed(
        "configuration_loads",
        f"project {config.project.name} {config.project.version}; "
        f"mlflow.enabled={str(config.mlops.mlflow.enabled).lower()}; "
        f"store_raw_video={str(config.capture.store_raw_video).lower()}",
    )


def _check_synthetic_dataset(context: _Context) -> SmokeCheckResult:
    from engagevr.__main__ import main

    for path, seed in (
        (context.reference_dataset, SMOKE_SEED),
        (context.current_dataset, SMOKE_DRIFT_SEED),
    ):
        with _quiet():
            status = main(
                [
                    "features-demo",
                    "--seed",
                    str(seed),
                    "--subjects",
                    str(SMOKE_SUBJECTS),
                    "--sessions-per-subject",
                    str(SMOKE_SESSIONS_PER_SUBJECT),
                    "--windows-per-session",
                    str(SMOKE_WINDOWS_PER_SESSION),
                    "--output",
                    str(path),
                ]
            )
        if status != 0:
            return _failed(
                "synthetic_dataset_generated",
                f"features-demo exited {status} while writing {path}",
            )
    return _passed(
        "synthetic_dataset_generated",
        f"two SYNTHETIC datasets of {SMOKE_SUBJECTS} subjects written",
    )


def _check_dataset_metadata(context: _Context) -> SmokeCheckResult:
    from engagevr.features.assembly import read_dataset_metadata

    metadata = read_dataset_metadata(context.reference_dataset)
    if metadata.scientific_evaluation_eligible:
        return _failed(
            "dataset_provenance_preserved",
            "the synthetic dataset claims scientific eligibility",
        )
    if set(metadata.data_source_counts) != {"synthetic"}:
        return _failed(
            "dataset_provenance_preserved",
            f"unexpected data sources {sorted(metadata.data_source_counts)}",
        )
    return _passed(
        "dataset_provenance_preserved",
        f"{metadata.row_count} rows, all synthetic, "
        f"fingerprint {metadata.dataset_fingerprint[:12]}..., "
        "scientific_evaluation_eligible=false",
    )


def _check_baseline_pipeline(context: _Context) -> SmokeCheckResult:
    from engagevr.__main__ import main

    with _quiet():
        status = main(
            [
                "baseline-demo",
                "--dataset",
                str(context.reference_dataset),
                "--target",
                "engagement_class",
                "--folds",
                str(SMOKE_FOLDS),
                "--seed",
                str(SMOKE_SEED),
                "--models",
                SMOKE_MODELS,
                "--permutation-repeats",
                str(SMOKE_PERMUTATION_REPEATS),
                "--no-ablations",
                "--output",
                str(context.run_directory),
            ]
        )
    if status != 0:
        return _failed("baseline_pipeline_ran", f"baseline-demo exited {status}")
    return _passed(
        "baseline_pipeline_ran",
        f"{SMOKE_FOLDS}-fold grouped run written to {context.run_directory.name}",
    )


def _check_artifact_integrity(context: _Context) -> SmokeCheckResult:
    from engagevr.training.artifacts import read_manifest, verify_checksums

    manifest = read_manifest(context.run_directory)
    if manifest.status.value != "completed":
        return _failed(
            "artifact_manifest_validated",
            f"run status is {manifest.status.value!r}",
        )
    if manifest.scientific_evaluation_eligible:
        return _failed(
            "artifact_manifest_validated",
            "a synthetic self-check run claims scientific eligibility",
        )
    mismatched = verify_checksums(context.run_directory)
    if mismatched:
        return _failed(
            "artifact_manifest_validated",
            f"checksum mismatch for {list(mismatched)}",
        )
    return _passed(
        "artifact_manifest_validated",
        f"run {manifest.run_id} completed; "
        f"{len(manifest.artifact_checksums)} checksums verified; "
        "scientific_evaluation_eligible=false",
    )


def _check_model_version(context: _Context) -> SmokeCheckResult:
    from engagevr.mlops.model_version import (
        build_model_versions,
        read_model_version,
        verify_model_version,
        write_model_versions,
    )

    versions = build_model_versions(context.run_directory, config=context.config)
    directory = context.directory / "model_versions"
    written = write_model_versions(versions, directory)
    reread = read_model_version(written[0])
    if reread.model_version_id != versions[0].model_version_id:
        return _failed(
            "model_version_manifest_validated",
            "a written model version did not read back identically",
        )
    mismatched = verify_model_version(reread, run_directory=context.run_directory)
    if mismatched:
        return _failed(
            "model_version_manifest_validated",
            f"model-version checksum mismatch for {list(mismatched)}",
        )
    if any(version.scientific_evaluation_eligible for version in versions):
        return _failed(
            "model_version_manifest_validated",
            "a synthetic model version claims scientific eligibility",
        )
    return _passed(
        "model_version_manifest_validated",
        f"{len(versions)} immutable, checksum-linked version(s); "
        f"first is {versions[0].model_version_id}",
    )


def _check_mlflow_tracking(context: _Context) -> SmokeCheckResult:
    """Log the smoke run to a throwaway local store.

    The store is a temporary directory that is removed when the check
    returns, so a smoke run never accumulates tracking state and never
    writes into the project's own ``mlruns/``.  It cannot live under the
    smoke output directory: that sits inside ``artifacts/``, and MLflow's
    file store refuses to serve runs from a path containing a component
    of that name.
    """
    import tempfile

    from engagevr.mlops.mlflow_tracking import TrackingError, log_run_directory
    from engagevr.schemas.mlops import REQUIRED_TRACKING_TAGS

    with tempfile.TemporaryDirectory(prefix="engagevr-smoke-mlflow-") as temporary:
        store = Path(temporary).resolve()
        try:
            summary = log_run_directory(
                context.run_directory,
                config=context.config,
                tracking_uri=store.as_uri(),
                experiment_name="engagevr-smoke",
            )
        except TrackingError as exc:
            return _skipped("mlflow_tracking_local", str(exc))
        return _validate_tracking_summary(summary, REQUIRED_TRACKING_TAGS)


def _validate_tracking_summary(
    summary: MLOpsRunSummary, required_tags: tuple[str, ...]
) -> SmokeCheckResult:
    """What a tracked synthetic run must carry before the check passes."""
    missing = [tag for tag in required_tags if tag not in summary.tags]
    if missing:
        return _failed("mlflow_tracking_local", f"missing provenance tags {missing}")
    if summary.scientific_evaluation_eligible:
        return _failed(
            "mlflow_tracking_local",
            "a tracked synthetic run claims scientific eligibility",
        )
    if summary.registered_model is not None:
        return _failed(
            "mlflow_tracking_local",
            "a model was registered; registration is bookkeeping",
        )
    return _passed(
        "mlflow_tracking_local",
        f"MLflow {summary.mlflow_version} local file store; run "
        f"{summary.mlflow_run_id[:12]}...; {len(summary.metrics)} metrics; "
        "is_synthetic=true; scientific_evaluation_eligible=false",
    )


def _check_drift_diagnostic(context: _Context) -> SmokeCheckResult:
    from engagevr.mlops.drift import compare_datasets

    report = compare_datasets(
        context.reference_dataset,
        context.current_dataset,
        thresholds=context.config.mlops.drift,
    )
    if report.scientific_evaluation_eligible:
        return _failed(
            "drift_diagnostic_ran",
            "a synthetic distribution-shift report claims scientific eligibility",
        )
    if report.features_compared_count == 0:
        return _failed("drift_diagnostic_ran", "no feature was compared")
    return _passed(
        "drift_diagnostic_ran",
        f"{report.features_compared_count} features compared, "
        f"{report.features_exceeding_count} crossed an engineering diagnostic "
        f"default, {report.features_unavailable_count} unavailable",
    )


def _check_dashboard_catalogue(context: _Context) -> SmokeCheckResult:
    from engagevr.dashboard.catalogue import build_catalogue

    catalogue = build_catalogue(context.run_directory.parent, validate_checksums=True)
    discovered = [
        run
        for run in catalogue.runs
        if run.directory_name == context.run_directory.name
    ]
    if not discovered:
        return _failed(
            "dashboard_catalogue_discovered_run",
            f"the catalogue did not discover {context.run_directory.name}",
        )
    run = discovered[0]
    if run.provenance.scientific_evaluation_eligible:
        return _failed(
            "dashboard_catalogue_discovered_run",
            "the catalogue reports a synthetic run as scientifically eligible",
        )
    return _passed(
        "dashboard_catalogue_discovered_run",
        f"family={run.provenance.family.value} "
        f"status={run.provenance.status.value} "
        f"integrity={run.provenance.integrity.value} "
        "eligible=false",
    )


def _check_backend_application(context: _Context) -> SmokeCheckResult:
    """Construct the Milestone 4 backend. No socket is bound."""
    from engagevr.api.app import WEBSOCKET_PATH, create_app

    sessions = context.directory / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    app = create_app(context.config, session_root=sessions)
    paths = _route_paths(app.routes)
    for required in ("/health", "/version", WEBSOCKET_PATH):
        if required not in paths:
            return _failed(
                "backend_application_created",
                f"the backend exposes no {required} route; it exposes {sorted(paths)}",
            )
    return _passed(
        "backend_application_created",
        f"FastAPI application built with {len(paths)} routes including "
        "/health, /version, and the session WebSocket; no socket was bound "
        "and no server was started",
    )


def _route_paths(routes: object, *, depth: int = 0) -> set[str]:
    """Every route path, including those inside an included router.

    Starlette wraps an included ``APIRouter`` in a container rather than
    flattening its routes onto the application, so a check that reads
    ``app.routes`` alone sees one empty path where the whole REST surface
    should be.
    """
    found: set[str] = set()
    if depth > 4 or not isinstance(routes, list | tuple):
        return found
    for route in routes:
        path = getattr(route, "path", None)
        if isinstance(path, str) and path:
            found.add(path)
        for attribute in ("routes", "original_router"):
            nested = getattr(route, attribute, None)
            if nested is None:
                continue
            found |= _route_paths(getattr(nested, "routes", nested), depth=depth + 1)
    return found


def _check_dashboard_module(_: _Context) -> SmokeCheckResult:
    """Import the dashboard and build its argv. Streamlit is not started."""
    from engagevr.dashboard.launch import app_path, build_command

    command = build_command(address="127.0.0.1", port=8501, headless=True)
    if not app_path().is_file():
        return _failed(
            "dashboard_module_imports", "the Streamlit application script is missing"
        )
    if "--server.headless" not in command or "true" not in command:
        return _failed(
            "dashboard_module_imports",
            "the dashboard command does not run headless",
        )
    return _passed(
        "dashboard_module_imports",
        "launch argv built, headless, no browser, no server started",
    )


def _check_protocol_artifacts(_: _Context) -> SmokeCheckResult:
    """The checked-in protocol artefacts still match the code."""
    from engagevr.mlops.fingerprints import REPOSITORY_ROOT, sha256_payload
    from engagevr.protocol.json_schema import build_protocol_json_schema
    from engagevr.protocol.version import PROTOCOL_VERSION

    schema_path = REPOSITORY_ROOT / "protocol" / "engagevr-protocol-v1.schema.json"
    if not schema_path.is_file():
        return _skipped(
            "protocol_artifacts_current",
            f"{schema_path} is absent from this source tree",
        )
    on_disk = json.loads(schema_path.read_text(encoding="utf-8"))
    generated = build_protocol_json_schema()
    if sha256_payload(on_disk) != sha256_payload(generated):
        return _failed(
            "protocol_artifacts_current",
            "the checked-in protocol schema differs from the one this build "
            "generates. Run scripts/generate_protocol_artifacts.py.",
        )
    return _passed(
        "protocol_artifacts_current",
        f"protocol {PROTOCOL_VERSION} JSON Schema matches the code",
    )


#: Checks in execution order. Later ones depend on earlier ones' output.
CHECKS: tuple[tuple[str, Callable[[_Context], SmokeCheckResult]], ...] = (
    ("package_imports", _check_package_imports),
    ("configuration_loads", _check_configuration_loads),
    ("synthetic_dataset_generated", _check_synthetic_dataset),
    ("dataset_provenance_preserved", _check_dataset_metadata),
    ("baseline_pipeline_ran", _check_baseline_pipeline),
    ("artifact_manifest_validated", _check_artifact_integrity),
    ("model_version_manifest_validated", _check_model_version),
    ("mlflow_tracking_local", _check_mlflow_tracking),
    ("drift_diagnostic_ran", _check_drift_diagnostic),
    ("dashboard_catalogue_discovered_run", _check_dashboard_catalogue),
    ("backend_application_created", _check_backend_application),
    ("dashboard_module_imports", _check_dashboard_module),
    ("protocol_artifacts_current", _check_protocol_artifacts),
)


def run_system_smoke(
    directory: Path,
    *,
    config: EngageVRConfig,
    skip_mlflow: bool = False,
) -> SmokeReport:
    """Run every check against a caller-supplied scratch directory.

    A check that raises is recorded as ``failed`` with the exception text
    and the remaining checks still run: one broken component should
    produce one red line, not a report that stops at the first problem
    and says nothing about the rest.
    """
    from engagevr.training.artifacts import engagevr_version, runtime_environment

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    context = _Context(
        directory=root,
        config=config,
        reference_dataset=root / "datasets" / "smoke-reference.parquet",
        current_dataset=root / "datasets" / "smoke-current.parquet",
        run_directory=root / "experiments" / "smoke-baseline",
    )

    results: list[SmokeCheckResult] = []
    for name, check in CHECKS:
        if skip_mlflow and name == "mlflow_tracking_local":
            results.append(
                _skipped(
                    name,
                    "tracking was skipped by request (--no-mlflow). Skipping is "
                    "not passing.",
                )
            )
            continue
        try:
            results.append(check(context))
        except Exception as exc:
            results.append(_failed(name, f"{type(exc).__name__}: {exc}"))

    failed = sum(1 for r in results if r.status is SmokeCheckStatus.FAILED)
    environment = runtime_environment()
    return SmokeReport(
        engagevr_version=engagevr_version(),
        python_version=environment["python_version"],
        checks=tuple(results),
        passed_count=sum(1 for r in results if r.status is SmokeCheckStatus.PASSED),
        failed_count=failed,
        skipped_count=sum(1 for r in results if r.status is SmokeCheckStatus.SKIPPED),
        status=SmokeCheckStatus.FAILED if failed else SmokeCheckStatus.PASSED,
        created_at_utc=datetime.now(UTC),
        disclaimers=(SELF_CHECK_DISCLAIMER, MLOPS_DISCLAIMER),
    )


__all__ = [
    "CHECKS",
    "SMOKE_DRIFT_SEED",
    "SMOKE_FOLDS",
    "SMOKE_MODELS",
    "SMOKE_PERMUTATION_REPEATS",
    "SMOKE_SEED",
    "SMOKE_SESSIONS_PER_SUBJECT",
    "SMOKE_SUBJECTS",
    "SMOKE_WINDOWS_PER_SESSION",
    "run_system_smoke",
]
