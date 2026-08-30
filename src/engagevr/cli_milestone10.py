"""Milestone 10 CLI: the operational commands.

::

    mlops-demo      run the whole deterministic SYNTHETIC pipeline
    model-manifest  derive immutable, checksum-linked model versions
    drift-check     distribution-shift diagnostic between two datasets
    mlflow-log      log a finished run to a LOCAL MLflow store
    repro-manifest  build the reproducibility manifest for a pipeline
    system-smoke    the integrated software self-check

One orchestrator plus five focused subcommands, because the DVC pipeline
needs each step addressable on its own and ``mlops-demo`` needs them in
one process.  Every command has ``--help``, an explicit ``--output``,
prints its synthetic and scientific status, and exits non-zero on a
validation failure.

None of these commands trains a model, fuses a modality, estimates
uncertainty, chooses an adaptation, or renders a dashboard.  They call
the Milestone 5--9 commands that do, and record what happened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engagevr.config import EngageVRConfig
from engagevr.mlops.pipeline import STAGE_NAMES
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER
from engagevr.schemas.mlops import NO_INFLATION_NOTE

#: Printed by every command in this module.
_MLOPS_BANNER = (
    "=== MILESTONE 10: OPERATIONS, NOT EVIDENCE ===\n"
    "Reproducibility is not validity. Tracking is not validation.\n"
    "Registration is not approval. Packaging is not production readiness.\n"
    "Drift alerts are engineering diagnostics. Nothing in this repository\n"
    "has been evaluated against a participant-provided engagement or\n"
    "cognitive-load label."
)


def add_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the Milestone 10 subcommands."""
    demo = sub.add_parser(
        "mlops-demo",
        help="Run the deterministic SYNTHETIC MLOps pipeline end to end.",
        description=(
            "Generate two SYNTHETIC datasets, run the Milestone 5 baseline "
            "and the Milestone 7 uncertainty pipelines on one of them, "
            "derive model versions, compute a distribution-shift "
            "diagnostic, discover the runs with the Milestone 9 catalogue, "
            "and write a reproducibility manifest. Starts no server."
        ),
    )
    demo.add_argument(
        "--pipeline-root",
        type=str,
        default=None,
        help="Output root. Defaults to mlops.pipeline_root in configs/defaults.yaml.",
    )
    demo.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target to model. Defaults to pipeline.target in params.yaml.",
    )
    demo.add_argument(
        "--mlflow",
        action="store_true",
        help=(
            "Also log every produced run to the LOCAL MLflow store. Opt-in: "
            "no other command in this repository writes to a tracking store."
        ),
    )
    demo.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Override the local tracking URI. Remote schemes are refused.",
    )

    manifest = sub.add_parser(
        "model-manifest",
        help="Derive immutable, checksum-linked model versions from a run.",
        description=(
            "Read a completed run directory and write one version record "
            "per persisted estimator. Reads only; the run is left "
            "byte-identical. No model file is ever loaded."
        ),
    )
    manifest.add_argument("--run", type=str, required=True, help="Run directory.")
    manifest.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory the version records are written to.",
    )
    manifest.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names. Default: every persisted estimator.",
    )
    manifest.add_argument(
        "--verify",
        action="store_true",
        help="Re-hash each referenced artifact and fail on any mismatch.",
    )

    drift = sub.add_parser(
        "drift-check",
        help="Distribution-shift diagnostic between two named datasets.",
        description=(
            "Compare a REFERENCE dataset with a CURRENT dataset, feature by "
            "feature. This is a diagnostic, not a diagnosis: a shift is not "
            "model degradation, not concept drift, and not a change in any "
            "person's state. Target columns take no part."
        ),
    )
    drift.add_argument(
        "--reference", type=str, required=True, help="Reference dataset (Parquet)."
    )
    drift.add_argument(
        "--current", type=str, required=True, help="Current dataset (Parquet)."
    )
    drift.add_argument("--output", type=str, required=True, help="Report path (JSON).")
    drift.add_argument(
        "--predictions",
        action="store_true",
        help=(
            "Compare the predicted_value column of two prediction tables "
            "instead. Reported as prediction-distribution shift, never as "
            "concept drift."
        ),
    )
    drift.add_argument(
        "--fail-on-shift",
        action="store_true",
        help=(
            "Exit non-zero when any statistic crosses its engineering "
            "diagnostic default. Off by default: a threshold crossing is an "
            "invitation to look, not a verdict."
        ),
    )

    track = sub.add_parser(
        "mlflow-log",
        help="Log a finished run directory to a LOCAL MLflow store.",
        description=(
            "Read a completed run's own documents and record them as an "
            "MLflow run: parameters, the metrics the run already computed, "
            "provenance tags, and the JSON artifacts. Writes nothing into "
            "the run directory and registers no model."
        ),
    )
    track.add_argument(
        "--run",
        type=str,
        required=True,
        action="append",
        help="Run directory (repeatable).",
    )
    track.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Local tracking URI. Defaults to mlops.mlflow.tracking_uri.",
    )
    track.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name. Defaults to mlops.mlflow.experiment_name.",
    )
    track.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for the JSON tracking summaries.",
    )

    repro = sub.add_parser(
        "repro-manifest",
        help="Build the reproducibility manifest for an executed pipeline.",
        description=(
            "Record the package versions, the effective configuration "
            "fingerprint, every stage's logical identity, and the checksums "
            "of the outputs declared byte-deterministic. Wall clocks, "
            "absolute paths, and MLflow identifiers are excluded from "
            "identity by construction."
        ),
    )
    repro.add_argument(
        "--pipeline-root",
        type=str,
        default=None,
        help="Pipeline root. Defaults to mlops.pipeline_root.",
    )
    repro.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target the pipeline modelled. Defaults to params.yaml.",
    )
    repro.add_argument(
        "--output", type=str, required=True, help="Manifest path (JSON)."
    )
    repro.add_argument(
        "--stages",
        type=str,
        default=None,
        help=(
            "Comma-separated stage names to describe. Default: every stage. "
            "Useful when only part of the pipeline has been executed; a "
            "stage whose deterministic record is absent is refused rather "
            "than reported as a row of absences."
        ),
    )
    repro.add_argument(
        "--compare",
        type=str,
        default=None,
        help=(
            "An earlier manifest to compare against. Exits non-zero when the "
            "two differ in LOGICAL identity."
        ),
    )

    record = sub.add_parser(
        "stage-record",
        help="Write the deterministic record for one executed pipeline stage.",
        description=(
            "Read what a stage produced and write its byte-stable, "
            "DVC-declared representation: the stage's logical identity plus "
            "a checksum for every artifact whose bytes are a function of the "
            "pipeline's inputs. The timestamped provenance the Milestone 5-8 "
            "runners write is listed by path and reason, never checksummed, "
            "so a creation time can never reach dvc.lock."
        ),
    )
    record.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=list(STAGE_NAMES),
        help="Which pipeline stage to record.",
    )
    record.add_argument(
        "--pipeline-root",
        type=str,
        default=None,
        help="Pipeline root. Defaults to mlops.pipeline_root.",
    )
    record.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target the pipeline modelled. Defaults to params.yaml.",
    )
    record.add_argument(
        "--output",
        type=str,
        default=None,
        help=("Record path. Defaults to <pipeline-root>/mlops/stages/<stage>.json."),
    )

    smoke = sub.add_parser(
        "system-smoke",
        help="Run the integrated software self-check and report.",
        description=(
            "Exercise the whole software stack without a webcam, a network, "
            "Unity, a browser, an external dataset, or a running server. "
            "A pass means the components interoperate. It does not mean a "
            "model is accurate, calibrated, useful, or validated."
        ),
    )
    smoke.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Directory for the scratch outputs and the report. Defaults to "
            "mlops.smoke_root in configs/defaults.yaml."
        ),
    )
    smoke.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip the local tracking check. Skipping is not passing.",
    )
    smoke.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON instead of a table.",
    )


def _config() -> EngageVRConfig:
    """The effective configuration, loaded once per command."""
    from engagevr.config import load_config

    return load_config()


def _write_json(path: Path, document: dict[str, object]) -> None:
    from engagevr.training.artifacts import write_json_atomic

    write_json_atomic(path, document)


def _write_sidecar(output: Path, produced_by: str) -> Path:
    """Record when a deterministic document was produced, beside it.

    The document itself carries no wall clock: it is a DVC-declared
    output, and a creation timestamp inside one would rewrite ``dvc.lock``
    on every reproduction. The sidecar is never declared.
    """
    from engagevr.mlops.execution import write_execution_sidecar
    from engagevr.mlops.fingerprints import repository_relative

    return write_execution_sidecar(
        output, describes=repository_relative(output), produced_by=produced_by
    )


# ---------------------------------------------------------------------------
# stage-record
# ---------------------------------------------------------------------------


def _layout_and_parameters(args: argparse.Namespace, config: EngageVRConfig):  # type: ignore[no-untyped-def]
    """The pipeline layout and parameters a command was asked to act on."""
    from engagevr.mlops.pipeline import default_layout, load_parameters

    parameters = load_parameters()
    target = getattr(args, "target", None) or parameters.target
    parameters = parameters.model_copy(update={"target": target})
    root = Path(getattr(args, "pipeline_root", None) or config.mlops.pipeline_root)
    return default_layout(root, target), parameters


def run_stage_record(args: argparse.Namespace) -> int:
    """Write one stage's deterministic, DVC-declared representation."""
    from engagevr.mlops.execution import write_execution_sidecar
    from engagevr.mlops.fingerprints import repository_relative
    from engagevr.mlops.pipeline import build_stages
    from engagevr.mlops.stage_record import (
        StageRecordError,
        build_stage_record,
        dataset_identity,
        run_identity,
        write_stage_record,
    )

    config = _config()
    layout, parameters = _layout_and_parameters(args, config)
    stage = next(
        (s for s in build_stages(layout, parameters) if s.name == args.stage), None
    )
    if stage is None:  # pragma: no cover - argparse restricts the choices
        print(f"Error: unknown stage {args.stage!r}", file=sys.stderr)
        return 2
    if stage.record is None:
        print(
            f"Error: stage {args.stage!r} produces its own deterministic "
            "document and needs no separate record.",
            file=sys.stderr,
        )
        return 2

    try:
        if stage.kind == "dataset":
            identity = dataset_identity(stage.recorded_targets[0])
        elif stage.kind == "experiment_run":
            identity = run_identity(stage.recorded_targets[0])
        else:  # pragma: no cover - only two kinds carry a record today
            identity = f"stage:{stage.name}"
        record = build_stage_record(
            stage_name=stage.name,
            stage_kind=stage.kind,
            command=stage.command,
            logical_identity=identity,
            targets=list(stage.recorded_targets),
            root=layout.root,
        )
    except StageRecordError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else stage.record
    write_stage_record(record, output)
    write_execution_sidecar(
        output,
        describes=repository_relative(output),
        produced_by=f"engagevr stage-record --stage {stage.name}",
    )

    print(f"Stage:                  {record.stage_name} ({record.stage_kind})")
    print(f"Logical identity:       {record.logical_identity}")
    print(f"Deterministic artifacts:{len(record.deterministic_artifacts):>4}")
    print(f"Volatile artifacts:     {len(record.volatile_artifacts):>4}")
    for path, reason in record.volatile_artifacts.items():
        print(f"  {path}")
        print(f"      not checksummed: {reason.split('.')[0]}.")
    print(f"Record:                 {output}")
    print(
        f"Scientific evaluation:  {str(record.scientific_evaluation_eligible).lower()}"
    )
    print(record.determinism_note)
    return 0


# ---------------------------------------------------------------------------
# mlops-demo
# ---------------------------------------------------------------------------


def run_mlops_demo(args: argparse.Namespace) -> int:
    """Run the whole deterministic SYNTHETIC pipeline in one process."""
    from engagevr.mlops.pipeline import (
        build_stages,
        default_layout,
        load_parameters,
        run_stage,
    )

    config = _config()
    parameters = load_parameters()
    target = args.target or parameters.target
    if target != parameters.target:
        parameters = parameters.model_copy(update={"target": target})
    root = Path(args.pipeline_root or config.mlops.pipeline_root)
    layout = default_layout(root, target)
    stages = build_stages(layout, parameters)

    print(_MLOPS_BANNER)
    print()
    print(SOFTWARE_SELF_CHECK_BANNER)
    print(f"Pipeline root:          {root}")
    print(f"Target:                 {target}")
    print(f"Reference seed:         {parameters.reference_seed}")
    print(f"Current seed:           {parameters.current_seed}")
    print(f"Folds:                  {parameters.folds}")
    print(f"Stages:                 {len(stages)}")
    print()

    for stage in stages:
        print(f"--- stage {stage.name} ---")
        print(f"    {stage.command}")
        status = run_stage(stage)
        if status != 0:
            print(
                f"Error: stage {stage.name!r} exited {status}.",
                file=sys.stderr,
            )
            return status
    print()

    summaries: list[dict[str, object]] = []
    if args.mlflow:
        from engagevr.mlops.mlflow_tracking import (
            TrackingError,
            log_run_directory,
            resolve_tracking_uri,
        )

        uri = args.tracking_uri or resolve_tracking_uri(config)
        for directory in (layout.baseline_run, layout.uncertainty_run):
            try:
                summary = log_run_directory(directory, config=config, tracking_uri=uri)
            except TrackingError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            summaries.append(summary.model_dump(mode="json"))
            print(
                f"Tracked {directory.name} as MLflow run "
                f"{summary.mlflow_run_id} in {summary.experiment_name}"
            )
        _write_json(layout.tracking_summaries, {"runs": summaries})
        print(f"Tracking summaries:     {layout.tracking_summaries}")
        print()

    from engagevr.mlops.reproducibility import read_manifest

    manifest = read_manifest(layout.reproducibility)
    print(f"Reference dataset:      {layout.reference_dataset}")
    print(f"Current dataset:        {layout.current_dataset}")
    print(f"Baseline run:           {layout.baseline_run}")
    print(f"Uncertainty run:        {layout.uncertainty_run}")
    print(f"Model versions:         {layout.model_versions}")
    print(f"Drift report:           {layout.drift_report}")
    print(f"Run catalogue:          {layout.catalogue}")
    print(f"Reproducibility:        {layout.reproducibility}")
    print(f"Config fingerprint:     {manifest.configuration.config_fingerprint}")
    print(f"Logical fingerprint:    {manifest.logical_fingerprint}")
    print("Data source:            SYNTHETIC")
    print(
        f"Scientific evaluation:  "
        f"{str(manifest.scientific_evaluation_eligible).lower()}"
    )
    print()
    print(SOFTWARE_SELF_CHECK_BANNER)
    print(NO_INFLATION_NOTE)
    return 0


# ---------------------------------------------------------------------------
# model-manifest
# ---------------------------------------------------------------------------


def run_model_manifest(args: argparse.Namespace) -> int:
    """Derive and write immutable model-version records."""
    from engagevr.mlops.model_version import (
        ModelVersionError,
        build_model_versions,
        summarise,
        verify_model_version,
        write_model_versions,
    )

    run_directory = Path(args.run)
    if not run_directory.is_dir():
        print(f"Error: no run directory {run_directory}", file=sys.stderr)
        return 2
    names = (
        tuple(n.strip() for n in args.models.split(",") if n.strip())
        if args.models
        else None
    )
    try:
        versions = build_model_versions(
            run_directory, config=_config(), model_names=names
        )
    except ModelVersionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.verify:
        for version in versions:
            mismatched = verify_model_version(version, run_directory=run_directory)
            if mismatched:
                print(
                    f"Error: {version.model_version_id} references artifacts "
                    f"whose bytes have changed: {list(mismatched)}",
                    file=sys.stderr,
                )
                return 1

    written = write_model_versions(versions, Path(args.output))
    print(_MLOPS_BANNER)
    print()
    print(f"Source run:             {run_directory}")
    print(f"Source run id:          {versions[0].source_run_id}")
    print(f"Run family:             {versions[0].source_run_family}")
    print(f"Dataset fingerprint:    {versions[0].dataset_fingerprint}")
    print(f"Split fingerprint:      {versions[0].split_fingerprint}")
    print(f"Feature schema:         {versions[0].feature_schema_fingerprint}")
    print(f"Config fingerprint:     {versions[0].configuration.config_fingerprint}")
    print(f"Versions written:       {len(written)} -> {Path(args.output)}")
    print()
    for version in versions:
        print(f"  {summarise(version)}")
    print()
    print(versions[0].limitation)
    print(
        "No version here is production, staging, champion, approved, or "
        "validated. Those words are refused by the schema."
    )
    return 0


# ---------------------------------------------------------------------------
# drift-check
# ---------------------------------------------------------------------------


def run_drift_check(args: argparse.Namespace) -> int:
    """Compute a distribution-shift diagnostic and write the report."""
    from engagevr.mlops.drift import (
        DriftError,
        compare_datasets,
        compare_predictions,
        exceeding_summary,
    )

    config = _config()
    try:
        if args.predictions:
            report = compare_predictions(
                Path(args.reference),
                Path(args.current),
                thresholds=config.mlops.drift,
            )
        else:
            report = compare_datasets(
                Path(args.reference),
                Path(args.current),
                thresholds=config.mlops.drift,
            )
    except DriftError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    output = Path(args.output)
    _write_json(output, report.model_dump(mode="json"))
    _write_sidecar(output, "engagevr drift-check")

    print(_MLOPS_BANNER)
    print()
    print(f"Report kind:            {report.report_kind.value}")
    print(f"Reference:              {report.reference.path}")
    print(f"  fingerprint:          {report.reference.dataset_fingerprint}")
    print(f"  rows:                 {report.reference.row_count}")
    print(f"Current:                {report.current.path}")
    print(f"  fingerprint:          {report.current.dataset_fingerprint}")
    print(f"  rows:                 {report.current.row_count}")
    print(f"Features compared:      {report.features_compared_count}")
    print(f"Features unavailable:   {report.features_unavailable_count}")
    print(f"Features over default:  {report.features_exceeding_count}")
    print(f"Minimum samples:        {report.minimum_samples}")
    print(f"Histogram bins:         {report.histogram_bin_count}")
    print(f"Report fingerprint:     {report.report_fingerprint}")
    print(
        f"Scientific evaluation:  {str(report.scientific_evaluation_eligible).lower()}"
    )
    print(f"Output:                 {Path(args.output)}")
    print()
    exceeded = exceeding_summary(report)
    if exceeded:
        print("Statistics over an engineering diagnostic default:")
        for line in exceeded[:40]:
            print(f"  {line}")
        if len(exceeded) > 40:
            print(f"  ... and {len(exceeded) - 40} more (see the report)")
        print()
    print(report.terminology_note)
    print(report.interpretation)
    if args.fail_on_shift and report.features_exceeding_count:
        print(
            "Exiting non-zero because --fail-on-shift was requested. This is "
            "a build gate, not a scientific finding.",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# mlflow-log
# ---------------------------------------------------------------------------


def run_mlflow_log(args: argparse.Namespace) -> int:
    """Log one or more finished runs to a local MLflow store."""
    from engagevr.mlops.mlflow_tracking import (
        TrackingError,
        log_run_directory,
        resolve_tracking_uri,
    )

    config = _config()
    uri = args.tracking_uri or resolve_tracking_uri(config)
    summaries = []
    for raw in args.run:
        directory = Path(raw)
        if not directory.is_dir():
            print(f"Error: no run directory {directory}", file=sys.stderr)
            return 2
        try:
            summaries.append(
                log_run_directory(
                    directory,
                    config=config,
                    tracking_uri=uri,
                    experiment_name=args.experiment,
                )
            )
        except TrackingError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.output is not None:
        _write_json(
            Path(args.output),
            {"runs": [s.model_dump(mode="json") for s in summaries]},
        )

    print(_MLOPS_BANNER)
    print()
    print(f"Tracking URI:           {uri}")
    print(f"Experiment:             {summaries[0].experiment_name}")
    print(f"MLflow version:         {summaries[0].mlflow_version}")
    print(f"Runs logged:            {len(summaries)}")
    print()
    for summary in summaries:
        print(f"  {summary.mlflow_run_name}")
        print(f"    mlflow run id:      {summary.mlflow_run_id}")
        print(f"    source run:         {summary.source_run_directory}")
        print(f"    parameters:         {len(summary.parameters)}")
        print(f"    metrics:            {len(summary.metrics)}")
        print(f"    artifacts:          {len(summary.logged_artifacts)}")
        print(f"    skipped metrics:    {len(summary.skipped_metrics)}")
        print(f"    is_synthetic:       {str(summary.is_synthetic).lower()}")
        print(
            f"    scientific_evaluation_eligible: "
            f"{str(summary.scientific_evaluation_eligible).lower()}"
        )
        print("    registered model:   none")
    if args.output is not None:
        print()
        print(f"Summaries:              {Path(args.output)}")
    print()
    print(
        "A run in a tracking store is bookkeeping. It is not validated, not "
        "approved, and not a release candidate."
    )
    print(NO_INFLATION_NOTE)
    return 0


# ---------------------------------------------------------------------------
# repro-manifest
# ---------------------------------------------------------------------------


def run_repro_manifest(args: argparse.Namespace) -> int:
    """Build (and optionally compare) the reproducibility manifest."""
    from engagevr.mlops.pipeline import (
        build_stages,
        default_layout,
        load_parameters,
    )
    from engagevr.mlops.reproducibility import (
        ReproducibilityError,
        build_manifest,
        compare,
        read_manifest,
    )

    config = _config()
    parameters = load_parameters()
    target = args.target or parameters.target
    root = Path(args.pipeline_root or config.mlops.pipeline_root)
    layout = default_layout(root, target)
    stages = build_stages(layout, parameters.model_copy(update={"target": target}))
    if args.stages:
        wanted = {name.strip() for name in args.stages.split(",") if name.strip()}
        unknown = wanted - {stage.name for stage in stages}
        if unknown:
            print(
                f"Error: unknown stage(s) {sorted(unknown)}; the pipeline "
                f"defines {[stage.name for stage in stages]}",
                file=sys.stderr,
            )
            return 2
        stages = tuple(stage for stage in stages if stage.name in wanted)

    try:
        manifest = build_manifest(stages, layout, config=config)
    except ReproducibilityError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output)
    _write_json(output, manifest.model_dump(mode="json"))
    _write_sidecar(output, "engagevr repro-manifest")

    print(_MLOPS_BANNER)
    print()
    print(f"Pipeline root:          {root}")
    print(f"EngageVR version:       {manifest.engagevr_version}")
    print(f"Python series:          {manifest.python_series}")
    print(f"Config fingerprint:     {manifest.configuration.config_fingerprint}")
    print(f"Logical fingerprint:    {manifest.logical_fingerprint}")
    print(f"Stages recorded:        {len(manifest.stages)}")
    print(
        f"Scientific evaluation:  "
        f"{str(manifest.scientific_evaluation_eligible).lower()}"
    )
    print(f"Output:                 {Path(args.output)}")
    print()
    for stage in manifest.stages:
        print(
            f"  {stage.name:<20} {stage.kind:<16} "
            f"deterministic={len(stage.deterministic_artifacts):<4} "
            f"volatile={len(stage.volatile_artifacts)}"
        )
        print(f"      identity: {stage.logical_identity}")
    print()
    print("Excluded from logical identity:")
    for item in manifest.excluded_from_identity:
        print(f"  - {item}")

    if args.compare is not None:
        earlier = read_manifest(Path(args.compare))
        differences = compare(earlier, manifest)
        print()
        if differences:
            print("Logical identity DIFFERS from the earlier manifest:")
            for difference in differences:
                print(f"  {difference}")
            return 1
        print(
            "Logical identity matches the earlier manifest. Timestamps and "
            "MLflow identifiers were not compared: they are excluded from "
            "identity by construction."
        )
    print()
    print(NO_INFLATION_NOTE)
    return 0


# ---------------------------------------------------------------------------
# system-smoke
# ---------------------------------------------------------------------------


def run_system_smoke_command(args: argparse.Namespace) -> int:
    """Run the integrated software self-check."""
    from engagevr.mlops.smoke import run_system_smoke
    from engagevr.schemas.mlops import SmokeCheckStatus

    config = _config()
    directory = Path(args.output or config.mlops.smoke_root)
    report = run_system_smoke(directory, config=config, skip_mlflow=args.no_mlflow)
    _write_json(directory / "smoke_report.json", report.model_dump(mode="json"))

    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
    else:
        print(_MLOPS_BANNER)
        print()
        print(report.banner)
        print()
        print(f"EngageVR version:       {report.engagevr_version}")
        print(f"Python version:         {report.python_version}")
        print(f"Output directory:       {directory}")
        print()
        for check in report.checks:
            marker = {
                SmokeCheckStatus.PASSED: "PASS",
                SmokeCheckStatus.FAILED: "FAIL",
                SmokeCheckStatus.SKIPPED: "SKIP",
            }[check.status]
            print(f"  [{marker}] {check.name}")
            reason = check.detail or check.failure_reason or check.skip_reason or ""
            if reason:
                print(f"         {reason}")
        print()
        print(
            f"Passed {report.passed_count}, failed {report.failed_count}, "
            f"skipped {report.skipped_count}. Overall: {report.status.value}."
        )
        print(f"Report:                 {directory / 'smoke_report.json'}")
        print()
        print(report.banner)
        print(
            f"scientific_evaluation_eligible="
            f"{str(report.scientific_evaluation_eligible).lower()}"
        )
        print(
            "A passing smoke check means the software components "
            "interoperate. It does not mean any model is accurate, "
            "calibrated, useful, or validated."
        )
    return 0 if report.status is SmokeCheckStatus.PASSED else 1


__all__ = [
    "add_parsers",
    "run_drift_check",
    "run_mlflow_log",
    "run_mlops_demo",
    "run_model_manifest",
    "run_repro_manifest",
    "run_stage_record",
    "run_system_smoke_command",
]
