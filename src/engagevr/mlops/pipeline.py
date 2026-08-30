"""The deterministic software demo: layout, parameters, and stage commands.

One definition, two runners
---------------------------
The stages below are the single description of the reproducible demo.
``dvc.yaml`` runs them as a DAG; ``engagevr mlops-demo`` runs the same
commands in one process.  Neither one contains modelling logic: every
stage invokes an existing Milestone 5--9 subcommand through the public
CLI, so there is no second training pipeline, no second uncertainty
engine, and nothing to drift out of step with the runners.

Parameters live in ``params.yaml`` at the repository root, which is both
what this module reads and what ``dvc.yaml`` templates from, so the two
cannot disagree.

Where the output goes
---------------------
Everything lands under ``mlops.pipeline_root`` (``artifacts/pipeline``),
which is inside the gitignored ``artifacts/`` tree and deliberately
outside ``dashboard.artifact_root``.  Rebuilding the pipeline must not
silently reshuffle the run catalogue a reader is looking at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator

from engagevr.mlops.fingerprints import REPOSITORY_ROOT
from engagevr.schemas.targets import TargetName

#: Repository-root parameter file shared with ``dvc.yaml``.
PARAMS_FILE = REPOSITORY_ROOT / "params.yaml"

#: Stage names, in dependency order. ``dvc.yaml`` must define exactly these.
STAGE_NAMES: tuple[str, ...] = (
    "dataset-reference",
    "dataset-current",
    "baseline",
    "uncertainty",
    "model-version",
    "drift",
    "integrity",
    "reproducibility",
)


class PipelineParameters(BaseModel):
    """Everything that makes the demo the demo.

    Small on purpose.  These sizes make a full ``dvc repro`` finish in
    well under a minute, which is what keeps reproducibility something a
    reviewer actually checks rather than something they read about.
    """

    model_config = {"extra": "forbid"}

    reference_seed: int = 42
    current_seed: int = 1337
    subjects: int = Field(default=24, ge=2)
    sessions_per_subject: int = Field(default=2, ge=1)
    windows_per_session: int = Field(default=8, ge=1)
    window_seconds: float = Field(default=10.0, gt=0.0)
    step_seconds: float = Field(default=10.0, gt=0.0)
    target: str = TargetName.ENGAGEMENT_CLASS.value
    folds: int = Field(default=3, ge=2)

    @model_validator(mode="after")
    def _check(self) -> Self:
        valid = sorted(t.value for t in TargetName)
        if self.target not in valid:
            raise ValueError(f"pipeline.target must be one of {valid}")
        if self.step_seconds > self.window_seconds:
            raise ValueError(
                "pipeline.step_seconds must not exceed pipeline.window_seconds"
            )
        if self.current_seed == self.reference_seed:
            raise ValueError(
                "pipeline.current_seed must differ from pipeline.reference_seed: "
                "the drift stage compares two SYNTHETIC draws, and comparing a "
                "draw with itself would report a shift diagnostic that could "
                "never be anything but zero"
            )
        return self


def load_parameters(path: Path | None = None) -> PipelineParameters:
    """Read ``params.yaml``, falling back to the documented defaults."""
    location = Path(path) if path is not None else PARAMS_FILE
    if not location.is_file():
        return PipelineParameters()
    document = yaml.safe_load(location.read_text(encoding="utf-8")) or {}
    section = document.get("pipeline", {})
    return PipelineParameters.model_validate(section)


@dataclass(frozen=True)
class PipelineLayout:
    """Every path the demo reads or writes, derived from one root."""

    root: Path
    target: str

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def mlops(self) -> Path:
        return self.root / "mlops"

    @property
    def reference_dataset(self) -> Path:
        return self.datasets / "reference.parquet"

    @property
    def current_dataset(self) -> Path:
        return self.datasets / "current.parquet"

    @property
    def baseline_run(self) -> Path:
        return self.experiments / f"baseline-{self.target}"

    @property
    def uncertainty_run(self) -> Path:
        return self.experiments / f"uncertainty-{self.target}"

    @property
    def model_versions(self) -> Path:
        return self.mlops / "model_versions"

    @property
    def drift_report(self) -> Path:
        return self.mlops / "drift_report.json"

    @property
    def catalogue(self) -> Path:
        return self.mlops / "catalogue.json"

    @property
    def reproducibility(self) -> Path:
        return self.mlops / "reproducibility.json"

    @property
    def stages(self) -> Path:
        """Where deterministic stage records live.

        Separate from the run directories they describe: a record is a
        Milestone 10 artifact and is DVC-declared, whereas the run
        directory is a Milestone 5-8 artifact and is not.
        """
        return self.mlops / "stages"

    def stage_record(self, stage_name: str) -> Path:
        """The deterministic record for one stage."""
        return self.stages / f"{stage_name}.json"

    @property
    def tracking_summaries(self) -> Path:
        return self.mlops / "mlflow_runs.json"


@dataclass(frozen=True)
class StageStep:
    """One command inside a stage."""

    argv: tuple[str, ...]
    redirect_stdout_to: Path | None = None

    @property
    def command(self) -> str:
        rendered = "uv run python -m engagevr " + " ".join(self.argv)
        if self.redirect_stdout_to is not None:
            rendered = f"{rendered} > {self.redirect_stdout_to.as_posix()}"
        return rendered


@dataclass(frozen=True)
class StageSpec:
    """One stage: what it runs, what it needs, and what it declares.

    ``outputs`` are the **DVC-declared** artifacts and are byte-stable by
    construction.  ``produces`` is everything the stage writes, including
    the timestamped provenance the Milestone 5-8 runners emit; those files
    stay on disk, intact, and are never declared, so their creation times
    never reach ``dvc.lock``.  See :mod:`engagevr.mlops.stage_record`.
    """

    name: str
    kind: str
    steps: tuple[StageStep, ...]
    dependencies: tuple[Path, ...]
    outputs: tuple[Path, ...]
    produces: tuple[Path, ...] = ()
    record: Path | None = None

    @property
    def command_lines(self) -> tuple[str, ...]:
        """One shell command per step, in order."""
        return tuple(step.command for step in self.steps)

    @property
    def command(self) -> str:
        """Every step, newline-joined, for display and for the record."""
        return "\n".join(self.command_lines)

    @property
    def recorded_targets(self) -> tuple[Path, ...]:
        """What the deterministic stage record classifies and checksums."""
        return self.produces or self.outputs


def _dataset_argv(
    parameters: PipelineParameters, *, seed: int, output: Path
) -> tuple[str, ...]:
    return (
        "features-demo",
        "--seed",
        str(seed),
        "--subjects",
        str(parameters.subjects),
        "--sessions-per-subject",
        str(parameters.sessions_per_subject),
        "--windows-per-session",
        str(parameters.windows_per_session),
        "--window-seconds",
        str(parameters.window_seconds),
        "--step-seconds",
        str(parameters.step_seconds),
        "--output",
        output.as_posix(),
    )


def _dataset_products(dataset: Path) -> tuple[Path, ...]:
    """Everything a dataset stage writes, timestamped metadata included."""
    return (
        dataset,
        dataset.with_name(f"{dataset.stem}.metadata.json"),
        dataset.with_name(f"{dataset.stem}.feature_catalog.json"),
    )


def _record_argv(layout: PipelineLayout, stage: str) -> tuple[str, ...]:
    return (
        "stage-record",
        "--stage",
        stage,
        "--pipeline-root",
        layout.root.as_posix(),
        "--target",
        layout.target,
    )


def build_stages(
    layout: PipelineLayout, parameters: PipelineParameters
) -> tuple[StageSpec, ...]:
    """The demo, as an ordered list of stages.

    Every stage's DVC-declared outputs are byte-stable.  Where a stage
    invokes a Milestone 5-8 runner that writes timestamped provenance, the
    runner's directory is *produced* but not *declared*, and a
    deterministic stage record is declared in its place.
    """
    return (
        StageSpec(
            name="dataset-reference",
            kind="dataset",
            steps=(
                StageStep(
                    _dataset_argv(
                        parameters,
                        seed=parameters.reference_seed,
                        output=layout.reference_dataset,
                    )
                ),
                StageStep(_record_argv(layout, "dataset-reference")),
            ),
            dependencies=(),
            outputs=(
                layout.reference_dataset,
                layout.reference_dataset.with_name(
                    f"{layout.reference_dataset.stem}.feature_catalog.json"
                ),
                layout.stage_record("dataset-reference"),
            ),
            produces=_dataset_products(layout.reference_dataset),
            record=layout.stage_record("dataset-reference"),
        ),
        StageSpec(
            name="dataset-current",
            kind="dataset",
            steps=(
                StageStep(
                    _dataset_argv(
                        parameters,
                        seed=parameters.current_seed,
                        output=layout.current_dataset,
                    )
                ),
                StageStep(_record_argv(layout, "dataset-current")),
            ),
            dependencies=(),
            outputs=(
                layout.current_dataset,
                layout.current_dataset.with_name(
                    f"{layout.current_dataset.stem}.feature_catalog.json"
                ),
                layout.stage_record("dataset-current"),
            ),
            produces=_dataset_products(layout.current_dataset),
            record=layout.stage_record("dataset-current"),
        ),
        StageSpec(
            name="baseline",
            kind="experiment_run",
            steps=(
                StageStep(
                    (
                        "baseline-demo",
                        "--dataset",
                        layout.reference_dataset.as_posix(),
                        "--target",
                        parameters.target,
                        "--folds",
                        str(parameters.folds),
                        "--seed",
                        str(parameters.reference_seed),
                        "--output",
                        layout.baseline_run.as_posix(),
                    )
                ),
                StageStep(_record_argv(layout, "baseline")),
            ),
            dependencies=(layout.reference_dataset,),
            outputs=(layout.stage_record("baseline"),),
            produces=(layout.baseline_run,),
            record=layout.stage_record("baseline"),
        ),
        StageSpec(
            name="uncertainty",
            kind="experiment_run",
            steps=(
                StageStep(
                    (
                        "uncertainty-demo",
                        "--dataset",
                        layout.reference_dataset.as_posix(),
                        "--target",
                        parameters.target,
                        "--folds",
                        str(parameters.folds),
                        "--seed",
                        str(parameters.reference_seed),
                        "--output",
                        layout.uncertainty_run.as_posix(),
                    )
                ),
                StageStep(_record_argv(layout, "uncertainty")),
            ),
            dependencies=(layout.reference_dataset,),
            outputs=(layout.stage_record("uncertainty"),),
            produces=(layout.uncertainty_run,),
            record=layout.stage_record("uncertainty"),
        ),
        StageSpec(
            name="model-version",
            kind="report",
            steps=(
                StageStep(
                    (
                        "model-manifest",
                        "--run",
                        layout.baseline_run.as_posix(),
                        "--output",
                        layout.model_versions.as_posix(),
                        "--verify",
                    )
                ),
            ),
            dependencies=(layout.stage_record("baseline"),),
            outputs=(layout.model_versions,),
        ),
        StageSpec(
            name="drift",
            kind="diagnostic",
            steps=(
                StageStep(
                    (
                        "drift-check",
                        "--reference",
                        layout.reference_dataset.as_posix(),
                        "--current",
                        layout.current_dataset.as_posix(),
                        "--output",
                        layout.drift_report.as_posix(),
                    )
                ),
            ),
            dependencies=(layout.reference_dataset, layout.current_dataset),
            outputs=(layout.drift_report,),
        ),
        StageSpec(
            name="integrity",
            kind="report",
            steps=(
                StageStep(
                    (
                        "dashboard-check",
                        "--artifact-root",
                        layout.experiments.as_posix(),
                        "--json",
                    ),
                    redirect_stdout_to=layout.catalogue,
                ),
            ),
            dependencies=(
                layout.stage_record("baseline"),
                layout.stage_record("uncertainty"),
            ),
            outputs=(layout.catalogue,),
        ),
        StageSpec(
            name="reproducibility",
            kind="report",
            steps=(
                StageStep(
                    (
                        "repro-manifest",
                        "--pipeline-root",
                        layout.root.as_posix(),
                        "--target",
                        parameters.target,
                        "--output",
                        layout.reproducibility.as_posix(),
                    )
                ),
            ),
            dependencies=(
                layout.stage_record("dataset-reference"),
                layout.stage_record("dataset-current"),
                layout.stage_record("baseline"),
                layout.stage_record("uncertainty"),
                layout.model_versions,
                layout.drift_report,
                layout.catalogue,
            ),
            outputs=(layout.reproducibility,),
        ),
    )


def stage_commands(
    layout: PipelineLayout, parameters: PipelineParameters
) -> dict[str, str]:
    """Stage name to the shell command that runs it."""
    return {stage.name: stage.command for stage in build_stages(layout, parameters)}


def default_layout(pipeline_root: Path | str, target: str) -> PipelineLayout:
    """Layout rooted at ``pipeline_root`` for one target."""
    return PipelineLayout(root=Path(pipeline_root), target=target)


def run_step(step: StageStep) -> int:
    """Run one step in this process, through the public CLI.

    ``engagevr.__main__.main`` is the documented entry point, so a step
    executed here takes exactly the path a user's shell command would.
    """
    import contextlib

    from engagevr.__main__ import main

    if step.redirect_stdout_to is None:
        return main(list(step.argv))
    step.redirect_stdout_to.parent.mkdir(parents=True, exist_ok=True)
    with step.redirect_stdout_to.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle):
            return main(list(step.argv))


def run_stage(stage: StageSpec) -> int:
    """Run every step of a stage in order, stopping at the first failure."""
    for step in stage.steps:
        status = run_step(step)
        if status != 0:
            return status
    return 0


def run_stages(stages: Sequence[StageSpec]) -> tuple[str, int]:
    """Run stages in order, stopping at the first non-zero exit.

    Returns the last stage attempted and its status, so a caller can say
    *which* stage failed rather than only that something did.
    """
    last = ("", 0)
    for stage in stages:
        status = run_stage(stage)
        last = (stage.name, status)
        if status != 0:
            return last
    return last


__all__ = [
    "PARAMS_FILE",
    "STAGE_NAMES",
    "PipelineLayout",
    "PipelineParameters",
    "StageSpec",
    "StageStep",
    "build_stages",
    "default_layout",
    "load_parameters",
    "run_stage",
    "run_stages",
    "run_step",
    "stage_commands",
]
