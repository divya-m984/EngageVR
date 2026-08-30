"""Reproducibility manifests and the pipeline definition.

The manifest's job is to separate *logical* identity — what must match
across two correct executions — from the *volatile* record of everything
else.  These tests check that the separation holds: a wall clock never
enters identity, a non-deterministic output is recorded with a reason
rather than quietly folded in, and two manifests over the same pipeline
compare equal even though their timestamps differ.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from engagevr.config import load_config
from engagevr.mlops.pipeline import (
    STAGE_NAMES,
    PipelineParameters,
    build_stages,
    default_layout,
    load_parameters,
    stage_commands,
)
from engagevr.mlops.reproducibility import (
    ReproducibilityError,
    build_manifest,
    build_stage_entries,
    compare,
    logical_fingerprint,
    read_manifest,
)
from engagevr.mlops.stage_record import (
    VOLATILE_ARTIFACT_REASONS,
    build_stage_record,
    run_identity,
    write_stage_record,
)
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER


@pytest.fixture
def executed_pipeline(m10_baseline_run: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    """A layout whose baseline stage output and record exist on disk.

    The baseline run is copied rather than re-fitted: this module tests
    the manifest, not the modelling.  The stage record is then built the
    way the pipeline builds it, because the manifest reads records rather
    than walking the filesystem.
    """
    import shutil

    parameters = load_parameters()
    layout = default_layout(tmp_path / "pipeline", parameters.target)
    layout.experiments.mkdir(parents=True)
    shutil.copytree(m10_baseline_run, layout.baseline_run)
    stage = next(s for s in build_stages(layout, parameters) if s.name == "baseline")
    write_stage_record(
        build_stage_record(
            stage_name=stage.name,
            stage_kind=stage.kind,
            command=stage.command,
            logical_identity=run_identity(layout.baseline_run),
            targets=list(stage.recorded_targets),
            root=layout.root,
        ),
        stage.record,
    )
    return layout, parameters, (stage,)


class TestPipelineDefinition:
    def test_the_stage_order_is_the_declared_one(self) -> None:
        parameters = load_parameters()
        layout = default_layout("artifacts/pipeline", parameters.target)
        assert tuple(s.name for s in build_stages(layout, parameters)) == STAGE_NAMES

    def test_every_command_invokes_the_public_cli(self) -> None:
        parameters = load_parameters()
        layout = default_layout("artifacts/pipeline", parameters.target)
        for name, command in stage_commands(layout, parameters).items():
            assert "python -m engagevr" in command, name

    def test_a_runner_stage_declares_a_record_rather_than_its_run_directory(
        self,
    ) -> None:
        parameters = load_parameters()
        layout = default_layout("artifacts/pipeline", parameters.target)
        for stage in build_stages(layout, parameters):
            if stage.name not in {"baseline", "uncertainty"}:
                continue
            declared = {o.as_posix() for o in stage.outputs}
            assert declared == {stage.record.as_posix()}
            assert layout.baseline_run.as_posix() not in declared
            assert layout.uncertainty_run.as_posix() not in declared

    def test_a_dataset_stage_does_not_declare_its_timestamped_metadata(
        self,
    ) -> None:
        parameters = load_parameters()
        layout = default_layout("artifacts/pipeline", parameters.target)
        for stage in build_stages(layout, parameters):
            if not stage.name.startswith("dataset-"):
                continue
            declared = {o.as_posix() for o in stage.outputs}
            assert not any(p.endswith(".metadata.json") for p in declared)
            produced = {o.as_posix() for o in stage.produces}
            assert any(p.endswith(".metadata.json") for p in produced)

    def test_the_two_synthetic_draws_use_different_seeds(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            PipelineParameters(reference_seed=42, current_seed=42)

    def test_an_unknown_target_is_refused(self) -> None:
        with pytest.raises(ValueError, match=re.escape("pipeline.target")):
            PipelineParameters(target="mood")

    def test_the_layout_keeps_the_pipeline_out_of_the_dashboard_root(self) -> None:
        config = load_config()
        layout = default_layout(config.mlops.pipeline_root, "engagement_class")
        assert not str(layout.root).startswith(config.dashboard.artifact_root)

    def test_only_the_integrity_stage_redirects_stdout(self) -> None:
        parameters = load_parameters()
        layout = default_layout("artifacts/pipeline", parameters.target)
        redirecting = [
            stage.name
            for stage in build_stages(layout, parameters)
            if any(step.redirect_stdout_to for step in stage.steps)
        ]
        assert redirecting == ["integrity"]


class TestStageEntries:
    def test_a_run_stage_is_identified_by_its_run_id(self, executed_pipeline) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        entries = build_stage_entries(stages, layout)
        baseline = next(e for e in entries if e.name == "baseline")
        manifest = json.loads(
            (layout.baseline_run / "manifest.json").read_text(encoding="utf-8")
        )
        assert baseline.logical_identity == f"run_id:{manifest['run_id']}"

    def test_the_reproducibility_stage_does_not_record_itself(
        self, executed_pipeline
    ) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        entries = build_stage_entries(stages, layout)
        assert "reproducibility" not in {e.name for e in entries}

    def test_a_timestamped_document_is_listed_volatile_without_a_checksum(
        self, executed_pipeline
    ) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        entries = build_stage_entries(stages, layout)
        baseline = next(e for e in entries if e.name == "baseline")
        volatile = set(baseline.volatile_artifacts)
        assert any(path.endswith("/manifest.json") for path in volatile)
        assert any(path.endswith("/dataset.json") for path in volatile)
        assert any(path.endswith("/checksums.json") for path in volatile)
        # And none of them carries a digest anywhere in the entry.
        checksummed = {a.path for a in baseline.deterministic_artifacts}
        assert checksummed & volatile == set()

    def test_metrics_and_splits_are_recorded_as_deterministic(
        self, executed_pipeline
    ) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        entries = build_stage_entries(stages, layout)
        baseline = next(e for e in entries if e.name == "baseline")
        paths = {a.path for a in baseline.deterministic_artifacts}
        for name in ("metrics.json", "splits.json"):
            assert any(path.endswith(f"/{name}") for path in paths), name

    def test_model_binaries_are_checksummed(self, executed_pipeline) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        entries = build_stage_entries(stages, layout)
        baseline = next(e for e in entries if e.name == "baseline")
        joblibs = [
            a for a in baseline.deterministic_artifacts if a.path.endswith(".joblib")
        ]
        assert joblibs
        for artifact in joblibs:
            assert len(artifact.sha256) == 64

    def test_paths_are_recorded_relative_to_the_pipeline_root(
        self, executed_pipeline
    ) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        entries = build_stage_entries(stages, layout)
        for entry in entries:
            for artifact in entry.deterministic_artifacts:
                assert not artifact.path.startswith("/")
                assert "tmp" not in artifact.path.split("/")[0]
            for path in entry.volatile_artifacts:
                assert not path.startswith("/")

    def test_a_stage_whose_record_was_never_written_is_refused(
        self, executed_pipeline
    ) -> None:  # type: ignore[no-untyped-def]
        layout, parameters, _stages = executed_pipeline
        stage = next(
            s for s in build_stages(layout, parameters) if s.name == "uncertainty"
        )
        with pytest.raises(ReproducibilityError, match="no readable deterministic"):
            build_stage_entries([stage], layout)

    def test_every_volatile_name_has_a_stated_reason(self) -> None:
        for name, reason in VOLATILE_ARTIFACT_REASONS.items():
            assert reason, name


class TestLogicalFingerprint:
    def test_it_is_stable_across_two_builds(self, executed_pipeline) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        first = build_manifest(stages, layout, config=load_config())
        second = build_manifest(stages, layout, config=load_config())
        assert first.logical_fingerprint == second.logical_fingerprint
        assert compare(first, second) == ()

    def test_it_ignores_a_changed_timestamp_inside_a_run_manifest(
        self, executed_pipeline
    ) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        before = build_manifest(stages, layout, config=load_config())
        # Rewriting only the timestamps inside the run manifest is what a
        # second correct execution does.
        path = layout.baseline_run / "manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["finished_at_utc"] = "2030-01-01T00:00:00Z"
        path.write_text(json.dumps(document), encoding="utf-8")
        after = build_manifest(stages, layout, config=load_config())
        assert after.logical_fingerprint == before.logical_fingerprint
        assert compare(before, after) == ()

    def test_it_notices_a_changed_deterministic_output(self, executed_pipeline) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        before = build_manifest(stages, layout, config=load_config())
        # A meaningful change must still propagate. Rewrite metrics.json
        # and rebuild the stage record the way the pipeline's second step
        # does; the record's checksum for that file then changes, so the
        # fingerprint changes and every downstream stage would re-run.
        (layout.baseline_run / "metrics.json").write_text("{}", encoding="utf-8")
        stage = stages[0]
        write_stage_record(
            build_stage_record(
                stage_name=stage.name,
                stage_kind=stage.kind,
                command=stage.command,
                logical_identity=run_identity(layout.baseline_run),
                targets=list(stage.recorded_targets),
                root=layout.root,
            ),
            stage.record,
        )
        after = build_manifest(stages, layout, config=load_config())
        assert after.logical_fingerprint != before.logical_fingerprint
        differences = compare(before, after)
        assert any("metrics.json" in d for d in differences)

    def test_it_notices_a_changed_run_identity(self, executed_pipeline) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        before = build_manifest(stages, layout, config=load_config())
        path = layout.baseline_run / "manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["run_id"] = "a-different-run"
        path.write_text(json.dumps(document), encoding="utf-8")
        stage = next(s for s in stages if s.name == "baseline")
        write_stage_record(
            build_stage_record(
                stage_name=stage.name,
                stage_kind=stage.kind,
                command=stage.command,
                logical_identity=run_identity(layout.baseline_run),
                targets=list(stage.recorded_targets),
                root=layout.root,
            ),
            stage.record,
        )
        after = build_manifest(stages, layout, config=load_config())
        assert any("logical identity differs" in d for d in compare(before, after))

    def test_an_empty_stage_list_is_refused(self, executed_pipeline) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, _stages = executed_pipeline
        with pytest.raises(ReproducibilityError, match="Run the pipeline"):
            build_manifest([], layout, config=load_config())

    def test_the_fingerprint_covers_only_deterministic_artifacts(self) -> None:
        from engagevr.schemas.mlops import ReproducibilityStage

        stage = ReproducibilityStage(
            name="s",
            kind="report",
            command="c",
            logical_identity="i",
            volatile_artifacts={"a/manifest.json": "records a wall clock"},
        )
        renamed = stage.model_copy(
            update={"volatile_artifacts": {"a/other.json": "also volatile"}}
        )
        assert logical_fingerprint([stage]) == logical_fingerprint([renamed])


class TestManifestContent:
    def test_it_records_the_environment_and_the_configuration(
        self, executed_pipeline
    ) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        manifest = build_manifest(stages, layout, config=load_config())
        assert manifest.engagevr_version
        assert manifest.python_series == "3.12"
        assert manifest.dependency_versions
        assert len(manifest.configuration.config_fingerprint) == 64

    def test_it_is_synthetic_and_never_eligible(self, executed_pipeline) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        manifest = build_manifest(stages, layout, config=load_config())
        assert manifest.is_synthetic is True
        assert manifest.scientific_evaluation_eligible is False
        assert any(SOFTWARE_SELF_CHECK_BANNER in d for d in manifest.disclaimers)

    def test_it_states_that_reproducibility_is_not_validity(
        self, executed_pipeline
    ) -> None:  # type: ignore[no-untyped-def]
        layout, _parameters, stages = executed_pipeline
        manifest = build_manifest(stages, layout, config=load_config())
        assert "Reproducibility is not validity" in manifest.note

    def test_it_round_trips_through_disk(
        self, executed_pipeline, tmp_path: Path
    ) -> None:  # type: ignore[no-untyped-def]
        from engagevr.training.artifacts import write_json_atomic

        layout, _parameters, stages = executed_pipeline
        manifest = build_manifest(stages, layout, config=load_config())
        path = tmp_path / "reproducibility.json"
        write_json_atomic(path, manifest.model_dump(mode="json"))
        assert read_manifest(path).logical_fingerprint == manifest.logical_fingerprint
