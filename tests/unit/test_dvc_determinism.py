"""The DVC-declared outputs are byte-stable, and so is ``dvc.lock``.

The invariant under test
------------------------
::

    clean source tree -> dvc repro -> dvc.lock byte-identical

given the same source, ``uv.lock``, configuration, synthetic seed, and
parameters.  ``dvc.lock`` is tracked, and a fresh reproduction must leave
it unchanged.

That holds because of a boundary, not because timestamps were deleted
from anywhere they belonged.  The Milestone 5--8 runners still write
``started_at_utc``, ``finished_at_utc``, and ``created_at_utc`` into their
own provenance documents; those documents are simply never DVC-declared.
A deterministic stage record is declared in their place, pinning the run
id and checksumming only the byte-stable files.

These tests exercise the property structurally and at the unit level.
The end-to-end proof — two independent fresh source trees producing the
same ``dvc.lock`` — is in ``tests/system/test_dvc_lock_stability.py``,
which is slower and lives beside the other system-level checks.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from engagevr.config import load_config
from engagevr.mlops.execution import (
    EXECUTION_SUFFIX,
    build_execution_metadata,
    sidecar_path,
    write_execution_sidecar,
)
from engagevr.mlops.model_version import build_model_versions
from engagevr.mlops.pipeline import build_stages, default_layout, load_parameters
from engagevr.mlops.stage_record import (
    VOLATILE_ARTIFACT_REASONS,
    VOLATILE_DATASET_SUFFIX,
    StageRecordError,
    build_stage_record,
    classify,
    is_volatile,
    normalize_command,
    read_stage_record,
    run_identity,
    write_stage_record,
)
from engagevr.schemas.mlops import (
    DeterministicStageRecord,
    ExecutionMetadata,
    assert_python_series,
    assert_relative_path,
    python_series,
)

ROOT = Path(__file__).resolve().parents[2]
DVC_YAML = ROOT / "dvc.yaml"

#: Field names that would put a wall clock into a deterministic document.
WALL_CLOCK_FIELD_TOKENS = ("created_at", "started_at", "finished_at", "timestamp")

#: An ISO-8601-ish date, which is what a leaked wall clock looks like.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def declared_outputs() -> list[str]:
    """Every path ``dvc.yaml`` declares as a stage output."""
    document = yaml.safe_load(DVC_YAML.read_text(encoding="utf-8"))
    paths: list[str] = []
    for stage in document["stages"].values():
        for out in stage["outs"]:
            paths.append(out if isinstance(out, str) else next(iter(out)))
    return paths


def walk_json(document: Any) -> list[tuple[str, Any]]:
    """Every ``(dotted key, value)`` pair in a JSON document."""
    pairs: list[tuple[str, Any]] = []

    def visit(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{prefix}[{index}]")
        else:
            pairs.append((prefix, node))

    visit(document, "")
    return pairs


@pytest.fixture(scope="module")
def recorded_baseline(
    tmp_path_factory: pytest.TempPathFactory, m10_baseline_run: Path
) -> tuple[Path, Any, Any]:
    """A pipeline layout with the baseline stage recorded, as DVC would."""
    import shutil

    parameters = load_parameters()
    layout = default_layout(
        tmp_path_factory.mktemp("determinism") / "pipeline", parameters.target
    )
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
    return layout, parameters, stage


# ---------------------------------------------------------------------------
# 1. Deterministic outputs contain no wall-clock-dependent values
# ---------------------------------------------------------------------------


class TestNoWallClockInDeterministicOutputs:
    def test_no_declared_output_is_a_timestamped_runner_document(self) -> None:
        for path in declared_outputs():
            assert not is_volatile(path), (
                f"dvc.yaml declares {path!r}, which records when it was "
                "written. Its bytes would change on every reproduction."
            )

    def test_no_run_directory_is_declared(self) -> None:
        # A run directory contains manifest.json, dataset.json, and
        # checksums.json, all of which carry a clock.
        for path in declared_outputs():
            assert "/experiments/" not in path, (
                f"dvc.yaml declares {path!r}, a Milestone 5-8 run directory. "
                "Declare its deterministic stage record instead."
            )

    def test_the_stage_record_model_has_no_wall_clock_field(self) -> None:
        for name in DeterministicStageRecord.model_fields:
            assert not any(token in name for token in WALL_CLOCK_FIELD_TOKENS), name

    def test_a_written_stage_record_contains_no_timestamp(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        _layout, _parameters, stage = recorded_baseline
        document = json.loads(stage.record.read_text(encoding="utf-8"))
        for key, value in walk_json(document):
            assert not any(token in key.lower() for token in WALL_CLOCK_FIELD_TOKENS), (
                key
            )
            if isinstance(value, str):
                assert not _ISO_DATE.search(value), (key, value)

    def test_a_model_version_record_contains_no_timestamp(
        self, m10_baseline_run: Path
    ) -> None:
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        for key, value in walk_json(version.model_dump(mode="json")):
            assert not any(token in key.lower() for token in WALL_CLOCK_FIELD_TOKENS), (
                key
            )
            if isinstance(value, str):
                assert not _ISO_DATE.search(value), (key, value)

    def test_the_current_year_appears_in_no_deterministic_record(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        _layout, _parameters, stage = recorded_baseline
        assert str(datetime.now(UTC).year) not in stage.record.read_text(
            encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# 2. Deterministic outputs contain no absolute repository or temp paths
# ---------------------------------------------------------------------------


class TestNoAbsolutePathsInDeterministicOutputs:
    def test_a_stage_record_stores_only_pipeline_relative_paths(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        _layout, _parameters, stage = recorded_baseline
        record = read_stage_record(stage.record)
        for artifact in record.deterministic_artifacts:
            assert not artifact.path.startswith("/")
            assert not artifact.path.startswith("~")
        for path in record.volatile_artifacts:
            assert not path.startswith("/")

    def test_the_temporary_root_never_appears_in_a_record(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        # The fixture runs under a tmp_path root, which is exactly the
        # case that would leak a machine-specific location into the
        # recorded command if it were stored verbatim.
        layout, _parameters, stage = recorded_baseline
        text = stage.record.read_text(encoding="utf-8")
        assert str(layout.root.resolve()) not in text
        assert "/tmp/" not in text

    def test_the_recorded_command_is_root_relative(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        layout, _parameters, stage = recorded_baseline
        record = read_stage_record(stage.record)
        assert "datasets/reference.parquet" in record.command
        assert str(layout.root) not in record.command

    def test_two_roots_produce_the_same_recorded_command(self, tmp_path: Path) -> None:
        parameters = load_parameters()
        first = default_layout(tmp_path / "a" / "pipeline", parameters.target)
        second = default_layout("artifacts/pipeline", parameters.target)
        stage_a = next(
            s for s in build_stages(first, parameters) if s.name == "baseline"
        )
        stage_b = next(
            s for s in build_stages(second, parameters) if s.name == "baseline"
        )
        assert normalize_command(stage_a.command, first.root) == normalize_command(
            stage_b.command, second.root
        )

    def test_a_model_version_record_stores_relative_paths(
        self, m10_baseline_run: Path
    ) -> None:
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        assert not version.source_run_directory.startswith("/")
        assert not version.model_artifact_path.startswith("/")
        assert str(m10_baseline_run) not in version.model_dump_json()

    @pytest.mark.parametrize(
        "path",
        ["/etc/passwd", "~/EngageVR/run", "C:\\Users\\me\\run", "a/../../escape"],
    )
    def test_a_machine_specific_path_is_refused(self, path: str) -> None:
        with pytest.raises(ValueError):
            assert_relative_path(path, field="test")

    def test_a_pipeline_relative_path_is_accepted(self) -> None:
        assert assert_relative_path("experiments/baseline/metrics.json", field="t")


# ---------------------------------------------------------------------------
# 3. Same source/config/seed -> byte-identical DVC-declared outputs
# ---------------------------------------------------------------------------


class TestByteIdenticalOutputs:
    def test_a_stage_record_is_byte_identical_when_rebuilt(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        layout, _parameters, stage = recorded_baseline
        first = stage.record.read_bytes()
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
        assert stage.record.read_bytes() == first

    def test_rebuilding_after_a_timestamp_change_is_still_byte_identical(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        layout, _parameters, stage = recorded_baseline
        first = stage.record.read_bytes()
        # Exactly what a second correct execution does to a run manifest.
        manifest_path = layout.baseline_run / "manifest.json"
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        document["started_at_utc"] = "2031-01-01T00:00:00Z"
        document["finished_at_utc"] = "2031-01-01T00:00:01Z"
        manifest_path.write_text(json.dumps(document), encoding="utf-8")
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
        assert stage.record.read_bytes() == first

    def test_model_versions_are_byte_identical_when_rebuilt(
        self, m10_baseline_run: Path
    ) -> None:
        config = load_config()
        first = build_model_versions(m10_baseline_run, config=config)
        second = build_model_versions(m10_baseline_run, config=config)
        assert [v.model_dump(mode="json") for v in first] == [
            v.model_dump(mode="json") for v in second
        ]


# ---------------------------------------------------------------------------
# 5-8. Identities are unchanged across rebuilds
# ---------------------------------------------------------------------------


class TestIdentitiesUnchanged:
    def test_the_model_version_logical_id_is_unchanged(
        self, m10_baseline_run: Path
    ) -> None:
        config = load_config()
        first = build_model_versions(m10_baseline_run, config=config)
        second = build_model_versions(m10_baseline_run, config=config)
        assert [v.model_version_id for v in first] == [
            v.model_version_id for v in second
        ]

    def test_the_config_fingerprint_is_unchanged(self, m10_baseline_run: Path) -> None:
        config = load_config()
        first = build_model_versions(m10_baseline_run, config=config)[0]
        second = build_model_versions(m10_baseline_run, config=load_config())[0]
        assert (
            first.configuration.config_fingerprint
            == second.configuration.config_fingerprint
        )

    def test_the_dataset_fingerprint_is_unchanged(self, m10_baseline_run: Path) -> None:
        config = load_config()
        first = build_model_versions(m10_baseline_run, config=config)[0]
        second = build_model_versions(m10_baseline_run, config=config)[0]
        assert first.dataset_fingerprint == second.dataset_fingerprint
        assert len(first.dataset_fingerprint) == 64

    def test_the_split_and_feature_schema_fingerprints_are_unchanged(
        self, m10_baseline_run: Path
    ) -> None:
        config = load_config()
        first = build_model_versions(m10_baseline_run, config=config)[0]
        second = build_model_versions(m10_baseline_run, config=config)[0]
        assert first.split_fingerprint == second.split_fingerprint
        assert first.feature_schema_fingerprint == second.feature_schema_fingerprint

    def test_synthetic_and_scientific_provenance_is_unchanged(
        self, recorded_baseline: tuple[Path, Any, Any], m10_baseline_run: Path
    ) -> None:
        _layout, _parameters, stage = recorded_baseline
        record = read_stage_record(stage.record)
        assert record.is_synthetic is True
        assert record.scientific_evaluation_eligible is False
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        assert version.is_synthetic is True
        assert version.scientific_evaluation_eligible is False

    def test_the_self_check_banner_survives_the_determinism_rework(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER

        _layout, _parameters, stage = recorded_baseline
        record = read_stage_record(stage.record)
        assert any(SOFTWARE_SELF_CHECK_BANNER in d for d in record.disclaimers)


# ---------------------------------------------------------------------------
# 9. MLflow identifiers are not part of DVC deterministic identity
# ---------------------------------------------------------------------------


class TestMlflowIsOutsideDeterministicIdentity:
    def test_no_declared_output_is_a_tracking_store(self) -> None:
        for path in declared_outputs():
            assert "mlruns" not in path
            assert "mlflow" not in path

    def test_the_tracking_summary_is_not_a_declared_output(self) -> None:
        parameters = load_parameters()
        layout = default_layout("artifacts/pipeline", parameters.target)
        assert layout.tracking_summaries.as_posix() not in declared_outputs()

    def test_no_stage_command_logs_to_mlflow(self) -> None:
        document = yaml.safe_load(DVC_YAML.read_text(encoding="utf-8"))
        for name, stage in document["stages"].items():
            command = stage["cmd"]
            rendered = command if isinstance(command, str) else " ".join(command)
            assert "mlflow" not in rendered, name

    def test_a_stage_record_carries_no_mlflow_identifier(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        # The prose legitimately says "no MLflow run identifier"; what
        # must be absent is an actual identifier — a 32-character hex run
        # id, or a tracking URI.
        _layout, _parameters, stage = recorded_baseline
        text = stage.record.read_text(encoding="utf-8")
        assert "mlruns" not in text
        assert "tracking_uri" not in text
        assert not re.search(r'"[0-9a-f]{32}"', text)

    def test_tracking_a_run_does_not_change_a_stage_record(
        self, recorded_baseline: tuple[Path, Any, Any], tmp_path: Path
    ) -> None:
        from engagevr.mlops.mlflow_tracking import log_run_directory

        layout, _parameters, stage = recorded_baseline
        before = stage.record.read_bytes()
        store = tmp_path / "store"
        store.mkdir()
        log_run_directory(
            layout.baseline_run,
            config=load_config(),
            tracking_uri=store.resolve().as_uri(),
            experiment_name="engagevr-determinism-test",
        )
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
        assert stage.record.read_bytes() == before


# ---------------------------------------------------------------------------
# The volatile/deterministic split itself
# ---------------------------------------------------------------------------


class TestTheSplit:
    def test_the_runner_artifacts_are_still_written_intact(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        # The repair must not have rewritten Milestone 5-8 semantics: a run
        # still records when it happened.
        layout, _parameters, _stage = recorded_baseline
        manifest = json.loads(
            (layout.baseline_run / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["started_at_utc"]
        assert manifest["finished_at_utc"]

    def test_every_timestamped_document_is_classified_volatile(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        _layout, _parameters, stage = recorded_baseline
        record = read_stage_record(stage.record)
        for name in VOLATILE_ARTIFACT_REASONS:
            assert any(
                path.endswith(f"/{name}") for path in record.volatile_artifacts
            ), name

    def test_a_volatile_document_carries_no_checksum(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        _layout, _parameters, stage = recorded_baseline
        record = read_stage_record(stage.record)
        deterministic = {a.path for a in record.deterministic_artifacts}
        assert deterministic & set(record.volatile_artifacts) == set()

    def test_a_dataset_metadata_document_is_volatile(self) -> None:
        assert is_volatile(f"datasets/reference{VOLATILE_DATASET_SUFFIX}")

    def test_a_metrics_document_is_not_volatile(self) -> None:
        assert not is_volatile("experiments/baseline/metrics.json")

    def test_an_unclassified_file_defaults_to_deterministic(
        self, tmp_path: Path
    ) -> None:
        # The failure mode to prefer: a file nobody classified is
        # checksummed, so if it turns out to vary the two-execution test
        # fails loudly rather than the guarantee weakening silently.
        target = tmp_path / "novel.json"
        target.write_text("{}", encoding="utf-8")
        deterministic, volatile = classify([target], tmp_path)
        assert [a.path for a in deterministic] == ["novel.json"]
        assert volatile == {}

    def test_a_stage_that_produced_nothing_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(StageRecordError, match="produced no file"):
            build_stage_record(
                stage_name="baseline",
                stage_kind="experiment_run",
                command="c",
                logical_identity="i",
                targets=[tmp_path / "absent"],
                root=tmp_path,
            )


# ---------------------------------------------------------------------------
# The execution sidecar: where the wall clock went
# ---------------------------------------------------------------------------


class TestExecutionSidecar:
    def test_the_sidecar_records_the_timestamp_the_document_does_not(self) -> None:
        metadata = build_execution_metadata(
            describes="mlops/stages/baseline.json", produced_by="test"
        )
        assert metadata.created_at_utc
        assert metadata.python_version.count(".") >= 2

    def test_the_sidecar_sits_beside_its_document_not_inside_it(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "mlops" / "drift_report.json"
        output.parent.mkdir(parents=True)
        output.write_text("{}", encoding="utf-8")
        path = write_execution_sidecar(
            output, describes="mlops/drift_report.json", produced_by="test"
        )
        assert path.name == f"drift_report{EXECUTION_SUFFIX}"
        assert path.parent == output.parent
        assert path != output

    def test_the_sidecar_for_a_directory_is_written_outside_it(
        self, tmp_path: Path
    ) -> None:
        # A sidecar written inside a directory output would be hashed
        # along with the directory and defeat the whole point.
        directory = tmp_path / "model_versions"
        directory.mkdir()
        assert directory not in sidecar_path(directory).parents

    def test_no_sidecar_is_a_declared_output(self) -> None:
        for path in declared_outputs():
            assert not path.endswith(EXECUTION_SUFFIX)

    def test_the_sidecar_states_that_it_is_not_an_identity(self) -> None:
        metadata = build_execution_metadata(describes="a/b.json", produced_by="t")
        note = metadata.note.lower()
        assert "never declared as a dvc output" in note
        assert "participates in any fingerprint" in note

    def test_the_sidecar_refuses_an_absolute_describes_path(self) -> None:
        with pytest.raises(ValueError, match="absolute"):
            ExecutionMetadata(
                describes="/tmp/pipeline/mlops/drift_report.json",
                produced_by="t",
                created_at_utc=datetime.now(UTC),
                engagevr_version="0.1.0",
                python_version="3.12.13",
                python_implementation="CPython",
            )


class TestPythonSeries:
    def test_a_patch_version_is_reduced_to_a_series(self) -> None:
        assert python_series("3.12.13") == "3.12"

    def test_a_series_is_accepted(self) -> None:
        assert assert_python_series("3.12", field="t") == "3.12"

    def test_a_patch_level_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match=re.escape("major.minor")):
            assert_python_series("3.12.13", field="t")

    def test_an_interpreter_patch_upgrade_cannot_dirty_the_lock(
        self, m10_baseline_run: Path
    ) -> None:
        # The compatibility contract is the series. Recording the patch
        # level would put every interpreter upgrade into the identity of
        # every deterministic document.
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        assert version.python_series.count(".") == 1
