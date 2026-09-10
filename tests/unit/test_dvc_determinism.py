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

The boundary was drawn in the wrong place once
-----------------------------------------------
DEC-104 originally checksummed *every* non-timestamped file a stage
produced, "models included".  That held on one machine and failed on
GitHub Actions, which reported a changed hash for
``artifacts/pipeline/mlops/model_versions`` — same size, same file count,
different bytes — propagating into ``stages/baseline.json`` and
``reproducibility.json``.

The cause is not a bug in the pipeline.  ``joblib.dump`` writes
scikit-learn's raw tree-node buffer, whose C struct has seven bytes of
padding per node that nothing initialises; the baseline run's
random-forest artifacts carry 112,826 such bytes each, about ten thousand
of them non-zero heap residue.  191 of the 200 trees that are identical
field-for-field between the plain and the calibrated artifact disagree in
that padding — so two serializations of *one model in one process*
already differ.  Hashing a ``.joblib`` hashes memory no pipeline input
determines.

DEC-105 therefore adds a third classification, ``execution_specific``,
and the tests below hold the corrected boundary: a serialized estimator
may never be portable deterministic identity, its real SHA-256 must still
be recorded, and a logical model version must not move when only those
bytes do.

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
    INTEGRITY_SUFFIX,
    build_execution_metadata,
    integrity_sidecar_path,
    sidecar_path,
    write_execution_sidecar,
    write_integrity_sidecar,
)
from engagevr.mlops.model_version import (
    build_model_artifact_integrity,
    build_model_versions,
    read_artifact_integrity,
    verify_model_version,
    write_model_versions,
)
from engagevr.mlops.pipeline import build_stages, default_layout, load_parameters
from engagevr.mlops.stage_record import (
    VOLATILE_ARTIFACT_REASONS,
    VOLATILE_DATASET_SUFFIX,
    StageRecordError,
    build_stage_record,
    classify,
    is_execution_specific,
    is_volatile,
    normalize_command,
    read_stage_record,
    run_identity,
    write_stage_record,
)
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER
from engagevr.schemas.mlops import (
    SERIALIZED_ESTIMATOR_SUFFIXES,
    ArtifactIntegrityRecord,
    DeterministicArtifact,
    DeterministicStageRecord,
    ExecutionMetadata,
    assert_python_series,
    assert_relative_path,
    is_serialized_estimator,
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
    record, integrity = build_stage_record(
        stage_name=stage.name,
        stage_kind=stage.kind,
        command=stage.command,
        logical_identity=run_identity(layout.baseline_run),
        targets=list(stage.recorded_targets),
        root=layout.root,
    )
    write_stage_record(record, stage.record)
    write_integrity_sidecar(
        stage.record,
        integrity,
        describes=f"mlops/stages/{stage.name}.json",
        produced_by=f"engagevr stage-record --stage {stage.name}",
        paths_relative_to="the pipeline root",
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
            )[0],
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
            )[0],
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
            )[0],
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
        classification = classify([target], tmp_path)
        assert [a.path for a in classification.deterministic] == ["novel.json"]
        assert classification.volatile == {}
        assert classification.execution_specific == {}
        assert classification.integrity == ()

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


# ---------------------------------------------------------------------------
# DEC-105: the corrected reproducibility boundary
#
# The regression coverage for the defect GitHub Actions found. Every test
# here would have failed on the code that produced the failing CI run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_versions_with_integrity(
    tmp_path_factory: pytest.TempPathFactory, m10_baseline_run: Path
) -> tuple[Path, Any, Any]:
    """A written model-version directory and its integrity sidecar."""
    directory = tmp_path_factory.mktemp("dec105") / "model_versions"
    versions = build_model_versions(m10_baseline_run, config=load_config())
    integrity = build_model_artifact_integrity(versions, m10_baseline_run)
    write_model_versions(versions, directory, integrity=integrity)
    return directory, versions, integrity


class TestSerializedEstimatorsAreNotPortableIdentity:
    """A ``.joblib`` checksum may not enter any DVC-declared document."""

    def test_a_model_file_is_classified_execution_specific(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        _layout, _parameters, stage = recorded_baseline
        record = read_stage_record(stage.record)
        models = [
            path
            for path in record.execution_specific_artifacts
            if path.endswith(".joblib")
        ]
        assert models, "the baseline run persists estimators; none was classified"
        for path in models:
            reason = record.execution_specific_artifacts[path]
            assert "never-initialised padding" in reason
            assert "serialized Python estimator" in reason

    def test_no_portable_stage_record_checksums_a_model_file(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        _layout, _parameters, stage = recorded_baseline
        record = read_stage_record(stage.record)
        offending = [
            a.path
            for a in record.deterministic_artifacts
            if is_serialized_estimator(a.path)
        ]
        assert offending == []

    @pytest.mark.parametrize("suffix", SERIALIZED_ESTIMATOR_SUFFIXES)
    def test_every_pickle_suffix_is_recognised(self, suffix: str) -> None:
        assert is_serialized_estimator(f"experiments/run/models/estimator{suffix}")
        assert is_execution_specific(f"experiments/run/models/estimator{suffix}")

    def test_the_schema_refuses_a_model_checksum_in_a_portable_record(self) -> None:
        # The last line of defence: even if classification were bypassed,
        # the document cannot be constructed. CI fails if somebody
        # reintroduces a raw serialized-model checksum into portable
        # identity.
        with pytest.raises(ValueError, match="serialized Python estimator"):
            DeterministicStageRecord(
                stage_name="baseline",
                stage_kind="experiment_run",
                command="c",
                logical_identity="run_id:x",
                deterministic_artifacts=(
                    DeterministicArtifact(
                        path="experiments/baseline/models/rf-fold0.joblib",
                        sha256="a" * 64,
                        size_bytes=1,
                    ),
                ),
                engagevr_version="0.1.0",
                python_series="3.12",
                is_synthetic=True,
                scientific_evaluation_eligible=False,
                disclaimers=(SOFTWARE_SELF_CHECK_BANNER,),
            )

    def test_a_file_cannot_be_both_deterministic_and_execution_specific(self) -> None:
        with pytest.raises(ValueError, match="exactly one classification"):
            DeterministicStageRecord(
                stage_name="baseline",
                stage_kind="experiment_run",
                command="c",
                logical_identity="run_id:x",
                deterministic_artifacts=(
                    DeterministicArtifact(
                        path="experiments/baseline/metrics.json",
                        sha256="a" * 64,
                        size_bytes=1,
                    ),
                ),
                execution_specific_artifacts={
                    "experiments/baseline/metrics.json": "why",
                },
                engagevr_version="0.1.0",
                python_series="3.12",
                is_synthetic=True,
                scientific_evaluation_eligible=False,
                disclaimers=(SOFTWARE_SELF_CHECK_BANNER,),
            )

    def test_classification_still_fails_closed_for_an_unknown_binary(
        self, tmp_path: Path
    ) -> None:
        # An unknown suffix is NOT quietly excluded. It is checksummed, so
        # a genuinely unstable new output breaks the reproduction test
        # loudly instead of joining an exclusion list nobody reviewed.
        target = tmp_path / "models" / "estimator.onnx"
        target.parent.mkdir()
        target.write_bytes(b"\x00\x01")
        classification = classify([target], tmp_path)
        assert [a.path for a in classification.deterministic] == [
            "models/estimator.onnx"
        ]
        assert classification.execution_specific == {}


class TestModelArtifactIntegrityIsPreserved:
    """The digest moved. It was not weakened and it was not lost."""

    def test_every_model_file_still_has_a_recorded_sha256(
        self, model_versions_with_integrity: tuple[Path, Any, Any]
    ) -> None:
        directory, versions, _integrity = model_versions_with_integrity
        record = read_artifact_integrity(integrity_sidecar_path(directory))
        recorded = {entry.path for entry in record.artifacts}
        # Every model file, and since DEC-106 the CPU-dependent numerical
        # documents the versions reference as well: both classes lost
        # their place in portable identity, so both keep their real
        # digest here or nowhere.
        assert recorded >= {v.model_artifact_path for v in versions}
        assert "metrics.json" in recorded
        for entry in record.artifacts:
            assert len(entry.sha256) == 64
            assert entry.size_bytes > 0
            assert entry.excluded_from_portable_identity

    def test_the_stage_record_sidecar_holds_the_same_digests(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        layout, _parameters, stage = recorded_baseline
        record = read_stage_record(stage.record)
        integrity = ArtifactIntegrityRecord.model_validate(
            json.loads(integrity_sidecar_path(stage.record).read_text(encoding="utf-8"))
        )
        # Exactly the two classes that are excluded from portable byte
        # identity, and nothing else: an artifact is either pinned by its
        # bytes in the record or digested here, never neither.
        assert {e.path for e in integrity.artifacts} == set(
            record.execution_specific_artifacts
        ) | {a.path for a in record.cpu_dependent_numeric_artifacts}
        for entry in integrity.artifacts:
            assert entry.sha256 == _sha256(layout.root / entry.path)

    def test_changing_the_model_binary_changes_its_integrity_record(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        import shutil

        run = tmp_path / "run"
        shutil.copytree(m10_baseline_run, run)
        versions = build_model_versions(run, config=load_config())
        before = build_model_artifact_integrity(versions, run)
        target = run / versions[0].model_artifact_path
        target.write_bytes(target.read_bytes() + b"\x00")
        after = build_model_artifact_integrity(versions, run)
        changed = [
            (b.path, b.sha256, a.sha256)
            for b, a in zip(before, after, strict=True)
            if b.sha256 != a.sha256
        ]
        assert len(changed) == 1
        assert changed[0][0] == versions[0].model_artifact_path

    def test_a_tampered_model_binary_still_fails_verification(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        import shutil

        run = tmp_path / "run"
        shutil.copytree(m10_baseline_run, run)
        versions = build_model_versions(run, config=load_config())
        integrity = build_model_artifact_integrity(versions, run)
        assert (
            verify_model_version(versions[0], run_directory=run, integrity=integrity)
            == ()
        )
        target = run / versions[0].model_artifact_path
        target.write_bytes(target.read_bytes() + b"\x00")
        assert verify_model_version(
            versions[0], run_directory=run, integrity=integrity
        ) == (versions[0].model_artifact_path,)

    def test_no_integrity_sidecar_is_a_declared_output(self) -> None:
        for path in declared_outputs():
            assert not path.endswith(INTEGRITY_SUFFIX)

    def test_no_model_digest_appears_in_the_portable_stage_record(
        self, recorded_baseline: tuple[Path, Any, Any]
    ) -> None:
        # The precise regression: a platform-sensitive binary hash must not
        # appear anywhere in the DVC-declared document, in any field.
        layout, _parameters, stage = recorded_baseline
        rendered = stage.record.read_text(encoding="utf-8")
        record = read_stage_record(stage.record)
        assert record.execution_specific_artifacts
        for relative in record.execution_specific_artifacts:
            assert _sha256(layout.root / relative) not in rendered, relative

    def test_the_integrity_sidecar_sits_outside_the_directory_it_describes(
        self, model_versions_with_integrity: tuple[Path, Any, Any]
    ) -> None:
        directory, _versions, _integrity = model_versions_with_integrity
        sidecar = integrity_sidecar_path(directory)
        assert sidecar.is_file()
        assert directory not in sidecar.parents

    def test_the_integrity_record_says_a_checksum_is_not_validity(
        self, model_versions_with_integrity: tuple[Path, Any, Any]
    ) -> None:
        directory, _versions, _integrity = model_versions_with_integrity
        record = read_artifact_integrity(integrity_sidecar_path(directory))
        assert "NOT SCIENTIFIC VALIDITY" in record.integrity_note.upper()
        assert record.scientific_evaluation_eligible is False
        assert record.is_synthetic is True


class TestLogicalModelVersionIdentity:
    """One logical version, N serialized instances."""

    def test_the_identifier_does_not_depend_on_the_serialized_bytes(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        # The exact defect: on the old code, appending a byte to a model
        # file renamed the model version. It must not now.
        import shutil

        run = tmp_path / "run"
        shutil.copytree(m10_baseline_run, run)
        before = _identifiers(run)
        target = run / "models" / "logistic_regression-fold0.joblib"
        target.write_bytes(target.read_bytes() + b"\x00")
        # checksums.json still records the old digest, so re-derivation
        # refuses: tamper detection is intact. Update it the way a genuine
        # re-serialization would, then re-derive.
        checksums_path = run / "checksums.json"
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
        checksums["models/logistic_regression-fold0.joblib"] = _sha256(target)
        checksums_path.write_text(json.dumps(checksums), encoding="utf-8")
        after = _identifiers(run)
        assert after == before

    def test_a_changed_estimator_configuration_changes_the_identifier(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        import shutil

        run = tmp_path / "run"
        shutil.copytree(m10_baseline_run, run)
        before = {
            v.model_name: v.model_version_id
            for v in build_model_versions(run, config=load_config())
        }
        path = run / "manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["model_parameters"]["logistic_regression"]["parameters"]["C"] = 0.5
        path.write_text(json.dumps(document), encoding="utf-8")
        after = {
            v.model_name: v.model_version_id
            for v in build_model_versions(run, config=load_config())
        }
        assert after["logistic_regression-fold0"] != before["logistic_regression-fold0"]
        assert after["dummy-fold0"] == before["dummy-fold0"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("run_id", "a-different-run"),
            ("dataset_fingerprint", "f" * 64),
            ("target_name", "engagement_score"),
            ("task_type", "regression"),
        ],
    )
    def test_changed_provenance_changes_the_identifier(
        self, tmp_path: Path, m10_baseline_run: Path, field: str, value: str
    ) -> None:
        import shutil

        run = tmp_path / "run"
        shutil.copytree(m10_baseline_run, run)
        before = _identifiers(run)
        path = run / "manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document[field] = value
        path.write_text(json.dumps(document), encoding="utf-8")
        after = _identifiers(run)
        assert after != before

    def test_a_changed_split_changes_the_identifier(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        import shutil

        run = tmp_path / "run"
        shutil.copytree(m10_baseline_run, run)
        before = _identifiers(run)
        path = run / "splits.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        # Move one subject from training to test in the first fold: a real
        # change to the split design, not a cosmetic edit to the document.
        fold = document["folds"][0]
        fold["test_groups"].append(fold["train_groups"].pop())
        path.write_text(json.dumps(document), encoding="utf-8")
        after = _identifiers(run)
        assert after != before

    def test_the_manifest_carries_no_serialized_artifact_checksum(
        self, model_versions_with_integrity: tuple[Path, Any, Any]
    ) -> None:
        directory, _versions, _integrity = model_versions_with_integrity
        for path in sorted(directory.glob("*.model-version.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            assert "model_artifact_sha256" not in document
            assert not any(
                is_serialized_estimator(name)
                for name in document["referenced_checksums"]
            )

    def test_the_manifest_still_points_at_its_artifact_and_its_integrity_record(
        self, model_versions_with_integrity: tuple[Path, Any, Any]
    ) -> None:
        directory, versions, _integrity = model_versions_with_integrity
        sidecar = integrity_sidecar_path(directory)
        for version in versions:
            assert version.model_artifact_path.startswith("models/")
            assert version.model_artifact_integrity_document == sidecar.name

    def test_the_model_version_directory_carries_no_platform_hash(
        self, model_versions_with_integrity: tuple[Path, Any, Any]
    ) -> None:
        # What CI actually reported: the whole directory's hash moved.
        # Nothing inside it may now be a fact about one machine.
        directory, _versions, integrity = model_versions_with_integrity
        digests = {entry.sha256 for entry in integrity}
        assert digests
        for path in sorted(directory.glob("*.json")):
            rendered = path.read_text(encoding="utf-8")
            for digest in digests:
                assert digest not in rendered, path.name

    def test_no_model_version_becomes_an_endorsement(
        self, model_versions_with_integrity: tuple[Path, Any, Any]
    ) -> None:
        directory, versions, _integrity = model_versions_with_integrity
        del directory
        for version in versions:
            assert version.scientific_evaluation_eligible is False
            assert version.is_synthetic is True
            for word in ("production", "champion", "approved", "validated"):
                assert word not in version.model_version_id.lower()


class TestModelVersionIdentityUnderTheNumericContract:
    """DEC-106 extended to model versions: structure pins, floats do not.

    ``metrics.json`` is a document of fitted scores.  Referencing it by
    raw digest renamed the *same fitted model* on a machine whose BLAS
    kernel rounds differently, which is DEC-105's defect one layer up.
    It is referenced by structure digest instead.
    """

    def _versions(self, run: Path) -> dict[str, Any]:
        return {
            v.model_name: v for v in build_model_versions(run, config=load_config())
        }

    def _copy(self, tmp_path: Path, source: Path) -> Path:
        import shutil

        run = tmp_path / "run"
        shutil.copytree(source, run)
        return run

    def _nudge_metrics(self, run: Path, *, delta: float) -> None:
        """Move every float in metrics.json by ``delta``, as a CPU would."""
        path = run / "metrics.json"
        document = json.loads(path.read_text(encoding="utf-8"))

        def walk(node: Any) -> Any:
            if isinstance(node, bool):
                return node
            if isinstance(node, float):
                return node + delta
            if isinstance(node, dict):
                return {key: walk(value) for key, value in node.items()}
            if isinstance(node, list):
                return [walk(value) for value in node]
            return node

        path.write_text(json.dumps(walk(document), indent=2), encoding="utf-8")

    def test_metrics_is_referenced_by_structure_not_by_raw_bytes(
        self, model_versions_with_integrity: tuple[Path, Any, Any]
    ) -> None:
        _directory, versions, _integrity = model_versions_with_integrity
        for version in versions:
            assert "metrics.json" not in version.referenced_checksums
            assert "metrics.json" in version.referenced_structure_digests

    def test_the_exact_documents_are_still_referenced_by_raw_bytes(
        self, model_versions_with_integrity: tuple[Path, Any, Any]
    ) -> None:
        # splits.json and feature_catalog.json reproduce byte for byte on
        # every CPU measured. They keep the strict rule.
        _directory, versions, _integrity = model_versions_with_integrity
        for version in versions:
            assert "splits.json" in version.referenced_checksums
            assert "feature_catalog.json" in version.referenced_checksums

    def test_the_identifier_is_stable_within_the_declared_tolerance(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        run = self._copy(tmp_path, m10_baseline_run)
        before = _identifiers(run)
        before_structures = {
            name: v.referenced_structure_digests.get("metrics.json")
            for name, v in self._versions(run).items()
        }
        self._nudge_metrics(run, delta=1e-12)
        after = _identifiers(run)
        after_structures = {
            name: v.referenced_structure_digests.get("metrics.json")
            for name, v in self._versions(run).items()
        }
        assert after == before, (
            "a last-bit change in a fitted score must not rename the model"
        )
        assert after_structures == before_structures

    def test_a_structural_change_to_metrics_changes_the_reference(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        run = self._copy(tmp_path, m10_baseline_run)
        before = self._versions(run)["logistic_regression-fold0"]
        path = run / "metrics.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["fold_count"] = 99
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        after = self._versions(run)["logistic_regression-fold0"]
        assert (
            after.referenced_structure_digests["metrics.json"]
            != before.referenced_structure_digests["metrics.json"]
        ), "an integer is not a float; a changed count must propagate"

    def test_verification_still_detects_a_structural_change(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        run = self._copy(tmp_path, m10_baseline_run)
        version = self._versions(run)["logistic_regression-fold0"]
        integrity = build_model_artifact_integrity([version], run)
        assert (
            verify_model_version(version, run_directory=run, integrity=integrity) == ()
        )
        path = run / "metrics.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["run_id"] = "a-different-run"
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        assert "metrics.json" in verify_model_version(
            version, run_directory=run, integrity=integrity
        )

    def test_verification_still_detects_a_truncated_metrics_document(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        run = self._copy(tmp_path, m10_baseline_run)
        version = self._versions(run)["logistic_regression-fold0"]
        integrity = build_model_artifact_integrity([version], run)
        (run / "metrics.json").write_text("{not json", encoding="utf-8")
        assert "metrics.json" in verify_model_version(
            version, run_directory=run, integrity=integrity
        )

    def test_the_raw_metrics_digest_is_still_recorded_in_the_integrity_record(
        self, model_versions_with_integrity: tuple[Path, Any, Any]
    ) -> None:
        # The digest moved; it was not deleted. Corruption of metrics.json
        # is still detectable byte for byte on the machine that wrote it.
        _directory, _versions, integrity = model_versions_with_integrity
        recorded = {entry.path for entry in integrity}
        assert "metrics.json" in recorded

    def test_a_beyond_tolerance_change_is_caught_by_comparison_not_by_the_id(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        # An honest boundary. A model version is derived from ONE run, so
        # it has nothing to compare a float against and cannot, by
        # itself, notice that a score moved. Detecting that needs a
        # reference execution, which is what `engagevr numeric-check`
        # supplies and what CI runs. Asserted here so the limit is a
        # tested fact rather than an assumption. See DEC-106.
        from engagevr.mlops.numeric_contract import compare_artifacts

        run = self._copy(tmp_path, m10_baseline_run)
        reference_metrics = run / "metrics.json"
        original = reference_metrics.read_bytes()
        before = _identifiers(run)

        self._nudge_metrics(run, delta=1e-3)
        after = _identifiers(run)
        assert after == before, "documented: the identifier alone cannot see this"

        moved = tmp_path / "moved-metrics.json"
        moved.write_bytes(reference_metrics.read_bytes())
        reference = tmp_path / "reference-metrics.json"
        reference.write_bytes(original)
        differences = compare_artifacts(reference, moved)
        assert differences, "the comparison must see what the identifier cannot"


class TestNothingRewritesTheLock:
    """`dvc.lock` is generated. No formatter may edit it."""

    def test_the_lock_wraps_commands_with_trailing_whitespace(self) -> None:
        # Not a defect to fix: DVC writes it, and the point of the two
        # tests here is that nothing else may take it away.
        lock = (ROOT / "dvc.lock").read_text(encoding="utf-8")
        assert any(line.endswith(" ") for line in lock.splitlines())

    @pytest.mark.parametrize("hook", ["trailing-whitespace", "end-of-file-fixer"])
    def test_the_whitespace_hooks_exclude_the_lock(self, hook: str) -> None:
        # A hook that strips those trailing spaces makes the committed lock
        # differ from the one `dvc repro` produces, so CI fails its
        # cross-environment lock check on an edit nobody made deliberately.
        document = yaml.safe_load(
            (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        )
        entries = [
            entry
            for repo in document["repos"]
            for entry in repo["hooks"]
            if entry["id"] == hook
        ]
        assert entries, f"{hook} is no longer configured"
        for entry in entries:
            pattern = entry.get("exclude")
            assert pattern, f"{hook} does not exclude anything"
            assert re.search(pattern, "dvc.lock"), hook


def _identifiers(run: Path) -> list[str]:
    """Every logical model-version identifier a run yields, in order."""
    return [
        version.model_version_id
        for version in build_model_versions(run, config=load_config())
    ]


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
