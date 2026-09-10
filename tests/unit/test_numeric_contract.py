"""The fourth classification: CPU-dependent numerical artifacts (DEC-106).

What is under test
------------------
DEC-105 predicted this failure and left it open::

    Reproducing the baseline stage with only the OpenBLAS kernel changed
    changed metrics.json, predictions.parquet, and
    feature_importance.parquet as well as the model files. [...] if a
    runner's CPU ever produces different numbers the lock check fails
    loudly and says so.

On PR #10 it did.  Ten artifacts moved between the machine that committed
``dvc.lock`` and the GitHub runner of CI run 34403534631, and none of
them was a serialized estimator: the ``uncertainty`` stage persists no
``.joblib`` at all, yet seven of its thirteen artifacts changed.

The repair is a fourth classification, not a fourth exclusion list.  What
differs across CPUs is *only the floating-point values*; the schema, the
ordering, the dtypes, the predicted labels, the integer counts, and the
positions of missing and non-finite values are identical, and stay pinned
exactly.  These tests hold that line from both directions:

- a last-bit difference must **pass** — that is the whole point;
- everything else must **fail**, including a float that moved too far,
  a reordered row, a changed dtype, a relabelled prediction, a moved
  missing value, and an unclassified artifact that changed at all.

The tolerance is an engineering portability allowance.  It is not a
scientific uncertainty interval, not a confidence bound, and not evidence
about any model: nothing in this repository has been evaluated against a
participant-provided label.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from engagevr.mlops.numeric_contract import (
    DEFAULT_TOLERANCE,
    MEASURED_CROSS_KERNEL_DEVIATION,
    NUMERIC_PORTABILITY_ATOL,
    NUMERIC_PORTABILITY_RTOL,
    NumericContractError,
    NumericTolerance,
    compare_artifacts,
    is_supported,
    structure_digest,
)
from engagevr.mlops.stage_record import (
    CPU_DEPENDENT_NUMERIC_REASONS,
    classify,
    is_cpu_dependent_numeric,
    is_execution_specific,
    record_digest,
)
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER
from engagevr.schemas.mlops import (
    CpuDependentNumericArtifact,
    DeterministicArtifact,
    DeterministicStageRecord,
)


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures: a small artifact of each kind, shaped like the real ones
# ---------------------------------------------------------------------------


def write_predictions(path: Path, *, probabilities: list[float]) -> Path:
    """A predictions table shaped like the pipeline's own."""
    frame = pd.DataFrame(
        {
            "window_id": [f"w{i:03d}" for i in range(len(probabilities))],
            "fold_index": list(range(len(probabilities))),
            "predicted_label": ["low", "high", "medium"][: len(probabilities)],
            "probability__high": probabilities,
            "interval_width": [0.1, None, 0.3][: len(probabilities)],
        }
    )
    frame.to_parquet(path, index=False)
    return path


def write_metrics(path: Path, *, value: float) -> Path:
    """A metrics document shaped like the pipeline's own."""
    document = {
        "run_id": "engagement_class-selfcheck-abc123",
        "random_seed": 42,
        "fold_count": 3,
        "scientific_evaluation_eligible": False,
        "results": [
            {
                "model_name": "logistic_regression",
                "aggregate": [{"mean": value, "fold_values": [value, value]}],
            }
        ],
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def predictions(tmp_path: Path) -> Path:
    return write_predictions(
        tmp_path / "predictions.parquet", probabilities=[0.25, 0.5, 0.75]
    )


@pytest.fixture
def metrics(tmp_path: Path) -> Path:
    return write_metrics(tmp_path / "metrics.json", value=0.7041209873714114)


# ---------------------------------------------------------------------------
# 1. A last-bit difference passes, and the raw digest still records it
# ---------------------------------------------------------------------------


class TestUlpScalePerturbationIsPortable:
    """The case the whole classification exists for."""

    def test_a_one_ulp_json_difference_passes_equivalence(
        self, tmp_path: Path, metrics: Path
    ) -> None:
        base = 0.7041209873714114
        nudged = math.nextafter(base, math.inf)
        other = write_metrics(tmp_path / "metrics_b.json", value=nudged)
        assert base != nudged, "the fixture must actually differ"
        assert compare_artifacts(metrics, other) == []

    def test_a_realistic_cross_kernel_difference_passes(
        self, tmp_path: Path, metrics: Path
    ) -> None:
        # The worst deviation measured across three OpenBLAS kernels.
        moved = 0.7041209873714114 + MEASURED_CROSS_KERNEL_DEVIATION
        other = write_metrics(tmp_path / "metrics_b.json", value=moved)
        assert compare_artifacts(metrics, other) == []

    def test_the_structure_digest_is_unchanged_by_a_float(
        self, tmp_path: Path, metrics: Path
    ) -> None:
        moved = 0.7041209873714114 + MEASURED_CROSS_KERNEL_DEVIATION
        other = write_metrics(tmp_path / "metrics_b.json", value=moved)
        assert structure_digest(metrics) == structure_digest(other)

    def test_the_raw_digest_does_differ_and_is_still_recorded(
        self, tmp_path: Path
    ) -> None:
        # Nothing is being hidden: the bytes really are different, and the
        # integrity record really does say so. That is the difference
        # between classifying an artifact and dropping it.
        run = tmp_path / "experiments" / "baseline-engagement_class"
        run.mkdir(parents=True)
        write_metrics(run / "metrics.json", value=0.5)
        first = classify([run], tmp_path)

        write_metrics(run / "metrics.json", value=0.5 + 1e-12)
        second = classify([run], tmp_path)

        [entry_a] = [e for e in first.integrity if e.path.endswith("metrics.json")]
        [entry_b] = [e for e in second.integrity if e.path.endswith("metrics.json")]
        assert entry_a.sha256 != entry_b.sha256, (
            "the raw digest must still change; the integrity record is where "
            "corruption stays detectable"
        )
        assert entry_a.excluded_from_portable_identity

    def test_the_stage_identity_does_not_move_for_a_last_bit_change(
        self, tmp_path: Path
    ) -> None:
        run = tmp_path / "experiments" / "baseline-engagement_class"
        run.mkdir(parents=True)
        write_metrics(run / "metrics.json", value=0.5)
        before = classify([run], tmp_path).cpu_numeric
        write_metrics(run / "metrics.json", value=0.5 + 1e-12)
        after = classify([run], tmp_path).cpu_numeric
        assert [a.structure_sha256 for a in before] == [
            a.structure_sha256 for a in after
        ]


# ---------------------------------------------------------------------------
# 2-7. Everything that is not a last-bit difference must fail
# ---------------------------------------------------------------------------


class TestBeyondToleranceFails:
    """A numerical change large enough to read differently is a change."""

    def test_a_json_float_beyond_tolerance_fails(
        self, tmp_path: Path, metrics: Path
    ) -> None:
        moved = 0.7041209873714114 + 1e-3
        other = write_metrics(tmp_path / "metrics_b.json", value=moved)
        differences = compare_artifacts(metrics, other)
        assert differences, "a 1e-3 change must not pass a 1e-6 tolerance"
        assert "exceeds tolerance" in differences[0]

    def test_a_parquet_float_beyond_tolerance_fails(
        self, tmp_path: Path, predictions: Path
    ) -> None:
        other = write_predictions(
            tmp_path / "b.parquet", probabilities=[0.25, 0.5 + 1e-3, 0.75]
        )
        differences = compare_artifacts(predictions, other)
        assert differences
        assert "probability__high" in differences[0]

    def test_the_boundary_is_where_it_is_declared(self) -> None:
        tolerance = NumericTolerance(atol=1e-6, rtol=0.0)
        assert tolerance.holds(1.0, 1.0 + 9e-7)
        assert not tolerance.holds(1.0, 1.0 + 2e-6)

    def test_a_nan_that_became_a_number_is_never_within_tolerance(self) -> None:
        assert not DEFAULT_TOLERANCE.holds(float("nan"), 0.0)
        assert not DEFAULT_TOLERANCE.holds(0.0, float("nan"))
        assert DEFAULT_TOLERANCE.holds(float("nan"), float("nan"))

    def test_an_infinity_that_changed_sign_is_never_within_tolerance(self) -> None:
        assert not DEFAULT_TOLERANCE.holds(float("inf"), float("-inf"))
        assert DEFAULT_TOLERANCE.holds(float("inf"), float("inf"))


class TestStructuralChangesFail:
    """Structure is exact. No tolerance reaches any of these."""

    def test_a_schema_change_fails(self, tmp_path: Path, predictions: Path) -> None:
        frame = pd.read_parquet(predictions)
        frame["unexpected_column"] = 1
        other = tmp_path / "b.parquet"
        frame.to_parquet(other, index=False)
        assert structure_digest(predictions) != structure_digest(other)
        differences = compare_artifacts(predictions, other)
        assert differences and "columns" in differences[0]

    def test_a_column_reordering_fails(self, tmp_path: Path, predictions: Path) -> None:
        frame = pd.read_parquet(predictions)
        other = tmp_path / "b.parquet"
        frame[list(reversed(frame.columns))].to_parquet(other, index=False)
        assert structure_digest(predictions) != structure_digest(other)
        assert compare_artifacts(predictions, other)

    def test_a_row_reordering_fails(self, tmp_path: Path, predictions: Path) -> None:
        frame = pd.read_parquet(predictions)
        other = tmp_path / "b.parquet"
        frame.iloc[::-1].reset_index(drop=True).to_parquet(other, index=False)
        assert structure_digest(predictions) != structure_digest(other), (
            "row order participates: a shuffled table is a different result"
        )
        assert compare_artifacts(predictions, other)

    def test_a_dtype_change_fails(self, tmp_path: Path, predictions: Path) -> None:
        frame = pd.read_parquet(predictions)
        frame["fold_index"] = frame["fold_index"].astype("float64")
        other = tmp_path / "b.parquet"
        frame.to_parquet(other, index=False)
        differences = compare_artifacts(predictions, other)
        assert differences and "dtype" in differences[0], (
            "an integer silently promoted to a float must not then be "
            "compared with a tolerance"
        )

    def test_a_predicted_label_change_fails(
        self, tmp_path: Path, predictions: Path
    ) -> None:
        frame = pd.read_parquet(predictions)
        frame.loc[0, "predicted_label"] = "high"
        other = tmp_path / "b.parquet"
        frame.to_parquet(other, index=False)
        assert structure_digest(predictions) != structure_digest(other)
        differences = compare_artifacts(predictions, other)
        assert differences and "predicted_label" in differences[0]

    def test_a_missing_value_moving_fails(
        self, tmp_path: Path, predictions: Path
    ) -> None:
        frame = pd.read_parquet(predictions)
        frame["interval_width"] = [None, 0.1, 0.3]
        other = tmp_path / "b.parquet"
        frame.to_parquet(other, index=False)
        assert structure_digest(predictions) != structure_digest(other), (
            "where a value is absent is structural, not numerical"
        )
        differences = compare_artifacts(predictions, other)
        assert differences and "interval_width" in differences[0]

    def test_a_non_finite_value_appearing_fails(self, tmp_path: Path) -> None:
        a = write_predictions(tmp_path / "a.parquet", probabilities=[0.1, 0.2, 0.3])
        b = write_predictions(
            tmp_path / "b.parquet", probabilities=[0.1, float("nan"), 0.3]
        )
        assert structure_digest(a) != structure_digest(b)
        assert compare_artifacts(a, b)

    def test_a_json_identifier_is_never_compared_with_a_tolerance(
        self, tmp_path: Path, metrics: Path
    ) -> None:
        document = json.loads(metrics.read_text(encoding="utf-8"))
        document["run_id"] = "engagement_class-selfcheck-DIFFERENT"
        other = tmp_path / "b.json"
        other.write_text(json.dumps(document, indent=2), encoding="utf-8")
        assert structure_digest(metrics) != structure_digest(other)
        assert compare_artifacts(metrics, other)

    def test_a_json_integer_is_never_compared_with_a_tolerance(
        self, tmp_path: Path, metrics: Path
    ) -> None:
        document = json.loads(metrics.read_text(encoding="utf-8"))
        document["fold_count"] = 4
        other = tmp_path / "b.json"
        other.write_text(json.dumps(document, indent=2), encoding="utf-8")
        assert compare_artifacts(metrics, other), (
            "3 and 4 differ by less than the absolute tolerance would allow "
            "for a float; a count is not a float"
        )

    def test_a_json_bool_is_not_a_number(self, tmp_path: Path, metrics: Path) -> None:
        document = json.loads(metrics.read_text(encoding="utf-8"))
        document["scientific_evaluation_eligible"] = True
        other = tmp_path / "b.json"
        other.write_text(json.dumps(document, indent=2), encoding="utf-8")
        assert compare_artifacts(metrics, other)


# ---------------------------------------------------------------------------
# 8. Fail closed on anything unclassified
# ---------------------------------------------------------------------------


class TestFailsClosed:
    """An artifact nobody classified is held to the strictest rule."""

    def test_an_unknown_json_output_is_portable_deterministic(
        self, tmp_path: Path
    ) -> None:
        run = tmp_path / "experiments" / "baseline-engagement_class"
        run.mkdir(parents=True)
        target = run / "brand_new_scores.json"
        target.write_text('{"value": 0.5}', encoding="utf-8")
        classification = classify([run], tmp_path)
        assert [a.path for a in classification.deterministic] == [
            "experiments/baseline-engagement_class/brand_new_scores.json"
        ]
        assert classification.cpu_numeric == (), (
            "a new artifact must not join the tolerant class by resembling "
            "one; it is checksummed and will fail loudly if it is unstable"
        )

    def test_an_unknown_parquet_output_is_portable_deterministic(
        self, tmp_path: Path
    ) -> None:
        run = tmp_path / "experiments" / "baseline-engagement_class"
        run.mkdir(parents=True)
        write_predictions(run / "brand_new_table.parquet", probabilities=[0.1])
        classification = classify([run], tmp_path)
        assert classification.cpu_numeric == ()
        assert len(classification.deterministic) == 1

    def test_membership_is_an_exact_path_not_a_basename(self) -> None:
        # DEC-107. A basename allowlist would hand the exemption to any
        # future artifact called metrics.json, at any path, without
        # anybody having measured it.
        assert is_cpu_dependent_numeric(
            "experiments/baseline-engagement_class/metrics.json"
        )
        assert not is_cpu_dependent_numeric("metrics.json")
        assert not is_cpu_dependent_numeric("experiments/run/metrics.json")
        assert not is_cpu_dependent_numeric(
            "experiments/baseline-engagement_score/metrics.json"
        )
        assert not is_cpu_dependent_numeric(
            "experiments/baseline-engagement_class/metrics_extra.json"
        )
        assert not is_cpu_dependent_numeric(
            "experiments/baseline-engagement_class/ablations.json"
        )

    def test_a_same_named_artifact_elsewhere_is_portable_deterministic(
        self, tmp_path: Path
    ) -> None:
        # The concrete consequence: a new stage writing its own
        # metrics.json is checksummed by its raw bytes and will fail
        # loudly if it is not byte-stable, rather than silently
        # inheriting an exemption measured for a different file.
        run = tmp_path / "experiments" / "some-other-stage"
        run.mkdir(parents=True)
        write_metrics(run / "metrics.json", value=0.5)
        write_predictions(run / "predictions.parquet", probabilities=[0.1, 0.2])
        classification = classify([run], tmp_path)
        assert classification.cpu_numeric == ()
        assert sorted(a.path for a in classification.deterministic) == [
            "experiments/some-other-stage/metrics.json",
            "experiments/some-other-stage/predictions.parquet",
        ]

    def test_the_allowlist_holds_exactly_the_ten_measured_paths(self) -> None:
        assert len(CPU_DEPENDENT_NUMERIC_REASONS) == 10
        assert set(CPU_DEPENDENT_NUMERIC_REASONS) == {
            "experiments/baseline-engagement_class/metrics.json",
            "experiments/baseline-engagement_class/predictions.parquet",
            "experiments/baseline-engagement_class/feature_importance.parquet",
            "experiments/uncertainty-engagement_class/metrics.json",
            "experiments/uncertainty-engagement_class/predictions.parquet",
            "experiments/uncertainty-engagement_class/selective_metrics.json",
            "experiments/uncertainty-engagement_class/selective_predictions.parquet",
            "experiments/uncertainty-engagement_class/thresholds.json",
            "experiments/uncertainty-engagement_class/uncertainty.json",
            "experiments/uncertainty-engagement_class/adaptation_gate.parquet",
        }

    def test_every_classified_path_is_pipeline_relative(self) -> None:
        for path in CPU_DEPENDENT_NUMERIC_REASONS:
            assert not path.startswith("/")
            assert path.startswith("experiments/")
            assert path.count("/") == 2

    def test_every_classified_name_is_readable_by_the_contract(self) -> None:
        # A file cannot be excused from byte identity unless it can
        # actually be compared instead.
        for name in CPU_DEPENDENT_NUMERIC_REASONS:
            assert is_supported(name), f"{name} cannot be read or compared"

    def test_every_classified_name_states_its_evidence(self) -> None:
        for name, reason in CPU_DEPENDENT_NUMERIC_REASONS.items():
            assert len(reason) > 40, f"{name} has no stated reason"

    def test_an_unreadable_artifact_is_refused_rather_than_skipped(
        self, tmp_path: Path
    ) -> None:
        broken = tmp_path / "metrics.json"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(NumericContractError):
            structure_digest(broken)

    def test_an_unsupported_suffix_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "estimator.onnx"
        target.write_bytes(b"\x00")
        with pytest.raises(NumericContractError, match="cannot"):
            structure_digest(target)


# ---------------------------------------------------------------------------
# 9-10. The other two classifications are unchanged
# ---------------------------------------------------------------------------


class TestTheOtherClassificationsAreUnchanged:
    """DEC-105 and DEC-104 keep doing exactly what they did."""

    def test_a_joblib_stays_execution_specific(self, tmp_path: Path) -> None:
        run = tmp_path / "experiments" / "baseline-engagement_class"
        (run / "models").mkdir(parents=True)
        target = run / "models" / "rf-fold0.joblib"
        target.write_bytes(b"\x80\x05\x00")
        classification = classify([run], tmp_path)
        assert list(classification.execution_specific) == [
            "experiments/baseline-engagement_class/models/rf-fold0.joblib"
        ]
        assert classification.cpu_numeric == (), (
            "a pickle has no readable structure; DEC-105 still governs it"
        )
        assert is_execution_specific("models/rf-fold0.joblib")

    def test_joblib_bytes_changing_does_not_move_portable_identity(
        self, tmp_path: Path
    ) -> None:
        run = tmp_path / "experiments" / "baseline-engagement_class"
        (run / "models").mkdir(parents=True)
        target = run / "models" / "rf-fold0.joblib"
        target.write_bytes(b"\x80\x05\x00")
        first = classify([run], tmp_path)
        target.write_bytes(b"\x80\x05\x01")
        second = classify([run], tmp_path)
        assert first.deterministic == second.deterministic
        assert first.cpu_numeric == second.cpu_numeric
        assert first.integrity[0].sha256 != second.integrity[0].sha256

    def test_a_deterministic_artifact_changing_still_fails_exact_identity(
        self, tmp_path: Path
    ) -> None:
        # splits.json is portable deterministic and must stay that way: no
        # tolerance, no structure digest, just bytes.
        run = tmp_path / "experiments" / "baseline-engagement_class"
        run.mkdir(parents=True)
        target = run / "splits.json"
        target.write_text('{"folds": 3}', encoding="utf-8")
        first = classify([run], tmp_path)
        target.write_text('{"folds": 4}', encoding="utf-8")
        second = classify([run], tmp_path)
        assert first.deterministic[0].sha256 != second.deterministic[0].sha256

    def test_a_volatile_document_is_still_never_checksummed(
        self, tmp_path: Path
    ) -> None:
        run = tmp_path / "experiments" / "baseline-engagement_class"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text('{"run_id": "x"}', encoding="utf-8")
        classification = classify([run], tmp_path)
        assert list(classification.volatile) == [
            "experiments/baseline-engagement_class/manifest.json"
        ]
        assert classification.deterministic == ()
        assert classification.cpu_numeric == ()


# ---------------------------------------------------------------------------
# The schema refuses the mistake even if classification were bypassed
# ---------------------------------------------------------------------------


class TestTheSchemaRefusesMisclassification:
    def test_a_file_cannot_be_both_exact_and_cpu_numeric(self) -> None:
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
                cpu_dependent_numeric_artifacts=(
                    CpuDependentNumericArtifact(
                        path="experiments/baseline/metrics.json",
                        structure_sha256="b" * 64,
                        excluded_from_portable_identity="why",
                        atol=1e-6,
                        rtol=1e-6,
                        numeric_tolerance="spec",
                    ),
                ),
                engagevr_version="0.1.0",
                python_series="3.12",
                is_synthetic=True,
                scientific_evaluation_eligible=False,
                disclaimers=(SOFTWARE_SELF_CHECK_BANNER,),
            )

    def test_a_serialized_estimator_cannot_be_cpu_numeric(self) -> None:
        with pytest.raises(ValueError, match="serialized Python estimator"):
            CpuDependentNumericArtifact(
                path="experiments/baseline/models/rf.joblib",
                structure_sha256="b" * 64,
                excluded_from_portable_identity="why",
                atol=1e-6,
                rtol=1e-6,
                numeric_tolerance="spec",
            )

    def test_a_cpu_numeric_artifact_records_no_size(self) -> None:
        # A one-ULP change alters a float's decimal rendering length, so
        # the file size is as CPU-dependent as the bytes.
        assert "size_bytes" not in CpuDependentNumericArtifact.model_fields

    def test_a_cpu_numeric_artifact_must_state_its_tolerance(self) -> None:
        with pytest.raises(ValueError):
            CpuDependentNumericArtifact(
                path="experiments/baseline/metrics.json",
                structure_sha256="b" * 64,
                excluded_from_portable_identity="why",
                numeric_tolerance="",
            )


# ---------------------------------------------------------------------------
# The stage identity carries the structure digest, and only that
# ---------------------------------------------------------------------------


class TestStageIdentityUsesStructure:
    def _record(self, structure: str) -> DeterministicStageRecord:
        return DeterministicStageRecord(
            stage_name="baseline",
            stage_kind="experiment_run",
            command="c",
            logical_identity="run_id:x",
            cpu_dependent_numeric_artifacts=(
                CpuDependentNumericArtifact(
                    path="experiments/baseline/metrics.json",
                    structure_sha256=structure,
                    excluded_from_portable_identity="why",
                    atol=1e-6,
                    rtol=1e-6,
                    numeric_tolerance="spec",
                ),
            ),
            engagevr_version="0.1.0",
            python_series="3.12",
            is_synthetic=True,
            scientific_evaluation_eligible=False,
            disclaimers=(SOFTWARE_SELF_CHECK_BANNER,),
        )

    def test_a_changed_structure_changes_the_stage_digest(self) -> None:
        assert record_digest(self._record("a" * 64)) != record_digest(
            self._record("b" * 64)
        )

    def test_the_same_structure_gives_the_same_stage_digest(self) -> None:
        assert record_digest(self._record("a" * 64)) == record_digest(
            self._record("a" * 64)
        )


class TestTheToleranceIsDeclaredAndJustified:
    def test_the_tolerance_is_well_above_the_measured_deviation(self) -> None:
        assert NUMERIC_PORTABILITY_ATOL > MEASURED_CROSS_KERNEL_DEVIATION * 10, (
            "a tolerance at the measured noise floor would be flaky"
        )

    def test_the_tolerance_is_not_so_wide_it_hides_a_regression(self) -> None:
        # A probability reported to six decimals must still be checked at
        # six decimals. Anything looser would pass a real change.
        assert NUMERIC_PORTABILITY_ATOL <= 1e-6
        assert NUMERIC_PORTABILITY_RTOL <= 1e-6

    def test_the_tolerance_describes_itself_as_engineering_not_science(self) -> None:
        described = DEFAULT_TOLERANCE.describe()
        assert "portability" in described
        assert "not a" in described and "scientific" in described


class TestRealPipelineArtifactsAreCovered:
    """The ten artifacts that actually moved in CI run 34403534631."""

    OBSERVED: tuple[str, ...] = tuple(sorted(CPU_DEPENDENT_NUMERIC_REASONS))

    @pytest.mark.parametrize("path", OBSERVED)
    def test_each_observed_artifact_is_classified(self, path: str) -> None:
        assert is_cpu_dependent_numeric(path)

    @pytest.mark.parametrize(
        "name",
        (
            "splits.json",
            "calibration.json",
            "ablations.json",
            "feature_catalog.json",
            "coverage_curve.json",
            "uncertainty_config.json",
        ),
    )
    def test_artifacts_that_did_not_move_stay_exact(self, name: str) -> None:
        # Measured across three OpenBLAS kernels and a clean Ubuntu
        # container: byte-identical every time. They keep the strict rule.
        for stage in ("baseline-engagement_class", "uncertainty-engagement_class"):
            assert not is_cpu_dependent_numeric(f"experiments/{stage}/{name}")


class TestContainerCellsAreHandled:
    """A list-valued cell is compared elementwise, not by truthiness."""

    def _frame(self, path: Path, reasons: list[Any], width: Any) -> Path:
        pd.DataFrame(
            {
                "decision": ["eligible", "blocked"],
                "reasons": reasons,
                "maximum_interval_width": width,
            }
        ).to_parquet(path, index=False)
        return path

    def test_an_empty_array_cell_does_not_raise(self, tmp_path: Path) -> None:
        a = self._frame(tmp_path / "a.parquet", [[], ["x"]], [None, 0.5])
        assert structure_digest(a)
        assert compare_artifacts(a, a) == []

    def test_a_changed_reason_code_fails(self, tmp_path: Path) -> None:
        a = self._frame(tmp_path / "a.parquet", [[], ["x"]], [None, 0.5])
        b = self._frame(tmp_path / "b.parquet", [[], ["y"]], [None, 0.5])
        assert structure_digest(a) != structure_digest(b)
        assert compare_artifacts(a, b)

    def test_a_float_inside_an_object_column_gets_the_tolerance(
        self, tmp_path: Path
    ) -> None:
        # maximum_interval_width is an object column of None and float in
        # the real adaptation_gate.parquet. Comparing it exactly would put
        # CPU-dependent values back under an exact test.
        a = self._frame(tmp_path / "a.parquet", [[], ["x"]], [None, 0.5])
        b = self._frame(tmp_path / "b.parquet", [[], ["x"]], [None, 0.5 + 1e-12])
        assert compare_artifacts(a, b) == []
        assert sha256_bytes(a) != sha256_bytes(b)

    def test_a_float_inside_an_object_column_still_fails_beyond_tolerance(
        self, tmp_path: Path
    ) -> None:
        a = self._frame(tmp_path / "a.parquet", [[], ["x"]], [None, 0.5])
        b = self._frame(tmp_path / "b.parquet", [[], ["x"]], [None, 0.9])
        assert compare_artifacts(a, b)
