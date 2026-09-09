"""Distribution-shift diagnostics.

The properties under test are the ones that would let a diagnostic
mislead if they failed:

- identical inputs report no shift, and a deliberately shifted feature
  reports one;
- an absent, empty, or too-thin column is reported *unavailable*, never
  as zero shift, because zero means "these agree";
- a target column never takes part, so no statistic can be computed from
  a label;
- the report is deterministic and carries its own provenance;
- the vocabulary never claims concept drift, model failure, or anything
  about a person.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from engagevr.config import DriftThresholdSettings, load_config
from engagevr.mlops.drift import (
    DriftError,
    compare_datasets,
    compare_predictions,
    exceeding_summary,
    population_stability_index,
    standardized_mean_difference,
    total_variation_distance,
)
from engagevr.schemas.mlops import (
    DriftMethod,
    DriftReportKind,
    DriftStatus,
)


@pytest.fixture
def thresholds() -> DriftThresholdSettings:
    return load_config().mlops.drift


def write_table(path: Path, columns: dict[str, list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def numeric_column(values: list[float | None]) -> list[object]:
    return list(values)


def result_for(report: object, name: str) -> object:
    for entry in report.results:  # type: ignore[attr-defined]
        if entry.feature_name == name:
            return entry
    raise AssertionError(f"no result for {name!r}")


def statistic_for(result: object, method: DriftMethod) -> object:
    for statistic in result.statistics:  # type: ignore[attr-defined]
        if statistic.method is method:
            return statistic
    raise AssertionError(f"no {method.value} statistic")


class TestFormulas:
    def test_standardized_mean_difference_is_zero_for_identical_samples(
        self,
    ) -> None:
        values = np.linspace(0.0, 1.0, 50)
        assert standardized_mean_difference(values, values) == pytest.approx(0.0)

    def test_standardized_mean_difference_is_none_without_variance(self) -> None:
        constant = np.full(50, 0.5)
        assert standardized_mean_difference(constant, constant) is None

    def test_standardized_mean_difference_sign_follows_the_current_dataset(
        self,
    ) -> None:
        reference = np.linspace(0.0, 1.0, 50)
        assert standardized_mean_difference(reference, reference + 1.0) > 0
        assert standardized_mean_difference(reference, reference - 1.0) < 0

    def test_psi_is_zero_for_identical_samples(self) -> None:
        values = np.linspace(0.0, 1.0, 200)
        assert population_stability_index(values, values, bins=10) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_psi_grows_when_mass_moves(self) -> None:
        reference = np.linspace(0.0, 1.0, 200)
        near = population_stability_index(reference, reference + 0.05, bins=10)
        far = population_stability_index(reference, reference + 0.5, bins=10)
        assert near is not None and far is not None
        assert far > near > 0.0

    def test_psi_is_finite_with_empty_bins(self) -> None:
        reference = np.linspace(0.0, 1.0, 200)
        disjoint = np.linspace(5.0, 6.0, 200)
        value = population_stability_index(reference, disjoint, bins=10)
        assert value is not None
        assert math.isfinite(value)

    def test_psi_is_none_for_a_constant_reference(self) -> None:
        constant = np.full(200, 0.5)
        assert population_stability_index(
            constant, np.linspace(0, 1, 200), bins=10
        ) is (None)

    def test_total_variation_is_zero_for_identical_shares(self) -> None:
        values = ["a"] * 30 + ["b"] * 70
        assert total_variation_distance(values, values) == pytest.approx(0.0)

    def test_total_variation_is_one_for_disjoint_categories(self) -> None:
        assert total_variation_distance(["a"] * 10, ["b"] * 10) == pytest.approx(1.0)

    def test_total_variation_counts_a_new_category(self) -> None:
        assert total_variation_distance(["a"] * 10, ["a"] * 5 + ["b"] * 5) == (
            pytest.approx(0.5)
        )


class TestIdenticalInputs:
    def test_a_dataset_compared_with_itself_reports_no_shift(
        self, m5_dataset: Path, thresholds: DriftThresholdSettings
    ) -> None:
        report = compare_datasets(m5_dataset, m5_dataset, thresholds=thresholds)
        assert report.features_compared_count > 0
        assert report.features_exceeding_count == 0
        assert exceeding_summary(report) == ()

    def test_every_computed_statistic_is_zero(
        self, m5_dataset: Path, thresholds: DriftThresholdSettings
    ) -> None:
        report = compare_datasets(m5_dataset, m5_dataset, thresholds=thresholds)
        for result in report.results:
            if result.status is not DriftStatus.COMPUTED:
                continue
            for statistic in result.statistics:
                if statistic.statistic is not None:
                    assert statistic.statistic == pytest.approx(0.0, abs=1e-9)


class TestShiftedInput:
    def test_a_deliberately_shifted_feature_is_detected(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        rows = 200
        stable = list(np.linspace(0.0, 1.0, rows))
        reference = write_table(
            tmp_path / "reference.parquet",
            {"feat__stable": stable, "feat__moved": stable},
        )
        current = write_table(
            tmp_path / "current.parquet",
            {
                "feat__stable": stable,
                "feat__moved": [value + 5.0 for value in stable],
            },
        )
        report = compare_datasets(reference, current, thresholds=thresholds)

        moved = result_for(report, "feat__moved")
        assert moved.exceeded_methods  # type: ignore[attr-defined]
        assert DriftMethod.KOLMOGOROV_SMIRNOV in moved.exceeded_methods  # type: ignore[attr-defined]
        assert DriftMethod.STANDARDIZED_MEAN_DIFFERENCE in moved.exceeded_methods  # type: ignore[attr-defined]

        stable_result = result_for(report, "feat__stable")
        assert stable_result.exceeded_methods == ()  # type: ignore[attr-defined]

    def test_a_missingness_change_is_detected_and_named_availability(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        rows = 200
        reference = write_table(
            tmp_path / "reference.parquet",
            {"feat__x": numeric_column([0.5] * rows)},
        )
        current = write_table(
            tmp_path / "current.parquet",
            {"feat__x": numeric_column([0.5] * 100 + [None] * 100)},
        )
        report = compare_datasets(reference, current, thresholds=thresholds)
        statistic = statistic_for(
            result_for(report, "feat__x"), DriftMethod.MISSINGNESS_RATE_DIFFERENCE
        )
        assert statistic.statistic == pytest.approx(0.5)  # type: ignore[attr-defined]
        assert statistic.exceeded is True  # type: ignore[attr-defined]
        assert "NEVER disengagement" in statistic.interpretation  # type: ignore[attr-defined]


class TestUnavailability:
    def test_a_column_missing_from_the_current_dataset_is_unavailable(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        values = list(np.linspace(0.0, 1.0, 100))
        reference = write_table(
            tmp_path / "reference.parquet",
            {"feat__a": values, "feat__gone": values},
        )
        current = write_table(tmp_path / "current.parquet", {"feat__a": values})
        report = compare_datasets(reference, current, thresholds=thresholds)

        gone = result_for(report, "feat__gone")
        assert gone.status is DriftStatus.UNAVAILABLE_MISSING_IN_CURRENT  # type: ignore[attr-defined]
        assert gone.statistics == ()  # type: ignore[attr-defined]
        assert gone.exceeded_methods == ()  # type: ignore[attr-defined]
        assert "not zero shift" in (gone.unavailable_reason or "")  # type: ignore[attr-defined]
        assert "feat__gone" in report.unavailable_features

    def test_a_column_missing_from_the_reference_is_unavailable(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        values = list(np.linspace(0.0, 1.0, 100))
        reference = write_table(tmp_path / "reference.parquet", {"feat__a": values})
        current = write_table(
            tmp_path / "current.parquet", {"feat__a": values, "feat__new": values}
        )
        report = compare_datasets(reference, current, thresholds=thresholds)
        new = result_for(report, "feat__new")
        assert new.status is DriftStatus.UNAVAILABLE_MISSING_IN_REFERENCE  # type: ignore[attr-defined]

    def test_an_all_missing_feature_is_unavailable_not_zero(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        rows = 100
        reference = write_table(
            tmp_path / "reference.parquet",
            {"feat__empty": numeric_column([None] * rows)},
        )
        current = write_table(
            tmp_path / "current.parquet",
            {"feat__empty": numeric_column([None] * rows)},
        )
        report = compare_datasets(reference, current, thresholds=thresholds)
        result = result_for(report, "feat__empty")
        distribution = statistic_for(result, DriftMethod.KOLMOGOROV_SMIRNOV)
        assert distribution.status is DriftStatus.UNAVAILABLE_ALL_VALUES_MISSING  # type: ignore[attr-defined]
        assert distribution.statistic is None  # type: ignore[attr-defined]
        assert distribution.exceeded is None  # type: ignore[attr-defined]
        # The missingness rate is still computable and is legitimately zero:
        # the measurement was absent equally often on both sides.
        availability = statistic_for(result, DriftMethod.MISSINGNESS_RATE_DIFFERENCE)
        assert availability.statistic == pytest.approx(0.0)  # type: ignore[attr-defined]

    def test_too_few_samples_is_unavailable_not_zero(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        few = [0.1, 0.2, 0.3]
        reference = write_table(tmp_path / "reference.parquet", {"feat__x": few})
        current = write_table(tmp_path / "current.parquet", {"feat__x": few})
        report = compare_datasets(reference, current, thresholds=thresholds)
        statistic = statistic_for(
            result_for(report, "feat__x"), DriftMethod.KOLMOGOROV_SMIRNOV
        )
        assert statistic.status is DriftStatus.UNAVAILABLE_INSUFFICIENT_SAMPLES  # type: ignore[attr-defined]
        assert statistic.statistic is None  # type: ignore[attr-defined]
        assert str(thresholds.minimum_samples) in (statistic.unavailable_reason or "")  # type: ignore[attr-defined]

    def test_a_constant_feature_reports_zero_variance_not_a_number(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        constant = [0.5] * 100
        reference = write_table(tmp_path / "reference.parquet", {"feat__c": constant})
        current = write_table(tmp_path / "current.parquet", {"feat__c": constant})
        report = compare_datasets(reference, current, thresholds=thresholds)
        result = result_for(report, "feat__c")
        smd = statistic_for(result, DriftMethod.STANDARDIZED_MEAN_DIFFERENCE)
        assert smd.status is DriftStatus.UNAVAILABLE_ZERO_VARIANCE  # type: ignore[attr-defined]
        psi = statistic_for(result, DriftMethod.POPULATION_STABILITY_INDEX)
        assert psi.status is DriftStatus.UNAVAILABLE_ZERO_VARIANCE  # type: ignore[attr-defined]

    def test_a_type_mismatch_is_unavailable(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        reference = write_table(
            tmp_path / "reference.parquet", {"feat__mixed": [0.5] * 100}
        )
        current = write_table(
            tmp_path / "current.parquet", {"feat__mixed": ["pos"] * 100}
        )
        report = compare_datasets(reference, current, thresholds=thresholds)
        result = result_for(report, "feat__mixed")
        assert result.status is DriftStatus.UNAVAILABLE_TYPE_MISMATCH  # type: ignore[attr-defined]

    def test_a_missing_dataset_is_an_error_not_an_empty_report(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        present = write_table(tmp_path / "a.parquet", {"feat__x": [0.1] * 40})
        with pytest.raises(DriftError, match="does not exist"):
            compare_datasets(
                present, tmp_path / "absent.parquet", thresholds=thresholds
            )
        with pytest.raises(DriftError, match="does not exist"):
            compare_datasets(
                tmp_path / "absent.parquet", present, thresholds=thresholds
            )


class TestNoTargetLeakage:
    def test_no_target_column_is_ever_compared(
        self,
        m5_dataset: Path,
        m10_shifted_dataset: Path,
        thresholds: DriftThresholdSettings,
    ) -> None:
        report = compare_datasets(
            m5_dataset, m10_shifted_dataset, thresholds=thresholds
        )
        for name in report.compared_features:
            assert not name.startswith("target__")
            assert not name.startswith("target_meta__")
        for result in report.results:
            assert not result.feature_name.startswith("target")

    def test_target_columns_are_excluded_with_a_stated_reason(
        self,
        m5_dataset: Path,
        m10_shifted_dataset: Path,
        thresholds: DriftThresholdSettings,
    ) -> None:
        report = compare_datasets(
            m5_dataset, m10_shifted_dataset, thresholds=thresholds
        )
        targets = [n for n in report.excluded_features if n.startswith("target__")]
        assert targets
        for name in targets:
            assert "leakage" in report.excluded_features[name]

    def test_identity_columns_are_excluded(
        self,
        m5_dataset: Path,
        m10_shifted_dataset: Path,
        thresholds: DriftThresholdSettings,
    ) -> None:
        report = compare_datasets(
            m5_dataset, m10_shifted_dataset, thresholds=thresholds
        )
        for name in ("window_id", "subject_id", "session_id"):
            assert name in report.excluded_features


class TestDeterminismAndProvenance:
    def test_the_report_fingerprint_is_deterministic(
        self,
        m5_dataset: Path,
        m10_shifted_dataset: Path,
        thresholds: DriftThresholdSettings,
    ) -> None:
        first = compare_datasets(m5_dataset, m10_shifted_dataset, thresholds=thresholds)
        second = compare_datasets(
            m5_dataset, m10_shifted_dataset, thresholds=thresholds
        )
        assert first.report_fingerprint == second.report_fingerprint
        # The wall clock takes no part, because the report contains none:
        # two reports built at different moments are byte-identical.
        assert "Excludes wall-clock time" in first.report_fingerprint_inputs
        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_the_report_contains_no_wall_clock_anywhere(
        self, m5_dataset: Path, thresholds: DriftThresholdSettings
    ) -> None:
        report = compare_datasets(m5_dataset, m5_dataset, thresholds=thresholds)
        rendered = report.model_dump_json()
        assert "created_at_utc" not in rendered
        assert str(datetime.now(UTC).year) not in rendered

    def test_swapping_the_two_sides_changes_the_fingerprint(
        self,
        m5_dataset: Path,
        m10_shifted_dataset: Path,
        thresholds: DriftThresholdSettings,
    ) -> None:
        forward = compare_datasets(
            m5_dataset, m10_shifted_dataset, thresholds=thresholds
        )
        backward = compare_datasets(
            m10_shifted_dataset, m5_dataset, thresholds=thresholds
        )
        assert forward.report_fingerprint != backward.report_fingerprint

    def test_both_sides_are_named_explicitly(
        self,
        m5_dataset: Path,
        m10_shifted_dataset: Path,
        thresholds: DriftThresholdSettings,
    ) -> None:
        report = compare_datasets(
            m5_dataset, m10_shifted_dataset, thresholds=thresholds
        )
        assert report.reference.role == "reference"
        assert report.current.role == "current"
        assert report.reference.dataset_fingerprint
        assert report.current.dataset_fingerprint
        assert (
            report.reference.dataset_fingerprint != report.current.dataset_fingerprint
        )
        assert report.reference.row_count > 0
        assert report.current.row_count > 0

    def test_synthetic_provenance_is_preserved_and_never_inflated(
        self,
        m5_dataset: Path,
        m10_shifted_dataset: Path,
        thresholds: DriftThresholdSettings,
    ) -> None:
        report = compare_datasets(
            m5_dataset, m10_shifted_dataset, thresholds=thresholds
        )
        assert report.is_synthetic is True
        assert report.scientific_evaluation_eligible is False
        assert report.reference.scientific_evaluation_eligible is False
        assert report.current.scientific_evaluation_eligible is False

    def test_the_thresholds_and_method_versions_are_recorded(
        self, m5_dataset: Path, thresholds: DriftThresholdSettings
    ) -> None:
        report = compare_datasets(m5_dataset, m5_dataset, thresholds=thresholds)
        assert report.thresholds == thresholds.as_mapping()
        assert report.minimum_samples == thresholds.minimum_samples
        assert report.histogram_bin_count == thresholds.histogram_bins
        assert "ENGINEERING DIAGNOSTIC DEFAULTS" in report.threshold_policy
        for result in report.results:
            for statistic in result.statistics:
                assert statistic.method_version


class TestTerminology:
    def test_the_report_kind_is_a_distribution_shift(
        self, m5_dataset: Path, thresholds: DriftThresholdSettings
    ) -> None:
        report = compare_datasets(m5_dataset, m5_dataset, thresholds=thresholds)
        assert report.report_kind is DriftReportKind.FEATURE_DISTRIBUTION_SHIFT

    def test_no_document_claims_concept_drift_or_a_failure(
        self, m5_dataset: Path, thresholds: DriftThresholdSettings
    ) -> None:
        report = compare_datasets(m5_dataset, m5_dataset, thresholds=thresholds)
        rendered = report.model_dump_json().lower()
        assert "not concept drift" in rendered
        for forbidden in (
            "participant drift",
            "cognitive decline",
            "disengagement drift",
            "psychological change",
            "model failed",
        ):
            assert forbidden not in rendered

    def test_the_interpretation_denies_every_overclaim(
        self, m5_dataset: Path, thresholds: DriftThresholdSettings
    ) -> None:
        note = compare_datasets(
            m5_dataset, m5_dataset, thresholds=thresholds
        ).interpretation.lower()
        assert "is not model degradation" in note
        assert "is not concept drift" in note
        assert "engineering diagnostic default" in note


class TestPredictionShift:
    def test_a_prediction_comparison_is_never_called_concept_drift(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        reference = write_table(
            tmp_path / "reference.parquet",
            {"predicted_value": ["low"] * 60 + ["high"] * 40},
        )
        current = write_table(
            tmp_path / "current.parquet",
            {"predicted_value": ["low"] * 20 + ["high"] * 80},
        )
        report = compare_predictions(reference, current, thresholds=thresholds)
        assert report.report_kind is DriftReportKind.PREDICTION_DISTRIBUTION_SHIFT
        assert "concept_drift" not in report.model_dump_json()
        result = result_for(report, "predicted_value")
        assert DriftMethod.CATEGORICAL_TOTAL_VARIATION in result.exceeded_methods  # type: ignore[attr-defined]

    def test_a_numeric_prediction_column_uses_the_numeric_methods(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        values = list(np.linspace(0.0, 1.0, 100))
        reference = write_table(
            tmp_path / "reference.parquet", {"predicted_value": values}
        )
        current = write_table(
            tmp_path / "current.parquet",
            {"predicted_value": [v + 0.4 for v in values]},
        )
        report = compare_predictions(reference, current, thresholds=thresholds)
        result = result_for(report, "predicted_value")
        methods = {s.method for s in result.statistics}  # type: ignore[attr-defined]
        assert DriftMethod.KOLMOGOROV_SMIRNOV in methods
        assert DriftMethod.POPULATION_STABILITY_INDEX in methods

    def test_a_missing_prediction_column_is_an_error(
        self, tmp_path: Path, thresholds: DriftThresholdSettings
    ) -> None:
        table = write_table(tmp_path / "a.parquet", {"something_else": [1.0] * 40})
        with pytest.raises(DriftError, match="no 'predicted_value' column"):
            compare_predictions(table, table, thresholds=thresholds)
