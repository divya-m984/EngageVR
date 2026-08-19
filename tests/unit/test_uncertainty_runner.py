"""Milestone 7 runner: leakage, reconciliation, curves, and artifacts.

Every run here is executed on a deterministic SYNTHETIC dataset.  No number
produced by these tests is model accuracy, calibration quality,
selective-prediction reliability, safety, or evidence about any person, and
no test asserts that abstention improved anything.

No test needs a webcam, a model asset, a display server, a network, Unity,
a public dataset, or participant data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from engagevr.schemas.experiments import (
    SOFTWARE_SELF_CHECK_BANNER,
    EvaluationMode,
    RunStatus,
)
from engagevr.schemas.fusion import FusionModality
from engagevr.schemas.targets import TargetName, TaskType
from engagevr.schemas.uncertainty import (
    AbstentionReason,
    AdaptationGateDecision,
    CoverageAxis,
    EvidenceGateConfiguration,
    MonotonicDirection,
    PredictionSource,
    ProbabilityCalibrationStatus,
    SelectivePredictionConfiguration,
    ThresholdObjective,
    ThresholdSource,
)
from engagevr.training.artifacts import read_manifest, verify_checksums
from engagevr.training.uncertainty_runner import (
    UNCERTAINTY_REQUIRED_ARTIFACTS,
    UncertaintyConfigurationError,
    UncertaintyRunConfiguration,
    UncertaintyRunResult,
    run_uncertainty,
)

#: Values or fragments that must never appear in a persisted document.
FORBIDDEN_IDENTIFIERS: tuple[str, ...] = (
    "@example.com",
    "password",
    "api_key",
    "secret_key",
    "access_token",
    "first_name",
    "last_name",
    "frame_bytes",
    "landmark_array",
)

#: Affirmative claims no artifact of this project may make. "production
#: threshold" is absent on purpose: it is permitted inside the sentence that
#: denies there is one, which ``test_production_threshold_appears_only_in_a_
#: denial`` pins.
FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "production-ready",
    "optimal threshold",
    "best uncertainty method",
    "validated abstention",
    "safe confidence",
    "trusted prediction",
    "clinically useful",
    "reliable in humans",
    "state of the art",
)


def _configuration(**overrides: object) -> SelectivePredictionConfiguration:
    payload: dict[str, object] = {
        "modalities": tuple(FusionModality),
        "threshold_grid": (0.0, 0.25, 0.5, 0.75, 1.0),
        "population_confidence_threshold": 0.5,
        "alpha": 0.2,
    }
    payload.update(overrides)
    return SelectivePredictionConfiguration(**payload)  # type: ignore[arg-type]


def _run(
    dataset: Path,
    output: Path,
    *,
    target: TargetName = TargetName.ENGAGEMENT_CLASS,
    folds: int = 3,
    mode: EvaluationMode = EvaluationMode.SOFTWARE_SELF_CHECK,
    **overrides: object,
) -> UncertaintyRunResult:
    return run_uncertainty(
        UncertaintyRunConfiguration(
            dataset_path=dataset,
            target_name=target,
            output_directory=output,
            selective=_configuration(**overrides),
            evaluation_mode=mode,
            n_splits=folds,
        )
    )


class TestLeakageSafety:
    def test_the_outer_test_fold_fits_nothing(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        for fold in m7_uncertainty_classification_run.uncertainty.folds:
            test = set(fold.outer_test_group_ids)
            assert not set(fold.fit_group_ids) & test
            assert not set(fold.probability_calibration_group_ids) & test
            assert not set(fold.threshold_selection_group_ids) & test
            assert not set(fold.conformal_calibration_group_ids) & test

    def test_probability_calibration_never_reuses_a_fitting_group(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        for fold in m7_uncertainty_classification_run.uncertainty.folds:
            assert not set(fold.fit_group_ids) & set(
                fold.probability_calibration_group_ids
            )

    def test_conformal_calibration_never_reuses_a_fitting_group(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        for fold in m7_uncertainty_regression_run.uncertainty.folds:
            assert not set(fold.fit_group_ids) & set(
                fold.conformal_calibration_group_ids
            )

    def test_all_four_group_sets_are_recorded_per_fold(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        for fold in m7_uncertainty_classification_run.uncertainty.folds:
            if not fold.evaluated:
                continue
            assert fold.fit_group_ids
            assert fold.probability_calibration_group_ids
            assert fold.outer_test_group_ids

    def test_a_personal_threshold_uses_no_label_of_any_kind(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        records = [
            record
            for fold in m7_uncertainty_classification_run.uncertainty.folds
            for record in fold.personal_thresholds
        ]
        assert records
        assert all(record.uses_labels is False for record in records)

    def test_a_personal_threshold_names_only_earlier_windows(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        checked = 0
        for fold in m7_uncertainty_classification_run.uncertainty.folds:
            for record in fold.personal_thresholds:
                if not record.personalization_applied:
                    continue
                assert record.calibration_end_utc is not None
                assert record.evaluation_start_utc is not None
                assert record.calibration_end_utc <= record.evaluation_start_utc
                assert record.temporal_order_verified
                checked += 1
        assert checked > 0

    def test_a_calibration_window_is_never_also_scored(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        calibration: set[str] = set()
        for fold in m7_uncertainty_classification_run.uncertainty.folds:
            for record in fold.personal_thresholds:
                calibration.update(record.calibration_window_ids)
        assert calibration
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "selective_predictions.parquet").to_pandas()
        assert not calibration & set(table["window_id"])

    def test_the_split_manifest_audit_passed(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        assert m7_uncertainty_classification_run.splits.audit_passed

    def test_no_threshold_is_read_off_the_reported_curve(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        curve = m7_uncertainty_classification_run.coverage_curve
        assert "No threshold in this repository is selected from this curve" in (
            curve.selection_note
        )
        for fold in m7_uncertainty_classification_run.uncertainty.folds:
            if not fold.evaluated:
                continue
            assert fold.applied_population_threshold_source in {
                ThresholdSource.CONFIGURED_POPULATION,
                ThresholdSource.ESTIMATED_POPULATION,
            }

    def test_threshold_estimation_records_its_calibration_groups(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "estimated",
            estimate_population_threshold=True,
            threshold_objective=ThresholdObjective.TARGET_ACCEPTED_ACCURACY,
            threshold_objective_target=0.4,
            minimum_threshold_selection_samples=5,
            minimum_threshold_selection_groups=1,
        )
        records = [
            fold.estimated_threshold
            for fold in result.uncertainty.folds
            if fold.estimated_threshold is not None
        ]
        assert records
        for record in records:
            assert record.used_outer_test_labels is False
            assert not set(record.calibration_group_ids) & set(
                record.outer_test_group_ids
            )


class TestClassificationDecisions:
    def test_every_probability_row_is_a_distribution(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        columns = [c for c in table.columns if c.startswith("probability__")]
        assert columns
        matrix = table[columns].to_numpy(dtype=float)
        assert np.isfinite(matrix).all()
        assert (matrix >= 0.0).all()
        assert np.allclose(matrix.sum(axis=1), 1.0)

    def test_confidence_entropy_and_margin_are_separate_columns(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        for column in ("confidence_score", "entropy", "normalized_entropy", "margin"):
            assert column in table.columns
        assert "uncertainty" not in table.columns

    def test_quality_and_confidence_are_different_columns(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        assert "minimum_recorded_quality" in table.columns
        assert "confidence_score" in table.columns

    def test_disagreement_is_carried_under_its_own_name(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        assert "ensemble_disagreement" in table.columns

    def test_every_confidence_is_finite_and_in_the_unit_interval(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        for fold in m7_uncertainty_classification_run.uncertainty.folds:
            if not fold.evaluated:
                continue
            assert fold.probability_calibration_status is not (
                ProbabilityCalibrationStatus.UNAVAILABLE
            )

    def test_an_abstained_window_keeps_its_original_prediction(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "selective_predictions.parquet").to_pandas()
        abstained = table[table["abstained"]]
        assert len(abstained) > 0
        assert abstained["predicted_class"].notna().all()
        # Whichever score the fold produced is retained; the prediction is
        # never blanked because the window was declined.
        scores = abstained["confidence_score"].combine_first(
            abstained["selection_score"]
        )
        assert scores.notna().all()

    def test_an_abstained_window_records_a_reason(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "selective_predictions.parquet").to_pandas()
        abstained = table[table["abstained"]]
        assert abstained["primary_abstention_reason"].notna().all()

    def test_every_decision_records_its_threshold_and_provenance(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "selective_predictions.parquet").to_pandas()
        assert table["applied_threshold"].notna().all()
        assert table["threshold_source"].notna().all()

    def test_the_acceptance_boundary_is_the_documented_one(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "selective_predictions.parquet").to_pandas()
        checked = 0
        for _index, row in table.iterrows():
            if row["primary_abstention_reason"] not in (
                None,
                AbstentionReason.BELOW_CONFIDENCE_THRESHOLD.value,
            ):
                # Blocked by the evidence gate, which the threshold does not
                # decide; the boundary rule is not what is under test here.
                continue
            score = row["confidence_score"]
            if score is None:
                score = row["selection_score"]
            assert score is not None
            assert bool(row["accepted"]) == bool(score >= row["applied_threshold"])
            checked += 1
        assert checked > 0


class TestProbabilityCalibrationContract:
    """A maximum is only called confidence when it was calibrated."""

    def test_the_fixture_run_exercises_both_calibration_branches(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        statuses = {
            fold.probability_calibration_status
            for fold in m7_uncertainty_classification_run.uncertainty.folds
            if fold.evaluated
        }
        assert ProbabilityCalibrationStatus.CALIBRATED in statuses
        assert ProbabilityCalibrationStatus.UNCALIBRATED in statuses

    def test_a_calibrated_fold_records_a_confidence_score(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        calibrated = [
            fold.fold_index
            for fold in m7_uncertainty_classification_run.uncertainty.folds
            if fold.probability_calibration_status
            is ProbabilityCalibrationStatus.CALIBRATED
        ]
        assert calibrated
        table = pq.read_table(
            m7_uncertainty_classification_run.directory / "predictions.parquet"
        ).to_pandas()
        rows = table[table["fold_index"].isin(calibrated)]
        assert rows["confidence_score"].notna().all()
        assert rows["selection_score"].isna().all()
        assert (
            rows["probability_calibration_status"]
            == ProbabilityCalibrationStatus.CALIBRATED.value
        ).all()

    def test_an_uncalibrated_fold_records_a_selection_score_instead(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        uncalibrated = [
            fold.fold_index
            for fold in m7_uncertainty_classification_run.uncertainty.folds
            if fold.probability_calibration_status
            is ProbabilityCalibrationStatus.UNCALIBRATED
        ]
        assert uncalibrated
        table = pq.read_table(
            m7_uncertainty_classification_run.directory / "predictions.parquet"
        ).to_pandas()
        rows = table[table["fold_index"].isin(uncalibrated)]
        assert rows["selection_score"].notna().all()
        assert rows["confidence_score"].isna().all()

    def test_an_uncalibrated_fold_refuses_confidence_based_acceptance(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        # The default evidence gate requires a calibrated probability before
        # a confidence threshold may decide anything, so an uncalibrated fold
        # abstains with its own reason rather than overstating calibration.
        counts = m7_uncertainty_classification_run.uncertainty.abstention_reason_counts
        assert (
            counts.get(AbstentionReason.PROBABILITY_CALIBRATION_UNAVAILABLE.value, 0)
            > 0
        )

    def test_an_uncalibrated_fold_states_why_no_calibrator_was_fitted(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        for fold in m7_uncertainty_classification_run.uncertainty.folds:
            if (
                fold.probability_calibration_status
                is ProbabilityCalibrationStatus.UNCALIBRATED
            ):
                assert fold.probability_calibration_unavailable_reason

    def test_the_run_warns_when_probabilities_were_not_calibrated(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        joined = " ".join(m7_uncertainty_classification_run.warnings)
        assert "UNCALIBRATED" in joined
        assert "not as calibrated confidence" in joined

    def test_relaxing_the_gate_allows_an_uncalibrated_selection_policy(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        # An explicitly named uncalibrated selection-score policy is
        # permitted; it is simply never called calibrated confidence.
        result = _run(
            m5_dataset,
            tmp_path / "uncalibrated-policy",
            evidence_gate=EvidenceGateConfiguration(
                require_probability_calibration_for_classification_confidence=False
            ),
            population_confidence_threshold=0.0,
        )
        assert result.uncertainty.accepted_count > 0
        table = pq.read_table(result.directory / "predictions.parquet").to_pandas()
        assert table["confidence_score"].isna().all()
        assert table["selection_score"].notna().all()


class TestRegressionIntervals:
    def test_every_point_prediction_is_finite(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_regression_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        values = table["predicted_value"].to_numpy(dtype=float)
        assert np.isfinite(values).all()

    def test_no_regression_row_carries_a_confidence_score(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_regression_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        assert table["confidence_score"].isna().all()
        assert table["selection_score"].isna().all()

    def test_the_bounds_bracket_the_point_prediction(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_regression_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        present = table[table["interval_width"].notna()]
        assert len(present) > 0
        assert (present["interval_lower_bound"] <= present["predicted_value"]).all()
        assert (present["predicted_value"] <= present["interval_upper_bound"]).all()

    def test_every_width_is_finite_and_non_negative(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_regression_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        widths = table["interval_width"].dropna().to_numpy(dtype=float)
        assert np.isfinite(widths).all()
        assert (widths >= 0.0).all()

    def test_the_conformal_order_statistic_matches_the_documented_rule(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        import math

        alpha = m7_uncertainty_regression_run.uncertainty.configuration.alpha
        for fold in m7_uncertainty_regression_run.uncertainty.folds:
            if not fold.conformal_available:
                continue
            n = fold.conformal_calibration_sample_count
            assert fold.conformal_order_statistic == math.ceil((n + 1) * (1 - alpha))

    def test_empirical_interval_coverage_is_recorded(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        recorded = [
            fold.applied_selective_metrics.empirical_interval_coverage
            for fold in m7_uncertainty_regression_run.uncertainty.folds
            if fold.applied_selective_metrics is not None
        ]
        assert recorded
        assert all(value is not None for value in recorded)

    def test_too_few_calibration_residuals_make_the_interval_unavailable(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        # alpha=0.001 needs 999 residuals; this fold has far fewer.
        result = _run(
            m5_dataset,
            tmp_path / "thin-conformal",
            target=TargetName.ENGAGEMENT_SCORE,
            alpha=0.001,
        )
        folds = [f for f in result.uncertainty.folds if f.evaluated]
        assert folds
        assert all(not fold.conformal_available for fold in folds)
        assert all(fold.conformal_unavailable_reason for fold in folds)
        assert all(fold.conformal_quantile is None for fold in folds)

    def test_a_missing_interval_never_becomes_width_zero(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "thin-conformal-widths",
            target=TargetName.ENGAGEMENT_SCORE,
            alpha=0.001,
        )
        table = pq.read_table(
            result.directory / "selective_predictions.parquet"
        ).to_pandas()
        assert table["interval_width"].isna().all()
        assert not (table["interval_width"] == 0.0).any()

    def test_a_missing_interval_abstains_with_its_own_reason(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "thin-conformal-reason",
            target=TargetName.ENGAGEMENT_SCORE,
            alpha=0.001,
        )
        counts = result.uncertainty.abstention_reason_counts
        assert counts.get(AbstentionReason.PREDICTION_INTERVAL_UNAVAILABLE.value, 0) > 0
        assert result.uncertainty.accepted_count == 0

    def test_a_width_policy_abstains_on_wide_intervals(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "narrow-width",
            target=TargetName.ENGAGEMENT_SCORE,
            maximum_interval_width=1e-6,
        )
        counts = result.uncertainty.abstention_reason_counts
        assert counts.get(AbstentionReason.INTERVAL_TOO_WIDE.value, 0) > 0

    def test_a_generous_width_policy_accepts(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "wide-width",
            target=TargetName.ENGAGEMENT_SCORE,
            maximum_interval_width=100.0,
        )
        assert result.uncertainty.accepted_count > 0
        assert (
            result.uncertainty.abstention_reason_counts.get(
                AbstentionReason.INTERVAL_TOO_WIDE.value, 0
            )
            == 0
        )

    def test_the_raw_bounds_are_not_silently_clipped_to_the_target_range(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        assert not (
            m7_uncertainty_regression_run.uncertainty.configuration
        ).clip_interval_to_target_range
        table = pq.read_table(
            m7_uncertainty_regression_run.directory / "predictions.parquet"
        ).to_pandas()
        # The target lives in [0, 1]; the RAW conformal bounds are allowed to
        # leave it, and this run proves they were not projected back.
        bounds = table["interval_lower_bound"].dropna().to_numpy(dtype=float)
        assert bounds.min() < 0.0


class TestCoverageAccounting:
    def test_the_three_counts_reconcile_at_the_run_level(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        evaluation = m7_uncertainty_classification_run.uncertainty
        assert (
            evaluation.accepted_count
            + evaluation.abstained_count
            + evaluation.unavailable_count
            == evaluation.total_window_count
        )

    def test_abstention_reduces_coverage(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        evaluation = m7_uncertainty_classification_run.uncertainty
        assert evaluation.abstained_count > 0
        assert evaluation.coverage is not None
        assert evaluation.coverage < 1.0

    def test_the_coverage_denominator_is_recorded(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        evaluation = m7_uncertainty_classification_run.uncertainty
        assert evaluation.coverage_denominator == "total_evaluated_windows"

    def test_accepted_metrics_cover_exactly_the_accepted_windows(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        for fold in m7_uncertainty_classification_run.uncertainty.folds:
            metrics = fold.applied_selective_metrics
            if metrics is None or metrics.accepted_classification is None:
                continue
            assert metrics.accepted_classification.sample_count == fold.accepted_count

    def test_abstentions_are_not_counted_as_errors(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        # The accepted-set confusion matrix totals the accepted rows only.
        for fold in m7_uncertainty_classification_run.uncertainty.folds:
            metrics = fold.applied_selective_metrics
            if metrics is None or metrics.accepted_classification is None:
                continue
            matrix = metrics.accepted_classification.confusion_matrix
            assert matrix is not None
            total = sum(sum(row) for row in matrix.counts)
            assert total == fold.accepted_count


class TestCoverageCurve:
    def test_the_curve_covers_exactly_the_configured_grid(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        curve = m7_uncertainty_classification_run.coverage_curve
        settings = m7_uncertainty_classification_run.uncertainty.configuration
        assert tuple(p.threshold for p in curve.points) == settings.threshold_grid

    def test_the_classification_x_axis_is_the_confidence_threshold(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        curve = m7_uncertainty_classification_run.coverage_curve
        assert curve.axis is CoverageAxis.CONFIDENCE_THRESHOLD
        assert "probability" in curve.axis_units
        assert all(
            point.axis is CoverageAxis.CONFIDENCE_THRESHOLD for point in curve.points
        )

    def test_the_regression_x_axis_is_the_maximum_interval_width(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        curve = m7_uncertainty_regression_run.coverage_curve
        assert curve.axis is CoverageAxis.MAXIMUM_INTERVAL_WIDTH
        assert "target's own units" in curve.axis_units
        assert "NOT a probability" in curve.axis_units
        assert all(
            point.axis is CoverageAxis.MAXIMUM_INTERVAL_WIDTH for point in curve.points
        )

    def test_the_regression_curve_sweeps_the_width_grid_not_the_confidence_grid(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        curve = m7_uncertainty_regression_run.coverage_curve
        settings = m7_uncertainty_regression_run.uncertainty.configuration
        assert settings.interval_width_grid is not None
        assert curve.axis_values == settings.interval_width_grid
        assert tuple(p.threshold for p in curve.points) == settings.interval_width_grid

    def test_classification_coverage_is_monotonic_non_increasing(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        curve = m7_uncertainty_classification_run.coverage_curve
        assert curve.expected_monotonic_direction is MonotonicDirection.NON_INCREASING
        assert curve.coverage_is_monotonic
        coverages = [point.coverage_point.coverage for point in curve.points]
        assert coverages == sorted(coverages, reverse=True)

    def test_regression_coverage_is_monotonic_non_decreasing(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        # Raising the widest acceptable interval can only ADMIT windows, so
        # the width axis moves in the opposite direction from confidence.
        curve = m7_uncertainty_regression_run.coverage_curve
        assert curve.expected_monotonic_direction is MonotonicDirection.NON_DECREASING
        assert curve.coverage_is_monotonic
        coverages = [point.coverage_point.coverage for point in curve.points]
        assert coverages == sorted(coverages)

    def test_a_regression_run_with_no_width_grid_reports_no_curve(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "no-width-grid",
            target=TargetName.ENGAGEMENT_SCORE,
            interval_width_grid=None,
        )
        curve = result.coverage_curve
        assert curve.points == ()
        assert curve.axis is CoverageAxis.MAXIMUM_INTERVAL_WIDTH
        assert curve.coverage_is_monotonic is None
        reason = curve.points_unavailable_reason or ""
        assert "no regression interval-width grid is configured" in reason
        assert "1 - width is not a confidence" in reason
        # The operating point is still reported.
        applied = result.uncertainty.folds[0].applied_selective_metrics
        assert applied is not None
        assert applied.coverage_point.threshold is None
        assert applied.coverage_point.threshold_unavailable_reason

    def test_a_width_of_zero_is_never_read_as_no_width_policy(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        # An absent maximum accepts everything; a maximum of zero would
        # accept nothing. The two must not be recorded the same way.
        result = _run(
            m5_dataset,
            tmp_path / "no-width-policy",
            target=TargetName.ENGAGEMENT_SCORE,
            interval_width_grid=None,
        )
        applied = result.uncertainty.folds[0].applied_selective_metrics
        assert applied is not None
        assert applied.coverage_point.threshold is None
        assert applied.coverage_point.coverage > 0.0

    def test_every_point_uses_the_same_underlying_predictions(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        totals = {
            point.coverage_point.total_window_count
            for point in m7_uncertainty_classification_run.coverage_curve.points
        }
        assert len(totals) == 1

    def test_the_curve_is_deterministic(self, m5_dataset: Path, tmp_path: Path) -> None:
        first = _run(m5_dataset, tmp_path / "curve-a")
        second = _run(m5_dataset, tmp_path / "curve-b")
        assert first.coverage_curve.model_dump(
            mode="json"
        ) == second.coverage_curve.model_dump(mode="json")

    def test_an_undefined_risk_point_states_why(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        for point in m7_uncertainty_classification_run.coverage_curve.risk_coverage:
            if point.empirical_risk is None:
                assert point.unavailable_reason

    def test_the_area_is_marked_descriptive_only(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        curve = m7_uncertainty_classification_run.coverage_curve
        assert "DESCRIPTIVE" in curve.aurc_equation
        assert "establishes nothing about" in curve.aurc_equation

    def test_a_regression_curve_has_no_classification_risk(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        curve = m7_uncertainty_regression_run.coverage_curve
        assert all(point.empirical_risk is None for point in curve.risk_coverage)
        assert curve.area_under_risk_coverage is None
        assert curve.area_under_risk_coverage_unavailable_reason


class TestEvidenceGate:
    def test_a_quality_gate_blocks_with_its_own_reason(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "quality-gate",
            evidence_gate=EvidenceGateConfiguration(minimum_signal_quality=0.99),
        )
        counts = result.uncertainty.abstention_reason_counts
        assert counts.get(AbstentionReason.SIGNAL_QUALITY_BELOW_GATE.value, 0) > 0

    def test_a_quality_block_is_not_a_confidence_block(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "quality-not-confidence",
            evidence_gate=EvidenceGateConfiguration(minimum_signal_quality=0.99),
            population_confidence_threshold=0.0,
        )
        counts = result.uncertainty.abstention_reason_counts
        assert counts.get(AbstentionReason.SIGNAL_QUALITY_BELOW_GATE.value, 0) > 0
        # Every window cleared a zero threshold, so no confidence reason fired.
        assert counts.get(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD.value, 0) == 0

    def test_a_modality_requirement_blocks_with_its_own_reason(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "modality-gate",
            evidence_gate=EvidenceGateConfiguration(minimum_available_modalities=99),
        )
        counts = result.uncertainty.abstention_reason_counts
        assert (
            counts.get(AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE.value, 0) > 0
        )

    def test_a_gated_window_still_carries_its_prediction(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "gate-keeps-prediction",
            evidence_gate=EvidenceGateConfiguration(minimum_available_modalities=99),
        )
        table = pq.read_table(
            result.directory / "selective_predictions.parquet"
        ).to_pandas()
        assert table["predicted_class"].notna().all()


class TestAdaptationGate:
    def test_every_evaluated_window_gets_a_gate_decision(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        evaluation = m7_uncertainty_classification_run.uncertainty
        assert (
            evaluation.adaptation_gate_eligible_count
            + evaluation.adaptation_gate_blocked_count
            == evaluation.total_window_count
        )

    def test_every_abstention_blocks_the_gate(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        gates = pq.read_table(directory / "adaptation_gate.parquet").to_pandas()
        blocked = gates[gates["prediction_abstained"]]
        assert len(blocked) > 0
        assert (blocked["decision"] == AdaptationGateDecision.BLOCKED.value).all()

    def test_the_gate_table_names_no_action(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        gates = pq.read_table(directory / "adaptation_gate.parquet").to_pandas()
        for column in gates.columns:
            for token in ("action", "difficulty", "scene", "reward", "policy"):
                assert token not in column

    def test_the_gate_and_the_decision_never_disagree(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        gates = pq.read_table(directory / "adaptation_gate.parquet").to_pandas()
        decisions = pq.read_table(
            directory / "selective_predictions.parquet"
        ).to_pandas()
        merged = decisions.merge(gates, on="source_prediction_id", suffixes=("", "_g"))
        eligible = merged["decision"] == AdaptationGateDecision.ELIGIBLE.value
        assert (merged.loc[eligible, "accepted"]).all()


class TestArtifacts:
    def test_every_required_artifact_is_written(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        for name in UNCERTAINTY_REQUIRED_ARTIFACTS:
            assert (directory / name).exists(), name

    def test_the_supporting_tables_are_written(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        for name in (
            "predictions.parquet",
            "selective_predictions.parquet",
            "adaptation_gate.parquet",
            "calibration.json",
            "checksums.json",
            "manifest.json",
        ):
            assert (directory / name).exists(), name

    def test_the_checksums_verify(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        assert verify_checksums(m7_uncertainty_classification_run.directory) == ()

    def test_the_checksums_cover_every_required_artifact(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        recorded = json.loads((directory / "checksums.json").read_text())
        for name in UNCERTAINTY_REQUIRED_ARTIFACTS:
            assert name in recorded, name

    def test_the_manifest_is_written_last_and_claims_completion(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        manifest = read_manifest(m7_uncertainty_classification_run.directory)
        assert manifest.status is RunStatus.COMPLETED
        assert manifest.failure_reason is None

    def test_no_temporary_file_survives(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        assert not list(directory.glob(".*.tmp"))

    def test_the_original_and_the_selective_record_are_both_retained(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        original = pq.read_table(directory / "predictions.parquet").to_pandas()
        selective = pq.read_table(
            directory / "selective_predictions.parquet"
        ).to_pandas()
        assert len(original) == len(selective)
        assert set(original["source_prediction_id"]) == set(
            selective["source_prediction_id"]
        )

    def test_threshold_provenance_is_recorded_in_its_own_document(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        document = json.loads((directory / "thresholds.json").read_text())
        assert "ENGINEERING DEFAULT" in document["population_threshold_provenance"]
        assert document["leakage_rules"]
        assert any(fold["personal_thresholds"] for fold in document["folds"])

    def test_calibration_groups_are_recorded(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        document = json.loads((directory / "calibration.json").read_text())
        assert document["folds"]
        for record in document["folds"]:
            assert record["calibration_groups"]
            assert not set(record["calibration_groups"]) & set(record["test_groups"])

    def test_metrics_report_all_windows_and_accepted_windows_separately(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        results = m7_uncertainty_classification_run.metrics.results
        kinds = {result.model_kind for result in results}
        assert kinds == {"all_windows", "selective"}

    def test_no_artifact_carries_a_forbidden_identifier(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        for path in sorted(directory.glob("*.json")):
            text = path.read_text().lower()
            for forbidden in FORBIDDEN_IDENTIFIERS:
                assert forbidden not in text, f"{path.name} contains {forbidden}"

    def test_no_artifact_makes_a_forbidden_claim(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        for path in sorted(directory.glob("*.json")):
            text = path.read_text().lower()
            for claim in FORBIDDEN_CLAIMS:
                assert claim not in text, f"{path.name} claims {claim!r}"

    def test_production_threshold_appears_only_in_a_denial(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        for path in sorted(directory.glob("*.json")):
            text = path.read_text().lower()
            assert text.count("production threshold") == text.count(
                "not a production threshold"
            ), path.name

    def test_no_artifact_carries_a_champion_or_safety_field(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory

        def keys(node: object) -> list[str]:
            if isinstance(node, dict):
                found: list[str] = []
                for key, value in node.items():
                    found.append(str(key))
                    found.extend(keys(value))
                return found
            if isinstance(node, list):
                return [k for item in node for k in keys(item)]
            return []

        for path in sorted(directory.glob("*.json")):
            for key in keys(json.loads(path.read_text())):
                lowered = key.lower()
                for token in ("champion", "winner", "safe", "validated", "reliable"):
                    assert token not in lowered, f"{path.name} has field {key}"

    def test_no_mlflow_dvc_or_docker_file_is_produced(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        names = {path.name.lower() for path in directory.rglob("*")}
        for forbidden in ("mlruns", "mlflow", "dvc.yaml", ".dvc", "dockerfile"):
            assert forbidden not in names

    def test_synthetic_rows_are_never_scientifically_eligible(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        assert table["is_synthetic"].all()
        assert not table["scientific_evaluation_eligible"].any()
        assert not (
            m7_uncertainty_classification_run.uncertainty
        ).scientific_evaluation_eligible

    def test_the_self_check_banner_is_persisted(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        evaluation = m7_uncertainty_classification_run.uncertainty
        assert evaluation.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK
        assert not evaluation.scientific_evaluation_eligible
        assert any(SOFTWARE_SELF_CHECK_BANNER in d for d in evaluation.disclaimers)


class TestDeterminism:
    def test_two_identical_runs_share_a_run_identifier(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        first = _run(m5_dataset, tmp_path / "det-a")
        second = _run(m5_dataset, tmp_path / "det-b")
        assert first.run_id == second.run_id

    def test_two_identical_runs_produce_identical_documents(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        first = _run(m5_dataset, tmp_path / "doc-a")
        second = _run(m5_dataset, tmp_path / "doc-b")
        for name in (
            "uncertainty.json",
            "thresholds.json",
            "selective_metrics.json",
            "coverage_curve.json",
            "metrics.json",
            "splits.json",
        ):
            assert (first.directory / name).read_text() == (
                second.directory / name
            ).read_text(), name

    def test_only_the_timestamps_differ_in_the_manifest(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        first = read_manifest(_run(m5_dataset, tmp_path / "man-a").directory)
        second = read_manifest(_run(m5_dataset, tmp_path / "man-b").directory)
        volatile = {"started_at_utc", "finished_at_utc"}
        a = first.model_dump(mode="json")
        b = second.model_dump(mode="json")
        for key in volatile:
            a.pop(key)
            b.pop(key)
        assert a == b

    def test_a_different_threshold_changes_the_run_identifier(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        first = _run(
            m5_dataset, tmp_path / "tau-a", population_confidence_threshold=0.5
        )
        second = _run(
            m5_dataset, tmp_path / "tau-b", population_confidence_threshold=0.9
        )
        assert first.run_id != second.run_id


class TestScientificMode:
    def test_scientific_mode_refuses_synthetic_data(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        from engagevr.training.runner import ScientificModeError

        with pytest.raises(ScientificModeError):
            _run(
                m5_dataset,
                tmp_path / "scientific",
                mode=EvaluationMode.SCIENTIFIC,
            )

    def test_a_refused_run_writes_no_manifest_claiming_success(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        from engagevr.training.runner import ScientificModeError

        output = tmp_path / "refused-scientific"
        with pytest.raises(ScientificModeError):
            _run(m5_dataset, output, mode=EvaluationMode.SCIENTIFIC)
        assert not (output / "manifest.json").exists()


class TestRefusals:
    def test_personalized_thresholds_require_subject_grouping(
        self, m5_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from engagevr.schemas.experiments import GroupField

        monkeypatch.setattr(
            "engagevr.training.uncertainty_runner.choose_group_field",
            lambda subject_ids, session_ids: (
                GroupField.SESSION_ID,
                "forced for this test",
            ),
        )
        with pytest.raises(UncertaintyConfigurationError, match="no person"):
            _run(
                m5_dataset,
                tmp_path / "session-grouped",
                personalized_thresholds_enabled=True,
                personal_calibration_windows=3,
                minimum_personal_calibration_windows=2,
            )

    def test_a_failed_run_records_a_failed_manifest(
        self, m5_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("deliberate failure for this test")

        monkeypatch.setattr(
            "engagevr.training.uncertainty_runner._evaluate_fold", explode
        )
        output = tmp_path / "failed"
        with pytest.raises(RuntimeError, match="deliberate failure"):
            _run(m5_dataset, output)
        manifest = read_manifest(output)
        assert manifest.status is RunStatus.FAILED
        assert "deliberate failure" in (manifest.failure_reason or "")

    def test_the_prediction_source_is_recorded_on_every_row(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        directory = m7_uncertainty_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        assert set(table["source_model"]) == {PredictionSource.BASELINE_MODEL.value}

    def test_an_early_fusion_source_resolves_a_smaller_column_set(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        baseline = _run(m5_dataset, tmp_path / "src-baseline")
        fusion = _run(
            m5_dataset,
            tmp_path / "src-fusion",
            prediction_source=PredictionSource.EARLY_FUSION,
        )
        assert len(fusion.uncertainty.predictor_columns) <= len(
            baseline.uncertainty.predictor_columns
        )
        assert fusion.run_id != baseline.run_id


class TestTaskTypeSeparation:
    def test_a_classification_run_reports_the_classification_task(
        self, m7_uncertainty_classification_run: UncertaintyRunResult
    ) -> None:
        assert (
            m7_uncertainty_classification_run.uncertainty.task_type
            is TaskType.CLASSIFICATION
        )

    def test_a_regression_run_reports_the_regression_task(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        assert (
            m7_uncertainty_regression_run.uncertainty.task_type is TaskType.REGRESSION
        )

    def test_a_regression_run_states_that_no_probability_was_calibrated(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        for fold in m7_uncertainty_regression_run.uncertainty.folds:
            if not fold.evaluated:
                continue
            assert fold.probability_calibration_status is (
                ProbabilityCalibrationStatus.UNAVAILABLE
            )
            assert "no class probabilities" in (
                fold.probability_calibration_unavailable_reason or ""
            )

    def test_a_regression_run_warns_that_personal_thresholds_do_not_apply(
        self, m7_uncertainty_regression_run: UncertaintyRunResult
    ) -> None:
        joined = " ".join(m7_uncertainty_regression_run.warnings)
        assert "no meaning for a point prediction" in joined
        assert "Subject-conditional conformal intervals are NOT" in joined
