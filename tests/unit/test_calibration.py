"""Offline calibration tests.

The disjointness requirement is the point: a calibrator fitted on the base
estimator's own rows, or on the outer test fold, produces a number that
describes nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from engagevr.training.calibration import (
    MINIMUM_ISOTONIC_SAMPLES,
    MINIMUM_ISOTONIC_SAMPLES_PER_CLASS,
    CalibrationError,
    CalibrationMethod,
    aligned_probabilities,
    assert_calibration_disjoint,
    calibrate_classifier,
    isotonic_is_supported,
)
from engagevr.training.metrics import calibration_metrics
from engagevr.training.models import CLASSIFICATION_MODELS, build_pipeline

COLUMNS = ["feat__task_correct_proportion", "feat__task_reaction_time_mean_ms"]
CLASSES = ("low", "medium", "high")


def population(n: int = 240) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rng = np.random.default_rng(3)
    accuracy = rng.uniform(0.0, 1.0, n)
    table = pd.DataFrame(
        {
            "feat__task_correct_proportion": accuracy,
            "feat__task_reaction_time_mean_ms": 900.0
            - 400.0 * accuracy
            + rng.normal(0.0, 40.0, n),
        }
    )
    edges = np.quantile(accuracy, [1 / 3, 2 / 3])
    y = np.asarray(
        [CLASSES[int(np.searchsorted(edges, v, side="right"))] for v in accuracy],
        dtype=object,
    )
    groups = [f"g{i % 12:02d}" for i in range(n)]
    return table, y, groups


def partition(
    groups: list[str],
) -> tuple[list[str], list[str], list[str]]:
    distinct = sorted(set(groups))
    return distinct[:6], distinct[6:9], distinct[9:]


def rows_for(groups: list[str], wanted: list[str]) -> np.ndarray:
    members = set(wanted)
    return np.asarray([i for i, g in enumerate(groups) if g in members], dtype=int)


class TestDisjointness:
    def test_disjoint_group_sets_are_accepted(self) -> None:
        assert_calibration_disjoint(
            fit_groups=["a"], calibration_groups=["b"], test_groups=["c"]
        )

    def test_calibration_on_the_fit_groups_is_refused(self) -> None:
        with pytest.raises(CalibrationError, match="memorised predictions"):
            assert_calibration_disjoint(
                fit_groups=["a", "b"],
                calibration_groups=["a"],
                test_groups=["c"],
            )

    def test_calibration_on_the_outer_test_fold_is_refused(self) -> None:
        with pytest.raises(CalibrationError, match="meaningless"):
            assert_calibration_disjoint(
                fit_groups=["a"], calibration_groups=["c"], test_groups=["c"]
            )

    def test_calibration_refuses_before_fitting_anything(self) -> None:
        table, y, _groups = population(60)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["logistic_regression"], COLUMNS, random_seed=0
        )
        with pytest.raises(CalibrationError):
            calibrate_classifier(
                pipeline,
                X_fit=table,
                y_fit=list(y),
                X_calibration=table,
                y_calibration=list(y),
                method=CalibrationMethod.SIGMOID,
                fit_groups=["a"],
                calibration_groups=["a"],
                test_groups=["b"],
            )


class TestSigmoidCalibration:
    def test_a_calibrator_is_produced_on_disjoint_groups(self) -> None:
        table, y, groups = population()
        fit, calibration, test = partition(groups)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["logistic_regression"], COLUMNS, random_seed=0
        )
        fit_idx = rows_for(groups, fit)
        calibration_idx = rows_for(groups, calibration)
        base, outcome = calibrate_classifier(
            pipeline,
            X_fit=table.iloc[fit_idx],
            y_fit=list(y[fit_idx]),
            X_calibration=table.iloc[calibration_idx],
            y_calibration=list(y[calibration_idx]),
            method=CalibrationMethod.SIGMOID,
            fit_groups=fit,
            calibration_groups=calibration,
            test_groups=test,
        )
        assert outcome.available is True
        assert outcome.method is CalibrationMethod.SIGMOID
        assert outcome.fit_group_count == len(fit)
        assert outcome.calibration_group_count == len(calibration)
        assert hasattr(base, "predict_proba")

    def test_calibrated_and_uncalibrated_probabilities_are_both_available(
        self,
    ) -> None:
        table, y, groups = population()
        fit, calibration, test = partition(groups)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["logistic_regression"], COLUMNS, random_seed=0
        )
        fit_idx = rows_for(groups, fit)
        calibration_idx = rows_for(groups, calibration)
        test_idx = rows_for(groups, test)
        base, outcome = calibrate_classifier(
            pipeline,
            X_fit=table.iloc[fit_idx],
            y_fit=list(y[fit_idx]),
            X_calibration=table.iloc[calibration_idx],
            y_calibration=list(y[calibration_idx]),
            method=CalibrationMethod.SIGMOID,
            fit_groups=fit,
            calibration_groups=calibration,
            test_groups=test,
        )
        raw = aligned_probabilities(base, table.iloc[test_idx], CLASSES)
        calibrated = aligned_probabilities(
            outcome.calibrated_estimator, table.iloc[test_idx], CLASSES
        )
        assert raw is not None and calibrated is not None
        assert np.allclose(raw.sum(axis=1), 1.0)
        assert np.allclose(calibrated.sum(axis=1), 1.0)
        assert not np.allclose(raw, calibrated)

    def test_the_supported_frozen_estimator_api_is_used(self) -> None:
        table, y, groups = population()
        fit, calibration, test = partition(groups)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["logistic_regression"], COLUMNS, random_seed=0
        )
        fit_idx = rows_for(groups, fit)
        calibration_idx = rows_for(groups, calibration)
        _base, outcome = calibrate_classifier(
            pipeline,
            X_fit=table.iloc[fit_idx],
            y_fit=list(y[fit_idx]),
            X_calibration=table.iloc[calibration_idx],
            y_calibration=list(y[calibration_idx]),
            method=CalibrationMethod.SIGMOID,
            fit_groups=fit,
            calibration_groups=calibration,
            test_groups=test,
        )
        calibrated = outcome.calibrated_estimator
        assert calibrated is not None
        assert isinstance(calibrated.estimator, FrozenEstimator)
        # A frozen estimator means one calibrator fitted on all the
        # calibration data, not an internal cross-validation.
        assert len(calibrated.calibrated_classifiers_) == 1

    def test_none_method_skips_calibration_but_still_fits_the_base(self) -> None:
        table, y, groups = population(60)
        fit, calibration, test = partition(groups)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["logistic_regression"], COLUMNS, random_seed=0
        )
        fit_idx = rows_for(groups, fit)
        base, outcome = calibrate_classifier(
            pipeline,
            X_fit=table.iloc[fit_idx],
            y_fit=list(y[fit_idx]),
            X_calibration=table.iloc[rows_for(groups, calibration)],
            y_calibration=list(y[rows_for(groups, calibration)]),
            method=CalibrationMethod.NONE,
            fit_groups=fit,
            calibration_groups=calibration,
            test_groups=test,
        )
        assert outcome.available is False
        assert outcome.unavailable_reason == "calibration was not requested"
        assert base.predict(table.iloc[fit_idx]) is not None

    def test_an_empty_calibration_set_skips_rather_than_reuses_training_rows(
        self,
    ) -> None:
        table, y, groups = population(60)
        fit, _calibration, test = partition(groups)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["logistic_regression"], COLUMNS, random_seed=0
        )
        fit_idx = rows_for(groups, fit)
        _base, outcome = calibrate_classifier(
            pipeline,
            X_fit=table.iloc[fit_idx],
            y_fit=list(y[fit_idx]),
            X_calibration=table.iloc[[]],
            y_calibration=[],
            method=CalibrationMethod.SIGMOID,
            fit_groups=fit,
            calibration_groups=[],
            test_groups=test,
        )
        assert outcome.available is False
        assert "estimator's own training rows" in (outcome.unavailable_reason or "")

    def test_a_calibration_set_missing_a_class_is_refused(self) -> None:
        table, y, groups = population(120)
        fit, calibration, test = partition(groups)
        fit_idx = rows_for(groups, fit)
        calibration_idx = rows_for(groups, calibration)
        thin = np.asarray(["low"] * len(calibration_idx), dtype=object)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["logistic_regression"], COLUMNS, random_seed=0
        )
        _base, outcome = calibrate_classifier(
            pipeline,
            X_fit=table.iloc[fit_idx],
            y_fit=list(y[fit_idx]),
            X_calibration=table.iloc[calibration_idx],
            y_calibration=list(thin),
            method=CalibrationMethod.SIGMOID,
            fit_groups=fit,
            calibration_groups=calibration,
            test_groups=test,
        )
        assert outcome.available is False
        assert "no example of class" in (outcome.unavailable_reason or "")


class TestIsotonicMinimumData:
    def test_a_small_calibration_set_is_refused(self) -> None:
        supported, reason = isotonic_is_supported(["low", "medium", "high"] * 3)
        assert supported is False
        assert f"at least {MINIMUM_ISOTONIC_SAMPLES}" in reason
        assert "step function to noise" in reason

    def test_a_thin_class_is_refused(self) -> None:
        y = ["low"] * 80 + ["medium"] * 2
        supported, reason = isotonic_is_supported(y)
        assert supported is False
        assert f"at least {MINIMUM_ISOTONIC_SAMPLES_PER_CLASS}" in reason

    def test_a_sufficient_calibration_set_is_accepted(self) -> None:
        y = ["low", "medium", "high"] * 40
        supported, reason = isotonic_is_supported(y)
        assert supported is True
        assert "meets the isotonic minimum-data thresholds" in reason

    def test_isotonic_is_skipped_with_a_reason_when_data_is_thin(self) -> None:
        table, y, groups = population(60)
        fit, calibration, test = partition(groups)
        fit_idx = rows_for(groups, fit)
        calibration_idx = rows_for(groups, calibration)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["logistic_regression"], COLUMNS, random_seed=0
        )
        _base, outcome = calibrate_classifier(
            pipeline,
            X_fit=table.iloc[fit_idx],
            y_fit=list(y[fit_idx]),
            X_calibration=table.iloc[calibration_idx],
            y_calibration=list(y[calibration_idx]),
            method=CalibrationMethod.ISOTONIC,
            fit_groups=fit,
            calibration_groups=calibration,
            test_groups=test,
        )
        assert outcome.available is False
        assert "isotonic calibration requires" in (outcome.unavailable_reason or "")

    def test_isotonic_runs_when_the_calibration_set_is_large_enough(self) -> None:
        table, y, groups = population(600)
        fit, calibration, test = partition(groups)
        fit_idx = rows_for(groups, fit)
        calibration_idx = rows_for(groups, calibration)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["logistic_regression"], COLUMNS, random_seed=0
        )
        _base, outcome = calibrate_classifier(
            pipeline,
            X_fit=table.iloc[fit_idx],
            y_fit=list(y[fit_idx]),
            X_calibration=table.iloc[calibration_idx],
            y_calibration=list(y[calibration_idx]),
            method=CalibrationMethod.ISOTONIC,
            fit_groups=fit,
            calibration_groups=calibration,
            test_groups=test,
        )
        assert outcome.available is True


class TestProbabilityAlignment:
    def test_columns_are_reordered_to_the_class_vocabulary(self) -> None:
        table, y, _groups = population(90)
        estimator = LogisticRegression(max_iter=500).fit(table, y)
        aligned = aligned_probabilities(estimator, table, CLASSES)
        assert aligned is not None
        raw = estimator.predict_proba(table)
        known = [str(c) for c in estimator.classes_]
        for column, label in enumerate(known):
            assert np.allclose(aligned[:, CLASSES.index(label)], raw[:, column])

    def test_an_unseen_class_gets_a_zero_column_and_rows_still_sum_to_one(
        self,
    ) -> None:
        table, y, _groups = population(90)
        y = np.asarray(["low" if v != "high" else "medium" for v in y], dtype=object)
        estimator = LogisticRegression(max_iter=500).fit(table, y)
        aligned = aligned_probabilities(estimator, table, CLASSES)
        assert aligned is not None
        assert np.allclose(aligned[:, CLASSES.index("high")], 0.0)
        assert np.allclose(aligned.sum(axis=1), 1.0)

    def test_a_model_without_probabilities_returns_none(self) -> None:
        class NoProba:
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(X))

        assert (
            aligned_probabilities(NoProba(), pd.DataFrame({"a": [1]}), CLASSES) is None
        )


class TestCalibrationIsNotSignalQuality:
    def test_the_scored_document_says_so_explicitly(self) -> None:
        metrics = calibration_metrics(
            label="sigmoid",
            probabilities=np.asarray([[0.6, 0.3, 0.1]]),
            y_true=["low"],
            labels=CLASSES,
        )
        assert "not signal quality" in metrics.note

    def test_no_calibration_field_carries_a_quality_score(self) -> None:
        from engagevr.schemas.experiments import CalibrationMetrics

        assert not any("quality" in name for name in CalibrationMetrics.model_fields)
