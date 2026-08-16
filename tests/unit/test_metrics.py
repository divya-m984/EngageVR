"""Metric tests: labelled matrices, undefined values, and honest aggregation.

The recurring rule under test is that an undefined metric stays
unavailable with a stated reason. Zero is a legitimate score and must
never stand in for "not computable".
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from engagevr.schemas.experiments import (
    SELF_CHECK_DISCLAIMER,
    SOFTWARE_SELF_CHECK_BANNER,
    ConfusionMatrix,
    EvaluationMode,
    MetricsDocument,
    ModelResult,
)
from engagevr.training.metrics import (
    aggregate_calibration_metrics,
    aggregate_fold_metrics,
    build_confusion_matrix,
    calibration_metrics,
    classification_metrics,
    expected_calibration_error,
    multiclass_brier_score,
    regression_metrics,
    reliability_bins,
    sum_confusion_matrices,
)

CLASSES = ("low", "medium", "high")


class TestConfusionMatrix:
    def test_the_matrix_is_stored_with_its_labels(self) -> None:
        matrix = build_confusion_matrix(
            ["low", "high", "low"], ["low", "low", "medium"], CLASSES
        )
        assert matrix.labels == CLASSES
        assert matrix.rows_are == "true_label"
        assert matrix.columns_are == "predicted_label"

    def test_counts_are_placed_at_the_named_cells(self) -> None:
        matrix = build_confusion_matrix(
            ["low", "high", "low"], ["low", "low", "medium"], CLASSES
        )
        cells = matrix.as_labelled_cells()
        assert cells["true=low|predicted=low"] == 1
        assert cells["true=low|predicted=medium"] == 1
        assert cells["true=high|predicted=low"] == 1
        assert sum(cells.values()) == 3

    def test_a_ragged_matrix_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="rows but"):
            ConfusionMatrix(labels=CLASSES, counts=((1, 0, 0), (0, 1, 0)))

    def test_matrices_sum_element_wise(self) -> None:
        first = build_confusion_matrix(["low"], ["low"], CLASSES)
        second = build_confusion_matrix(["low"], ["medium"], CLASSES)
        total = sum_confusion_matrices([first, second])
        assert total is not None
        assert total.as_labelled_cells()["true=low|predicted=low"] == 1
        assert total.as_labelled_cells()["true=low|predicted=medium"] == 1

    def test_matrices_with_different_labels_do_not_sum(self) -> None:
        first = build_confusion_matrix(["low"], ["low"], CLASSES)
        second = build_confusion_matrix(["a"], ["a"], ("a", "b"))
        assert sum_confusion_matrices([first, second]) is None


class TestClassificationMetrics:
    def test_all_declared_metrics_are_computed(self) -> None:
        truth = ["low", "low", "medium", "high", "high", "medium"]
        predicted = ["low", "medium", "medium", "high", "low", "medium"]
        metrics = classification_metrics(
            y_true=truth,
            y_predicted=predicted,
            labels=CLASSES,
            group_ids=["a", "a", "b", "b", "c", "c"],
        )
        assert metrics.sample_count == 6
        assert metrics.independent_group_count == 3
        assert metrics.class_support == {"low": 2, "medium": 2, "high": 2}
        assert metrics.accuracy == pytest.approx(4 / 6)
        assert metrics.balanced_accuracy is not None
        assert metrics.macro_precision is not None
        assert metrics.macro_recall is not None
        assert metrics.macro_f1 is not None
        assert metrics.weighted_f1 is not None
        assert metrics.confusion_matrix is not None
        assert len(metrics.per_class) == 3

    def test_an_undefined_per_class_precision_stays_unavailable(self) -> None:
        # "high" is never predicted, so its precision is undefined.
        metrics = classification_metrics(
            y_true=["low", "medium", "high"],
            y_predicted=["low", "medium", "medium"],
            labels=CLASSES,
            group_ids=["a", "b", "c"],
        )
        high = next(entry for entry in metrics.per_class if entry.label == "high")
        assert high.precision is None
        assert high.support == 1

    def test_undefined_classes_are_excluded_from_the_macro_mean(self) -> None:
        metrics = classification_metrics(
            y_true=["low", "medium", "high"],
            y_predicted=["low", "medium", "medium"],
            labels=CLASSES,
            group_ids=["a", "b", "c"],
        )
        assert metrics.macro_precision is not None
        # Excluded, not counted as zero.
        assert metrics.macro_precision > 0.0
        assert "macro_precision_excluded_classes" in metrics.unavailable_metrics

    def test_an_empty_evaluation_set_produces_no_metrics(self) -> None:
        metrics = classification_metrics(
            y_true=[], y_predicted=[], labels=CLASSES, group_ids=[]
        )
        assert metrics.accuracy is None
        assert metrics.confusion_matrix is None
        assert (
            metrics.unavailable_metrics["accuracy"]
            == "no samples in this evaluation set"
        )


class TestBalancedAccuracy:
    """Balanced accuracy is the mean recall over classes present in the truth.

    A heavy missing-modality scenario routinely leaves a class predicted
    that the surviving truth does not contain.  ``balanced_accuracy_score``
    warns in exactly that case; this pipeline computes the same quantity
    from its own recall vector, so the number is identical and the warning
    does not appear in a run log.
    """

    def test_a_class_predicted_but_absent_from_the_truth_emits_no_warning(
        self,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            metrics = classification_metrics(
                y_true=["low", "low", "medium", "medium"],
                y_predicted=["low", "high", "medium", "high"],
                labels=CLASSES,
                group_ids=["a", "a", "b", "b"],
            )
        assert metrics.balanced_accuracy is not None

    def test_the_value_matches_the_scikit_learn_definition(self) -> None:
        truth = ["low", "low", "medium", "medium", "high"]
        predicted = ["low", "high", "medium", "low", "high"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = float(balanced_accuracy_score(truth, predicted))
        metrics = classification_metrics(
            y_true=truth,
            y_predicted=predicted,
            labels=CLASSES,
            group_ids=["a"] * 5,
        )
        assert metrics.balanced_accuracy == pytest.approx(expected)

    def test_it_agrees_with_scikit_learn_when_a_class_is_absent(self) -> None:
        """The heavy-dropout condition: 'high' never appears in the truth."""
        truth = ["low", "low", "medium", "medium"]
        predicted = ["low", "high", "medium", "high"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = float(balanced_accuracy_score(truth, predicted))
        metrics = classification_metrics(
            y_true=truth,
            y_predicted=predicted,
            labels=CLASSES,
            group_ids=["a"] * 4,
        )
        assert metrics.balanced_accuracy == pytest.approx(expected)
        assert metrics.class_support["high"] == 0

    def test_it_equals_macro_recall_by_construction(self) -> None:
        metrics = classification_metrics(
            y_true=["low", "low", "medium", "high"],
            y_predicted=["low", "medium", "medium", "low"],
            labels=CLASSES,
            group_ids=["a"] * 4,
        )
        assert metrics.balanced_accuracy == pytest.approx(metrics.macro_recall)


class TestRegressionMetrics:
    def test_all_declared_metrics_are_computed(self) -> None:
        metrics = regression_metrics(
            y_true=[0.0, 1.0, 2.0, 3.0],
            y_predicted=[0.1, 1.1, 1.9, 3.2],
            group_ids=["a", "a", "b", "b"],
        )
        assert metrics.sample_count == 4
        assert metrics.independent_group_count == 2
        assert metrics.mean_absolute_error == pytest.approx(0.125)
        assert metrics.root_mean_squared_error is not None
        assert metrics.median_absolute_error is not None
        assert metrics.r_squared is not None

    def test_r_squared_is_unavailable_with_zero_variance(self) -> None:
        metrics = regression_metrics(
            y_true=[2.0, 2.0, 2.0],
            y_predicted=[1.0, 2.0, 3.0],
            group_ids=["a", "b", "c"],
        )
        assert metrics.r_squared is None
        assert "zero variance" in metrics.unavailable_metrics["r_squared"]
        assert metrics.mean_absolute_error is not None

    def test_r_squared_is_unavailable_with_one_sample(self) -> None:
        metrics = regression_metrics(y_true=[1.0], y_predicted=[0.5], group_ids=["a"])
        assert metrics.r_squared is None

    def test_a_non_finite_prediction_is_an_error_not_a_metric(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            regression_metrics(
                y_true=[1.0, 2.0],
                y_predicted=[float("nan"), 2.0],
                group_ids=["a", "b"],
            )

    def test_an_empty_evaluation_set_produces_no_metrics(self) -> None:
        metrics = regression_metrics(y_true=[], y_predicted=[], group_ids=[])
        assert metrics.mean_absolute_error is None
        assert "no samples" in metrics.unavailable_metrics["mean_absolute_error"]


class TestCalibrationScoring:
    def test_the_brier_score_matches_its_documented_formula(self) -> None:
        proba = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
        score = multiclass_brier_score(proba, ["low", "medium"], CLASSES)
        # Row 1 is perfect (0); row 2 is (0 + 0.25 + 0.25) = 0.5.
        assert score == pytest.approx(0.25)

    def test_a_perfect_prediction_scores_zero(self) -> None:
        proba = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert multiclass_brier_score(proba, ["low", "medium"], CLASSES) == 0.0

    def test_reliability_bins_cover_the_unit_interval(self) -> None:
        proba = np.asarray([[0.9, 0.05, 0.05], [0.4, 0.4, 0.2]])
        bins = reliability_bins(proba, ["low", "medium"], CLASSES, bin_count=10)
        assert len(bins) == 10
        assert bins[0].lower_edge == 0.0
        assert bins[-1].upper_edge == 1.0
        assert sum(b.count for b in bins) == 2

    def test_empty_bins_are_retained_with_null_summaries(self) -> None:
        proba = np.asarray([[0.9, 0.05, 0.05]])
        bins = reliability_bins(proba, ["low"], CLASSES, bin_count=10)
        empty = [b for b in bins if b.count == 0]
        assert empty
        assert all(b.mean_confidence is None for b in empty)

    def test_the_ece_formula_is_the_weighted_gap(self) -> None:
        proba = np.asarray(
            [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.9, 0.05, 0.05]]
        )
        # Three of four "low" predictions are correct: confidence 0.9,
        # accuracy 0.75, one populated bin -> ECE = |0.75 - 0.9| = 0.15.
        bins = reliability_bins(
            proba, ["low", "low", "low", "medium"], CLASSES, bin_count=10
        )
        assert expected_calibration_error(bins) == pytest.approx(0.15)

    def test_ece_is_unavailable_with_no_samples(self) -> None:
        bins = reliability_bins(np.zeros((0, 3)), [], CLASSES, bin_count=5)
        assert expected_calibration_error(bins) is None

    def test_calibration_metrics_bundle_every_score(self) -> None:
        proba = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        metrics = calibration_metrics(
            label="sigmoid",
            probabilities=proba,
            y_true=["low", "medium", "high"],
            labels=CLASSES,
        )
        assert metrics.brier_score is not None
        assert metrics.log_loss is not None
        assert metrics.expected_calibration_error is not None
        assert metrics.ece_bin_count == 10
        assert metrics.bins
        assert "sum_m" in metrics.ece_formula

    def test_a_model_without_probabilities_is_reported_unavailable(self) -> None:
        metrics = calibration_metrics(
            label="uncalibrated",
            probabilities=None,
            y_true=["low"],
            labels=CLASSES,
        )
        assert metrics.brier_score is None
        assert "does not produce class probabilities" in (
            metrics.unavailable_reason or ""
        )

    def test_a_non_finite_probability_is_refused(self) -> None:
        proba = np.asarray([[float("nan"), 0.5, 0.5]])
        metrics = calibration_metrics(
            label="sigmoid", probabilities=proba, y_true=["low"], labels=CLASSES
        )
        assert metrics.brier_score is None
        assert "non-finite" in (metrics.unavailable_reason or "")

    def test_calibration_is_never_described_as_signal_quality(self) -> None:
        metrics = calibration_metrics(
            label="sigmoid",
            probabilities=np.asarray([[0.7, 0.2, 0.1]]),
            y_true=["low"],
            labels=CLASSES,
        )
        assert "not certainty" in metrics.note
        assert "not signal quality" in metrics.note

    def test_log_loss_column_order_follows_the_class_vocabulary(self) -> None:
        """A perfectly confident correct prediction must score near zero.

        This fails loudly if the probability columns are paired with the
        wrong classes, which is what happens when an unsorted vocabulary is
        handed to a metric that binarises in sorted order.
        """
        proba = np.asarray([[0.98, 0.01, 0.01], [0.01, 0.01, 0.98]])
        metrics = calibration_metrics(
            label="uncalibrated",
            probabilities=proba,
            y_true=["low", "high"],
            labels=CLASSES,
        )
        assert metrics.log_loss is not None
        assert metrics.log_loss < 0.05


class TestAggregation:
    def _folds(self) -> list:  # type: ignore[type-arg]
        return [
            regression_metrics(
                y_true=[0.0, 1.0, 2.0],
                y_predicted=[0.0, 1.0, 2.0],
                group_ids=["a", "b", "c"],
            ),
            regression_metrics(
                y_true=[0.0, 1.0, 2.0],
                y_predicted=[0.5, 1.5, 2.5],
                group_ids=["d", "e", "f"],
            ),
        ]

    def test_mean_standard_deviation_and_fold_count_are_reported(self) -> None:
        aggregates = aggregate_fold_metrics(
            self._folds(), ("mean_absolute_error",), total_fold_count=3
        )
        entry = aggregates[0]
        assert entry.mean == pytest.approx(0.25)
        assert entry.standard_deviation == pytest.approx(0.25)
        assert entry.valid_fold_count == 2
        assert entry.total_fold_count == 3
        assert entry.aggregation == "unweighted_mean_over_valid_folds"

    def test_undefined_folds_are_excluded_and_counted(self) -> None:
        constant = regression_metrics(
            y_true=[2.0, 2.0], y_predicted=[1.0, 1.0], group_ids=["g", "h"]
        )
        folds = [*self._folds(), constant]
        aggregates = aggregate_fold_metrics(folds, ("r_squared",))
        entry = aggregates[0]
        assert entry.valid_fold_count == 2
        assert entry.total_fold_count == 3
        assert entry.fold_values[2] is None

    def test_a_metric_undefined_everywhere_has_no_aggregate(self) -> None:
        folds = [
            regression_metrics(
                y_true=[2.0, 2.0], y_predicted=[1.0, 1.0], group_ids=["a", "b"]
            )
        ]
        entry = aggregate_fold_metrics(folds, ("r_squared",))[0]
        assert entry.mean is None
        assert entry.unavailable_reason is not None
        assert "undefined in every" in entry.unavailable_reason

    def test_calibration_aggregation_selects_by_label(self) -> None:
        proba = np.asarray([[0.7, 0.2, 0.1]])
        per_fold = [
            (
                calibration_metrics(
                    label="uncalibrated",
                    probabilities=proba,
                    y_true=["low"],
                    labels=CLASSES,
                ),
                calibration_metrics(
                    label="sigmoid",
                    probabilities=proba,
                    y_true=["low"],
                    labels=CLASSES,
                ),
            )
        ]
        aggregates = aggregate_calibration_metrics(
            per_fold, label="sigmoid", total_fold_count=1
        )
        assert [entry.name for entry in aggregates] == [
            "sigmoid.brier_score",
            "sigmoid.log_loss",
            "sigmoid.expected_calibration_error",
        ]
        assert aggregates[0].mean is not None


class TestDocumentDisclaimers:
    def _document(self, **overrides: object) -> MetricsDocument:
        fields: dict[str, object] = {
            "run_id": "r",
            "evaluation_mode": EvaluationMode.SOFTWARE_SELF_CHECK,
            "scientific_evaluation_eligible": False,
            "target_name": "engagement_class",
            "task_type": "classification",
            "dataset_fingerprint": "0" * 64,
            "group_field": "subject_id",
            "group_count": 5,
            "fold_count": 3,
            "random_seed": 42,
            "results": (),
            "disclaimers": (SELF_CHECK_DISCLAIMER,),
        }
        fields.update(overrides)
        return MetricsDocument(**fields)  # type: ignore[arg-type]

    def test_a_self_check_document_carries_the_banner(self) -> None:
        document = self._document()
        assert any(SOFTWARE_SELF_CHECK_BANNER in d for d in document.disclaimers)

    def test_a_self_check_without_the_banner_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must carry the banner"):
            self._document(disclaimers=("some other note",))

    def test_a_document_with_no_disclaimer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one disclaimer"):
            self._document(disclaimers=())

    def test_a_self_check_can_never_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValueError, match="never be scientifically eligible"):
            self._document(scientific_evaluation_eligible=True)

    def test_the_disclaimer_forbids_comparison_with_published_results(self) -> None:
        assert "NEVER be compared with a published result" in SELF_CHECK_DISCLAIMER
        assert "NOT model accuracy" in SELF_CHECK_DISCLAIMER

    def test_no_public_dataset_result_is_representable(self) -> None:
        """A metrics document has no field to hold a fabricated dataset score."""
        fields = set(MetricsDocument.model_fields)
        for banned in ("ubfc", "public_dataset_accuracy", "published_score"):
            assert banned not in fields

    def test_model_results_record_the_software_check_flag(self) -> None:
        result = ModelResult(
            model_name="rule_software_check",
            model_kind="rule",
            is_software_check_baseline=True,
        )
        document = self._document(results=(result,))
        assert document.results[0].is_software_check_baseline is True
