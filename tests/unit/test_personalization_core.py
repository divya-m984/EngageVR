"""Milestone 6 personalization algebra: splits, baselines, and corrections.

Every test here is deterministic and works on hand-built windows or small
in-memory frames.  None needs a webcam, a model asset, a display server, a
network, Unity, a public dataset, or participant data.

The subject identifiers below (``s1``, ``s2``, ...) are SYNTHETIC labels
invented for these tests.  No person is described anywhere in this file.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from engagevr.features.catalog import get_catalog
from engagevr.schemas.fusion import FusionModality
from engagevr.schemas.personalization import (
    CLASSIFICATION_CORRECTION_EQUATION,
    REGRESSION_CORRECTION_EQUATION,
    PersonalizationMethod,
)
from engagevr.schemas.targets import TaskType
from engagevr.training.personalization import (
    PersonalizationError,
    SubjectWindow,
    apply_classification_correction,
    apply_personal_baseline,
    apply_regression_correction,
    assert_personalizable,
    build_calibration_split,
    build_personalization_run_id,
    classification_correction,
    parse_method,
    personal_baseline_statistics,
    personalizable_columns,
    regression_correction,
    subject_windows,
)

_START = datetime(2026, 8, 16, 9, 0, 0, tzinfo=UTC)
_VOCABULARY = ("low", "medium", "high")


def _window(
    index: int,
    *,
    duration: float = 10.0,
    step: float = 10.0,
    session: str = "sess-a",
) -> SubjectWindow:
    start = _START + timedelta(seconds=step * index)
    return SubjectWindow(
        row_index=index,
        window_id=f"w{index:02d}",
        session_id=session,
        start_utc=start,
        end_utc=start + timedelta(seconds=duration),
        window_index=index,
    )


def _windows(count: int, **kwargs: float | str) -> tuple[SubjectWindow, ...]:
    return tuple(_window(index, **kwargs) for index in range(count))  # type: ignore[arg-type]


class TestCalibrationSplit:
    def test_the_first_windows_form_the_calibration_region(self) -> None:
        split, calibration, evaluation = build_calibration_split(
            _windows(10),
            subject_id="s1",
            fold_index=0,
            calibration_windows=3,
            minimum_evaluation_windows=1,
        )
        assert split.available
        assert [w.window_id for w in calibration] == ["w00", "w01", "w02"]
        assert [w.window_id for w in evaluation] == [f"w{i:02d}" for i in range(3, 10)]
        assert split.calibration_window_ids == ("w00", "w01", "w02")
        assert split.evaluation_window_ids == tuple(f"w{i:02d}" for i in range(3, 10))

    def test_calibration_precedes_evaluation_in_time(self) -> None:
        split, _calibration, _evaluation = build_calibration_split(
            _windows(8),
            subject_id="s1",
            fold_index=0,
            calibration_windows=4,
            minimum_evaluation_windows=1,
        )
        assert split.calibration_end_utc is not None
        assert split.evaluation_start_utc is not None
        assert split.calibration_end_utc <= split.evaluation_start_utc
        assert split.temporal_order_verified

    def test_the_two_regions_never_share_a_window(self) -> None:
        split, _c, _e = build_calibration_split(
            _windows(9),
            subject_id="s1",
            fold_index=0,
            calibration_windows=2,
            minimum_evaluation_windows=1,
        )
        assert not set(split.calibration_window_ids) & set(split.evaluation_window_ids)

    def test_an_overlapping_window_is_excluded_not_moved(self) -> None:
        # 20-second windows stepped every 10 seconds: the window that starts
        # immediately after the calibration region still overlaps it in time.
        split, _c, evaluation = build_calibration_split(
            _windows(10, duration=20.0, step=10.0),
            subject_id="s1",
            fold_index=0,
            calibration_windows=3,
            minimum_evaluation_windows=1,
        )
        assert split.excluded_overlap_window_ids == ("w03",)
        assert "w03" not in {w.window_id for w in evaluation}
        assert "w03" not in split.evaluation_window_ids
        assert split.calibration_end_utc is not None
        assert split.evaluation_start_utc is not None
        assert split.calibration_end_utc <= split.evaluation_start_utc

    def test_the_split_is_deterministic(self) -> None:
        windows = _windows(10)
        first = build_calibration_split(
            windows,
            subject_id="s1",
            fold_index=0,
            calibration_windows=4,
            minimum_evaluation_windows=1,
        )[0]
        second = build_calibration_split(
            tuple(reversed(windows)),
            subject_id="s1",
            fold_index=0,
            calibration_windows=4,
            minimum_evaluation_windows=1,
        )[0]
        assert first.calibration_window_ids == second.calibration_window_ids
        assert first.evaluation_window_ids == second.evaluation_window_ids

    def test_too_few_windows_is_refused_with_a_reason(self) -> None:
        split, calibration, evaluation = build_calibration_split(
            _windows(3),
            subject_id="s1",
            fold_index=0,
            calibration_windows=3,
            minimum_evaluation_windows=1,
        )
        assert not split.available
        assert split.unavailable_reason is not None
        assert "cannot supply 3 calibration window(s)" in split.unavailable_reason
        assert calibration == () and evaluation == ()

    def test_too_few_evaluation_windows_is_refused_with_a_reason(self) -> None:
        split, _c, _e = build_calibration_split(
            _windows(6),
            subject_id="s1",
            fold_index=0,
            calibration_windows=3,
            minimum_evaluation_windows=5,
        )
        assert not split.available
        assert split.unavailable_reason is not None
        assert "fewer than the 5 required" in split.unavailable_reason

    def test_zero_calibration_windows_is_an_explicit_cold_start(self) -> None:
        split, calibration, evaluation = build_calibration_split(
            _windows(6),
            subject_id="s1",
            fold_index=0,
            calibration_windows=0,
            minimum_evaluation_windows=1,
        )
        assert split.available
        assert split.cold_start
        assert split.cold_start_reason is not None
        assert "calibration_windows=0" in split.cold_start_reason
        assert calibration == ()
        assert len(evaluation) == 6

    def test_a_window_without_a_timestamp_is_refused(self) -> None:
        broken = (
            _window(0),
            SubjectWindow(
                row_index=1,
                window_id="w01",
                session_id="sess-a",
                start_utc=None,
                end_utc=None,
                window_index=1,
            ),
        )
        with pytest.raises(PersonalizationError, match="no start or end"):
            build_calibration_split(
                broken,
                subject_id="s1",
                fold_index=0,
                calibration_windows=1,
                minimum_evaluation_windows=1,
            )

    def test_windows_from_several_sessions_are_ordered_by_time(self) -> None:
        first = _windows(3, session="sess-a")
        later = tuple(
            SubjectWindow(
                row_index=10 + index,
                window_id=f"x{index:02d}",
                session_id="sess-b",
                start_utc=_START + timedelta(hours=1, seconds=10 * index),
                end_utc=_START + timedelta(hours=1, seconds=10 * index + 10),
                window_index=index,
            )
            for index in range(3)
        )
        split, _c, _e = build_calibration_split(
            (*later, *first),
            subject_id="s1",
            fold_index=0,
            calibration_windows=3,
            minimum_evaluation_windows=1,
        )
        assert split.calibration_window_ids == ("w00", "w01", "w02")
        assert split.evaluation_window_ids == ("x00", "x01", "x02")
        assert split.session_ids == ("sess-a", "sess-b")

    def test_subject_windows_are_assembled_from_frame_columns(self) -> None:
        windows = subject_windows(
            row_indices=[1, 3],
            window_ids=("a", "b", "c", "d"),
            session_ids=("s", "s", "s", "s"),
            window_start_utc=(_START, _START, _START, _START),
            window_end_utc=(_START, _START, _START, _START),
            window_indices=(0, 1, 2, 3),
        )
        assert [w.window_id for w in windows] == ["b", "d"]
        assert [w.row_index for w in windows] == [1, 3]


class TestPersonalizableColumns:
    def test_only_catalogued_measured_features_qualify(self) -> None:
        catalog = get_catalog()
        columns = (
            "feat__task_correct_proportion",
            "avail__task_correct_proportion",
            "modality_available__task",
            "modality_quality__task",
        )
        selected = personalizable_columns(
            (FusionModality.TASK, FusionModality.RPPG), columns, catalog
        )
        assert selected == ("feat__task_correct_proportion",)

    def test_quality_features_are_not_selected(self) -> None:
        catalog = get_catalog()
        selected = personalizable_columns(
            tuple(FusionModality),
            ("feat__capture_brightness_mean", "feat__task_correct_proportion"),
            catalog,
        )
        assert selected == ("feat__task_correct_proportion",)

    @pytest.mark.parametrize(
        "column",
        [
            "subject_id",
            "session_id",
            "window_id",
            "window_start_utc",
            "data_source",
            "target__engagement_class",
            "target_meta__engagement_class__source_type",
            "avail__task_correct_proportion",
            "modality_available__task",
            "modality_quality__rppg",
        ],
    )
    def test_a_forbidden_column_cannot_be_personalized(self, column: str) -> None:
        with pytest.raises(PersonalizationError, match="never be personalised"):
            assert_personalizable([column])

    def test_an_uncatalogued_column_is_refused(self) -> None:
        with pytest.raises(PersonalizationError, match="not a catalogued"):
            assert_personalizable(["something_else"])


class TestPersonalBaseline:
    def _frame(self, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"feat__task_correct_proportion": values})

    def test_statistics_use_only_the_rows_supplied(self) -> None:
        catalog = get_catalog()
        statistics = personal_baseline_statistics(
            self._frame([0.2, 0.4, 0.6]),
            subject_id="s1",
            fold_index=0,
            columns=("feat__task_correct_proportion",),
            catalog=catalog,
            source_window_ids=["w00", "w01", "w02"],
            minimum_samples=2,
            zero_variance_epsilon=1e-9,
        )
        assert len(statistics) == 1
        record = statistics[0]
        assert record.normalized
        assert record.mean == pytest.approx(0.4)
        assert record.scale == pytest.approx(float(np.std([0.2, 0.4, 0.6], ddof=0)))
        assert record.source_window_ids == ("w00", "w01", "w02")
        assert record.calibration_sample_count == 3

    def test_a_later_window_cannot_reach_the_statistics(self) -> None:
        """The baseline of the first three rows ignores a fourth, later row."""
        catalog = get_catalog()
        early = personal_baseline_statistics(
            self._frame([0.2, 0.4, 0.6]),
            subject_id="s1",
            fold_index=0,
            columns=("feat__task_correct_proportion",),
            catalog=catalog,
            source_window_ids=["w00", "w01", "w02"],
            minimum_samples=2,
            zero_variance_epsilon=1e-9,
        )[0]
        with_future = personal_baseline_statistics(
            self._frame([0.2, 0.4, 0.6, 99.0]),
            subject_id="s1",
            fold_index=0,
            columns=("feat__task_correct_proportion",),
            catalog=catalog,
            source_window_ids=["w00", "w01", "w02", "w03"],
            minimum_samples=2,
            zero_variance_epsilon=1e-9,
        )[0]
        assert early.mean == pytest.approx(0.4)
        assert with_future.mean != pytest.approx(0.4)
        assert "w03" not in early.source_window_ids

    def test_the_z_score_formula_is_applied_exactly(self) -> None:
        catalog = get_catalog()
        statistics = personal_baseline_statistics(
            self._frame([0.2, 0.4, 0.6]),
            subject_id="s1",
            fold_index=0,
            columns=("feat__task_correct_proportion",),
            catalog=catalog,
            source_window_ids=["w00", "w01", "w02"],
            minimum_samples=2,
            zero_variance_epsilon=1e-9,
        )
        record = statistics[0]
        applied = apply_personal_baseline(self._frame([0.2, 0.4, 0.6, 1.0]), statistics)
        expected = (np.asarray([0.2, 0.4, 0.6, 1.0]) - record.mean) / record.scale
        assert applied["feat__task_correct_proportion"].to_numpy() == pytest.approx(
            expected
        )

    def test_zero_variance_is_centred_but_not_scaled(self) -> None:
        catalog = get_catalog()
        record = personal_baseline_statistics(
            self._frame([0.5, 0.5, 0.5]),
            subject_id="s1",
            fold_index=0,
            columns=("feat__task_correct_proportion",),
            catalog=catalog,
            source_window_ids=["w00", "w01", "w02"],
            minimum_samples=2,
            zero_variance_epsilon=1e-9,
        )[0]
        assert record.normalized
        assert record.mean == pytest.approx(0.5)
        assert record.scale == 1.0
        assert record.scale_source == "unit_scale_zero_variance"
        assert record.observed_standard_deviation == pytest.approx(0.0)

    def test_missing_values_stay_missing(self) -> None:
        catalog = get_catalog()
        statistics = personal_baseline_statistics(
            self._frame([0.2, float("nan"), 0.6]),
            subject_id="s1",
            fold_index=0,
            columns=("feat__task_correct_proportion",),
            catalog=catalog,
            source_window_ids=["w00", "w01", "w02"],
            minimum_samples=2,
            zero_variance_epsilon=1e-9,
        )
        assert statistics[0].finite_sample_count == 2
        applied = apply_personal_baseline(
            self._frame([0.2, float("nan"), 0.6]), statistics
        )
        assert bool(applied["feat__task_correct_proportion"].isna().tolist()[1])

    def test_too_little_evidence_leaves_the_feature_untouched(self) -> None:
        catalog = get_catalog()
        record = personal_baseline_statistics(
            self._frame([0.4]),
            subject_id="s1",
            fold_index=0,
            columns=("feat__task_correct_proportion",),
            catalog=catalog,
            source_window_ids=["w00"],
            minimum_samples=3,
            zero_variance_epsilon=1e-9,
        )[0]
        assert not record.normalized
        assert record.mean == 0.0
        assert record.scale == 1.0
        assert record.unavailable_reason is not None
        assert "fewer than the 3 required" in record.unavailable_reason
        applied = apply_personal_baseline(self._frame([0.4, 0.9]), [record])
        assert applied["feat__task_correct_proportion"].to_numpy() == pytest.approx(
            [0.4, 0.9]
        )

    def test_a_forbidden_column_is_refused_before_any_statistic(self) -> None:
        catalog = get_catalog()
        with pytest.raises(PersonalizationError, match="never be personalised"):
            personal_baseline_statistics(
                pd.DataFrame({"modality_quality__rppg": [0.1, 0.2]}),
                subject_id="s1",
                fold_index=0,
                columns=("modality_quality__rppg",),
                catalog=catalog,
                source_window_ids=["w00", "w01"],
                minimum_samples=1,
                zero_variance_epsilon=1e-9,
            )


class TestRegressionCorrection:
    def test_the_documented_bias_equation_is_used(self) -> None:
        correction = regression_correction(
            subject_id="s1",
            fold_index=0,
            method=PersonalizationMethod.FEW_SHOT_CORRECTION,
            calibration_window_ids=["w00", "w01", "w02"],
            calibration_targets=[0.5, 0.6, 0.7],
            population_predictions=[0.3, 0.4, 0.5],
            minimum_windows=2,
        )
        assert correction.available
        assert correction.bias == pytest.approx(0.2)
        assert correction.equation == REGRESSION_CORRECTION_EQUATION
        assert apply_regression_correction(0.45, correction.bias or 0.0) == (
            pytest.approx(0.65)
        )

    def test_only_calibration_labels_are_recorded(self) -> None:
        correction = regression_correction(
            subject_id="s1",
            fold_index=0,
            method=PersonalizationMethod.FEW_SHOT_CORRECTION,
            calibration_window_ids=["w00", "w01"],
            calibration_targets=[0.4, 0.6],
            population_predictions=[0.4, 0.6],
            minimum_windows=2,
        )
        assert set(correction.calibration_targets) == {"w00", "w01"}
        assert correction.calibration_window_ids == ("w00", "w01")
        assert correction.bias == pytest.approx(0.0)

    def test_thin_evidence_is_refused_with_a_reason(self) -> None:
        correction = regression_correction(
            subject_id="s1",
            fold_index=0,
            method=PersonalizationMethod.FEW_SHOT_CORRECTION,
            calibration_window_ids=["w00"],
            calibration_targets=[0.4],
            population_predictions=[0.3],
            minimum_windows=3,
        )
        assert not correction.available
        assert correction.bias is None
        assert correction.unavailable_reason is not None
        assert "fewer than the 3 required" in correction.unavailable_reason

    def test_a_corrected_prediction_stays_finite(self) -> None:
        assert math.isfinite(apply_regression_correction(0.4, -0.9))


class TestClassificationCorrection:
    def _uniform(self, rows: int) -> np.ndarray:
        return np.full((rows, 3), 1.0 / 3.0)

    def test_the_shift_is_zero_when_labels_match_the_population_average(
        self,
    ) -> None:
        # One label of each class against a uniform population prediction:
        # the smoothed observed and expected frequencies coincide exactly.
        correction = classification_correction(
            subject_id="s1",
            fold_index=0,
            method=PersonalizationMethod.FEW_SHOT_CORRECTION,
            calibration_window_ids=["w00", "w01", "w02"],
            calibration_labels=["low", "medium", "high"],
            population_probabilities=self._uniform(3),
            vocabulary=_VOCABULARY,
            smoothing=1.0,
            shrinkage_constant=5.0,
            minimum_windows=2,
            minimum_classes=2,
        )
        assert correction.available
        for value in correction.log_odds_shift.values():
            assert value == pytest.approx(0.0, abs=1e-12)
        assert correction.equation == CLASSIFICATION_CORRECTION_EQUATION

    def test_the_documented_equation_is_reproduced(self) -> None:
        probabilities = np.asarray(
            [[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]]
        )
        labels = ["low", "low", "medium", "high"]
        smoothing, kappa = 1.0, 5.0
        correction = classification_correction(
            subject_id="s1",
            fold_index=0,
            method=PersonalizationMethod.FEW_SHOT_CORRECTION,
            calibration_window_ids=["w00", "w01", "w02", "w03"],
            calibration_labels=labels,
            population_probabilities=probabilities,
            vocabulary=_VOCABULARY,
            smoothing=smoothing,
            shrinkage_constant=kappa,
            minimum_windows=2,
            minimum_classes=2,
        )
        n, k = 4, 3
        denominator = n + smoothing * k
        shrinkage = n / (n + kappa)
        for index, label in enumerate(_VOCABULARY):
            observed = (labels.count(label) + smoothing) / denominator
            expected = (float(probabilities[:, index].sum()) + smoothing) / denominator
            delta = shrinkage * (math.log(observed) - math.log(expected))
            assert correction.log_odds_shift[label] == pytest.approx(delta)
        assert correction.shrinkage == pytest.approx(shrinkage)

    def test_shrinkage_grows_with_calibration_evidence(self) -> None:
        def _shrinkage(count: int) -> float:
            correction = classification_correction(
                subject_id="s1",
                fold_index=0,
                method=PersonalizationMethod.FEW_SHOT_CORRECTION,
                calibration_window_ids=[f"w{i:02d}" for i in range(count)],
                calibration_labels=["low", "high"] * (count // 2),
                population_probabilities=self._uniform(count),
                vocabulary=_VOCABULARY,
                smoothing=1.0,
                shrinkage_constant=5.0,
                minimum_windows=2,
                minimum_classes=2,
            )
            assert correction.shrinkage is not None
            return correction.shrinkage

        assert _shrinkage(2) < _shrinkage(10) < _shrinkage(40) < 1.0

    def test_corrected_probabilities_are_finite_and_sum_to_one(self) -> None:
        probabilities = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        shifted = apply_classification_correction(
            probabilities, {"low": 1.5, "medium": -2.0, "high": 0.25}, _VOCABULARY
        )
        assert np.isfinite(shifted).all()
        assert (shifted >= 0.0).all()
        assert shifted.sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-12)

    def test_a_zero_shift_leaves_probabilities_unchanged(self) -> None:
        probabilities = np.asarray([[0.7, 0.2, 0.1]])
        shifted = apply_classification_correction(
            probabilities, dict.fromkeys(_VOCABULARY, 0.0), _VOCABULARY
        )
        assert shifted == pytest.approx(probabilities)

    def test_thin_evidence_is_refused_with_a_reason(self) -> None:
        correction = classification_correction(
            subject_id="s1",
            fold_index=0,
            method=PersonalizationMethod.FEW_SHOT_CORRECTION,
            calibration_window_ids=["w00"],
            calibration_labels=["low"],
            population_probabilities=self._uniform(1),
            vocabulary=_VOCABULARY,
            smoothing=1.0,
            shrinkage_constant=5.0,
            minimum_windows=3,
            minimum_classes=2,
        )
        assert not correction.available
        assert not correction.log_odds_shift
        assert correction.unavailable_reason is not None
        assert "fewer than the 3 required" in correction.unavailable_reason

    def test_insufficient_class_support_falls_back_explicitly(self) -> None:
        correction = classification_correction(
            subject_id="s1",
            fold_index=0,
            method=PersonalizationMethod.FEW_SHOT_CORRECTION,
            calibration_window_ids=["w00", "w01", "w02"],
            calibration_labels=["low", "low", "low"],
            population_probabilities=self._uniform(3),
            vocabulary=_VOCABULARY,
            smoothing=1.0,
            shrinkage_constant=5.0,
            minimum_windows=2,
            minimum_classes=2,
        )
        assert not correction.available
        assert correction.unavailable_reason is not None
        assert "fewer than the 2 required" in correction.unavailable_reason
        assert correction.calibration_class_support == {
            "low": 3,
            "medium": 0,
            "high": 0,
        }

    def test_a_label_outside_the_vocabulary_is_refused(self) -> None:
        correction = classification_correction(
            subject_id="s1",
            fold_index=0,
            method=PersonalizationMethod.FEW_SHOT_CORRECTION,
            calibration_window_ids=["w00", "w01"],
            calibration_labels=["low", "enormous"],
            population_probabilities=self._uniform(2),
            vocabulary=_VOCABULARY,
            smoothing=1.0,
            shrinkage_constant=5.0,
            minimum_windows=2,
            minimum_classes=2,
        )
        assert not correction.available
        assert correction.unavailable_reason is not None
        assert "outside the declared class vocabulary" in correction.unavailable_reason

    def test_a_non_finite_probability_matrix_is_refused(self) -> None:
        probabilities = np.asarray([[0.5, 0.5, 0.0], [float("nan"), 0.5, 0.5]])
        correction = classification_correction(
            subject_id="s1",
            fold_index=0,
            method=PersonalizationMethod.FEW_SHOT_CORRECTION,
            calibration_window_ids=["w00", "w01"],
            calibration_labels=["low", "high"],
            population_probabilities=probabilities,
            vocabulary=_VOCABULARY,
            smoothing=1.0,
            shrinkage_constant=5.0,
            minimum_windows=2,
            minimum_classes=2,
        )
        assert not correction.available
        assert correction.unavailable_reason is not None
        assert "non-finite" in correction.unavailable_reason

    def test_only_calibration_labels_are_recorded_by_window(self) -> None:
        correction = classification_correction(
            subject_id="s1",
            fold_index=0,
            method=PersonalizationMethod.FEW_SHOT_CORRECTION,
            calibration_window_ids=["w00", "w01"],
            calibration_labels=["low", "high"],
            population_probabilities=self._uniform(2),
            vocabulary=_VOCABULARY,
            smoothing=1.0,
            shrinkage_constant=5.0,
            minimum_windows=2,
            minimum_classes=2,
        )
        assert correction.calibration_targets == {"w00": "low", "w01": "high"}
        assert correction.task_type is TaskType.CLASSIFICATION


class TestMethodParsing:
    def test_hyphens_and_underscores_are_both_accepted(self) -> None:
        assert (
            parse_method("personal-baseline") is PersonalizationMethod.PERSONAL_BASELINE
        )
        assert (
            parse_method("few_shot_correction")
            is PersonalizationMethod.FEW_SHOT_CORRECTION
        )

    def test_an_unknown_method_is_refused_with_the_valid_list(self) -> None:
        with pytest.raises(PersonalizationError, match="unknown personalization"):
            parse_method("deep-personal-transformer")


class TestRunIdentity:
    def _identifier(self, **overrides: object) -> str:
        from engagevr.schemas.personalization import PersonalizationConfiguration

        configuration = PersonalizationConfiguration(
            **{"calibration_windows": 5, **overrides}  # type: ignore[arg-type]
        )
        return build_personalization_run_id(
            target_name="engagement_class",
            task_type="classification",
            evaluation_mode="software_self_check",
            dataset_fingerprint="abc123",
            split_manifest_fingerprint="def456",
            random_seed=42,
            configuration=configuration,
            calibration_method="sigmoid",
            engagevr_version="0.1.0",
        )

    def test_the_same_configuration_reproduces_the_identifier(self) -> None:
        assert self._identifier() == self._identifier()

    def test_a_different_calibration_window_count_changes_it(self) -> None:
        assert self._identifier() != self._identifier(calibration_windows=7)

    def test_a_different_method_changes_it(self) -> None:
        assert self._identifier() != self._identifier(
            method=PersonalizationMethod.POPULATION_ONLY
        )

    def test_the_identifier_names_the_target_and_the_mode(self) -> None:
        assert self._identifier().startswith(
            "engagement_class-personalization-selfcheck-"
        )
