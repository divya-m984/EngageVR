"""Leakage tests across the four failure modes.

Target leakage, identifier leakage, post-window leakage, and preprocessing
leakage are separate faults with separate defences, so each is tested on
its own rather than through a single end-to-end assertion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engagevr.features.catalog import FEATURE_CATALOG
from engagevr.features.validation import (
    LeakageError,
    assert_no_leakage,
    is_post_window_column,
    select_predictor_columns,
)
from engagevr.schemas.features import NON_PREDICTOR_COLUMNS
from engagevr.schemas.targets import TargetName
from engagevr.training.calibration import (
    CalibrationError,
    assert_calibration_disjoint,
)
from engagevr.training.preprocessing import (
    ImputationStrategy,
    PreprocessingError,
    build_preprocessor,
    load_modelling_frame,
    transformed_feature_names,
)


@pytest.fixture(scope="module")
def predictor_columns(m5_dataset: Path) -> tuple[str, ...]:
    frame = load_modelling_frame(
        m5_dataset,
        target_name=TargetName.ENGAGEMENT_CLASS,
        catalog=FEATURE_CATALOG,
    )
    return frame.predictor_columns


class TestTargetLeakage:
    def test_the_target_column_is_never_a_predictor(
        self, predictor_columns: tuple[str, ...]
    ) -> None:
        for target in TargetName:
            assert f"target__{target.value}" not in predictor_columns

    def test_no_target_provenance_column_is_a_predictor(
        self, predictor_columns: tuple[str, ...]
    ) -> None:
        assert not [c for c in predictor_columns if c.startswith("target_meta__")]

    def test_a_target_column_in_the_matrix_is_refused(self) -> None:
        with pytest.raises(LeakageError, match="carries the target"):
            assert_no_leakage(
                ["feat__task_correct_proportion", "target__engagement_class"],
                FEATURE_CATALOG,
                target_name=TargetName.ENGAGEMENT_CLASS,
            )

    def test_a_target_provenance_column_is_refused(self) -> None:
        with pytest.raises(LeakageError, match="or its"):
            assert_no_leakage(
                ["target_meta__engagement_class__source_type"], FEATURE_CATALOG
            )

    def test_a_target_derived_column_from_another_target_is_refused(self) -> None:
        with pytest.raises(LeakageError, match="target"):
            assert_no_leakage(
                ["feat__task_correct_proportion", "target__cognitive_load_score"],
                FEATURE_CATALOG,
                target_name=TargetName.ENGAGEMENT_CLASS,
            )


class TestIdentifierLeakage:
    @pytest.mark.parametrize(
        "column",
        [
            "subject_id",
            "session_id",
            "window_id",
            "window_index",
            "window_start_utc",
            "window_end_utc",
            "window_start_monotonic_seconds",
            "data_source",
            "experiment_condition",
            "feature_schema_version",
        ],
    )
    def test_identifier_and_timestamp_columns_are_refused(self, column: str) -> None:
        assert column in NON_PREDICTOR_COLUMNS
        with pytest.raises(LeakageError, match="identifier, timestamp, or"):
            assert_no_leakage([column], FEATURE_CATALOG)

    def test_no_identifier_reaches_the_predictor_matrix(
        self, predictor_columns: tuple[str, ...]
    ) -> None:
        assert not set(predictor_columns) & NON_PREDICTOR_COLUMNS


class TestPostWindowLeakage:
    @pytest.mark.parametrize(
        "column",
        [
            "feat__session_total_score",
            "feat__final_accuracy",
            "feat__next_trial_correct",
            "feat__adaptation_outcome",
            "feat__whole_session_mean",
            "feat__end_of_session_flag",
            "feat__post_window_rating",
        ],
    )
    def test_post_window_names_are_detected(self, column: str) -> None:
        assert is_post_window_column(column) is True
        with pytest.raises(LeakageError, match="after the predicted window"):
            assert_no_leakage([column], FEATURE_CATALOG)

    def test_ordinary_feature_names_are_not_flagged(
        self, predictor_columns: tuple[str, ...]
    ) -> None:
        assert not [c for c in predictor_columns if is_post_window_column(c)]

    def test_no_catalog_feature_is_a_post_window_name(self) -> None:
        for name in FEATURE_CATALOG.names():
            assert is_post_window_column(f"feat__{name}") is False, name


class TestUndeclaredColumns:
    def test_an_undeclared_feature_column_is_refused(self) -> None:
        with pytest.raises(LeakageError, match="not declared in feature catalog"):
            assert_no_leakage(["feat__invented_signal"], FEATURE_CATALOG)

    def test_a_non_permitted_feature_is_refused(self) -> None:
        with pytest.raises(LeakageError, match="not permitted as a predictor"):
            assert_no_leakage(["feat__rppg_method"], FEATURE_CATALOG)

    def test_an_unrecognised_column_convention_is_refused(self) -> None:
        with pytest.raises(LeakageError, match="recognised dataset column"):
            assert_no_leakage(["random_column"], FEATURE_CATALOG)

    def test_availability_columns_referring_to_unknown_features_are_refused(
        self,
    ) -> None:
        with pytest.raises(LeakageError, match="not in feature catalog"):
            assert_no_leakage(["avail__invented_signal"], FEATURE_CATALOG)

    def test_selection_keeps_only_permitted_features(self) -> None:
        columns = [
            "feat__task_correct_proportion",
            "avail__task_correct_proportion",
            "feat__rppg_method",
            "modality_quality__rppg",
            "modality_available__task",
            "subject_id",
            "target__engagement_class",
        ]
        selected = select_predictor_columns(columns, FEATURE_CATALOG)
        assert "feat__rppg_method" not in selected
        assert "subject_id" not in selected
        assert "target__engagement_class" not in selected
        assert "feat__task_correct_proportion" in selected
        assert "modality_quality__rppg" in selected


class TestPreprocessingLeakage:
    def test_imputation_statistics_come_from_the_fitted_rows_only(self) -> None:
        columns = ["feat__a", "feat__b"]
        train = pd.DataFrame(
            {"feat__a": [1.0, 2.0, 3.0, np.nan], "feat__b": [0.0, 0.0, 0.0, 0.0]}
        )
        test = pd.DataFrame({"feat__a": [1000.0, np.nan], "feat__b": [5.0, 5.0]})
        preprocessor = build_preprocessor(
            columns, strategy=ImputationStrategy.MEDIAN_WITH_INDICATOR, scale=False
        )
        preprocessor.fit(train)
        imputer = preprocessor.named_transformers_["measured"].named_steps["impute"]
        assert imputer.statistics_[0] == pytest.approx(2.0)
        transformed = preprocessor.transform(test)
        # The test row's missing value takes the TRAINING median, not a
        # statistic recomputed on the test rows.
        assert np.asarray(transformed)[1, 0] == pytest.approx(2.0)

    def test_scaling_parameters_come_from_the_fitted_rows_only(self) -> None:
        columns = ["feat__a"]
        train = pd.DataFrame({"feat__a": [0.0, 2.0]})
        preprocessor = build_preprocessor(
            columns, strategy=ImputationStrategy.MEDIAN_WITH_INDICATOR, scale=True
        )
        preprocessor.fit(train)
        scaler = preprocessor.named_transformers_["measured"].named_steps["scale"]
        assert scaler.mean_[0] == pytest.approx(1.0)
        shifted = pd.DataFrame({"feat__a": [100.0, 102.0]})
        transformed = np.asarray(preprocessor.transform(shifted))
        assert transformed[0, 0] == pytest.approx(99.0)

    def test_native_nan_strategy_does_not_impute(self) -> None:
        columns = ["feat__a"]
        frame = pd.DataFrame({"feat__a": [1.0, np.nan]})
        preprocessor = build_preprocessor(
            columns, strategy=ImputationStrategy.NATIVE_NAN, scale=False
        )
        transformed = np.asarray(preprocessor.fit_transform(frame))
        assert np.isnan(transformed[1, 0])

    def test_feature_names_survive_transformation(self) -> None:
        columns = ["feat__a", "avail__a"]
        frame = pd.DataFrame({"feat__a": [1.0, np.nan], "avail__a": [1.0, 0.0]})
        preprocessor = build_preprocessor(
            columns, strategy=ImputationStrategy.MEDIAN_WITH_INDICATOR, scale=True
        )
        preprocessor.fit(frame)
        names = transformed_feature_names(preprocessor)
        assert "feat__a" in names
        assert "avail__a" in names
        assert any("missing" in name for name in names)

    def test_an_unfitted_preprocessor_cannot_report_names(self) -> None:
        preprocessor = build_preprocessor(
            ["feat__a"],
            strategy=ImputationStrategy.MEDIAN_WITH_INDICATOR,
            scale=False,
        )
        with pytest.raises(PreprocessingError, match="unfitted preprocessor"):
            transformed_feature_names(preprocessor)


class TestCalibrationLeakage:
    def test_disjoint_groups_are_accepted(self) -> None:
        assert_calibration_disjoint(
            fit_groups=["a", "b"],
            calibration_groups=["c"],
            test_groups=["d"],
        )

    def test_calibration_overlapping_the_fit_groups_is_refused(self) -> None:
        with pytest.raises(CalibrationError, match="own training rows"):
            assert_calibration_disjoint(
                fit_groups=["a", "b"],
                calibration_groups=["b"],
                test_groups=["d"],
            )

    def test_calibration_overlapping_the_test_groups_is_refused(self) -> None:
        with pytest.raises(CalibrationError, match="test fold"):
            assert_calibration_disjoint(
                fit_groups=["a"],
                calibration_groups=["d"],
                test_groups=["d"],
            )

    def test_training_overlapping_the_test_groups_is_refused(self) -> None:
        with pytest.raises(CalibrationError, match="overlap the outer test"):
            assert_calibration_disjoint(
                fit_groups=["a", "d"],
                calibration_groups=["c"],
                test_groups=["d"],
            )
