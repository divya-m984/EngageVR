"""Baseline model registry and estimator tests.

Every registered model must fit, predict, and behave deterministically
under a fixed seed. The rule baselines get extra scrutiny: they are
software checks and must be labelled as such everywhere they appear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from engagevr.schemas.targets import TaskType
from engagevr.training.models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    RULE_BASELINE_DISCLAIMER,
    ModelKind,
    RuleBasedThresholdClassifier,
    RuleBasedThresholdRegressor,
    UnsupportedModelError,
    build_pipeline,
    describe_parameters,
    get_model_spec,
    registry_for,
)

COLUMNS = [
    "feat__task_correct_proportion",
    "feat__task_reaction_time_mean_ms",
    "feat__eye_openness_proxy_mean",
    "avail__task_correct_proportion",
    "modality_available__task",
    "modality_quality__task",
]
CLASSES = ("low", "medium", "high")


def frame(n: int = 60, *, with_nan: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    accuracy = np.clip(rng.normal(0.6, 0.2, n), 0.0, 1.0)
    data = {
        "feat__task_correct_proportion": accuracy,
        "feat__task_reaction_time_mean_ms": 900.0 - 400.0 * accuracy,
        "feat__eye_openness_proxy_mean": 0.2 + 0.1 * accuracy,
        "avail__task_correct_proportion": np.ones(n),
        "modality_available__task": np.ones(n),
        "modality_quality__task": np.full(n, 0.9),
    }
    table = pd.DataFrame(data)
    if with_nan:
        table.loc[: n // 10, "feat__eye_openness_proxy_mean"] = np.nan
    return table


def labels(table: pd.DataFrame) -> np.ndarray:
    accuracy = table["feat__task_correct_proportion"].to_numpy()
    edges = np.quantile(accuracy, [1 / 3, 2 / 3])
    return np.asarray(
        [
            CLASSES[int(np.searchsorted(edges, value, side="right"))]
            for value in accuracy
        ],
        dtype=object,
    )


def targets(table: pd.DataFrame) -> np.ndarray:
    return table["feat__task_correct_proportion"].to_numpy()


class TestRegistries:
    def test_both_registries_declare_the_required_families(self) -> None:
        for registry in (CLASSIFICATION_MODELS, REGRESSION_MODELS):
            kinds = {spec.kind for spec in registry.values()}
            assert ModelKind.DUMMY in kinds
            assert ModelKind.LINEAR in kinds
            assert ModelKind.TREE in kinds
            assert ModelKind.RULE in kinds

    def test_a_dummy_baseline_is_always_present(self) -> None:
        assert "dummy" in CLASSIFICATION_MODELS
        assert "dummy" in REGRESSION_MODELS

    def test_gradient_boosting_is_scikit_learns_implementation(self) -> None:
        spec = CLASSIFICATION_MODELS["hist_gradient_boosting"]
        pipeline = build_pipeline(spec, COLUMNS, random_seed=0)
        estimator = pipeline.named_steps["estimator"]
        assert type(estimator).__module__.startswith("sklearn.")
        assert "XGB" not in type(estimator).__name__

    def test_no_neural_or_temporal_model_is_registered(self) -> None:
        for registry in (CLASSIFICATION_MODELS, REGRESSION_MODELS):
            for name, spec in registry.items():
                lowered = f"{name} {type(spec.build(0)).__name__}".lower()
                for banned in ("mlp", "lstm", "gru", "tcn", "transformer", "neural"):
                    assert banned not in lowered, name

    def test_registry_lookup_by_task_type(self) -> None:
        assert registry_for(TaskType.CLASSIFICATION) is CLASSIFICATION_MODELS
        assert registry_for(TaskType.REGRESSION) is REGRESSION_MODELS

    def test_an_unsupported_model_is_refused(self) -> None:
        with pytest.raises(UnsupportedModelError, match="not implemented"):
            get_model_spec("deep_fusion_net", TaskType.CLASSIFICATION)

    def test_hyperparameter_grids_stay_small(self) -> None:
        for registry in (CLASSIFICATION_MODELS, REGRESSION_MODELS):
            for spec in registry.values():
                total = 1
                for values in spec.parameter_grid.values():
                    total *= len(values)
                assert total <= 8, spec.name

    def test_class_weighting_is_off_and_documented(self) -> None:
        spec = CLASSIFICATION_MODELS["logistic_regression"]
        estimator = spec.build(0)
        assert estimator.get_params()["class_weight"] is None


class TestClassifiers:
    @pytest.mark.parametrize("name", sorted(CLASSIFICATION_MODELS))
    def test_every_classifier_fits_and_predicts(self, name: str) -> None:
        table = frame()
        y = labels(table)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS[name],
            COLUMNS,
            random_seed=42,
            rule_class_order=CLASSES,
        )
        pipeline.fit(table, y)
        predictions = pipeline.predict(table)
        assert len(predictions) == len(table)
        assert set(map(str, predictions)) <= set(CLASSES)

    @pytest.mark.parametrize("name", sorted(CLASSIFICATION_MODELS))
    def test_probabilities_are_finite_and_normalised(self, name: str) -> None:
        table = frame()
        y = labels(table)
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS[name],
            COLUMNS,
            random_seed=42,
            rule_class_order=CLASSES,
        )
        pipeline.fit(table, y)
        proba = np.asarray(pipeline.predict_proba(table), dtype=float)
        assert np.isfinite(proba).all()
        assert (proba >= 0.0).all()
        assert np.allclose(proba.sum(axis=1), 1.0)

    @pytest.mark.parametrize("name", sorted(CLASSIFICATION_MODELS))
    def test_a_fixed_seed_produces_identical_predictions(self, name: str) -> None:
        table = frame()
        y = labels(table)
        first = build_pipeline(
            CLASSIFICATION_MODELS[name],
            COLUMNS,
            random_seed=42,
            rule_class_order=CLASSES,
        ).fit(table, y)
        second = build_pipeline(
            CLASSIFICATION_MODELS[name],
            COLUMNS,
            random_seed=42,
            rule_class_order=CLASSES,
        ).fit(table, y)
        assert list(first.predict(table)) == list(second.predict(table))

    def test_the_dummy_classifier_predicts_the_prior(self) -> None:
        table = frame()
        y = labels(table)
        y[:] = "low"
        y[-1] = "high"
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["dummy"], COLUMNS, random_seed=0
        ).fit(table, y)
        assert set(map(str, pipeline.predict(table))) == {"low"}


class TestRegressors:
    @pytest.mark.parametrize("name", sorted(REGRESSION_MODELS))
    def test_every_regressor_fits_and_predicts_finite_values(self, name: str) -> None:
        table = frame()
        y = targets(table)
        pipeline = build_pipeline(REGRESSION_MODELS[name], COLUMNS, random_seed=42)
        pipeline.fit(table, y)
        predictions = np.asarray(pipeline.predict(table), dtype=float)
        assert predictions.shape[0] == len(table)
        assert np.isfinite(predictions).all()

    @pytest.mark.parametrize("name", sorted(REGRESSION_MODELS))
    def test_a_fixed_seed_produces_identical_predictions(self, name: str) -> None:
        table = frame()
        y = targets(table)
        first = build_pipeline(REGRESSION_MODELS[name], COLUMNS, random_seed=42).fit(
            table, y
        )
        second = build_pipeline(REGRESSION_MODELS[name], COLUMNS, random_seed=42).fit(
            table, y
        )
        assert np.allclose(first.predict(table), second.predict(table))

    def test_the_dummy_regressor_predicts_the_training_mean(self) -> None:
        table = frame()
        y = targets(table)
        pipeline = build_pipeline(
            REGRESSION_MODELS["dummy"], COLUMNS, random_seed=0
        ).fit(table, y)
        assert np.allclose(pipeline.predict(table), y.mean())


class TestRuleBaselines:
    def test_the_rule_classifier_uses_the_requested_feature(self) -> None:
        table = frame()
        estimator = RuleBasedThresholdClassifier(
            "feat__task_reaction_time_mean_ms", class_order=CLASSES
        )
        estimator.fit(table, labels(table))
        assert estimator.resolved_feature_ == "feat__task_reaction_time_mean_ms"

    def test_the_rule_classifier_falls_back_visibly(self) -> None:
        table = frame()
        estimator = RuleBasedThresholdClassifier("feat__absent", class_order=CLASSES)
        estimator.fit(table, labels(table))
        assert estimator.resolved_feature_ != "feat__absent"
        assert estimator.resolved_feature_.startswith("feat__")

    def test_the_rule_classifier_respects_the_class_order(self) -> None:
        table = frame()
        estimator = RuleBasedThresholdClassifier(
            "feat__task_correct_proportion", class_order=CLASSES
        )
        estimator.fit(table, labels(table))
        assert list(estimator.classes_) == list(CLASSES)
        highest = table["feat__task_correct_proportion"].idxmax()
        assert estimator.predict(table.loc[[highest]])[0] == "high"

    def test_the_rule_classifier_handles_missing_values(self) -> None:
        table = frame()
        estimator = RuleBasedThresholdClassifier(
            "feat__eye_openness_proxy_mean", class_order=CLASSES
        )
        estimator.fit(table, labels(table))
        predictions = estimator.predict(table)
        assert len(predictions) == len(table)
        assert set(map(str, predictions)) <= set(CLASSES)

    def test_rule_probabilities_are_normalised(self) -> None:
        table = frame()
        estimator = RuleBasedThresholdClassifier(
            "feat__task_correct_proportion", class_order=CLASSES
        )
        estimator.fit(table, labels(table))
        proba = estimator.predict_proba(table)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.isfinite(proba).all()

    def test_the_rule_regressor_maps_into_the_training_range(self) -> None:
        table = frame()
        y = targets(table)
        estimator = RuleBasedThresholdRegressor("feat__task_correct_proportion")
        estimator.fit(table, y)
        predictions = estimator.predict(table)
        assert np.isfinite(predictions).all()
        assert predictions.min() >= y.min() - 1e-9
        assert predictions.max() <= y.max() + 1e-9

    def test_the_rule_regressor_survives_a_constant_feature(self) -> None:
        table = frame()
        table["feat__task_correct_proportion"] = 0.5
        y = targets(table)
        estimator = RuleBasedThresholdRegressor("feat__task_correct_proportion")
        estimator.fit(table, y)
        assert np.allclose(estimator.predict(table), y.mean())

    def test_rule_baselines_are_flagged_and_disclaimed(self) -> None:
        for registry in (CLASSIFICATION_MODELS, REGRESSION_MODELS):
            spec = registry["rule_software_check"]
            assert spec.is_software_check_baseline is True
            assert RULE_BASELINE_DISCLAIMER in spec.notes
            assert "not a validated indicator" in spec.description

    def test_no_other_model_claims_to_be_a_software_check(self) -> None:
        for registry in (CLASSIFICATION_MODELS, REGRESSION_MODELS):
            flagged = [
                name
                for name, spec in registry.items()
                if spec.is_software_check_baseline
            ]
            assert flagged == ["rule_software_check"]


class TestPipelineConstruction:
    def test_the_preprocessor_is_inside_the_pipeline(self) -> None:
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["logistic_regression"], COLUMNS, random_seed=0
        )
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["preprocess", "estimator"]

    def test_tree_models_are_not_standardised(self) -> None:
        for name in ("random_forest", "hist_gradient_boosting"):
            assert CLASSIFICATION_MODELS[name].scale is False

    def test_linear_models_are_standardised(self) -> None:
        assert CLASSIFICATION_MODELS["logistic_regression"].scale is True
        assert REGRESSION_MODELS["ridge"].scale is True

    def test_native_nan_models_are_not_imputed(self) -> None:
        spec = CLASSIFICATION_MODELS["hist_gradient_boosting"]
        assert spec.imputation.value == "native_nan"

    def test_parameters_are_describable_without_fitting(self) -> None:
        pipeline = build_pipeline(
            CLASSIFICATION_MODELS["random_forest"], COLUMNS, random_seed=3
        )
        described = describe_parameters(pipeline)
        assert described["estimator_class"] == "RandomForestClassifier"
        assert described["parameters"]["random_state"] == 3


class TestNoChampionSelection:
    def test_no_registry_entry_is_marked_champion_or_production_ready(self) -> None:
        for registry in (CLASSIFICATION_MODELS, REGRESSION_MODELS):
            for spec in registry.values():
                blob = " ".join([spec.name, spec.description, *spec.notes]).lower()
                for banned in (
                    "champion",
                    "production-ready",
                    "production ready",
                    "best model",
                    "validated indicator of",
                    "state of the art",
                ):
                    assert banned not in blob or "not a validated indicator" in blob

    def test_the_registry_exposes_no_selection_helper(self) -> None:
        import engagevr.training.models as module

        for attribute in dir(module):
            assert "champion" not in attribute.lower()
            assert "best_model" not in attribute.lower()
