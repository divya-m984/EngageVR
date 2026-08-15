"""Baseline model registries and the rule-based software-check estimators.

Two registries are kept, one per task type, because a classifier and a
regressor are not interchangeable and a single registry invites a caller
to pick the wrong one.

Model selection policy
----------------------
Only interpretable, well-understood estimators appear here: a dummy
baseline, a linear model, a random forest, scikit-learn's histogram
gradient boosting, and a deterministic rule.  No neural network, no
temporal model, and no fusion architecture is present; those belong to
later milestones.

XGBoost is deliberately **not** a dependency.  Scikit-learn's
:class:`~sklearn.ensemble.HistGradientBoostingClassifier` implements the
same histogram-based boosted-tree algorithm, handles ``NaN`` natively,
and is already installed, so adding a second gradient-boosting library
would add a dependency without adding a capability.

The rule baseline
-----------------
:class:`RuleBasedThresholdClassifier` and
:class:`RuleBasedThresholdRegressor` are **software checks**.  They exist
to prove that the harness can carry an estimator that does no learning,
so that a "good" score from a learned model can be compared against
something trivial.  They are not validated indicators of anything, the
feature each one thresholds is an arbitrary implementation choice, and
their outputs are not calibrated probabilities.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from engagevr.schemas.targets import TaskType
from engagevr.training.preprocessing import ImputationStrategy, build_preprocessor

#: Disclaimer attached to every rule-baseline result.
RULE_BASELINE_DISCLAIMER = (
    "Deterministic rule-based SOFTWARE-CHECK baseline. It thresholds one "
    "arbitrarily chosen feature. It is not a validated indicator of "
    "engagement or cognitive load, its probabilities are not calibrated, "
    "and it must never be described as scientifically validated."
)


class UnsupportedModelError(ValueError):
    """A requested model is not implemented for the requested task type."""


class ModelKind(enum.StrEnum):
    """Coarse family of an estimator, used for reporting and interpretation."""

    DUMMY = "dummy"
    LINEAR = "linear"
    TREE = "tree"
    RULE = "rule"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything needed to build and describe one baseline."""

    name: str
    kind: ModelKind
    task_type: TaskType
    description: str
    imputation: ImputationStrategy
    scale: bool
    supports_probability: bool
    build: Callable[[int], BaseEstimator]
    parameter_grid: dict[str, list[Any]] = field(default_factory=dict)
    is_software_check_baseline: bool = False
    notes: tuple[str, ...] = ()


class RuleBasedThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic threshold rule over a single feature.

    Fitting computes two quantiles of one feature **on the training rows
    only** and uses them as class boundaries.  Nothing else is learned.

    This is a software-check baseline, not a validated indicator.  See
    :data:`RULE_BASELINE_DISCLAIMER`.
    """

    def __init__(
        self,
        feature_name: str | None = None,
        *,
        class_order: Sequence[str] | None = None,
        higher_is_greater: bool = True,
        confidence: float = 0.8,
    ) -> None:
        self.feature_name = feature_name
        self.class_order = class_order
        self.higher_is_greater = higher_is_greater
        self.confidence = confidence

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: Sequence[str]
    ) -> RuleBasedThresholdClassifier:
        """Fit the two quantile boundaries on the training rows only."""
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.resolved_feature_ = _resolve_feature(frame, self.feature_name)
        labels = np.asarray(y, dtype=object)
        observed = sorted({str(v) for v in labels})
        if self.class_order is not None:
            ordered = [c for c in self.class_order if c in set(observed)]
            ordered.extend(c for c in observed if c not in set(ordered))
        else:
            ordered = observed
        self.classes_ = np.asarray(ordered, dtype=object)

        values = pd.to_numeric(frame[self.resolved_feature_], errors="coerce")
        finite = values.dropna().to_numpy(dtype=float)
        if finite.size == 0:
            self.thresholds_ = np.asarray([], dtype=float)
        else:
            quantiles = np.linspace(0.0, 1.0, len(self.classes_) + 1)[1:-1]
            self.thresholds_ = (
                np.quantile(finite, quantiles)
                if quantiles.size
                else (np.asarray([], dtype=float))
            )

        counts = {label: int(np.sum(labels == label)) for label in self.classes_}
        total = max(1, sum(counts.values()))
        self.class_prior_ = np.asarray(
            [counts[label] / total for label in self.classes_], dtype=float
        )
        self.fallback_class_ = str(self.classes_[int(np.argmax(self.class_prior_))])
        return self

    def _bucket(self, value: float) -> int:
        index = int(np.searchsorted(self.thresholds_, value, side="right"))
        if not self.higher_is_greater:
            index = len(self.classes_) - 1 - index
        return int(np.clip(index, 0, len(self.classes_) - 1))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict a class label per row."""
        frame = _as_frame(X, self.feature_names_in_)
        values = pd.to_numeric(frame[self.resolved_feature_], errors="coerce")
        fallback = int(np.argmax(self.class_prior_))
        predictions = [
            self.classes_[fallback if np.isnan(v) else self._bucket(float(v))]
            for v in values.to_numpy(dtype=float)
        ]
        return np.asarray(predictions, dtype=object)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return deterministic pseudo-probabilities.

        The chosen class receives ``confidence``; the remainder is spread
        across the other classes in proportion to their training
        frequency.  These are **not** calibrated probabilities and must
        not be read as such; they exist so the rule baseline can be passed
        through the same probability-scoring code as every other model.
        """
        predictions = self.predict(X)
        n_classes = len(self.classes_)
        if n_classes == 1:
            return np.ones((len(predictions), 1), dtype=float)
        others = 1.0 - self.confidence
        proba = np.empty((len(predictions), n_classes), dtype=float)
        index = {label: position for position, label in enumerate(self.classes_)}
        for row, label in enumerate(predictions):
            position = index[label]
            remaining = self.class_prior_.copy()
            remaining[position] = 0.0
            total = remaining.sum()
            if total <= 0.0:
                remaining = np.full(n_classes, 1.0 / (n_classes - 1), dtype=float)
                remaining[position] = 0.0
                total = remaining.sum()
            proba[row] = others * remaining / total
            proba[row, position] = self.confidence
        return proba

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class RuleBasedThresholdRegressor(RegressorMixin, BaseEstimator):
    """Deterministic linear rescaling of a single feature onto the target range.

    A software-check baseline only.  See :data:`RULE_BASELINE_DISCLAIMER`.
    """

    def __init__(
        self,
        feature_name: str | None = None,
        *,
        higher_is_greater: bool = True,
    ) -> None:
        self.feature_name = feature_name
        self.higher_is_greater = higher_is_greater

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: Sequence[float]
    ) -> RuleBasedThresholdRegressor:
        """Fit the source and target ranges on the training rows only."""
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.resolved_feature_ = _resolve_feature(frame, self.feature_name)
        values = pd.to_numeric(frame[self.resolved_feature_], errors="coerce")
        finite = values.dropna().to_numpy(dtype=float)
        targets = np.asarray(y, dtype=float)
        self.target_mean_ = float(targets.mean()) if targets.size else 0.0
        if finite.size == 0 or float(finite.max() - finite.min()) <= 0.0:
            self.source_minimum_ = 0.0
            self.source_span_ = 0.0
        else:
            self.source_minimum_ = float(finite.min())
            self.source_span_ = float(finite.max() - finite.min())
        self.target_minimum_ = float(targets.min()) if targets.size else 0.0
        self.target_span_ = (
            float(targets.max() - targets.min()) if targets.size else 0.0
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict a numeric value per row."""
        frame = _as_frame(X, self.feature_names_in_)
        values = pd.to_numeric(frame[self.resolved_feature_], errors="coerce").to_numpy(
            dtype=float
        )
        if self.source_span_ <= 0.0:
            return np.full(values.shape[0], self.target_mean_, dtype=float)
        scaled = (values - self.source_minimum_) / self.source_span_
        if not self.higher_is_greater:
            scaled = 1.0 - scaled
        predictions = self.target_minimum_ + np.clip(scaled, 0.0, 1.0) * (
            self.target_span_
        )
        return np.where(np.isnan(values), self.target_mean_, predictions)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


def _as_frame(
    X: pd.DataFrame | np.ndarray, columns: np.ndarray | None = None
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if columns is None:
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(np.asarray(X).shape[1])])
    return pd.DataFrame(X, columns=list(columns))


def _resolve_feature(frame: pd.DataFrame, requested: str | None) -> str:
    """Resolve the rule's feature, falling back to the first column.

    The fallback keeps the baseline usable inside an ablation that removed
    the preferred feature.  Which feature was actually used is recorded in
    ``resolved_feature_`` and reported, so the substitution is never
    silent.
    """
    columns = [str(c) for c in frame.columns]
    if not columns:
        raise UnsupportedModelError(
            "the rule baseline needs at least one predictor column"
        )
    if requested is not None and requested in columns:
        return requested
    measured = [c for c in columns if c.startswith("feat__")]
    return measured[0] if measured else columns[0]


def _logistic(seed: int) -> BaseEstimator:
    # class_weight is left at None deliberately: re-weighting changes what
    # the fitted probabilities mean, and there is no documented reason in
    # this project to prefer balanced errors over calibrated frequencies.
    return LogisticRegression(
        max_iter=2000,
        random_state=seed,
        class_weight=None,
    )


CLASSIFICATION_MODELS: dict[str, ModelSpec] = {
    "dummy": ModelSpec(
        name="dummy",
        kind=ModelKind.DUMMY,
        task_type=TaskType.CLASSIFICATION,
        description=(
            "Predicts the training class prior. The floor any real model "
            "must clear before its score means anything."
        ),
        imputation=ImputationStrategy.MEDIAN_WITH_INDICATOR,
        scale=False,
        supports_probability=True,
        build=lambda seed: DummyClassifier(strategy="prior", random_state=seed),
    ),
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        kind=ModelKind.LINEAR,
        task_type=TaskType.CLASSIFICATION,
        description=(
            "Multinomial logistic regression on standardised features. "
            "Coefficients are directly readable in standard-deviation units."
        ),
        imputation=ImputationStrategy.MEDIAN_WITH_INDICATOR,
        scale=True,
        supports_probability=True,
        build=_logistic,
        parameter_grid={"estimator__C": [0.1, 1.0, 10.0]},
        notes=(
            "Features are standardised, so a coefficient is the change in "
            "log-odds per one training-fold standard deviation.",
        ),
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        kind=ModelKind.TREE,
        task_type=TaskType.CLASSIFICATION,
        description="Random forest of 200 trees with a fixed random state.",
        imputation=ImputationStrategy.MEDIAN_WITH_INDICATOR,
        scale=False,
        supports_probability=True,
        build=lambda seed: RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=1
        ),
        parameter_grid={"estimator__max_depth": [None, 6]},
    ),
    "hist_gradient_boosting": ModelSpec(
        name="hist_gradient_boosting",
        kind=ModelKind.TREE,
        task_type=TaskType.CLASSIFICATION,
        description=(
            "Histogram-based gradient boosting. Handles missing values "
            "natively, so its inputs are not imputed."
        ),
        imputation=ImputationStrategy.NATIVE_NAN,
        scale=False,
        supports_probability=True,
        build=lambda seed: HistGradientBoostingClassifier(random_state=seed),
        parameter_grid={"estimator__learning_rate": [0.05, 0.1]},
        notes=(
            "Missing values reach the estimator unimputed; the split "
            "algorithm sends them to whichever child reduces the loss.",
        ),
    ),
    "rule_software_check": ModelSpec(
        name="rule_software_check",
        kind=ModelKind.RULE,
        task_type=TaskType.CLASSIFICATION,
        description=(
            "Deterministic quantile-threshold rule over one feature. A "
            "software check, not a validated indicator."
        ),
        imputation=ImputationStrategy.NATIVE_NAN,
        scale=False,
        supports_probability=True,
        build=lambda seed: RuleBasedThresholdClassifier(),
        is_software_check_baseline=True,
        notes=(RULE_BASELINE_DISCLAIMER,),
    ),
}


REGRESSION_MODELS: dict[str, ModelSpec] = {
    "dummy": ModelSpec(
        name="dummy",
        kind=ModelKind.DUMMY,
        task_type=TaskType.REGRESSION,
        description="Predicts the training mean.",
        imputation=ImputationStrategy.MEDIAN_WITH_INDICATOR,
        scale=False,
        supports_probability=False,
        build=lambda seed: DummyRegressor(strategy="mean"),
    ),
    "ridge": ModelSpec(
        name="ridge",
        kind=ModelKind.LINEAR,
        task_type=TaskType.REGRESSION,
        description=(
            "L2-penalised linear regression on standardised features. "
            "Chosen over ElasticNet because it has one hyperparameter and "
            "no sparsity pattern to over-interpret on correlated inputs."
        ),
        imputation=ImputationStrategy.MEDIAN_WITH_INDICATOR,
        scale=True,
        supports_probability=False,
        build=lambda seed: Ridge(alpha=1.0, random_state=seed),
        parameter_grid={"estimator__alpha": [0.1, 1.0, 10.0]},
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        kind=ModelKind.TREE,
        task_type=TaskType.REGRESSION,
        description="Random forest of 200 trees with a fixed random state.",
        imputation=ImputationStrategy.MEDIAN_WITH_INDICATOR,
        scale=False,
        supports_probability=False,
        build=lambda seed: RandomForestRegressor(
            n_estimators=200, random_state=seed, n_jobs=1
        ),
        parameter_grid={"estimator__max_depth": [None, 6]},
    ),
    "hist_gradient_boosting": ModelSpec(
        name="hist_gradient_boosting",
        kind=ModelKind.TREE,
        task_type=TaskType.REGRESSION,
        description=(
            "Histogram-based gradient boosting. Handles missing values "
            "natively, so its inputs are not imputed."
        ),
        imputation=ImputationStrategy.NATIVE_NAN,
        scale=False,
        supports_probability=False,
        build=lambda seed: HistGradientBoostingRegressor(random_state=seed),
        parameter_grid={"estimator__learning_rate": [0.05, 0.1]},
    ),
    "rule_software_check": ModelSpec(
        name="rule_software_check",
        kind=ModelKind.RULE,
        task_type=TaskType.REGRESSION,
        description=(
            "Deterministic linear rescaling of one feature onto the "
            "training target range. A software check, not a validated "
            "indicator."
        ),
        imputation=ImputationStrategy.NATIVE_NAN,
        scale=False,
        supports_probability=False,
        build=lambda seed: RuleBasedThresholdRegressor(),
        is_software_check_baseline=True,
        notes=(RULE_BASELINE_DISCLAIMER,),
    ),
}


def registry_for(task_type: TaskType) -> dict[str, ModelSpec]:
    """Return the model registry for ``task_type``."""
    return (
        CLASSIFICATION_MODELS
        if task_type is TaskType.CLASSIFICATION
        else REGRESSION_MODELS
    )


def get_model_spec(name: str, task_type: TaskType) -> ModelSpec:
    """Return the spec for ``name``.

    Raises
    ------
    UnsupportedModelError
        If the model is not implemented for this task type.
    """
    registry = registry_for(task_type)
    try:
        return registry[name]
    except KeyError as exc:
        raise UnsupportedModelError(
            f"model {name!r} is not implemented for {task_type.value} "
            f"targets; available: {', '.join(sorted(registry))}"
        ) from exc


def build_pipeline(
    spec: ModelSpec,
    columns: Sequence[str],
    *,
    random_seed: int,
    rule_feature: str | None = None,
    rule_class_order: Sequence[str] | None = None,
) -> Pipeline:
    """Build the unfitted preprocessing-plus-estimator pipeline for ``spec``.

    The preprocessor is inside the pipeline on purpose: every statistic it
    learns is fitted on whatever rows the pipeline is fitted on, which is
    always a training fold and never the test fold.
    """
    preprocessor = build_preprocessor(
        columns, strategy=spec.imputation, scale=spec.scale
    )
    estimator = spec.build(random_seed)
    if isinstance(
        estimator, RuleBasedThresholdClassifier | RuleBasedThresholdRegressor
    ):
        if rule_feature is not None:
            estimator.set_params(feature_name=rule_feature)
        if rule_class_order is not None and isinstance(
            estimator, RuleBasedThresholdClassifier
        ):
            # Without the ordered vocabulary the rule would bucket into
            # alphabetically sorted classes, so "high" would sit below "low".
            estimator.set_params(class_order=tuple(rule_class_order))
    pipeline = Pipeline([("preprocess", preprocessor), ("estimator", estimator)])
    # Pandas output keeps column names alive through the transformer, which
    # is what lets coefficients and importances be reported by name.
    pipeline.set_output(transform="pandas")
    return pipeline


def describe_parameters(pipeline: Pipeline) -> dict[str, object]:
    """Readable parameter snapshot of a built pipeline's estimator."""
    estimator = pipeline.named_steps["estimator"]
    params = estimator.get_params(deep=False)
    return {
        "estimator_class": type(estimator).__name__,
        "parameters": {key: _describe(value) for key, value in sorted(params.items())},
    }


def _describe(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return repr(value)
