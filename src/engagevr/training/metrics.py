"""Metric computation with explicit unavailability.

A metric is computed only when its prerequisites hold.  When they do not,
the field is ``None`` and the reason is recorded in
``unavailable_metrics``.  Zero is never substituted: zero is a legitimate
score, and a reader cannot tell a genuine zero from a placeholder.

Documented formulas
-------------------
- **Macro precision / recall / F1** — the unweighted mean over classes for
  which the quantity is *defined*.  A class with no predicted instances
  has undefined precision; it is excluded from the mean rather than
  counted as zero, and its exclusion is recorded.
- **Multiclass Brier score** — ``mean_i || p_i - onehot(y_i) ||^2``, the
  mean squared Euclidean distance between the predicted probability
  vector and the one-hot true vector.  Range ``[0, 2]``.  The alternative
  convention (half this value) is not used; the definition is stated
  wherever the number is stored.
- **Expected calibration error** —
  ``ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|`` over ``M``
  equal-width bins of the maximum predicted class probability.
  ``acc(B_m)`` is the fraction of samples in bin ``m`` whose ``argmax``
  prediction is correct; ``conf(B_m)`` is the mean maximum predicted
  probability in bin ``m``.  Empty bins contribute nothing.
- **Fold aggregation** — the unweighted mean and population standard
  deviation over the folds in which the metric was defined, together with
  the count of such folds.  Folds are weighted equally so that one large
  group cannot dominate the summary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)

from engagevr.schemas.experiments import (
    AggregateMetric,
    CalibrationBin,
    CalibrationMetrics,
    ClassificationMetrics,
    ConfusionMatrix,
    PerClassMetrics,
    RegressionMetrics,
)

#: Default number of equal-width reliability bins.
DEFAULT_CALIBRATION_BINS = 10


def build_confusion_matrix(
    y_true: Sequence[str],
    y_predicted: Sequence[str],
    labels: Sequence[str],
) -> ConfusionMatrix:
    """Labelled confusion matrix; rows are true labels, columns predicted."""
    counts = confusion_matrix(list(y_true), list(y_predicted), labels=list(labels))
    return ConfusionMatrix(
        labels=tuple(labels),
        counts=tuple(tuple(int(v) for v in row) for row in counts),
    )


def sum_confusion_matrices(
    matrices: Sequence[ConfusionMatrix],
) -> ConfusionMatrix | None:
    """Element-wise sum of matrices sharing one label ordering."""
    usable = [m for m in matrices if m is not None]
    if not usable:
        return None
    labels = usable[0].labels
    if any(m.labels != labels for m in usable):
        return None
    size = len(labels)
    total = [[0] * size for _ in range(size)]
    for matrix in usable:
        for row in range(size):
            for column in range(size):
                total[row][column] += matrix.counts[row][column]
    return ConfusionMatrix(labels=labels, counts=tuple(tuple(row) for row in total))


def classification_metrics(
    *,
    y_true: Sequence[str],
    y_predicted: Sequence[str],
    labels: Sequence[str],
    group_ids: Sequence[str],
    calibration: Sequence[CalibrationMetrics] = (),
) -> ClassificationMetrics:
    """Compute classification metrics for one fold or one pooled set."""
    unavailable: dict[str, str] = {}
    n = len(y_true)
    support = {
        label: int(sum(1 for value in y_true if value == label)) for label in labels
    }

    if n == 0:
        return ClassificationMetrics(
            sample_count=0,
            independent_group_count=len(set(group_ids)),
            class_support=support,
            unavailable_metrics={
                key: "no samples in this evaluation set"
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                    "confusion_matrix",
                )
            },
        )

    truth = list(y_true)
    predicted = list(y_predicted)
    accuracy = float(accuracy_score(truth, predicted))

    present = [label for label in labels if support[label] > 0]
    if present:
        balanced = float(balanced_accuracy_score(truth, predicted))
    else:  # pragma: no cover - unreachable while n > 0
        balanced = None
        unavailable["balanced_accuracy"] = "no class is present in the true labels"

    precision, recall, f1, _support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(labels),
        average=None,
        zero_division=np.nan,
    )
    per_class = tuple(
        PerClassMetrics(
            label=label,
            support=support[label],
            precision=_finite(precision[index]),
            recall=_finite(recall[index]),
            f1=_finite(f1[index]),
        )
        for index, label in enumerate(labels)
    )
    macro_precision, undefined_precision = _macro(precision, labels)
    macro_recall, undefined_recall = _macro(recall, labels)
    macro_f1, undefined_f1 = _macro(f1, labels)
    for name, undefined in (
        ("macro_precision", undefined_precision),
        ("macro_recall", undefined_recall),
        ("macro_f1", undefined_f1),
    ):
        if undefined:
            unavailable[f"{name}_excluded_classes"] = (
                f"class(es) {undefined} had no defined value and were excluded "
                "from the macro mean rather than counted as zero"
            )
    if macro_precision is None:
        unavailable["macro_precision"] = "undefined for every class"
    if macro_recall is None:
        unavailable["macro_recall"] = "undefined for every class"
    if macro_f1 is None:
        unavailable["macro_f1"] = "undefined for every class"

    weighted = float(
        f1_score(
            truth,
            predicted,
            labels=list(labels),
            average="weighted",
            zero_division=0.0,
        )
    )

    return ClassificationMetrics(
        sample_count=n,
        independent_group_count=len(set(group_ids)),
        class_support=support,
        accuracy=accuracy,
        balanced_accuracy=balanced,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted,
        per_class=per_class,
        confusion_matrix=build_confusion_matrix(truth, predicted, labels),
        calibration=tuple(calibration),
        unavailable_metrics=unavailable,
    )


def _macro(values: np.ndarray, labels: Sequence[str]) -> tuple[float | None, list[str]]:
    defined = [
        (label, float(value))
        for label, value in zip(labels, values, strict=True)
        if np.isfinite(value)
    ]
    undefined = [
        label
        for label, value in zip(labels, values, strict=True)
        if not np.isfinite(value)
    ]
    if not defined:
        return None, undefined
    return float(np.mean([value for _label, value in defined])), undefined


def _finite(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def regression_metrics(
    *,
    y_true: Sequence[float],
    y_predicted: Sequence[float],
    group_ids: Sequence[str],
) -> RegressionMetrics:
    """Compute regression metrics for one fold or one pooled set."""
    unavailable: dict[str, str] = {}
    truth = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_predicted), dtype=float)

    if truth.size == 0:
        return RegressionMetrics(
            sample_count=0,
            independent_group_count=len(set(group_ids)),
            unavailable_metrics={
                key: "no samples in this evaluation set"
                for key in (
                    "mean_absolute_error",
                    "root_mean_squared_error",
                    "median_absolute_error",
                    "r_squared",
                )
            },
        )
    if not np.isfinite(predicted).all():
        raise ValueError(
            "regression predictions contain a non-finite value; a model that "
            "cannot produce a finite prediction must fail rather than emit one"
        )

    mae = float(mean_absolute_error(truth, predicted))
    rmse = float(np.sqrt(np.mean((truth - predicted) ** 2)))
    medae = float(median_absolute_error(truth, predicted))

    variance = float(np.var(truth, ddof=0))
    if truth.size < 2 or variance <= 0.0:
        r_squared = None
        unavailable["r_squared"] = (
            "R-squared is undefined when the true values have zero variance "
            "or fewer than two samples: the denominator of the formula is zero"
        )
    else:
        r_squared = float(r2_score(truth, predicted))

    return RegressionMetrics(
        sample_count=int(truth.size),
        independent_group_count=len(set(group_ids)),
        mean_absolute_error=mae,
        root_mean_squared_error=rmse,
        median_absolute_error=medae,
        r_squared=r_squared,
        target_mean=float(truth.mean()),
        target_std=float(np.sqrt(variance)),
        unavailable_metrics=unavailable,
    )


def multiclass_brier_score(
    probabilities: np.ndarray,
    y_true: Sequence[str],
    labels: Sequence[str],
) -> float:
    """Mean squared Euclidean distance between predicted and one-hot vectors.

    Range ``[0, 2]``. See the module docstring for the convention used.
    """
    index = {label: position for position, label in enumerate(labels)}
    onehot = np.zeros_like(probabilities, dtype=float)
    for row, label in enumerate(y_true):
        onehot[row, index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def reliability_bins(
    probabilities: np.ndarray,
    y_true: Sequence[str],
    labels: Sequence[str],
    *,
    bin_count: int = DEFAULT_CALIBRATION_BINS,
) -> tuple[CalibrationBin, ...]:
    """Equal-width reliability bins over the maximum predicted probability."""
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(labels, dtype=object)[probabilities.argmax(axis=1)]
    correct = np.asarray(
        [p == t for p, t in zip(predicted, y_true, strict=True)], dtype=float
    )
    edges = np.linspace(0.0, 1.0, bin_count + 1)

    bins: list[CalibrationBin] = []
    for index in range(bin_count):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bin_count - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        bins.append(
            CalibrationBin(
                bin_index=index,
                lower_edge=lower,
                upper_edge=upper,
                count=count,
                mean_confidence=(float(confidence[mask].mean()) if count else None),
                empirical_accuracy=float(correct[mask].mean()) if count else None,
            )
        )
    return tuple(bins)


def expected_calibration_error(bins: Sequence[CalibrationBin]) -> float | None:
    """ECE from reliability bins, using the formula in the module docstring."""
    total = sum(b.count for b in bins)
    if total == 0:
        return None
    error = 0.0
    for b in bins:
        if b.count == 0 or b.mean_confidence is None or b.empirical_accuracy is None:
            continue
        error += (b.count / total) * abs(b.empirical_accuracy - b.mean_confidence)
    return float(error)


def calibration_metrics(
    *,
    label: str,
    probabilities: np.ndarray | None,
    y_true: Sequence[str],
    labels: Sequence[str],
    bin_count: int = DEFAULT_CALIBRATION_BINS,
) -> CalibrationMetrics:
    """Score one probability set. Calibration is not certainty.

    A calibrated probability describes how often outcomes of this kind
    occurred at this predicted probability across the evaluated folds.  It
    is unrelated to signal quality, which describes the measurement.
    """
    if probabilities is None:
        return CalibrationMetrics(
            label=label,
            sample_count=len(y_true),
            unavailable_reason="the model does not produce class probabilities",
        )
    if len(y_true) == 0:
        return CalibrationMetrics(
            label=label,
            sample_count=0,
            unavailable_reason="no samples in this evaluation set",
        )
    if probabilities.shape[1] != len(labels):
        return CalibrationMetrics(
            label=label,
            sample_count=len(y_true),
            unavailable_reason=(
                f"probability matrix has {probabilities.shape[1]} columns but "
                f"{len(labels)} classes are declared"
            ),
        )
    if not np.isfinite(probabilities).all():
        return CalibrationMetrics(
            label=label,
            sample_count=len(y_true),
            unavailable_reason="the probability matrix contains a non-finite value",
        )

    bins = reliability_bins(probabilities, y_true, labels, bin_count=bin_count)
    unavailable: str | None = None
    # scikit-learn's log_loss binarises the labels internally in sorted
    # order. Passing an unsorted vocabulary would silently pair the wrong
    # probability column with each class, so the columns are reordered to
    # match sorted labels rather than the ordered class vocabulary.
    order = sorted(range(len(labels)), key=lambda index: labels[index])
    sorted_labels = [labels[index] for index in order]
    sorted_probabilities = probabilities[:, order]
    try:
        loss = float(log_loss(list(y_true), sorted_probabilities, labels=sorted_labels))
    except ValueError as exc:  # pragma: no cover - guarded by the checks above
        loss = None
        unavailable = f"log loss is undefined: {exc}"

    return CalibrationMetrics(
        label=label,
        sample_count=len(y_true),
        brier_score=multiclass_brier_score(probabilities, y_true, labels),
        log_loss=loss,
        expected_calibration_error=expected_calibration_error(bins),
        ece_bin_count=bin_count,
        bins=bins,
        unavailable_reason=unavailable,
    )


#: Classification metric fields that are aggregated across folds.
CLASSIFICATION_AGGREGATE_FIELDS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)

#: Regression metric fields that are aggregated across folds.
REGRESSION_AGGREGATE_FIELDS: tuple[str, ...] = (
    "mean_absolute_error",
    "root_mean_squared_error",
    "median_absolute_error",
    "r_squared",
)


def aggregate_fold_metrics(
    fold_metrics: Sequence[ClassificationMetrics | RegressionMetrics],
    fields: Sequence[str],
    *,
    total_fold_count: int | None = None,
) -> tuple[AggregateMetric, ...]:
    """Aggregate fold metrics using the unweighted mean over valid folds."""
    total = total_fold_count if total_fold_count is not None else len(fold_metrics)
    aggregates: list[AggregateMetric] = []
    for name in fields:
        values = [getattr(metrics, name, None) for metrics in fold_metrics]
        defined = [float(v) for v in values if v is not None]
        if defined:
            mean = float(np.mean(defined))
            deviation = float(np.std(defined, ddof=0))
            reason = None
        else:
            mean = None
            deviation = None
            reason = (
                f"{name} was undefined in every evaluated fold, so no aggregate exists"
            )
        aggregates.append(
            AggregateMetric(
                name=name,
                mean=mean,
                standard_deviation=deviation,
                valid_fold_count=len(defined),
                total_fold_count=total,
                fold_values=tuple(float(v) if v is not None else None for v in values),
                unavailable_reason=reason,
            )
        )
    return tuple(aggregates)


def aggregate_calibration_metrics(
    calibrations: Sequence[Sequence[CalibrationMetrics]],
    *,
    label: str,
    total_fold_count: int,
) -> tuple[AggregateMetric, ...]:
    """Aggregate one calibration variant's scores across folds."""
    selected = [
        next((c for c in fold if c.label == label), None) for fold in calibrations
    ]
    fields = ("brier_score", "log_loss", "expected_calibration_error")
    aggregates: list[AggregateMetric] = []
    for field_name in fields:
        values = [
            getattr(entry, field_name) if entry is not None else None
            for entry in selected
        ]
        defined = [float(v) for v in values if v is not None]
        aggregates.append(
            AggregateMetric(
                name=f"{label}.{field_name}",
                mean=float(np.mean(defined)) if defined else None,
                standard_deviation=(
                    float(np.std(defined, ddof=0)) if defined else None
                ),
                valid_fold_count=len(defined),
                total_fold_count=total_fold_count,
                fold_values=tuple(float(v) if v is not None else None for v in values),
                unavailable_reason=(
                    None
                    if defined
                    else f"{field_name} was undefined in every evaluated fold"
                ),
            )
        )
    return tuple(aggregates)
