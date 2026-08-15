"""Offline probability calibration on disjoint groups.

Milestone 5 scope
-----------------
This module fits and *evaluates* calibrators offline.  It does not
implement abstention, coverage-versus-performance analysis, or any online
confidence policy; those belong to Milestone 7.

Disjointness
------------
A calibrator fitted on the same rows that fitted the base estimator
learns to correct predictions the estimator has already memorised, and
the correction does not transfer.  So, within each outer fold:

- the base estimator is fitted on the **fit groups**;
- the calibrator is fitted on the **calibration groups**, which are drawn
  from the training groups and are disjoint from the fit groups;
- the **outer test groups** are touched only for the final evaluation and
  are never used to fit anything.

API note
--------
``CalibratedClassifierCV(cv="prefit")`` was deprecated in scikit-learn 1.6
and removed in 1.8.  The supported way to calibrate an already-fitted
estimator is to wrap it in :class:`sklearn.frozen.FrozenEstimator`, which
is what this module does.  With a frozen estimator, ``ensemble="auto"``
resolves to ``False`` and all supplied data is used for calibration.

Calibration is not certainty, and it is not signal quality.  A calibrated
probability states how often outcomes of this kind occurred at this
predicted probability across the evaluated folds.  It says nothing about
whether the camera signal was usable.
"""

from __future__ import annotations

import enum
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

#: Minimum calibration rows before isotonic regression is attempted.
#: Isotonic regression is non-parametric and will happily interpolate a
#: step function through a handful of points, producing a calibration
#: curve that describes the calibration sample and nothing else.
MINIMUM_ISOTONIC_SAMPLES = 50

#: Minimum calibration rows per class before isotonic regression is used.
MINIMUM_ISOTONIC_SAMPLES_PER_CLASS = 10


class CalibrationError(ValueError):
    """A calibration set is not usable."""


class CalibrationMethod(enum.StrEnum):
    """Calibration methods offered in this milestone."""

    NONE = "none"
    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"


@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    """Result of attempting to calibrate one fitted estimator."""

    method: CalibrationMethod
    calibrated_estimator: BaseEstimator | None
    fit_row_count: int
    calibration_row_count: int
    fit_group_count: int
    calibration_group_count: int
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """Whether a calibrated estimator was produced."""
        return self.calibrated_estimator is not None


def assert_calibration_disjoint(
    *,
    fit_groups: Sequence[str],
    calibration_groups: Sequence[str],
    test_groups: Sequence[str],
) -> None:
    """Assert the three group sets are pairwise disjoint.

    Raises
    ------
    CalibrationError
        On the first overlap found, naming the offending groups.
    """
    fit = set(fit_groups)
    calibration = set(calibration_groups)
    test = set(test_groups)
    overlap = fit & calibration
    if overlap:
        raise CalibrationError(
            "calibration groups overlap the groups used to fit the base "
            f"estimator: {sorted(overlap)[:5]}. A calibrator fitted on the "
            "estimator's own training rows corrects memorised predictions."
        )
    overlap = calibration & test
    if overlap:
        raise CalibrationError(
            "calibration groups overlap the outer test groups: "
            f"{sorted(overlap)[:5]}. Calibrating on the test fold makes every "
            "calibration metric computed on it meaningless."
        )
    overlap = fit & test
    if overlap:
        raise CalibrationError(
            f"training groups overlap the outer test groups: {sorted(overlap)[:5]}"
        )


def isotonic_is_supported(
    y_calibration: Sequence[str],
    *,
    minimum_samples: int = MINIMUM_ISOTONIC_SAMPLES,
    minimum_per_class: int = MINIMUM_ISOTONIC_SAMPLES_PER_CLASS,
) -> tuple[bool, str]:
    """Whether the calibration set can support isotonic regression."""
    total = len(y_calibration)
    if total < minimum_samples:
        return False, (
            f"isotonic calibration requires at least {minimum_samples} "
            f"calibration rows; {total} are available. Isotonic regression is "
            "non-parametric and would fit a step function to noise."
        )
    counts: dict[str, int] = {}
    for label in y_calibration:
        counts[str(label)] = counts.get(str(label), 0) + 1
    thin = sorted(label for label, count in counts.items() if count < minimum_per_class)
    if thin:
        return False, (
            f"isotonic calibration requires at least {minimum_per_class} "
            f"calibration rows per class; class(es) {thin} have fewer"
        )
    return True, "the calibration set meets the isotonic minimum-data thresholds"


def calibrate_classifier(
    pipeline: BaseEstimator,
    *,
    X_fit: pd.DataFrame,
    y_fit: Sequence[str],
    X_calibration: pd.DataFrame,
    y_calibration: Sequence[str],
    method: CalibrationMethod,
    fit_groups: Sequence[str],
    calibration_groups: Sequence[str],
    test_groups: Sequence[str],
) -> tuple[BaseEstimator, CalibrationOutcome]:
    """Fit a base estimator and, when possible, a calibrator on disjoint groups.

    Returns the fitted **base** estimator and the calibration outcome.  The
    base estimator is always returned so that uncalibrated probabilities
    remain available for comparison.

    Raises
    ------
    CalibrationError
        If the group sets are not disjoint.
    """
    assert_calibration_disjoint(
        fit_groups=fit_groups,
        calibration_groups=calibration_groups,
        test_groups=test_groups,
    )

    base = clone(pipeline)
    base.fit(X_fit, list(y_fit))

    if method is CalibrationMethod.NONE:
        return base, CalibrationOutcome(
            method=method,
            calibrated_estimator=None,
            fit_row_count=len(X_fit),
            calibration_row_count=len(X_calibration),
            fit_group_count=len(set(fit_groups)),
            calibration_group_count=len(set(calibration_groups)),
            unavailable_reason="calibration was not requested",
        )

    if len(X_calibration) == 0:
        return base, CalibrationOutcome(
            method=method,
            calibrated_estimator=None,
            fit_row_count=len(X_fit),
            calibration_row_count=0,
            fit_group_count=len(set(fit_groups)),
            calibration_group_count=0,
            unavailable_reason=(
                "no calibration rows are available; calibration is skipped "
                "rather than fitted on the estimator's own training rows"
            ),
        )

    fit_classes = sorted({str(v) for v in y_fit})
    calibration_classes = sorted({str(v) for v in y_calibration})
    missing = [c for c in fit_classes if c not in set(calibration_classes)]
    if missing:
        return base, CalibrationOutcome(
            method=method,
            calibrated_estimator=None,
            fit_row_count=len(X_fit),
            calibration_row_count=len(X_calibration),
            fit_group_count=len(set(fit_groups)),
            calibration_group_count=len(set(calibration_groups)),
            unavailable_reason=(
                f"the calibration set contains no example of class(es) "
                f"{missing}, so their probabilities cannot be calibrated"
            ),
        )

    if method is CalibrationMethod.ISOTONIC:
        supported, reason = isotonic_is_supported([str(v) for v in y_calibration])
        if not supported:
            return base, CalibrationOutcome(
                method=method,
                calibrated_estimator=None,
                fit_row_count=len(X_fit),
                calibration_row_count=len(X_calibration),
                fit_group_count=len(set(fit_groups)),
                calibration_group_count=len(set(calibration_groups)),
                unavailable_reason=reason,
            )

    calibrated = CalibratedClassifierCV(
        FrozenEstimator(base), method=method.value, ensemble="auto"
    )
    with warnings.catch_warnings():
        # With a frozen estimator, ``ensemble="auto"`` resolves to False and
        # the whole calibration set is used to fit a single calibrator
        # (``len(calibrated_classifiers_) == 1``). scikit-learn still builds
        # a default CV splitter on the way in and warns about thin classes
        # even though it never splits. Suppressing that specific message
        # keeps a misleading warning out of the run log; every genuine
        # thin-class condition is caught by the explicit checks above.
        warnings.filterwarnings(
            "ignore",
            message="The least populated class in y has only",
            category=UserWarning,
        )
        calibrated.fit(X_calibration, list(y_calibration))

    return base, CalibrationOutcome(
        method=method,
        calibrated_estimator=calibrated,
        fit_row_count=len(X_fit),
        calibration_row_count=len(X_calibration),
        fit_group_count=len(set(fit_groups)),
        calibration_group_count=len(set(calibration_groups)),
    )


def aligned_probabilities(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    labels: Sequence[str],
) -> np.ndarray | None:
    """Predicted probabilities reordered to ``labels``.

    Returns ``None`` when the estimator produces no probabilities.  Columns
    for classes the estimator never saw are filled with zero and the rows
    are renormalised, so a probability row always sums to one.
    """
    if not hasattr(estimator, "predict_proba"):
        return None
    raw = np.asarray(estimator.predict_proba(X), dtype=float)
    known = [str(c) for c in getattr(estimator, "classes_", labels)]
    aligned = np.zeros((raw.shape[0], len(labels)), dtype=float)
    position = {label: index for index, label in enumerate(labels)}
    for column, label in enumerate(known):
        if label in position:
            aligned[:, position[label]] = raw[:, column]
    totals = aligned.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0.0, totals, 1.0)
    return aligned / safe
