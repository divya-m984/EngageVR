"""Evaluation metrics for rPPG heart-rate estimates.

Metrics are computed **only** against genuine reference physiological
measurements read from a documented dataset.  There is no code path that
produces an error metric from a synthetic trace: a synthetic "error" is
a property of the generator, not evidence about the estimator, and
reporting one as performance would be fabrication.

Every metric function returns ``None`` rather than a placeholder when no
valid reference exists, and :class:`RppgEvaluationMetrics` records
``reference_available`` so a consumer cannot mistake an absent metric for
a zero one.
"""

from __future__ import annotations

import numpy as np

from engagevr.schemas.rppg import (
    DatasetEvaluationRecord,
    RppgEvaluationMetrics,
    RppgMethod,
)


def _errors(records: list[DatasetEvaluationRecord]) -> list[float]:
    """Signed estimate-minus-reference errors over comparable records."""
    return [
        r.estimated_bpm - r.reference_bpm
        for r in records
        if not r.abstained
        and r.estimated_bpm is not None
        and r.reference_bpm is not None
    ]


def aggregate_metrics(
    records: list[DatasetEvaluationRecord],
    *,
    dataset: str,
    method: RppgMethod,
    subject_id: str | None = None,
    provenance: str = "",
) -> RppgEvaluationMetrics:
    """Aggregate evaluation records into coverage and error metrics.

    Coverage and window counts are always reported, because they describe
    the estimator's abstention behaviour and are meaningful with or
    without a reference signal.  Error metrics are populated only when at
    least one window has both an estimate and a real reference value.
    """
    attempted = len(records)
    valid = sum(1 for r in records if not r.abstained and r.estimated_bpm is not None)
    abstained = attempted - valid

    metrics = RppgEvaluationMetrics(
        dataset=dataset,
        method=method,
        subject_id=subject_id,
        n_windows_attempted=attempted,
        n_windows_valid=valid,
        n_windows_abstained=abstained,
        valid_window_pct=(100.0 * valid / attempted) if attempted else 0.0,
        coverage=(valid / attempted) if attempted else 0.0,
        provenance=provenance,
    )

    errors = _errors(records)
    metrics.n_windows_with_reference = sum(
        1 for r in records if r.reference_bpm is not None
    )
    if not errors:
        # No genuine reference comparison exists. Leave every error metric
        # None rather than emitting a misleading zero.
        metrics.reference_available = False
        return metrics

    arr = np.asarray(errors, dtype=np.float64)
    metrics.reference_available = True
    metrics.mae_bpm = float(np.mean(np.abs(arr)))
    metrics.rmse_bpm = float(np.sqrt(np.mean(arr**2)))
    metrics.bias_bpm = float(np.mean(arr))
    metrics.error_std_bpm = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return metrics


def per_subject_metrics(
    records: list[DatasetEvaluationRecord],
    *,
    dataset: str,
    method: RppgMethod,
    provenance: str = "",
) -> list[RppgEvaluationMetrics]:
    """Aggregate separately for each subject.

    Subjects are kept separate deliberately.  Pooling windows across
    subjects hides between-subject variation and, in a modelling context,
    would mix what should be distinct train/test units.
    """
    by_subject: dict[str, list[DatasetEvaluationRecord]] = {}
    for record in records:
        by_subject.setdefault(record.subject_id, []).append(record)

    return [
        aggregate_metrics(
            subject_records,
            dataset=dataset,
            method=method,
            subject_id=subject_id,
            provenance=provenance,
        )
        for subject_id, subject_records in sorted(by_subject.items())
    ]
