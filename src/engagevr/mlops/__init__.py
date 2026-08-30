"""Milestone 10: the operational layer around Milestones 5--9.

This package **orchestrates**.  It does not model.  There is no second
training pipeline here, no second fusion implementation, no second
uncertainty engine, no second adaptation controller, and no second
dashboard: every stage calls an existing public runner or CLI and then
records what happened.

    source + config + synthetic generators
              |
        reproducible DVC stages
              |
       persisted experiment artifacts        (Milestones 5-8, unchanged)
              |
       MLflow experiment-tracking metadata   (mlflow_tracking.py)
              |
       versioned model/artifact manifests    (model_version.py)
              |
        drift / system verification          (drift.py, smoke.py)
              |
       Docker / CI / release workflow

Two boundaries are load-bearing.

**Nothing here mutates an existing artifact.**  Tracking and versioning
read a run directory and write separate records elsewhere.  A run that
has been logged to MLflow is byte-identical to one that has not.

**Nothing here confers scientific status.**  Reproducibility is not
validity, tracking is not validation, registration is not approval,
packaging is not production readiness, and a distribution-shift statistic
is an engineering diagnostic.  Every synthetic record produced by this
package carries ``scientific_evaluation_eligible=false`` and the
``SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION`` banner, and the
schemas refuse to record anything else.

MLflow is imported lazily, inside the functions that need it, so that
importing this package -- or any Milestone 5--9 command -- neither loads
the tracking client nor touches a tracking store.  Tracking is opt-in:
``mlops.mlflow.enabled`` is ``false`` in ``configs/defaults.yaml``.
"""

from __future__ import annotations

__all__ = [
    "drift",
    "fingerprints",
    "mlflow_tracking",
    "model_version",
    "pipeline",
    "reproducibility",
    "smoke",
]
