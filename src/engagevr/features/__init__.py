"""Windowed feature datasets for the EngageVR modelling pipeline.

Every feature this package can produce is declared in
:mod:`engagevr.features.catalog`.  A value that is not in the catalog
cannot enter a dataset, and a dataset column that is not in the catalog
cannot reach a model.

Nothing in this package infers engagement, cognition, emotion, fatigue,
stress, or attention.  Aggregation produces observable summaries with
explicit availability; interpretation is a separate, later, and currently
unvalidated step.
"""

from __future__ import annotations

from engagevr.features.catalog import (
    FEATURE_CATALOG,
    FEATURE_CATALOG_VERSION,
    feature_names_for_modalities,
    get_catalog,
)
from engagevr.features.windowing import (
    PartialWindowPolicy,
    WindowingError,
    WindowSpec,
    build_windows,
)

__all__ = [
    "FEATURE_CATALOG",
    "FEATURE_CATALOG_VERSION",
    "PartialWindowPolicy",
    "WindowSpec",
    "WindowingError",
    "build_windows",
    "feature_names_for_modalities",
    "get_catalog",
]
