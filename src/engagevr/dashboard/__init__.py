"""Milestone 9: the read-only research dashboard.

The package is layered so that only the top layer knows about Streamlit:

``catalogue`` → ``loaders`` → ``views_*`` → ``components`` → ``pages`` → ``app``

with the session modes following the same shape:

``session_reader`` → ``session_catalogue`` → ``views_session`` /
``session_report`` → ``session_pages``

Everything up to and including ``views_*`` is framework-free and pure,
which is why the unit tests run without a browser, a socket, or a
server.  Importing this package therefore does **not** import Streamlit;
the modules that need it import it themselves.

The dashboard has three evidence modes, kept apart on purpose:
persisted **experiment artifacts** (the primary observatory), **live
observation** of a session recording as it is written, and **replay** of
a recording already on disk.  All three are read-only presentation.  Live
observation does not mean inference: no model is loaded, no camera is
opened, and no estimate is produced anywhere in this package.

Nothing in this package writes, deletes, or modifies an artifact or a
recording, opens a model file, runs a simulator, re-emits a message,
dispatches an adaptation command, or performs any Git operation.  Those
absences are checked by tests rather than left to discipline.
"""

from __future__ import annotations

from engagevr.dashboard.aggregation import (
    curve_series,
    group_counts,
    histogram,
    histogram_series,
    residuals,
)
from engagevr.dashboard.catalogue import (
    FAMILY_SIGNATURES,
    ArtifactReadError,
    build_catalogue,
    detect_family,
    inspect_run,
    verify_integrity,
)
from engagevr.dashboard.formatting import (
    build_table,
    count,
    format_value,
    metric,
    text,
)
from engagevr.dashboard.loaders import (
    DEFAULT_MAX_TABLE_ROWS,
    ArtifactColumnError,
    load_document,
    read_parquet,
)
from engagevr.dashboard.presentation import (
    LIMITATIONS,
    TERMINOLOGY,
    term_caption,
    unresolved_limitations,
)
from engagevr.dashboard.session_catalogue import (
    SessionRead,
    build_session_catalogue,
    read_session,
)
from engagevr.dashboard.session_reader import (
    SessionReadError,
    decode_records,
    read_stream,
    sequence_observations,
)
from engagevr.dashboard.session_report import (
    build_report,
    report_to_json,
    report_to_markdown,
)
from engagevr.dashboard.views_adaptation import load_adaptation
from engagevr.dashboard.views_dataset import (
    load_dataset_provenance,
    load_signal_quality,
)
from engagevr.dashboard.views_fusion import load_fusion, load_personalization
from engagevr.dashboard.views_models import load_classification, load_regression
from engagevr.dashboard.views_session import (
    catalogue_table,
    record_table,
    replay_state_for,
)
from engagevr.dashboard.views_uncertainty import load_uncertainty

__all__ = [
    "DEFAULT_MAX_TABLE_ROWS",
    "FAMILY_SIGNATURES",
    "LIMITATIONS",
    "TERMINOLOGY",
    "ArtifactColumnError",
    "ArtifactReadError",
    "SessionRead",
    "SessionReadError",
    "build_catalogue",
    "build_report",
    "build_session_catalogue",
    "build_table",
    "catalogue_table",
    "count",
    "curve_series",
    "decode_records",
    "detect_family",
    "format_value",
    "group_counts",
    "histogram",
    "histogram_series",
    "inspect_run",
    "load_adaptation",
    "load_classification",
    "load_dataset_provenance",
    "load_document",
    "load_fusion",
    "load_personalization",
    "load_regression",
    "load_signal_quality",
    "load_uncertainty",
    "metric",
    "read_parquet",
    "read_session",
    "read_stream",
    "record_table",
    "replay_state_for",
    "report_to_json",
    "report_to_markdown",
    "residuals",
    "sequence_observations",
    "term_caption",
    "text",
    "unresolved_limitations",
    "verify_integrity",
]
