"""Streamlit entry point for the EngageVR research dashboard.

Run it through the project CLI::

    uv run python -m engagevr dashboard

or directly, which is the development path::

    uv run streamlit run src/engagevr/dashboard/app.py

The artifact root and the session root come from ``configs/defaults.yaml``
and can each be overridden for one session with an environment variable
(:data:`ARTIFACT_ROOT_ENV`, :data:`SESSION_ROOT_ENV`).  They are passed
that way rather than as command-line arguments because Streamlit owns
``sys.argv``, and reading a variable is easier to test than parsing
around a framework.

Three evidence modes
--------------------
The sidebar's first control chooses between **experiment artifacts**,
**live session**, and **session replay**.  They are never merged into one
ambiguous state: each mode has its own catalogue, its own pages, and its
own statement of what its evidence is.  The artifact observatory remains
the primary view and is unchanged by the other two.

Everything this module does is read-only.  It has no button that
retrains a model, recalibrates a probability, dispatches an adaptation,
edits a manifest, deletes a run, or writes to a recording, and it has no
code path that could acquire one without a reviewer noticing: the
modules it imports contain no write function at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from engagevr.config import DashboardConfig, load_config
from engagevr.dashboard import components as ui
from engagevr.dashboard import pages, session_pages
from engagevr.dashboard.pages import PAGES, PageContext
from engagevr.dashboard.session_catalogue import build_session_catalogue
from engagevr.dashboard.session_pages import SESSION_PAGES, SessionPageContext
from engagevr.dashboard.views_session import catalogue_table, mode_statement
from engagevr.schemas.dashboard import (
    DASHBOARD_DISCLAIMER,
    DASHBOARD_PURPOSE,
    DashboardCatalogue,
    DashboardRunFamily,
    DashboardRunStatus,
    DashboardRunSummary,
)
from engagevr.schemas.dashboard_session import (
    DashboardSessionCatalogue,
    DashboardSessionMode,
    DashboardSessionSummary,
)

#: Environment variable that overrides the configured artifact root.
ARTIFACT_ROOT_ENV = "ENGAGEVR_DASHBOARD_ARTIFACT_ROOT"

#: Environment variable that overrides the configured session root.
SESSION_ROOT_ENV = "ENGAGEVR_DASHBOARD_SESSION_ROOT"

#: Title of the browser tab and the sidebar.
PAGE_TITLE = "EngageVR research dashboard"

#: The three evidence modes, in sidebar order. The artifact observatory
#: is first because it is the primary research view.
MODE_LABELS: tuple[tuple[str, DashboardSessionMode], ...] = (
    ("Experiment artifacts", DashboardSessionMode.ARTIFACT),
    ("Live session", DashboardSessionMode.LIVE),
    ("Session replay", DashboardSessionMode.REPLAY),
)


def resolve_settings() -> tuple[DashboardConfig, Path]:
    """The dashboard settings and the artifact root to scan."""
    settings = load_config().dashboard
    override = os.environ.get(ARTIFACT_ROOT_ENV)
    root = Path(override) if override else Path(settings.artifact_root)
    return settings, root


def resolve_session_root(settings: DashboardConfig) -> Path:
    """The session root the live and replay modes scan."""
    override = os.environ.get(SESSION_ROOT_ENV)
    return Path(override) if override else Path(settings.session_root)


@st.cache_data(show_spinner="Scanning artifact root...")
def _cached_catalogue(
    artifact_root: str, validate_checksums: bool, cache_key: tuple[tuple[str, int], ...]
) -> DashboardCatalogue:
    """Cached catalogue scan.

    ``cache_key`` carries the name and modification time of every
    candidate directory, so the cache invalidates when a run is written,
    replaced, or removed.  Only the parse is cached; no mutable state,
    no open file handle, and no side effect crosses this boundary.
    """
    del cache_key  # participates in the cache key only
    return pages.load_catalogue(
        Path(artifact_root), validate_checksums=validate_checksums
    )


def catalogue_cache_key(artifact_root: Path) -> tuple[tuple[str, int], ...]:
    """Name and modification time of every candidate run directory."""
    if not artifact_root.is_dir():
        return ()
    entries: list[tuple[str, int]] = []
    for child in sorted(artifact_root.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            try:
                entries.append((child.name, int(child.stat().st_mtime_ns)))
            except OSError:  # pragma: no cover - vanished mid-scan
                entries.append((child.name, 0))
    return tuple(entries)


def _session_catalogue(
    session_root: Path, mode: DashboardSessionMode
) -> DashboardSessionCatalogue:
    """Scan the session root, deliberately without a cache.

    The experiment catalogue is cached because an experiment run is
    written once and then never changes.  A session recording is the
    opposite: it is appended to while it is being observed, and an append
    leaves the containing directory's modification time untouched.  A
    modification-time cache key would therefore go stale in exactly the
    mode that must not go stale, so this scan runs every time. It is
    cheap: a handful of small local files.
    """
    return build_session_catalogue(session_root, mode=mode)


def _run_label(run: DashboardRunSummary) -> str:
    """A selector label carrying enough provenance to be unambiguous."""
    provenance = run.provenance
    marks = [provenance.family.value]
    if provenance.target_name:
        marks.append(provenance.target_name)
    if provenance.status is not DashboardRunStatus.COMPLETED:
        marks.append(provenance.status.value.upper())
    if provenance.is_synthetic:
        marks.append("synthetic")
    return f"{run.directory_name}  ·  {' · '.join(marks)}"


def _select_run(
    catalogue: DashboardCatalogue, settings: DashboardConfig
) -> DashboardRunSummary | None:
    """The sidebar run selector, with deterministic filters."""
    st.sidebar.subheader("Run selection")
    families = ("(all families)", *(f.value for f in catalogue.families()))
    default_family = settings.resolved_family()
    family_index = 0
    if default_family is not None and default_family.value in families:
        family_index = families.index(default_family.value)
    chosen_family = st.sidebar.selectbox(
        "Run family", options=families, index=family_index
    )

    runs = list(catalogue.runs)
    if chosen_family != "(all families)":
        runs = [
            run
            for run in runs
            if run.provenance.family is DashboardRunFamily(chosen_family)
        ]

    targets = (
        "(all targets)",
        *sorted(
            {run.provenance.target_name for run in runs if run.provenance.target_name}
        ),
    )
    chosen_target = st.sidebar.selectbox("Target", options=targets, index=0)
    if chosen_target != "(all targets)":
        runs = [run for run in runs if run.provenance.target_name == chosen_target]

    tasks = (
        "(all task types)",
        *sorted({run.provenance.task_type for run in runs if run.provenance.task_type}),
    )
    chosen_task = st.sidebar.selectbox("Task type", options=tasks, index=0)
    if chosen_task != "(all task types)":
        runs = [run for run in runs if run.provenance.task_type == chosen_task]

    synthetic = st.sidebar.selectbox(
        "Provenance", options=("(all)", "synthetic only", "non-synthetic only"), index=0
    )
    if synthetic == "synthetic only":
        runs = [run for run in runs if run.provenance.is_synthetic]
    elif synthetic == "non-synthetic only":
        runs = [run for run in runs if not run.provenance.is_synthetic]

    statuses = (
        "(all statuses)",
        *sorted({run.provenance.status.value for run in runs}),
    )
    chosen_status = st.sidebar.selectbox("Status", options=statuses, index=0)
    if chosen_status != "(all statuses)":
        runs = [run for run in runs if run.provenance.status.value == chosen_status]

    if not runs:
        st.sidebar.info("No run matches these filters.")
        return None
    labels = [_run_label(run) for run in runs]
    chosen = st.sidebar.selectbox("Run", options=labels, index=0)
    return runs[labels.index(chosen)]


def _session_label(session: DashboardSessionSummary) -> str:
    """A selector label carrying enough state to be unambiguous.

    Led by the directory, because that is what addresses a recording. The
    recorded session id follows when it differs, so a copied recording is
    visibly a copy rather than a duplicate entry.
    """
    provenance = session.provenance
    marks = [provenance.status.value, f"{session.complete_record_count} records"]
    if not provenance.identifier_matches_directory:
        marks.append(f"recorded id {provenance.session_id}")
    if provenance.is_synthetic:
        marks.append("synthetic")
    if provenance.replayed_message_count:
        marks.append("contains replayed records")
    return f"{provenance.directory_name}  ·  {' · '.join(marks)}"


def _select_session(catalogue: DashboardSessionCatalogue) -> str | None:
    """The sidebar session selector, returning a directory name."""
    st.sidebar.subheader("Session selection")
    if not catalogue.sessions:
        st.sidebar.info("No recorded session was found under the session root.")
        return None
    labels = [_session_label(session) for session in catalogue.sessions]
    chosen = st.sidebar.selectbox("Session", options=labels, index=0)
    return catalogue.sessions[labels.index(chosen)].provenance.directory_name


def _render_artifact_mode(settings: DashboardConfig, artifact_root: Path) -> None:
    """The Milestone 5-8 experiment observatory, unchanged."""
    st.sidebar.text(f"Artifact root: {artifact_root}")
    catalogue = _cached_catalogue(
        str(artifact_root),
        settings.validate_checksums,
        catalogue_cache_key(artifact_root),
    )
    run = _select_run(catalogue, settings) if catalogue.runs else None

    st.sidebar.divider()
    page_names = [name for name, _ in PAGES]
    chosen_page = st.sidebar.radio("Page", options=page_names, index=0)
    st.sidebar.divider()
    st.sidebar.caption(DASHBOARD_DISCLAIMER)

    context = PageContext(
        catalogue=catalogue,
        run=run,
        max_table_rows=settings.max_table_rows,
        show_subject_ids=settings.show_subject_ids,
        validate_checksums=settings.validate_checksums,
    )
    handler = dict(PAGES)[chosen_page]
    getattr(pages, handler)(context)


def _render_session_mode(settings: DashboardConfig, mode: DashboardSessionMode) -> None:
    """The live-observation or replay mode."""
    session_root = resolve_session_root(settings)
    st.sidebar.text(f"Session root: {session_root}")
    catalogue = _session_catalogue(session_root, mode)
    directory_name = _select_session(catalogue)
    st.sidebar.divider()
    st.sidebar.caption(DASHBOARD_DISCLAIMER)

    context = SessionPageContext(
        mode=mode,
        session_root=session_root,
        catalogue=catalogue,
        session_directory_name=directory_name,
        max_table_rows=settings.max_table_rows,
        live_refresh_seconds=settings.live_refresh_seconds,
        enable_session_report_export=settings.enable_session_report_export,
    )
    with st.expander("Every recorded session under this root"):
        ui.render_table(catalogue_table(catalogue, max_rows=settings.max_table_rows))
    getattr(session_pages, SESSION_PAGES[mode])(context)


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    settings, artifact_root = resolve_settings()

    st.sidebar.title(PAGE_TITLE)
    st.sidebar.caption(DASHBOARD_PURPOSE)

    labels = [label for label, _ in MODE_LABELS]
    chosen_mode = st.sidebar.radio("Evidence source", options=labels, index=0)
    mode = dict(MODE_LABELS)[chosen_mode]
    st.sidebar.caption(mode_statement(mode))
    st.sidebar.divider()

    if mode is DashboardSessionMode.ARTIFACT:
        _render_artifact_mode(settings, artifact_root)
    else:
        _render_session_mode(settings, mode)


if __name__ == "__main__":
    main()
