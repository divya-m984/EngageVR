"""Milestone 9 CLI: ``dashboard``, ``dashboard-check``, ``dashboard-sessions``.

``dashboard`` starts the local Streamlit application.  ``dashboard-check``
does the interesting half without a server: it scans the artifact root,
classifies every run, reports integrity, and exits.
``dashboard-sessions`` does the same for recorded sessions and can build
one session's report to stdout.  Those two are what the test suite
exercises, because a unit test that started a long-running web server
would be testing Streamlit rather than this repository.

No command here writes anything.  ``dashboard-check`` and
``dashboard-sessions`` open JSON and Parquet for reading and print;
``dashboard`` builds an argv and hands control to Streamlit.  The session
report is **printed**, never saved: writing it would be the one file
operation this milestone does not have.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from engagevr.schemas.dashboard import (
    DASHBOARD_DISCLAIMER,
    DASHBOARD_PURPOSE,
    SYNTHETIC_BANNER,
    ArtifactIntegrityStatus,
    DashboardCatalogue,
    DashboardRunStatus,
)

#: Printed by both commands, so the boundary is stated at the terminal too.
READ_ONLY_NOTE = (
    "This dashboard is READ-ONLY. It does not train, calibrate, re-run, "
    "dispatch, acknowledge, modify, or delete anything, and it never opens a "
    "model file."
)


def add_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the Milestone 9 commands."""
    dashboard = sub.add_parser(
        "dashboard",
        help="Launch the local READ-ONLY research dashboard (Streamlit).",
        description=(
            "READ-ONLY. Launch the local research dashboard for inspecting "
            "experiment artifacts. It displays what a run recorded and "
            "changes nothing: no training, no calibration, no dispatch."
        ),
    )
    dashboard.add_argument(
        "--artifact-root",
        type=str,
        default=None,
        help=(
            "Directory to scan for run directories. Defaults to "
            "dashboard.artifact_root in configs/defaults.yaml."
        ),
    )
    dashboard.add_argument(
        "--address",
        type=str,
        default="127.0.0.1",
        help="Interface to bind. Loopback by default: this tool has no authentication.",
    )
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.add_argument(
        "--open-browser",
        action="store_true",
        help="Let Streamlit open a browser window. Off by default.",
    )
    dashboard.add_argument(
        "--print-command",
        action="store_true",
        help="Print the command that would be run, then exit without "
        "starting a server.",
    )

    check = sub.add_parser(
        "dashboard-check",
        help="Scan the artifact root and report every run, without a server.",
        description=(
            "Discover run directories, classify each by artifact signature, "
            "verify checksums, and report. Starts no server and writes "
            "nothing."
        ),
    )
    check.add_argument(
        "--artifact-root",
        type=str,
        default=None,
        help=(
            "Directory to scan. Defaults to dashboard.artifact_root in "
            "configs/defaults.yaml."
        ),
    )
    check.add_argument(
        "--no-validate-checksums",
        action="store_true",
        help=(
            "Skip checksum verification. The integrity status then reads "
            "'not checked', which is not the same as a passing check."
        ),
    )
    check.add_argument(
        "--json",
        action="store_true",
        help="Emit the catalogue as JSON instead of a table.",
    )

    sessions = sub.add_parser(
        "dashboard-sessions",
        help="List recorded sessions, or build one session report, read-only.",
        description=(
            "Discover recorded sessions and report their status, composition, "
            "and provenance. With --session, build that session's report "
            "instead. Starts no server, sends nothing, and writes nothing: "
            "the report is printed, not saved."
        ),
    )
    sessions.add_argument(
        "--session-root",
        type=str,
        default=None,
        help=(
            "Directory to scan for recorded sessions. Defaults to "
            "dashboard.session_root in configs/defaults.yaml."
        ),
    )
    sessions.add_argument(
        "--session",
        type=str,
        default=None,
        help="Build the report for this session id and print it.",
    )
    sessions.add_argument(
        "--mode",
        type=str,
        choices=("live", "replay"),
        default="replay",
        help=(
            "Which read-only presentation mode the report records. Both read "
            "the same persisted records; the mode states how they were viewed."
        ),
    )
    sessions.add_argument(
        "--format",
        type=str,
        choices=("json", "markdown"),
        default="json",
        help="Report format. Both carry the same content and provenance.",
    )


def _artifact_root(argument: str | None) -> Path:
    if argument is not None:
        return Path(argument)
    from engagevr.config import load_config

    return Path(load_config().dashboard.artifact_root)


def run_dashboard(args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard."""
    from engagevr.dashboard.launch import (
        DashboardLaunchError,
        build_command,
        build_environment,
        describe,
        launch,
    )

    try:
        root = _artifact_root(args.artifact_root)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        command = build_command(
            address=args.address,
            port=args.port,
            headless=not args.open_browser,
        )
    except DashboardLaunchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not root.exists():
        print(
            f"Note: the artifact root {root} does not exist yet. The "
            "dashboard will start and report that there is nothing to show.",
            file=sys.stderr,
        )

    environment = build_environment(root)
    if args.print_command:
        print(describe(command, root))
        return 0

    print(DASHBOARD_PURPOSE)
    print(READ_ONLY_NOTE)
    print(DASHBOARD_DISCLAIMER)
    print()
    print(f"Artifact root: {root}")
    print(f"Address:       http://{args.address}:{args.port}")
    print()
    try:
        return launch(command, environment)
    except DashboardLaunchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def run_dashboard_check(args: argparse.Namespace) -> int:
    """Scan the artifact root and report, without starting a server."""
    from engagevr.dashboard.catalogue import build_catalogue

    root = _artifact_root(args.artifact_root)
    catalogue = build_catalogue(root, validate_checksums=not args.no_validate_checksums)
    if args.json:
        print(json.dumps(_as_json(catalogue), indent=2))
        return 0
    _print_catalogue(catalogue)
    return 0


def _session_root(argument: str | None) -> Path:
    if argument is not None:
        return Path(argument)
    from engagevr.config import load_config

    return Path(load_config().dashboard.session_root)


def run_dashboard_sessions(args: argparse.Namespace) -> int:
    """List recorded sessions, or print one session's report."""
    from engagevr.dashboard.session_catalogue import (
        build_session_catalogue,
        read_session,
    )
    from engagevr.dashboard.session_report import (
        SessionReportError,
        build_report,
        report_to_json,
        report_to_markdown,
    )
    from engagevr.schemas.dashboard_session import DashboardSessionMode

    root = _session_root(args.session_root)
    mode = DashboardSessionMode(args.mode)

    if args.session is None:
        catalogue = build_session_catalogue(root, mode=mode)
        _print_sessions(catalogue)
        return 0

    directory = root / args.session
    if not directory.is_dir():
        print(
            f"Error: no session directory {directory}. Run "
            "'dashboard-sessions' with no --session to list what is there.",
            file=sys.stderr,
        )
        return 2
    read = read_session(directory, mode=mode)
    try:
        report = build_report(read, mode=mode)
    except SessionReportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.format == "markdown":
        print(report_to_markdown(report), end="")
    else:
        print(report_to_json(report), end="")
    return 0


def _print_sessions(catalogue: Any) -> None:
    print(DASHBOARD_PURPOSE)
    print(READ_ONLY_NOTE)
    print()
    print(f"Session root:   {catalogue.session_root}")
    print(f"Root exists:    {catalogue.root_exists}")
    print(f"Sessions found: {len(catalogue.sessions)}")
    print()
    if catalogue.sessions:
        header = (
            f"{'session':<28} {'status':<22} {'records':>8} "
            f"{'malformed':>10} {'synthetic':>10} {'replayed':>9} {'eligible':>9}"
        )
        print(header)
        print("-" * len(header))
        for session in catalogue.sessions:
            provenance = session.provenance
            print(
                f"{provenance.session_id[:27]:<28} "
                f"{provenance.status.value:<22} "
                f"{session.complete_record_count:>8} "
                f"{session.malformed_record_count:>10} "
                f"{provenance.synthetic_message_count:>10} "
                f"{provenance.replayed_message_count:>9} "
                f"{provenance.scientific_evaluation_eligible!s:>9}"
            )
        print()
    for entry in catalogue.warnings:
        print(f"[{entry.level.value}] {entry.message}")
    synthetic = sum(1 for s in catalogue.sessions if s.provenance.is_synthetic)
    if synthetic:
        print()
        print(SYNTHETIC_BANNER)
    print(DASHBOARD_DISCLAIMER)


def _as_json(catalogue: DashboardCatalogue) -> dict[str, Any]:
    return {
        "artifact_root": catalogue.artifact_root,
        "root_exists": catalogue.root_exists,
        "run_count": len(catalogue.runs),
        "read_only": True,
        "note": READ_ONLY_NOTE,
        "disclaimer": DASHBOARD_DISCLAIMER,
        "runs": [
            {
                "directory": run.directory_name,
                "run_id": run.provenance.run_id,
                "family": run.provenance.family.value,
                "status": run.provenance.status.value,
                "integrity": run.provenance.integrity.value,
                "is_synthetic": run.provenance.is_synthetic,
                "scientific_evaluation_eligible": (
                    run.provenance.scientific_evaluation_eligible
                ),
                "target_name": run.provenance.target_name,
                "task_type": run.provenance.task_type,
                "dataset_fingerprint": run.provenance.dataset_fingerprint,
                "missing_required_artifacts": list(run.missing_required_artifacts),
                "failure_reason": run.provenance.failure_reason,
                "warnings": [
                    {"level": w.level.value, "message": w.message}
                    for w in run.provenance.warnings
                ],
            }
            for run in catalogue.runs
        ],
        "warnings": [
            {"level": w.level.value, "message": w.message} for w in catalogue.warnings
        ],
    }


def _print_catalogue(catalogue: DashboardCatalogue) -> None:
    print(DASHBOARD_PURPOSE)
    print(READ_ONLY_NOTE)
    print()
    print(f"Artifact root:  {catalogue.artifact_root}")
    print(f"Root exists:    {catalogue.root_exists}")
    print(f"Runs found:     {len(catalogue.runs)}")
    print()

    if catalogue.runs:
        header = (
            f"{'directory':<34} {'family':<16} {'status':<12} "
            f"{'integrity':<28} {'synthetic':<10} {'eligible':<9}"
        )
        print(header)
        print("-" * len(header))
        for run in catalogue.runs:
            provenance = run.provenance
            print(
                f"{run.directory_name[:33]:<34} "
                f"{provenance.family.value:<16} "
                f"{provenance.status.value:<12} "
                f"{provenance.integrity.value:<28} "
                f"{provenance.is_synthetic!s:<10} "
                f"{provenance.scientific_evaluation_eligible!s:<9}"
            )
        print()

    problems = [
        run
        for run in catalogue.runs
        if run.provenance.status is not DashboardRunStatus.COMPLETED
        or run.provenance.integrity
        in (
            ArtifactIntegrityStatus.MISMATCHED,
            ArtifactIntegrityStatus.REFERENCED_FILE_MISSING,
            ArtifactIntegrityStatus.CHECKSUM_FILE_CORRUPT,
        )
    ]
    if problems:
        print("Runs needing attention:")
        for run in problems:
            for entry in run.provenance.warnings:
                print(f"  [{entry.level.value}] {run.directory_name}: {entry.message}")
            if not run.provenance.warnings:
                print(
                    f"  [{run.provenance.status.value}] {run.directory_name}: "
                    f"integrity {run.provenance.integrity.value}"
                )
        print()

    for entry in catalogue.warnings:
        print(f"[{entry.level.value}] {entry.message}")

    synthetic = sum(1 for run in catalogue.runs if run.provenance.is_synthetic)
    eligible = sum(
        1 for run in catalogue.runs if run.provenance.scientific_evaluation_eligible
    )
    print()
    print(f"Synthetic runs:                 {synthetic}/{len(catalogue.runs)}")
    print(f"Scientifically eligible runs:   {eligible}/{len(catalogue.runs)}")
    if synthetic:
        print()
        print(SYNTHETIC_BANNER)
    print(DASHBOARD_DISCLAIMER)


__all__ = [
    "READ_ONLY_NOTE",
    "add_parsers",
    "run_dashboard",
    "run_dashboard_check",
    "run_dashboard_sessions",
]
