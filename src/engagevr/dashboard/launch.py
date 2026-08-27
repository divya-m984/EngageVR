"""Construction of the ``streamlit run`` invocation.

Separated from the CLI so the interesting part is testable without
starting a server.  :func:`build_command` and :func:`build_environment`
are pure: they resolve paths, validate arguments, and return an argv
list and an environment overlay.  Only :func:`launch` actually spawns a
process, and nothing in the test suite calls it.

A browser is not opened automatically.  Streamlit's own default is
configured off here, because a research tool that seizes the display on
start is unpleasant to run over SSH and in a container.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from engagevr.dashboard.app import ARTIFACT_ROOT_ENV

#: The Streamlit script this command runs.
APP_MODULE_PATH = Path(__file__).resolve().parent / "app.py"

#: Documented development invocation, printed by ``--print-command``.
DEVELOPMENT_COMMAND = "uv run streamlit run src/engagevr/dashboard/app.py"


class DashboardLaunchError(RuntimeError):
    """The dashboard command cannot be constructed as requested."""


def app_path() -> Path:
    """Absolute path of the Streamlit application script."""
    if not APP_MODULE_PATH.is_file():  # pragma: no cover - packaging error
        raise DashboardLaunchError(
            f"the dashboard application script is missing from {APP_MODULE_PATH}"
        )
    return APP_MODULE_PATH


def build_command(
    *,
    address: str = "127.0.0.1",
    port: int = 8501,
    headless: bool = True,
    python_executable: str | None = None,
) -> tuple[str, ...]:
    """The argv that starts the dashboard.

    Bound to loopback by default.  This dashboard has no authentication,
    no authorisation, and no audit log, because it is a local research
    tool; binding it to a routable interface would publish a filesystem
    browser for the artifact root to anyone who can reach the port.
    """
    if not 1 <= port <= 65535:
        raise DashboardLaunchError(f"port must be in 1-65535, got {port}")
    if not address.strip():
        raise DashboardLaunchError("address must not be empty")
    executable = python_executable or sys.executable
    return (
        executable,
        "-m",
        "streamlit",
        "run",
        str(app_path()),
        "--server.address",
        address.strip(),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
        "--browser.gatherUsageStats",
        "false",
    )


def build_environment(
    artifact_root: Path | str | None, *, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment overlay handed to the Streamlit process.

    The artifact root travels as an environment variable rather than as
    a script argument because Streamlit owns ``sys.argv`` of the script
    it runs.
    """
    environment = dict(os.environ if base is None else base)
    if artifact_root is not None:
        environment[ARTIFACT_ROOT_ENV] = str(artifact_root)
    return environment


def describe(command: Sequence[str], artifact_root: Path | str | None) -> str:
    """A human-readable rendering of what :func:`launch` would run."""
    lines = [" ".join(command)]
    if artifact_root is not None:
        lines.insert(0, f"{ARTIFACT_ROOT_ENV}={artifact_root}")
    lines.append(f"Development equivalent: {DEVELOPMENT_COMMAND}")
    return "\n".join(lines)


def launch(
    command: Sequence[str], environment: Mapping[str, str]
) -> int:  # pragma: no cover - starts a server
    """Run the dashboard in the foreground and return its exit status.

    Not exercised by the test suite: a unit test that started a
    long-running server would be a test of Streamlit, not of this
    repository, and would need a browser or a socket to prove anything.
    """
    try:
        completed = subprocess.run(list(command), env=dict(environment), check=False)
    except FileNotFoundError as exc:
        raise DashboardLaunchError(
            f"could not start the dashboard: {exc}. Streamlit is a declared "
            "dependency; run `uv sync` and try again."
        ) from exc
    except KeyboardInterrupt:
        return 0
    return completed.returncode


__all__ = [
    "APP_MODULE_PATH",
    "DEVELOPMENT_COMMAND",
    "DashboardLaunchError",
    "app_path",
    "build_command",
    "build_environment",
    "describe",
    "launch",
]
