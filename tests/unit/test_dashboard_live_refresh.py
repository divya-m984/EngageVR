"""The live mode's automatic refresh, and its boundaries.

``docs/PROJECT_PLAN.md`` asks Milestone 9 for a *real-time* mode.  A
snapshot the reader has to re-request is a current view, not a real-time
one, so the live page refreshes on its own through Streamlit's native
``st.fragment(run_every=...)`` (DEC-094, revised).

Three things must hold and are checked here.

*Only the live page refreshes.*  Replay must not auto-advance and the
experiment-artifact observatory must not poll, so the fragment is
constructed inside :func:`~engagevr.dashboard.session_pages.live_session_page`
and nowhere else.  Both an AST check over the source and a behavioural
check through ``AppTest`` are made: the AST test fails when someone
writes a second ``run_every``, and the behavioural test fails when the
one that exists reaches the wrong mode.

*The configured interval is the one used.*  Not a hard-coded default, and
not a clamped substitute for a value that was refused.

*A refused interval starts no timer.*  Zero, negative, non-finite, and
sub-minimum intervals are refused with a stated reason.

No test here sleeps, starts a server, opens a socket, or waits on a
timer.  ``run_every`` is checked where it is configured — at the fragment
boundary — because a test that waited for a real firing would be a slow
test of Streamlit rather than a test of this repository.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import engagevr.dashboard as dashboard_package
from engagevr.config import DashboardConfig, load_config
from engagevr.dashboard import views_session as views
from engagevr.dashboard.session_pages import SESSION_PAGES
from engagevr.schemas.dashboard import DashboardError
from engagevr.schemas.dashboard_session import DashboardSessionMode
from tests.unit import session_fixtures as sfx

DASHBOARD = Path(dashboard_package.__file__).parent
SESSION_PAGES_SOURCE = DASHBOARD / "session_pages.py"

#: Generous: the script parses a handful of small local files.
TIMEOUT_SECONDS = 60.0

ARTIFACT_MODE = "Experiment artifacts"
LIVE_MODE = "Live session"
REPLAY_MODE = "Session replay"


# --- Interval validation -------------------------------------------------


class TestLiveRefreshInterval:
    def test_the_configured_default_is_accepted_unchanged(self) -> None:
        configured = load_config().dashboard.live_refresh_seconds
        assert views.live_refresh_interval(configured) == pytest.approx(configured)

    def test_the_minimum_is_conservative(self) -> None:
        assert views.MINIMUM_LIVE_REFRESH_SECONDS >= 2.0

    def test_the_minimum_itself_is_accepted(self) -> None:
        minimum = views.MINIMUM_LIVE_REFRESH_SECONDS
        assert views.live_refresh_interval(minimum) == pytest.approx(minimum)

    @pytest.mark.parametrize("value", [0.0, -1.0, -0.001])
    def test_zero_and_negative_intervals_are_refused(self, value: float) -> None:
        with pytest.raises(DashboardError, match="greater than zero"):
            views.live_refresh_interval(value)

    @pytest.mark.parametrize(
        "value", [math.nan, math.inf, -math.inf, float("nan"), float("inf")]
    )
    def test_non_finite_intervals_are_refused(self, value: float) -> None:
        with pytest.raises(DashboardError, match="finite"):
            views.live_refresh_interval(value)

    @pytest.mark.parametrize("value", [0.1, 0.5, 1.0, 1.999])
    def test_an_interval_below_the_minimum_is_refused(self, value: float) -> None:
        with pytest.raises(DashboardError, match="at least"):
            views.live_refresh_interval(value)

    @pytest.mark.parametrize("value", ["5", None, [5.0]])
    def test_a_non_numeric_interval_is_refused(self, value: object) -> None:
        with pytest.raises(DashboardError, match="number of seconds"):
            views.live_refresh_interval(value)  # type: ignore[arg-type]

    def test_a_boolean_is_not_a_number_of_seconds(self) -> None:
        with pytest.raises(DashboardError, match="number of seconds"):
            views.live_refresh_interval(True)  # type: ignore[arg-type]

    def test_a_refused_interval_is_never_clamped_to_the_minimum(self) -> None:
        """Substituting a default would refresh at a cadence nobody chose."""
        with pytest.raises(DashboardError):
            views.live_refresh_interval(0.25)

    def test_the_statement_names_the_interval_and_denies_inference(self) -> None:
        statement = views.refresh_statement(5.0)
        assert "Automatic refresh: every 5 seconds" in statement
        assert "runs no model" in statement
        assert "Real-time observation is not real-time inference." in statement


class TestConfiguredInterval:
    def test_the_configuration_refuses_a_sub_minimum_interval(self) -> None:
        with pytest.raises(ValueError, match="live_refresh_seconds"):
            DashboardConfig(live_refresh_seconds=0.5)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_the_configuration_refuses_zero_and_negative(self, value: float) -> None:
        with pytest.raises(ValueError, match="live_refresh_seconds"):
            DashboardConfig(live_refresh_seconds=value)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_the_configuration_refuses_non_finite(self, value: float) -> None:
        with pytest.raises(ValueError, match="live_refresh_seconds"):
            DashboardConfig(live_refresh_seconds=value)

    def test_the_shipped_default_passes_the_page_validator(self) -> None:
        """The two validators must not disagree about the same value."""
        configured = load_config().dashboard.live_refresh_seconds
        assert views.live_refresh_interval(configured) == pytest.approx(configured)


# --- Where the timer may and may not appear ------------------------------


def _function(source: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not found in {source}")


def _run_every_keywords(node: ast.AST) -> list[ast.keyword]:
    return [
        keyword
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
        if keyword.arg == "run_every"
    ]


def _fragment_calls(node: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        target = call.func
        if isinstance(target, ast.Attribute) and target.attr == "fragment":
            calls.append(call)
        elif isinstance(target, ast.Name) and target.id == "fragment":
            calls.append(call)
    return calls


class TestOnlyTheLivePageSchedulesARefresh:
    def test_the_live_page_constructs_a_fragment_with_run_every(self) -> None:
        page = _function(SESSION_PAGES_SOURCE, "live_session_page")
        fragments = _fragment_calls(page)
        assert fragments, "the live page constructs no st.fragment"
        assert _run_every_keywords(page), "the live fragment sets no run_every"

    def test_the_replay_page_constructs_no_refreshing_fragment(self) -> None:
        page = _function(SESSION_PAGES_SOURCE, "replay_page")
        assert not _fragment_calls(page)
        assert not _run_every_keywords(page)

    def test_the_replay_cursor_helper_schedules_nothing(self) -> None:
        for name in ("_replay_state", "_move_cursor"):
            helper = _function(SESSION_PAGES_SOURCE, name)
            assert not _run_every_keywords(helper), name

    def test_run_every_appears_exactly_once_in_the_whole_package(self) -> None:
        """One timer, in one place, so there is one thing to review."""
        occurrences = {
            path.name: len(
                _run_every_keywords(ast.parse(path.read_text(encoding="utf-8")))
            )
            for path in sorted(DASHBOARD.glob("*.py"))
        }
        assert occurrences.pop("session_pages.py") == 1
        assert set(occurrences.values()) == {0}, occurrences

    def test_the_artifact_pages_never_schedule_a_refresh(self) -> None:
        for name in ("pages.py", "app.py", "components.py"):
            tree = ast.parse((DASHBOARD / name).read_text(encoding="utf-8"))
            assert not _fragment_calls(tree), name
            assert not _run_every_keywords(tree), name

    def test_no_dashboard_module_sleeps(self) -> None:
        """A timer is Streamlit's job; blocking the script is nobody's."""
        for path in sorted(DASHBOARD.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            assert "time.sleep" not in text, path.name
            assert "sleep(" not in text, path.name

    def test_the_two_session_pages_are_still_distinct_handlers(self) -> None:
        assert SESSION_PAGES[DashboardSessionMode.LIVE] == "live_session_page"
        assert SESSION_PAGES[DashboardSessionMode.REPLAY] == "replay_page"


# --- The timer, observed at the fragment boundary ------------------------


@pytest.fixture
def session_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A session root with one completed and one still-active recording."""
    from engagevr.dashboard.app import ARTIFACT_ROOT_ENV, SESSION_ROOT_ENV

    root = tmp_path / "sessions"
    sfx.write_session(root, "synthetic-completed")
    sfx.write_session(root, "synthetic-active", with_summary=False, completed=False)
    monkeypatch.setenv(SESSION_ROOT_ENV, str(root))
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path / "experiments"))
    yield root


@pytest.fixture
def recorded_intervals(monkeypatch: pytest.MonkeyPatch) -> list[float | None]:
    """Record every ``run_every`` the app hands to ``st.fragment``.

    The real decorator still runs, so the fragment renders exactly as it
    would in a browser; only the interval is observed on the way past.
    This is how the refresh cadence is verified without waiting for one.
    """
    recorded: list[float | None] = []
    real = st.fragment

    def spy(
        func: Callable[..., Any] | None = None, **kwargs: Any
    ) -> Callable[..., Any]:
        recorded.append(kwargs.get("run_every"))
        return real(func, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(st, "fragment", spy)
    return recorded


def run_app(mode: str | None = None, session: str | None = None) -> AppTest:
    """Render the app, optionally selecting a mode and a session."""
    from engagevr.dashboard.launch import app_path

    app = AppTest.from_file(str(app_path()), default_timeout=TIMEOUT_SECONDS)
    app.run()
    if mode is not None:
        app.sidebar.radio[0].set_value(mode).run()
    if session is not None:
        labels = list(app.sidebar.selectbox[0].options)
        match = next(label for label in labels if label.startswith(session))
        app.sidebar.selectbox[0].set_value(match).run()
    return app


def rendered_text(app: AppTest) -> str:
    """Every string the page produced, joined."""
    parts: list[str] = []
    for collection in (
        app.markdown,
        app.error,
        app.warning,
        app.info,
        app.success,
        app.caption,
        app.title,
        app.subheader,
        app.text,
    ):
        parts.extend(str(element.value) for element in collection)
    return " ".join(parts)


class TestLiveModeRefreshesAutomatically:
    def test_the_live_page_schedules_a_refresh(
        self, session_root: Path, recorded_intervals: list[float | None]
    ) -> None:
        app = run_app(LIVE_MODE)
        assert not app.exception
        assert recorded_intervals, "the live page scheduled no automatic refresh"

    def test_the_scheduled_interval_is_the_configured_one(
        self, session_root: Path, recorded_intervals: list[float | None]
    ) -> None:
        configured = load_config().dashboard.live_refresh_seconds
        run_app(LIVE_MODE)
        assert recorded_intervals == [pytest.approx(configured)]

    def test_a_changed_configuration_changes_the_interval(
        self,
        session_root: Path,
        recorded_intervals: list[float | None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The cadence follows configuration, not a constant in the page."""
        import engagevr.config as config_module

        base = load_config()
        adjusted = base.model_copy(
            update={
                "dashboard": base.dashboard.model_copy(
                    update={"live_refresh_seconds": 30.0}
                )
            }
        )
        monkeypatch.setattr(config_module, "load_config", lambda *a, **k: adjusted)
        app = run_app(LIVE_MODE)
        assert not app.exception
        assert recorded_intervals == [pytest.approx(30.0)]
        assert "Automatic refresh: every 30 seconds" in rendered_text(app)

    def test_the_page_states_the_cadence_and_the_mode(self, session_root: Path) -> None:
        text = rendered_text(run_app(LIVE_MODE))
        assert "Mode: LIVE OBSERVATION" in text
        assert "Automatic refresh: every 5 seconds" in text

    def test_the_page_never_calls_the_refresh_real_time_inference(
        self, session_root: Path
    ) -> None:
        """The only place the phrase may appear is where it is denied."""
        text = rendered_text(run_app(LIVE_MODE))
        assert "Real-time observation is not real-time inference." in text
        remainder = text.replace(
            "Real-time observation is not real-time inference.", ""
        )
        assert "real-time inference" not in remainder.lower()
        assert "no model" in remainder.lower()

    def test_the_manual_control_survives_the_timer(self, session_root: Path) -> None:
        app = run_app(LIVE_MODE)
        assert "Read new records" in [button.label for button in app.button]

    def test_the_refreshing_page_still_renders_its_evidence(
        self, session_root: Path
    ) -> None:
        text = rendered_text(run_app(LIVE_MODE, session="synthetic-active"))
        assert "Session provenance" in text
        assert "may still be running" in text


class TestOtherModesDoNotRefresh:
    def test_replay_schedules_nothing(
        self, session_root: Path, recorded_intervals: list[float | None]
    ) -> None:
        app = run_app(REPLAY_MODE)
        assert not app.exception
        assert recorded_intervals == []

    def test_the_artifact_observatory_schedules_nothing(
        self, session_root: Path, recorded_intervals: list[float | None]
    ) -> None:
        app = run_app(ARTIFACT_MODE)
        assert not app.exception
        assert recorded_intervals == []

    def test_replay_says_it_does_not_advance_on_its_own(
        self, session_root: Path
    ) -> None:
        text = rendered_text(run_app(REPLAY_MODE))
        assert "Replay does not advance on its own" in text
        assert "Automatic refresh: every" not in text

    def test_stepping_through_replay_schedules_nothing(
        self, session_root: Path, recorded_intervals: list[float | None]
    ) -> None:
        app = run_app(REPLAY_MODE)
        forward = next(b for b in app.button if b.label == "Step forward")
        forward.click().run()
        assert not app.exception
        assert recorded_intervals == []


class TestARefusedIntervalStartsNoTimer:
    def _with_interval(self, monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
        """Force a refused interval past the configuration validator.

        ``DashboardConfig`` refuses these values, which is the first line
        of defence.  The page must refuse them too: a context is a plain
        dataclass, and the day one is built from something other than
        ``configs/defaults.yaml`` the page is the only thing left.
        """
        import engagevr.config as config_module

        base = load_config()
        dashboard = base.dashboard.model_copy(update={"live_refresh_seconds": seconds})
        adjusted = base.model_copy(update={"dashboard": dashboard})
        monkeypatch.setattr(config_module, "load_config", lambda *a, **k: adjusted)

    @pytest.mark.parametrize("seconds", [0.0, -5.0, 0.25, math.inf, math.nan])
    def test_a_refused_interval_schedules_no_fragment(
        self,
        session_root: Path,
        recorded_intervals: list[float | None],
        monkeypatch: pytest.MonkeyPatch,
        seconds: float,
    ) -> None:
        self._with_interval(monkeypatch, seconds)
        app = run_app(LIVE_MODE)
        assert not app.exception
        assert recorded_intervals == []

    def test_a_refused_interval_is_reported_with_its_reason(
        self, session_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._with_interval(monkeypatch, 0.25)
        app = run_app(LIVE_MODE)
        text = rendered_text(app)
        assert "automatic-refresh interval was refused" in text
        assert "at least 2s" in text

    def test_a_refused_interval_still_renders_the_live_evidence(
        self, session_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refusing the cadence must not blank the page."""
        self._with_interval(monkeypatch, 0.0)
        app = run_app(LIVE_MODE, session="synthetic-active")
        text = rendered_text(app)
        assert "Mode: LIVE OBSERVATION" in text
        assert "Session provenance" in text
        assert "Read new records" in [button.label for button in app.button]

    def test_a_refused_interval_never_becomes_the_minimum(
        self,
        session_root: Path,
        recorded_intervals: list[float | None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._with_interval(monkeypatch, 0.5)
        run_app(LIVE_MODE)
        assert views.MINIMUM_LIVE_REFRESH_SECONDS not in recorded_intervals
        assert recorded_intervals == []


class TestRefreshReadsTheSourceAgain:
    def test_an_appended_record_appears_on_the_next_pass(
        self, session_root: Path
    ) -> None:
        """Nothing between the page and the file may cache the read.

        This is the property the automatic refresh exists to deliver: a
        pass that re-rendered a cached read would show a live view that
        cannot show an appended record.  The pass is triggered here by
        the manual control rather than by waiting for the timer, because
        both go through the same body.
        """
        directory = session_root / "synthetic-active"
        app = run_app(LIVE_MODE, session="synthetic-active")
        assert not app.exception
        sfx.append_record(
            directory,
            sfx.stored(
                sfx.envelope(
                    session_id="synthetic-active",
                    message_type="heartbeat",
                    sequence_number=90,
                    payload={
                        "heartbeat_id": "synthetic-heartbeat-0090",
                        "client_monotonic_seconds": 9.0,
                    },
                    offset=50,
                ),
                sfx.ingestion(arrival_index=90, offset=50),
            ),
        )
        target = next(b for b in app.button if b.label == "Read new records")
        target.click().run()
        assert not app.exception
        assert "1 new complete record(s)" in rendered_text(app)

    def test_the_session_catalogue_is_not_cached(self) -> None:
        """An append leaves a directory's mtime alone; a cache would go stale."""
        source = (DASHBOARD / "app.py").read_text(encoding="utf-8")
        marker = source.index("def _session_catalogue")
        preceding = source[:marker].rsplit("\n\n\n", 1)[-1]
        assert "cache_data" not in preceding
        assert "cache_resource" not in preceding

    def test_no_session_module_caches_a_read(self) -> None:
        for name in (
            "session_reader.py",
            "session_catalogue.py",
            "session_pages.py",
            "views_session.py",
        ):
            text = (DASHBOARD / name).read_text(encoding="utf-8")
            assert "cache_data" not in text, name
            assert "cache_resource" not in text, name
