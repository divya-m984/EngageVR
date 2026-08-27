"""Live and replay page smoke tests through Streamlit's own testing API.

``AppTest`` runs the script in-process: no browser, no socket, no server,
no automation dependency.  These tests check that both session modes
render, that neither can be rendered under the other's heading, and that
a synthetic recording never appears without its banner.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from engagevr.dashboard.app import ARTIFACT_ROOT_ENV, SESSION_ROOT_ENV
from engagevr.schemas.dashboard import DASHBOARD_DISCLAIMER, SYNTHETIC_BANNER
from engagevr.schemas.dashboard_session import SESSION_ELIGIBILITY_NOTE
from tests.unit import session_fixtures as sfx

#: Generous: the script parses a handful of small local files.
TIMEOUT_SECONDS = 60.0

ARTIFACT_MODE = "Experiment artifacts"
LIVE_MODE = "Live session"
REPLAY_MODE = "Session replay"


@pytest.fixture
def session_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A session root holding a completed, an active, and a damaged recording."""
    root = tmp_path / "sessions"
    sfx.write_session(root, "synthetic-completed", with_adaptation=True)
    sfx.write_session(root, "synthetic-active", with_summary=False, completed=False)
    damaged = sfx.write_session(root, "synthetic-damaged")
    sfx.corrupt_line(damaged, 3)
    sfx.write_session(
        root,
        "synthetic-partial",
        with_summary=False,
        completed=False,
        partial_trailing='{"envelope": {"mess',
    )
    monkeypatch.setenv(SESSION_ROOT_ENV, str(root))
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path / "experiments"))
    yield root


def run_app(mode: str | None = None, session: str | None = None) -> AppTest:
    """Render the app, optionally selecting a mode and a session."""
    from engagevr.dashboard.launch import app_path

    app = AppTest.from_file(str(app_path()), default_timeout=TIMEOUT_SECONDS)
    app.run()
    if mode is not None:
        app.sidebar.radio[0].set_value(mode).run()
    if session is not None:
        labels = [option for option in app.sidebar.selectbox[0].options]
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


class TestModeSelector:
    def test_the_default_mode_is_the_artifact_observatory(
        self, session_root: Path
    ) -> None:
        app = run_app()
        assert not app.exception
        assert app.sidebar.radio[0].value == ARTIFACT_MODE

    def test_all_three_modes_are_offered(self, session_root: Path) -> None:
        app = run_app()
        assert list(app.sidebar.radio[0].options) == [
            ARTIFACT_MODE,
            LIVE_MODE,
            REPLAY_MODE,
        ]

    def test_the_artifact_mode_still_renders_with_no_runs(
        self, session_root: Path
    ) -> None:
        app = run_app(ARTIFACT_MODE)
        assert not app.exception
        assert any("Overview" in title.value for title in app.title)

    @pytest.mark.parametrize("mode", [LIVE_MODE, REPLAY_MODE])
    def test_each_session_mode_renders(self, session_root: Path, mode: str) -> None:
        app = run_app(mode)
        assert not app.exception, f"{mode} raised: {app.exception}"

    def test_each_mode_states_its_evidence_source(self, session_root: Path) -> None:
        statements = set()
        for mode in (ARTIFACT_MODE, LIVE_MODE, REPLAY_MODE):
            app = run_app(mode)
            statements.add(
                next(
                    caption.value
                    for caption in app.sidebar.caption
                    if "Evidence source" in str(caption.value)
                )
            )
        assert len(statements) == 3


class TestLiveMode:
    def test_the_live_page_declares_its_mode(self, session_root: Path) -> None:
        app = run_app(LIVE_MODE)
        text = rendered_text(app)
        assert "LIVE OBSERVATION" in text
        assert "Live session observation" in text

    def test_the_live_page_is_not_labelled_replay(self, session_root: Path) -> None:
        app = run_app(LIVE_MODE)
        assert not any("Session replay" == title.value for title in app.title)

    def test_the_live_page_shows_the_synthetic_banner(self, session_root: Path) -> None:
        app = run_app(LIVE_MODE)
        text = rendered_text(app)
        assert SYNTHETIC_BANNER in text
        assert DASHBOARD_DISCLAIMER in text

    def test_the_live_page_states_ineligibility(self, session_root: Path) -> None:
        app = run_app(LIVE_MODE)
        assert SESSION_ELIGIBILITY_NOTE in rendered_text(app)

    def test_the_live_page_denies_running_a_model(self, session_root: Path) -> None:
        text = rendered_text(run_app(LIVE_MODE))
        assert "runs no model" in text
        assert "Nothing on this page is a model output" in text

    def test_an_active_session_renders_without_crashing(
        self, session_root: Path
    ) -> None:
        app = run_app(LIVE_MODE, session="synthetic-active")
        assert not app.exception
        text = rendered_text(app)
        assert "may still be running" in text
        assert "neither is a failure" in text

    def test_a_partial_trailing_line_renders_as_transient(
        self, session_root: Path
    ) -> None:
        app = run_app(LIVE_MODE, session="synthetic-partial")
        assert not app.exception
        assert "not treated as corruption" in rendered_text(app)

    def test_a_malformed_record_is_reported_with_its_line(
        self, session_root: Path
    ) -> None:
        app = run_app(LIVE_MODE, session="synthetic-damaged")
        assert not app.exception
        assert "could not be decoded" in rendered_text(app)

    def test_the_manual_refresh_control_remains_alongside_the_timer(
        self, session_root: Path
    ) -> None:
        app = run_app(LIVE_MODE)
        labels = [button.label for button in app.button]
        assert "Read new records" in labels
        text = rendered_text(app)
        assert "Automatic refresh: every 5 seconds" in text
        assert "Mode: LIVE OBSERVATION" in text

    def test_pressing_refresh_re_reads_without_error(self, session_root: Path) -> None:
        app = run_app(LIVE_MODE)
        target = next(b for b in app.button if b.label == "Read new records")
        target.click().run()
        assert not app.exception
        assert "No new complete record" in rendered_text(app)

    def test_an_appended_record_is_visible_after_a_refresh(
        self, session_root: Path
    ) -> None:
        app = run_app(LIVE_MODE, session="synthetic-active")
        sfx.append_record(
            session_root / "synthetic-active",
            sfx.stored(
                sfx.envelope(
                    session_id="synthetic-active",
                    message_type="heartbeat",
                    sequence_number=80,
                    payload={
                        "heartbeat_id": "synthetic-heartbeat-0100",
                        "client_monotonic_seconds": 9.0,
                    },
                    offset=45,
                ),
                sfx.ingestion(arrival_index=80, offset=45),
            ),
        )
        target = next(b for b in app.button if b.label == "Read new records")
        target.click().run()
        assert not app.exception
        assert "1 new complete record(s)" in rendered_text(app)


class TestReplayMode:
    def test_the_replay_page_declares_its_mode(self, session_root: Path) -> None:
        text = rendered_text(run_app(REPLAY_MODE))
        assert "MODE: REPLAY" in text
        assert "Nothing is re-emitted" in text

    def test_the_replay_page_is_not_labelled_live(self, session_root: Path) -> None:
        app = run_app(REPLAY_MODE)
        assert not any("Live session observation" == title.value for title in app.title)
        assert "LIVE OBSERVATION" not in rendered_text(app)

    def test_the_replay_page_shows_the_synthetic_banner(
        self, session_root: Path
    ) -> None:
        text = rendered_text(run_app(REPLAY_MODE))
        assert SYNTHETIC_BANNER in text
        assert DASHBOARD_DISCLAIMER in text

    def test_the_navigation_controls_exist(self, session_root: Path) -> None:
        app = run_app(REPLAY_MODE)
        labels = {button.label for button in app.button}
        assert {
            "Jump to beginning",
            "Step backward",
            "Step forward",
            "Jump to end",
        } <= labels

    def test_the_first_position_disables_backward_navigation(
        self, session_root: Path
    ) -> None:
        app = run_app(REPLAY_MODE)
        backward = next(b for b in app.button if b.label == "Step backward")
        assert backward.disabled is True

    def test_stepping_forward_advances_the_position(self, session_root: Path) -> None:
        app = run_app(REPLAY_MODE)
        assert "Record 1 of" in rendered_text(app)
        forward = next(b for b in app.button if b.label == "Step forward")
        forward.click().run()
        assert not app.exception
        assert "Record 2 of" in rendered_text(app)

    def test_jumping_to_the_end_reaches_the_last_record(
        self, session_root: Path
    ) -> None:
        app = run_app(REPLAY_MODE, session="synthetic-completed")
        last = next(b for b in app.button if b.label == "Jump to end")
        last.click().run()
        assert not app.exception
        assert "Record 8 of 8" in rendered_text(app)

    def test_jumping_back_to_the_beginning_returns_to_the_first_record(
        self, session_root: Path
    ) -> None:
        app = run_app(REPLAY_MODE, session="synthetic-completed")
        next(b for b in app.button if b.label == "Jump to end").click().run()
        next(b for b in app.button if b.label == "Jump to beginning").click().run()
        assert not app.exception
        assert "Record 1 of 8" in rendered_text(app)

    def test_stepping_backward_returns_one_record(self, session_root: Path) -> None:
        app = run_app(REPLAY_MODE, session="synthetic-completed")
        next(b for b in app.button if b.label == "Step forward").click().run()
        next(b for b in app.button if b.label == "Step forward").click().run()
        assert "Record 3 of 8" in rendered_text(app)
        next(b for b in app.button if b.label == "Step backward").click().run()
        assert not app.exception
        assert "Record 2 of 8" in rendered_text(app)

    def test_stepping_forward_at_the_end_does_not_overflow(
        self, session_root: Path
    ) -> None:
        app = run_app(REPLAY_MODE, session="synthetic-completed")
        next(b for b in app.button if b.label == "Jump to end").click().run()
        forward = next(b for b in app.button if b.label == "Step forward")
        assert forward.disabled is True
        assert not app.exception
        assert "Record 8 of 8" in rendered_text(app)

    def test_an_interrupted_recording_can_be_replayed(self, session_root: Path) -> None:
        app = run_app(REPLAY_MODE, session="synthetic-active")
        assert not app.exception
        assert "Record 1 of" in rendered_text(app)

    def test_the_replay_page_keeps_the_synthetic_label_per_record(
        self, session_root: Path
    ) -> None:
        text = rendered_text(run_app(REPLAY_MODE))
        assert "Replaying it does not make it participant data" in text


class TestSessionReportExport:
    @pytest.mark.parametrize("mode", [LIVE_MODE, REPLAY_MODE])
    def test_both_download_controls_exist(self, session_root: Path, mode: str) -> None:
        app = run_app(mode)
        labels = {button.label for button in app.download_button}
        assert "Download session report (JSON)" in labels
        assert "Download session report (Markdown)" in labels

    def test_the_export_states_its_fingerprint(self, session_root: Path) -> None:
        assert "Report fingerprint" in rendered_text(run_app(REPLAY_MODE))

    def test_the_export_states_that_provenance_is_permanent(
        self, session_root: Path
    ) -> None:
        text = rendered_text(run_app(REPLAY_MODE))
        assert "There is no export path that removes them" in text

    def test_the_export_states_that_the_source_is_untouched(
        self, session_root: Path
    ) -> None:
        assert "recording on disk is not touched" in rendered_text(run_app(LIVE_MODE))


class TestEmptyAndBrokenRoots:
    def test_an_absent_session_root_renders_a_statement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SESSION_ROOT_ENV, str(tmp_path / "nowhere"))
        monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path / "experiments"))
        app = run_app(LIVE_MODE)
        assert not app.exception
        assert "does not exist" in rendered_text(app)

    def test_an_empty_session_root_renders_a_statement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        monkeypatch.setenv(SESSION_ROOT_ENV, str(root))
        monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path / "experiments"))
        app = run_app(REPLAY_MODE)
        assert not app.exception
        assert "No recorded session was found" in rendered_text(app)

    def test_a_session_with_no_manifest_does_not_crash_the_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "sessions"
        sfx.write_session(root, "synthetic-headless", with_manifest=False)
        monkeypatch.setenv(SESSION_ROOT_ENV, str(root))
        monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path / "experiments"))
        app = run_app(REPLAY_MODE)
        assert not app.exception
        assert "UNREADABLE" in rendered_text(app)


class TestReadingChangesNothing:
    def test_rendering_both_modes_modifies_no_recording(
        self, session_root: Path
    ) -> None:
        directory = session_root / "synthetic-completed"
        before = sfx.directory_digests(directory)
        run_app(LIVE_MODE)
        run_app(REPLAY_MODE)
        assert sfx.directory_digests(directory) == before


class TestCopiedRecording:
    """A recording under a differently-named folder still renders.

    This is the case that first broke: the selector returned a recorded
    session id, the page rebuilt a path from it, and a copied recording
    resolved to a directory that did not exist — rendering an empty
    session with no synthetic banner.
    """

    @pytest.fixture
    def copied_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        root = tmp_path / "sessions"
        original = sfx.write_session(root, "synthetic-original")
        copy = root / "synthetic-copy"
        copy.mkdir()
        for path in original.iterdir():
            (copy / path.name).write_bytes(path.read_bytes())
        monkeypatch.setenv(SESSION_ROOT_ENV, str(root))
        monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path / "experiments"))
        return root

    @pytest.mark.parametrize("mode", [LIVE_MODE, REPLAY_MODE])
    def test_the_copy_renders_with_its_banner(
        self, copied_root: Path, mode: str
    ) -> None:
        app = run_app(mode, session="synthetic-copy")
        assert not app.exception
        text = rendered_text(app)
        assert SYNTHETIC_BANNER in text
        assert "Session provenance" in text
        assert "Ordering anomalies" in text

    def test_both_names_are_shown(self, copied_root: Path) -> None:
        app = run_app(REPLAY_MODE, session="synthetic-copy")
        text = rendered_text(app)
        assert "declares session_id" in text

    def test_the_selector_lists_both_copies(self, copied_root: Path) -> None:
        app = run_app(REPLAY_MODE)
        options = list(app.sidebar.selectbox[0].options)
        assert any(option.startswith("synthetic-copy") for option in options)
        assert any(option.startswith("synthetic-original") for option in options)
