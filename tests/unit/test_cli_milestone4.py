"""CLI tests for serve, task-sim, session-inspect, and session-replay.

The WebSocket integration test starts a uvicorn server on an
OS-assigned loopback port inside this process.  It needs no external
service, no display, and no internet: loopback is not a network
dependency in the sense the milestone brief excludes, and the test skips
cleanly if a loopback port cannot be bound.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from engagevr.__main__ import main
from engagevr.config import load_config
from engagevr.protocol.version import PROTOCOL_VERSION


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --- task-sim --------------------------------------------------------------


class TestTaskSimCli:
    def test_offline_run_succeeds_and_creates_the_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "made" / "by" / "the" / "cli"
        code = main(
            [
                "task-sim",
                "--seed",
                "42",
                "--blocks",
                "2",
                "--trials-per-block",
                "5",
                "--speed",
                "0",
                "--session-id",
                "cli-session",
                "--output",
                str(root),
            ]
        )
        assert code == 0
        assert (root / "cli-session" / "manifest.json").is_file()
        assert (root / "cli-session" / "events.jsonl").is_file()
        assert (root / "cli-session" / "summary.json").is_file()

        out = capsys.readouterr().out
        assert "Session ID:" in out
        assert "cli-session" in out
        assert f"Protocol version:      {PROTOCOL_VERSION}" in out
        assert "Blocks:                2" in out
        assert "Trials:                10" in out
        assert "Emitted events:" in out
        assert "Synthetic responses:" in out
        assert "Timeouts:" in out
        assert str(root) in out

    def test_the_synthetic_disclaimer_is_always_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "task-sim",
                "--blocks",
                "1",
                "--trials-per-block",
                "2",
                "--session-id",
                "s1",
                "--output",
                str(tmp_path),
            ]
        )
        out = capsys.readouterr().out
        assert "SYNTHETIC" in out
        assert "No person performed this task" in out
        assert "not participant data" in out
        assert "not experimental evidence" in out

    def test_no_scientific_conclusion_is_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "task-sim",
                "--blocks",
                "1",
                "--trials-per-block",
                "2",
                "--session-id",
                "s2",
                "--output",
                str(tmp_path),
            ]
        )
        out = capsys.readouterr().out.lower()
        for claim in (
            "engagement level",
            "cognitive load is",
            "attention was",
            "the participant",
            "we conclude",
            "significant",
        ):
            assert claim not in out
        # The scope disclaimer is present instead.
        assert "not engagement, attention, cognitive-load, or fatigue" in out

    def test_two_runs_of_the_same_seed_record_the_same_events(
        self, tmp_path: Path
    ) -> None:
        for name in ("run-a", "run-b"):
            main(
                [
                    "task-sim",
                    "--seed",
                    "1234",
                    "--blocks",
                    "1",
                    "--trials-per-block",
                    "6",
                    "--session-id",
                    name,
                    "--output",
                    str(tmp_path),
                ]
            )

        def events(name: str) -> list[dict[str, object]]:
            path = tmp_path / name / "events.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            return [
                r["envelope"]["payload"]
                for r in records
                if r["envelope"]["message_type"] == "task_event"
            ]

        assert events("run-a") == events("run-b")

    def test_a_different_seed_gives_different_events(self, tmp_path: Path) -> None:
        for name, seed in (("seed-1", "1"), ("seed-2", "2")):
            main(
                [
                    "task-sim",
                    "--seed",
                    seed,
                    "--blocks",
                    "1",
                    "--trials-per-block",
                    "20",
                    "--session-id",
                    name,
                    "--output",
                    str(tmp_path),
                ]
            )

        def stimuli(name: str) -> list[object]:
            path = tmp_path / name / "events.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            return [
                r["envelope"]["payload"]["event"].get("stimulus_id")
                for r in records
                if r["envelope"]["message_type"] == "task_event"
            ]

        assert stimuli("seed-1") != stimuli("seed-2")

    def test_negative_speed_is_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["task-sim", "--speed", "-1", "--output", str(tmp_path)])
        assert code == 2
        assert "--speed must not be negative" in capsys.readouterr().err

    def test_output_and_connect_are_mutually_exclusive(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "task-sim",
                "--output",
                str(tmp_path),
                "--connect",
                "ws://127.0.0.1:1/ws/v1/sessions/x",
            ]
        )
        assert code == 2
        assert "not both" in capsys.readouterr().err

    def test_an_unsafe_session_id_is_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["task-sim", "--session-id", "../escape", "--output", str(tmp_path)]
        )
        assert code == 2
        assert "not permitted" in capsys.readouterr().err

    def test_invalid_task_geometry_is_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["task-sim", "--blocks", "0", "--output", str(tmp_path)])
        assert code == 2
        assert "invalid task configuration" in capsys.readouterr().err

    def test_an_unreachable_websocket_returns_non_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        port = free_loopback_port()
        code = main(
            [
                "task-sim",
                "--blocks",
                "1",
                "--trials-per-block",
                "1",
                "--connect",
                f"ws://127.0.0.1:{port}/ws/v1/sessions/nope",
            ]
        )
        assert code == 1
        assert "cannot connect" in capsys.readouterr().err


# --- session-inspect -------------------------------------------------------


@pytest.fixture
def recorded_session(tmp_path: Path) -> Path:
    """A real recording produced by the CLI itself."""
    root = tmp_path / "sessions"
    main(
        [
            "task-sim",
            "--seed",
            "42",
            "--blocks",
            "2",
            "--trials-per-block",
            "4",
            "--session-id",
            "demo-session",
            "--output",
            str(root),
        ]
    )
    return root / "demo-session"


class TestSessionInspectCli:
    def test_inspecting_a_recording(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["session-inspect", str(recorded_session)])
        assert code == 0

        out = capsys.readouterr().out
        assert "Session:             demo-session" in out
        assert f"Protocol version:    {PROTOCOL_VERSION}" in out
        assert "Events:" in out
        assert "Dropped messages:    0" in out
        assert "Completed:           True" in out
        assert "Message types:" in out
        assert "task_event" in out
        assert "Sources:" in out
        assert "python_simulator" in out
        assert "no engagement estimate" in out

    def test_json_output_is_machine_readable(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["session-inspect", str(recorded_session), "--json"])
        document = json.loads(capsys.readouterr().out)
        assert document["manifest"]["session_id"] == "demo-session"
        assert document["summary"]["completed"] is True
        assert document["summary"]["synthetic_message_count"] > 0

    def test_a_missing_session_returns_non_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["session-inspect", str(tmp_path / "ghost")])
        assert code == 1
        assert "no manifest" in capsys.readouterr().err

    def test_a_malformed_recording_reports_the_line_number(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = recorded_session / "events.jsonl"
        lines = path.read_text().splitlines()
        lines.insert(4, "{ this is not json")
        path.write_text("\n".join(lines) + "\n")

        code = main(["session-inspect", str(recorded_session)])
        assert code == 1
        error = capsys.readouterr().err
        assert "malformed recording" in error
        assert "events.jsonl:5" in error


# --- session-replay --------------------------------------------------------


class TestSessionReplayCli:
    def test_immediate_replay_reproduces_the_whole_recording(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        expected = len((recorded_session / "events.jsonl").read_text().splitlines())
        code = main(["session-replay", str(recorded_session), "--speed", "0"])
        assert code == 0

        out = capsys.readouterr().out
        assert "=== REPLAY ===" in out
        assert "REPLAY. These messages are a re-emission" in out
        assert f"Available messages:  {expected}" in out
        assert f"Emitted messages:    {expected}" in out
        assert "Speed:               0.0 (immediate)" in out
        assert "Replay label:        REPLAY" in out
        assert "was not modified" in out

    def test_the_immediate_flag_is_equivalent_to_speed_zero(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["session-replay", str(recorded_session), "--immediate"])
        assert "(immediate)" in capsys.readouterr().out

    def test_accelerated_replay_reports_its_mode(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["session-replay", str(recorded_session), "--speed", "1000"])
        out = capsys.readouterr().out
        assert "(accelerated)" in out

    def test_a_negative_speed_is_rejected(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["session-replay", str(recorded_session), "--speed", "-2"])
        assert code == 2
        assert "must not be negative" in capsys.readouterr().err

    def test_a_speed_above_the_maximum_is_rejected(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        maximum = load_config().replay.maximum_speed
        code = main(
            ["session-replay", str(recorded_session), "--speed", str(maximum + 1)]
        )
        assert code == 2
        assert "exceeds" in capsys.readouterr().err

    def test_filtering_by_message_type(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "session-replay",
                str(recorded_session),
                "--speed",
                "0",
                "--message-type",
                "session_start",
                "--message-type",
                "session_end",
            ]
        )
        out = capsys.readouterr().out
        assert "Emitted messages:    2" in out
        assert "session_start" in out

    def test_an_unknown_filter_value_is_rejected(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "session-replay",
                str(recorded_session),
                "--message-type",
                "not_a_type",
            ]
        )
        assert code == 2
        assert "unknown message type" in capsys.readouterr().err

    def test_replay_does_not_modify_the_recording(self, recorded_session: Path) -> None:
        before = {
            path.name: path.read_bytes()
            for path in sorted(recorded_session.iterdir())
            if path.is_file()
        }
        main(["session-replay", str(recorded_session), "--speed", "0"])
        after = {
            path.name: path.read_bytes()
            for path in sorted(recorded_session.iterdir())
            if path.is_file()
        }
        assert after == before

    def test_json_output_reports_the_labels(
        self, recorded_session: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["session-replay", str(recorded_session), "--speed", "0", "--json"])
        document = json.loads(capsys.readouterr().out)
        assert document["replay_label"] == "REPLAY"
        assert document["synthetic_message_count"] > 0
        assert "not live" in document["disclaimer"]


# --- serve -----------------------------------------------------------------


class TestServeCli:
    def test_public_bind_without_the_flag_is_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["serve", "--host", "0.0.0.0", "--session-root", str(tmp_path)])
        assert code == 2
        error = capsys.readouterr().err
        assert "invalid server configuration" in error
        assert "no authentication" in error

    def test_an_invalid_port_is_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["serve", "--port", "99999", "--session-root", str(tmp_path)])
        assert code == 2
        assert "invalid server configuration" in capsys.readouterr().err

    def test_port_zero_is_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["serve", "--port", "0", "--session-root", str(tmp_path)])
        assert code == 2

    def test_an_unusable_session_root_is_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("in the way")
        code = main(["serve", "--session-root", str(blocker / "child")])
        assert code == 2
        assert "cannot create session root" in capsys.readouterr().err

    def test_auto_reload_is_never_enabled(self) -> None:
        """Reading the source is the only way to assert a negative here."""
        source = Path("src/engagevr/cli_milestone4.py").read_text()
        assert "reload=False" in source
        assert "reload=True" not in source


# --- end-to-end over a real loopback WebSocket ------------------------------


@pytest.fixture
def live_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """A uvicorn server on a loopback port, in a background thread."""
    import uvicorn

    from engagevr.api.app import create_app

    session_root = tmp_path / "sessions"
    session_root.mkdir()
    port = free_loopback_port()
    config = load_config()

    app = create_app(config, session_root=session_root)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.05)
    if not server.started:  # pragma: no cover - environment problem
        server.should_exit = True
        thread.join(timeout=5.0)
        pytest.skip("could not start a loopback server in this environment")

    try:
        yield f"127.0.0.1:{port}", session_root
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


class TestWebSocketIntegration:
    def test_the_full_demo_workflow(
        self,
        live_server: tuple[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """serve -> task-sim over WebSocket -> inspect -> replay."""
        address, session_root = live_server
        url = f"ws://{address}/ws/v1/sessions/demo-session"

        code = main(
            [
                "task-sim",
                "--seed",
                "42",
                "--blocks",
                "2",
                "--trials-per-block",
                "5",
                "--connect",
                url,
            ]
        )
        assert code == 0
        simulator_out = capsys.readouterr().out
        assert "SYNTHETIC" in simulator_out
        assert url in simulator_out

        directory = session_root / "demo-session"
        deadline = time.monotonic() + 10.0
        while (
            time.monotonic() < deadline and not (directory / "summary.json").is_file()
        ):
            time.sleep(0.05)
        assert (directory / "manifest.json").is_file()
        assert (directory / "events.jsonl").is_file()
        assert (directory / "summary.json").is_file()

        assert main(["session-inspect", str(directory)]) == 0
        inspect_out = capsys.readouterr().out
        assert "Completed:           True" in inspect_out
        assert "python_simulator" in inspect_out

        assert main(["session-replay", str(directory), "--speed", "0"]) == 0
        replay_out = capsys.readouterr().out
        assert "Replay label:        REPLAY" in replay_out

    def test_replaying_into_the_server_labels_everything(
        self,
        live_server: tuple[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        address, session_root = live_server

        main(
            [
                "task-sim",
                "--seed",
                "7",
                "--blocks",
                "1",
                "--trials-per-block",
                "3",
                "--connect",
                f"ws://{address}/ws/v1/sessions/source-session",
            ]
        )
        source = session_root / "source-session"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not (source / "summary.json").is_file():
            time.sleep(0.05)
        capsys.readouterr()

        code = main(
            [
                "session-replay",
                str(source),
                "--speed",
                "0",
                "--connect",
                f"ws://{address}/ws/v1/sessions/replayed-session",
            ]
        )
        assert code == 0

        replayed = session_root / "replayed-session"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not (replayed / "summary.json").is_file():
            time.sleep(0.05)

        summary = json.loads((replayed / "summary.json").read_text())
        assert summary["event_count"] > 0
        assert summary["replay_message_count"] == summary["event_count"]
        assert summary["synthetic_message_count"] == summary["event_count"]

        for line in (replayed / "events.jsonl").read_text().splitlines():
            envelope = json.loads(line)["envelope"]
            assert envelope["replay"]["replay_label"] == "REPLAY"
            assert envelope["provenance"]["synthetic_label"] == "SYNTHETIC"
            assert envelope["replay"]["source_session_id"] == "source-session"
            assert envelope["replay"]["replay_session_id"] == "replayed-session"

    def test_the_recording_contains_no_forbidden_content(
        self,
        live_server: tuple[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        address, session_root = live_server
        main(
            [
                "task-sim",
                "--seed",
                "3",
                "--blocks",
                "1",
                "--trials-per-block",
                "3",
                "--connect",
                f"ws://{address}/ws/v1/sessions/privacy-session",
            ]
        )
        capsys.readouterr()
        directory = session_root / "privacy-session"
        deadline = time.monotonic() + 10.0
        while (
            time.monotonic() < deadline and not (directory / "summary.json").is_file()
        ):
            time.sleep(0.05)

        text = (directory / "events.jsonl").read_text().lower()
        for token in (
            "engagement",
            "cognitive_load",
            "attention",
            "fatigue",
            "frame_data",
            "pixels",
            "landmark",
            "prediction",
            "confidence",
            "email",
            "password",
        ):
            assert token not in text, f"the recording contains {token!r}"
        assert "synthetic" in text


# --- top-level CLI ---------------------------------------------------------


class TestCliDispatch:
    def test_the_new_commands_are_registered(self) -> None:
        from engagevr.__main__ import _build_parser

        parser = _build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        commands = set()
        for action in actions:
            commands.update(action.choices)
        for command in ("serve", "task-sim", "session-inspect", "session-replay"):
            assert command in commands

    def test_no_command_prints_help_and_returns_non_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == 1
        assert "task-sim" in capsys.readouterr().out

    def test_the_existing_commands_still_work(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "demo.json"
        assert main(["demo", "--seed", "1", "--output", str(output)]) == 0
        assert output.is_file()
        assert "SYNTHETIC" in capsys.readouterr().out


def _drain(loop: asyncio.AbstractEventLoop) -> None:  # pragma: no cover - helper
    with contextlib.suppress(Exception):
        loop.run_until_complete(asyncio.sleep(0))
