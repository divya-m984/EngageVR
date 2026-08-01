"""Tests for the CLI entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from engagevr.__main__ import main


class TestCliDemo:
    def test_demo_creates_output(self, tmp_path: Path):
        out = tmp_path / "session.json"
        rc = main(["demo", "--seed", "42", "--output", str(out)])
        assert rc == 0
        assert out.exists()

    def test_demo_output_is_valid_json(self, tmp_path: Path):
        out = tmp_path / "session.json"
        main(["demo", "--seed", "42", "--output", str(out)])
        data = json.loads(out.read_text())
        assert "session" in data
        assert "events" in data
        assert "predictions" in data

    def test_demo_output_labelled_synthetic(self, tmp_path: Path):
        out = tmp_path / "session.json"
        main(["demo", "--seed", "42", "--output", str(out)])
        data = json.loads(out.read_text())
        assert data["session"]["data_source"] == "synthetic"
        for event in data["events"]:
            assert event["data_source"] == "synthetic"
        for pred in data["predictions"]:
            assert pred["data_source"] == "synthetic"

    def test_demo_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "nested" / "dir" / "session.json"
        rc = main(["demo", "--seed", "1", "--output", str(out)])
        assert rc == 0
        assert out.exists()

    def test_demo_deterministic(self, tmp_path: Path):
        out1 = tmp_path / "s1.json"
        out2 = tmp_path / "s2.json"
        main(["demo", "--seed", "42", "--output", str(out1)])
        main(["demo", "--seed", "42", "--output", str(out2)])
        d1 = json.loads(out1.read_text())
        d2 = json.loads(out2.read_text())
        # Session IDs differ (uuid), but event timestamps/types should match
        assert len(d1["events"]) == len(d2["events"])
        for e1, e2 in zip(d1["events"], d2["events"], strict=True):
            assert e1["event_type"] == e2["event_type"]
            assert e1["monotonic_timestamp"] == e2["monotonic_timestamp"]

    def test_demo_custom_trials(self, tmp_path: Path):
        out = tmp_path / "session.json"
        main(["demo", "--seed", "1", "--trials", "3", "--output", str(out)])
        data = json.loads(out.read_text())
        assert len(data["predictions"]) == 3

    def test_demo_prints_summary(self, tmp_path: Path, capsys):
        out = tmp_path / "session.json"
        main(["demo", "--seed", "42", "--output", str(out)])
        captured = capsys.readouterr()
        assert "Session ID:" in captured.out
        assert "SYNTHETIC" in captured.out
        assert "Events:" in captured.out
        assert "Predictions:" in captured.out
        assert "Abstentions:" in captured.out

    def test_no_command_returns_nonzero(self):
        rc = main([])
        assert rc == 1


class TestCliSmoke:
    def test_module_invocation(self, tmp_path: Path):
        out = tmp_path / "smoke.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "engagevr",
                "demo",
                "--seed",
                "42",
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["session"]["data_source"] == "synthetic"
