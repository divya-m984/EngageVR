"""Shared test fixtures.

No fixture requires a webcam, a display server, network access, or any
real public dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engagevr.schemas.session import DataSource, ExperimentCondition, Session


@pytest.fixture
def synthetic_session() -> Session:
    """Return a minimal synthetic session for testing."""
    return Session(
        participant_id="test_participant_001",
        experiment_condition=ExperimentCondition.STATIC,
        data_source=DataSource.SYNTHETIC,
    )


@pytest.fixture
def ubfc_fixture_root(tmp_path: Path) -> Path:
    """A minimal UBFC-rPPG-shaped directory tree.

    SYNTHETIC FIXTURE. This is a structural stand-in used to exercise the
    adapter's discovery and parsing logic. It contains no real dataset
    content, no real video, and no real physiological measurement, and it
    must never be used to produce a metric.
    """
    root = tmp_path / "UBFC-rPPG-fixture"
    for name in ("subject1", "subject2"):
        subject = root / name
        subject.mkdir(parents=True)
        # Placeholder only: the adapter checks existence, never decodes here.
        (subject / "vid.avi").write_bytes(b"\x00" * 16)
        waveform = " ".join(f"{v:.4f}" for v in range(10))
        heart_rate = " ".join("72.0" for _ in range(10))
        timestamps = " ".join(f"{i / 30.0:.4f}" for i in range(10))
        (subject / "ground_truth.txt").write_text(
            f"{waveform}\n{heart_rate}\n{timestamps}\n"
        )
    return root
