"""Shared test fixtures.

No fixture requires a webcam, a display server, network access, or any
real public dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engagevr.features.assembly import write_dataset
from engagevr.features.synthetic import (
    SyntheticDatasetConfig,
    generate_synthetic_dataset,
)
from engagevr.schemas.features import FeatureWindow
from engagevr.schemas.session import DataSource, ExperimentCondition, Session
from engagevr.schemas.targets import TargetName


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


# ---------------------------------------------------------------------------
# Milestone 5: windowed feature datasets and interpretable baselines
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def m5_synthetic_config() -> SyntheticDatasetConfig:
    """A small, deterministic SYNTHETIC generator configuration.

    Deliberately small: these tests verify pipeline behaviour, not model
    quality, and a large dataset would only make the suite slow.
    """
    return SyntheticDatasetConfig(
        seed=42,
        subjects=12,
        sessions_per_subject=2,
        windows_per_session=5,
        window_duration_seconds=10.0,
        window_step_seconds=10.0,
    )


@pytest.fixture(scope="session")
def m5_synthetic_rows(
    m5_synthetic_config: SyntheticDatasetConfig,
) -> tuple[FeatureWindow, ...]:
    """Deterministic SYNTHETIC feature windows shared across tests."""
    return generate_synthetic_dataset(m5_synthetic_config)


@pytest.fixture(scope="session")
def m5_dataset(
    tmp_path_factory: pytest.TempPathFactory,
    m5_synthetic_config: SyntheticDatasetConfig,
    m5_synthetic_rows: tuple[FeatureWindow, ...],
) -> Path:
    """A written SYNTHETIC Parquet dataset with metadata and catalog."""
    directory = tmp_path_factory.mktemp("m5-dataset")
    path = directory / "m5-synthetic.parquet"
    write_dataset(
        m5_synthetic_rows,
        path,
        target_names=list(TargetName),
        window_duration_seconds=m5_synthetic_config.window_duration_seconds,
        window_step_seconds=m5_synthetic_config.window_step_seconds,
        windows_overlap=m5_synthetic_config.windows_overlap,
        creation_configuration=m5_synthetic_config.model_dump(mode="json"),
        random_seed=m5_synthetic_config.seed,
    )
    return path
