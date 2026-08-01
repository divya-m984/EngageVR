"""Shared test fixtures."""

from __future__ import annotations

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
