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
from engagevr.schemas.fusion import (
    FusionConfiguration,
    FusionModality,
    FusionStrategy,
    RobustnessConfiguration,
    StackingConfiguration,
)
from engagevr.schemas.personalization import (
    PersonalizationConfiguration,
    PersonalizationMethod,
)
from engagevr.schemas.session import DataSource, ExperimentCondition, Session
from engagevr.schemas.targets import TargetName
from engagevr.schemas.uncertainty import SelectivePredictionConfiguration
from engagevr.training.fusion_runner import (
    FusionRunConfiguration,
    FusionRunResult,
    run_fusion,
)
from engagevr.training.personalization_runner import (
    PersonalizationRunConfiguration,
    PersonalizationRunResult,
    run_personalization,
)
from engagevr.training.uncertainty_runner import (
    UncertaintyRunConfiguration,
    UncertaintyRunResult,
    run_uncertainty,
)


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


# ---------------------------------------------------------------------------
# Milestone 6: multimodal fusion
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def m6_fusion_configuration() -> FusionConfiguration:
    """The three default fusion strategies over the four modality groups."""
    return FusionConfiguration(
        strategies=(
            FusionStrategy.EARLY,
            FusionStrategy.UNIFORM_LATE,
            FusionStrategy.QUALITY_LATE,
        ),
        modalities=tuple(FusionModality),
    )


@pytest.fixture(scope="session")
def m6_classification_run(
    tmp_path_factory: pytest.TempPathFactory,
    m5_dataset: Path,
    m6_fusion_configuration: FusionConfiguration,
) -> FusionRunResult:
    """A completed SYNTHETIC classification fusion run, shared across tests."""
    directory = tmp_path_factory.mktemp("m6-classification")
    return run_fusion(
        FusionRunConfiguration(
            dataset_path=m5_dataset,
            target_name=TargetName.ENGAGEMENT_CLASS,
            output_directory=directory,
            fusion=m6_fusion_configuration,
            n_splits=3,
        )
    )


@pytest.fixture(scope="session")
def m6_regression_run(
    tmp_path_factory: pytest.TempPathFactory,
    m5_dataset: Path,
) -> FusionRunResult:
    """A completed SYNTHETIC regression fusion run, reference scenario only."""
    directory = tmp_path_factory.mktemp("m6-regression")
    configuration = FusionConfiguration(
        strategies=(
            FusionStrategy.EARLY,
            FusionStrategy.UNIFORM_LATE,
            FusionStrategy.QUALITY_LATE,
            FusionStrategy.VALIDATION_WEIGHTED_LATE,
            FusionStrategy.STACKED_LATE,
        ),
        modalities=tuple(FusionModality),
        stacking=StackingConfiguration(enabled=True),
        robustness=RobustnessConfiguration(enabled=False),
    )
    return run_fusion(
        FusionRunConfiguration(
            dataset_path=m5_dataset,
            target_name=TargetName.ENGAGEMENT_SCORE,
            output_directory=directory,
            fusion=configuration,
            n_splits=3,
        )
    )


# ---------------------------------------------------------------------------
# Milestone 6: personalization
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def m6_personalization_configuration() -> PersonalizationConfiguration:
    """Personal baseline plus supervised correction over all four modalities.

    Two calibration windows out of each subject's ten, so every subject in
    the shared fixture dataset can be split into a calibration region and a
    strictly later evaluation region.
    """
    return PersonalizationConfiguration(
        method=PersonalizationMethod.PERSONAL_BASELINE_AND_CORRECTION,
        modalities=tuple(FusionModality),
        calibration_windows=3,
        minimum_calibration_windows=2,
    )


@pytest.fixture(scope="session")
def m6_personalization_classification_run(
    tmp_path_factory: pytest.TempPathFactory,
    m5_dataset: Path,
    m6_personalization_configuration: PersonalizationConfiguration,
) -> PersonalizationRunResult:
    """A completed SYNTHETIC classification personalization run."""
    directory = tmp_path_factory.mktemp("m6-personalization-classification")
    return run_personalization(
        PersonalizationRunConfiguration(
            dataset_path=m5_dataset,
            target_name=TargetName.ENGAGEMENT_CLASS,
            output_directory=directory,
            personalization=m6_personalization_configuration,
            n_splits=3,
        )
    )


@pytest.fixture(scope="session")
def m6_personalization_regression_run(
    tmp_path_factory: pytest.TempPathFactory,
    m5_dataset: Path,
    m6_personalization_configuration: PersonalizationConfiguration,
) -> PersonalizationRunResult:
    """A completed SYNTHETIC regression personalization run."""
    directory = tmp_path_factory.mktemp("m6-personalization-regression")
    return run_personalization(
        PersonalizationRunConfiguration(
            dataset_path=m5_dataset,
            target_name=TargetName.ENGAGEMENT_SCORE,
            output_directory=directory,
            personalization=m6_personalization_configuration,
            n_splits=3,
        )
    )


# ---------------------------------------------------------------------------
# Milestone 7: uncertainty, selective prediction, abstention
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Milestone 10: MLOps, packaging, reproducibility
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def m10_baseline_run(
    tmp_path_factory: pytest.TempPathFactory,
    m5_dataset: Path,
) -> Path:
    """A completed SYNTHETIC baseline run directory, with persisted models.

    Two estimators rather than the full registry: the Milestone 10 tests
    read a run's documents and hash its model files, and a larger run
    would only make them slower without exercising a different path.
    """
    from engagevr.schemas.experiments import EvaluationMode
    from engagevr.training.calibration import CalibrationMethod
    from engagevr.training.runner import RunConfiguration, run_baselines

    directory = tmp_path_factory.mktemp("m10-baseline")
    run_baselines(
        RunConfiguration(
            dataset_path=m5_dataset,
            target_name=TargetName.ENGAGEMENT_CLASS,
            output_directory=directory,
            evaluation_mode=EvaluationMode.SOFTWARE_SELF_CHECK,
            n_splits=3,
            random_seed=42,
            model_names=("dummy", "logistic_regression"),
            calibration_method=CalibrationMethod.SIGMOID,
            run_ablations=False,
        )
    )
    return directory


@pytest.fixture(scope="session")
def m10_shifted_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A second SYNTHETIC draw, for the distribution-shift diagnostic."""
    configuration = SyntheticDatasetConfig(
        seed=1337,
        subjects=12,
        sessions_per_subject=2,
        windows_per_session=5,
        window_duration_seconds=10.0,
        window_step_seconds=10.0,
    )
    directory = tmp_path_factory.mktemp("m10-shifted")
    path = directory / "m10-shifted.parquet"
    write_dataset(
        generate_synthetic_dataset(configuration),
        path,
        target_names=list(TargetName),
        window_duration_seconds=configuration.window_duration_seconds,
        window_step_seconds=configuration.window_step_seconds,
        windows_overlap=configuration.windows_overlap,
        creation_configuration=configuration.model_dump(mode="json"),
        random_seed=configuration.seed,
    )
    return path


@pytest.fixture(scope="session")
def m7_selective_configuration() -> SelectivePredictionConfiguration:
    """A selective-prediction policy sized for the shared fixture dataset.

    Three calibration windows out of each subject's ten leaves a strictly
    later evaluation region for every subject, so the personalized-threshold
    path actually fires rather than falling back everywhere.  The grids are
    short on purpose: these tests verify the sweeps' behaviour, not the shape
    of any particular curve.

    The two grids are separate surfaces.  ``threshold_grid`` holds
    probabilities compared against a confidence score, and raising one is
    stricter.  ``interval_width_grid`` holds widths in the target's own
    units compared against a prediction-interval width, and raising one is
    more permissive.  They happen to span the same numbers here only
    because the fixture targets are scaled to ``[0, 1]``; neither is
    derived from the other.
    """
    return SelectivePredictionConfiguration(
        modalities=tuple(FusionModality),
        threshold_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        interval_width_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        population_confidence_threshold=0.5,
        personalized_thresholds_enabled=True,
        personal_calibration_windows=3,
        minimum_personal_calibration_windows=2,
        alpha=0.2,
    )


@pytest.fixture(scope="session")
def m7_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A SYNTHETIC dataset large enough to calibrate SOME folds and not others.

    The shared Milestone 5 fixture is deliberately tiny, and its calibration
    groups are too thin for ``CalibratedClassifierCV`` in every fold — which
    exercises the refusal path but never the calibrated one. Twenty subjects
    with sixteen windows each produces a run whose folds are *mixed*: some
    calibrate, at least one does not. Both branches of the confidence
    contract are therefore covered by one run.
    """
    configuration = SyntheticDatasetConfig(
        seed=42,
        subjects=20,
        sessions_per_subject=2,
        windows_per_session=8,
        window_duration_seconds=10.0,
        window_step_seconds=10.0,
    )
    directory = tmp_path_factory.mktemp("m7-dataset")
    path = directory / "m7-synthetic.parquet"
    write_dataset(
        generate_synthetic_dataset(configuration),
        path,
        target_names=list(TargetName),
        window_duration_seconds=configuration.window_duration_seconds,
        window_step_seconds=configuration.window_step_seconds,
        windows_overlap=configuration.windows_overlap,
        creation_configuration=configuration.model_dump(mode="json"),
        random_seed=configuration.seed,
    )
    return path


@pytest.fixture(scope="session")
def m7_uncertainty_classification_run(
    tmp_path_factory: pytest.TempPathFactory,
    m7_dataset: Path,
    m7_selective_configuration: SelectivePredictionConfiguration,
) -> UncertaintyRunResult:
    """A completed SYNTHETIC classification uncertainty run."""
    directory = tmp_path_factory.mktemp("m7-uncertainty-classification")
    return run_uncertainty(
        UncertaintyRunConfiguration(
            dataset_path=m7_dataset,
            target_name=TargetName.ENGAGEMENT_CLASS,
            output_directory=directory,
            selective=m7_selective_configuration,
            n_splits=3,
        )
    )


@pytest.fixture(scope="session")
def m7_uncertainty_regression_run(
    tmp_path_factory: pytest.TempPathFactory,
    m7_dataset: Path,
    m7_selective_configuration: SelectivePredictionConfiguration,
) -> UncertaintyRunResult:
    """A completed SYNTHETIC regression uncertainty run."""
    directory = tmp_path_factory.mktemp("m7-uncertainty-regression")
    return run_uncertainty(
        UncertaintyRunConfiguration(
            dataset_path=m7_dataset,
            target_name=TargetName.ENGAGEMENT_SCORE,
            output_directory=directory,
            selective=m7_selective_configuration,
            n_splits=3,
        )
    )
