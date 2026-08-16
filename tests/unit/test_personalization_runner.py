"""Milestone 6 personalization runner: leakage, reporting, and artifacts.

Every run here is executed on a deterministic SYNTHETIC dataset.  No
number produced by these tests is model accuracy, personalization benefit,
or evidence about any person, and no test asserts that personalization
improved anything.

No test needs a webcam, a model asset, a display server, a network, Unity,
a public dataset, or participant data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from engagevr.schemas.experiments import (
    SOFTWARE_SELF_CHECK_BANNER,
    EvaluationMode,
    GroupField,
)
from engagevr.schemas.fusion import FusionModality
from engagevr.schemas.personalization import (
    PersonalizationConfiguration,
    PersonalizationMethod,
    PersonalizedPrediction,
)
from engagevr.schemas.targets import TargetName, TaskType
from engagevr.training.artifacts import verify_checksums
from engagevr.training.personalization import PersonalizationError
from engagevr.training.personalization_runner import (
    PERSONALIZATION_REQUIRED_ARTIFACTS,
    PersonalizationConfigurationError,
    PersonalizationRunConfiguration,
    PersonalizationRunResult,
    _required_regression_value,
    run_personalization,
)
from engagevr.training.runner import ScientificModeError

#: Values or fragments that must never appear in a persisted document.
FORBIDDEN_IDENTIFIERS: tuple[str, ...] = (
    "@example.com",
    "password",
    "api_key",
    "secret_key",
    "access_token",
    "first_name",
    "last_name",
)

#: Affirmative claims no artifact of this project may make. "champion" is
#: absent on purpose: it is permitted inside the sentence that denies there
#: is one, which ``test_champion_appears_only_inside_a_denial`` pins.
FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "production-ready",
    "clinically useful",
    "personalization improves",
    "personalisation improves",
    "personalized model is better",
    "personalized baselines outperform",
    "state of the art",
)


def _configuration(**overrides: object) -> PersonalizationConfiguration:
    payload: dict[str, object] = {
        "method": PersonalizationMethod.PERSONAL_BASELINE_AND_CORRECTION,
        "modalities": tuple(FusionModality),
        "calibration_windows": 3,
        "minimum_calibration_windows": 2,
    }
    payload.update(overrides)
    return PersonalizationConfiguration(**payload)  # type: ignore[arg-type]


def _run(
    dataset: Path,
    output: Path,
    *,
    target: TargetName = TargetName.ENGAGEMENT_CLASS,
    folds: int = 3,
    **overrides: object,
) -> PersonalizationRunResult:
    return run_personalization(
        PersonalizationRunConfiguration(
            dataset_path=dataset,
            target_name=target,
            output_directory=output,
            personalization=_configuration(**overrides),
            n_splits=folds,
        )
    )


class TestLeakageSafety:
    def test_the_population_model_never_trains_on_a_held_out_subject(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        result = m6_personalization_classification_run
        for fold in result.splits.folds:
            training = set(fold.train_groups)
            evaluated = {
                split.subject_id
                for evaluation_fold in result.personalization.folds
                if evaluation_fold.fold_index == fold.fold_index
                for split in evaluation_fold.splits
            }
            assert not training & evaluated

    def test_calibration_and_evaluation_belong_to_the_same_subject(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        directory = m6_personalization_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        by_subject: dict[str, set[str]] = {}
        for fold in m6_personalization_classification_run.personalization.folds:
            for split in fold.splits:
                by_subject.setdefault(split.subject_id, set()).update(
                    split.calibration_window_ids
                )
        for _index, row in table.iterrows():
            for window_id in row["calibration_window_ids"]:
                assert window_id in by_subject[row["subject_id"]]

    def test_calibration_precedes_evaluation_in_every_split(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        checked = 0
        for fold in m6_personalization_classification_run.personalization.folds:
            for split in fold.splits:
                if not split.calibration_window_ids:
                    continue
                assert split.calibration_end_utc is not None
                assert split.evaluation_start_utc is not None
                assert split.calibration_end_utc <= split.evaluation_start_utc
                assert split.temporal_order_verified
                checked += 1
        assert checked > 0

    def test_no_evaluation_window_is_also_a_calibration_window(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        for fold in m6_personalization_classification_run.personalization.folds:
            for split in fold.splits:
                assert not set(split.calibration_window_ids) & set(
                    split.evaluation_window_ids
                )

    def test_a_correction_only_ever_saw_calibration_labels(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        evaluation_windows: set[str] = set()
        recorded: set[str] = set()
        for fold in m6_personalization_classification_run.personalization.folds:
            for split in fold.splits:
                evaluation_windows.update(split.evaluation_window_ids)
            for correction in fold.corrections:
                recorded.update(correction.calibration_targets)
        assert recorded
        assert not recorded & evaluation_windows

    def test_a_baseline_is_estimated_only_from_calibration_windows(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        calibration_by_subject: dict[tuple[str, int], set[str]] = {}
        for fold in m6_personalization_classification_run.personalization.folds:
            for split in fold.splits:
                calibration_by_subject[(split.subject_id, fold.fold_index)] = set(
                    split.calibration_window_ids
                )
        checked = 0
        for record in m6_personalization_classification_run.baselines.statistics:
            allowed = calibration_by_subject[(record.subject_id, record.fold_index)]
            assert set(record.source_window_ids) <= allowed
            checked += 1
        assert checked > 0

    def test_the_split_manifest_audit_passed(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        assert m6_personalization_classification_run.splits.audit_passed

    def test_no_forbidden_column_reaches_the_predictor_matrix(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        columns = (
            m6_personalization_classification_run.personalization.predictor_columns
        )
        assert columns
        for column in columns:
            assert not column.startswith(("target__", "target_meta__", "window_"))
            assert column not in {"subject_id", "session_id", "window_id"}

    def test_only_measured_features_are_personalized(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        evaluation = m6_personalization_classification_run.personalization
        assert evaluation.personalized_columns
        for column in evaluation.personalized_columns:
            assert column.startswith("feat__")
        assert not any(
            column.startswith("modality_quality__")
            for column in evaluation.personalized_columns
        )


class TestClassificationReporting:
    def test_both_reports_cover_identical_evaluation_rows(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        for fold in m6_personalization_classification_run.personalization.folds:
            if not fold.evaluated:
                continue
            population = fold.population_classification_metrics
            personalized = fold.personalized_classification_metrics
            assert population is not None and personalized is not None
            assert population.sample_count == personalized.sample_count
            assert population.class_support == personalized.class_support

    def test_metrics_are_stored_as_two_separate_results(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        results = m6_personalization_classification_run.metrics.results
        assert [r.model_name for r in results] == ["population", "personalized"]
        assert [r.model_kind for r in results] == ["population", "personalized"]

    def test_every_probability_row_is_a_distribution(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        directory = m6_personalization_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        population = [
            c for c in table.columns if c.startswith("population_probability__")
        ]
        personalized = [
            c for c in table.columns if c.startswith("personalized_probability__")
        ]
        assert population and len(population) == len(personalized)
        for names in (population, personalized):
            matrix = table[names].to_numpy(dtype=float)
            assert np.isfinite(matrix).all()
            assert (matrix >= 0.0).all()
            assert matrix.sum(axis=1) == pytest.approx(1.0, abs=1e-9)

    def test_both_predictions_are_retained_on_every_row(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        directory = m6_personalization_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        assert table["population_predicted_class"].notna().all()
        assert table["personalized_predicted_class"].notna().all()

    def test_a_cold_start_row_reproduces_the_population_prediction(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        directory = m6_personalization_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        cold = table[~table["personalization_applied"]]
        for _index, row in cold.iterrows():
            assert row["cold_start"]
            assert row["cold_start_reason"]
            assert (
                row["population_predicted_class"] == row["personalized_predicted_class"]
            )

    def test_insufficient_class_support_falls_back_explicitly(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "class-support",
            minimum_calibration_classes=3,
        )
        evaluation = result.personalization
        assert evaluation.cold_start_subject_count > 0
        reasons = {
            correction.unavailable_reason
            for fold in evaluation.folds
            for correction in fold.corrections
            if not correction.available
        }
        assert any(
            reason is not None and "fewer than the 3 required" in reason
            for reason in reasons
        )


class TestRegressionReporting:
    def test_both_reports_cover_identical_evaluation_rows(
        self, m6_personalization_regression_run: PersonalizationRunResult
    ) -> None:
        for fold in m6_personalization_regression_run.personalization.folds:
            if not fold.evaluated:
                continue
            population = fold.population_regression_metrics
            personalized = fold.personalized_regression_metrics
            assert population is not None and personalized is not None
            assert population.sample_count == personalized.sample_count

    def test_every_prediction_is_finite(
        self, m6_personalization_regression_run: PersonalizationRunResult
    ) -> None:
        directory = m6_personalization_regression_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        for column in ("population_predicted_value", "personalized_predicted_value"):
            values = table[column].to_numpy(dtype=float)
            assert np.isfinite(values).all()

    def test_the_documented_bias_equation_is_recorded(
        self, m6_personalization_regression_run: PersonalizationRunResult
    ) -> None:
        applied = [
            correction
            for fold in m6_personalization_regression_run.personalization.folds
            for correction in fold.corrections
            if correction.available
        ]
        assert applied
        for correction in applied:
            assert correction.bias is not None
            assert "b_s = mean" in (correction.equation or "")
            assert correction.task_type is TaskType.REGRESSION

    def test_the_task_type_is_recorded_as_regression(
        self, m6_personalization_regression_run: PersonalizationRunResult
    ) -> None:
        assert (
            m6_personalization_regression_run.personalization.task_type
            is TaskType.REGRESSION
        )


def _regression_prediction(**overrides: object) -> PersonalizedPrediction:
    """A valid regression record; overrides bypass validation on purpose."""
    record = PersonalizedPrediction(
        window_id="w05",
        subject_id="synthetic-subject-01",
        session_id="sess-a",
        target_name="engagement_score",
        task_type=TaskType.REGRESSION,
        fold_index=0,
        method=PersonalizationMethod.FEW_SHOT_CORRECTION,
        population_predicted_value=0.4,
        personalized_predicted_value=0.5,
        personalization_applied=True,
        cold_start=False,
        calibration_window_ids=("w00", "w01"),
        calibration_sample_count=2,
        data_source="synthetic",
        is_synthetic=True,
        scientific_evaluation_eligible=False,
    )
    if not overrides:
        return record
    # ``model_copy`` deliberately skips validation: the schema refuses a
    # missing or non-finite regression prediction, and these tests exist to
    # prove the scoring path refuses one too rather than imputing zero.
    return record.model_copy(update=overrides)


class TestRequiredRegressionValue:
    """A missing prediction must never be silently scored as ``0.0``."""

    @pytest.mark.parametrize("kind", ["population", "personalized"])
    def test_a_legitimate_zero_stays_zero(self, kind: str) -> None:
        record = _regression_prediction(
            population_predicted_value=0.0, personalized_predicted_value=0.0
        )
        value = getattr(record, f"{kind}_predicted_value")
        assert _required_regression_value(value, kind=kind, prediction=record) == 0.0

    @pytest.mark.parametrize("kind", ["population", "personalized"])
    @pytest.mark.parametrize("given", [-2.5, -0.0, 1e-12, 0.5, 1234.75])
    def test_a_finite_value_passes_through_unchanged(
        self, kind: str, given: float
    ) -> None:
        record = _regression_prediction(
            population_predicted_value=given, personalized_predicted_value=given
        )
        value = getattr(record, f"{kind}_predicted_value")
        assert _required_regression_value(value, kind=kind, prediction=record) == given

    @pytest.mark.parametrize("kind", ["population", "personalized"])
    def test_none_is_rejected_rather_than_converted_to_zero(self, kind: str) -> None:
        record = _regression_prediction(**{f"{kind}_predicted_value": None})
        with pytest.raises(PersonalizationError) as raised:
            _required_regression_value(None, kind=kind, prediction=record)
        message = str(raised.value)
        assert "not zero" in message
        assert kind in message
        assert record.window_id in message
        assert record.subject_id in message

    @pytest.mark.parametrize("kind", ["population", "personalized"])
    @pytest.mark.parametrize(
        "given", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
    )
    def test_a_non_finite_value_is_rejected(self, kind: str, given: float) -> None:
        record = _regression_prediction(**{f"{kind}_predicted_value": given})
        with pytest.raises(PersonalizationError, match="not a finite estimate"):
            _required_regression_value(given, kind=kind, prediction=record)

    def test_the_two_sides_are_symmetric(self) -> None:
        record = _regression_prediction()
        for kind in ("population", "personalized"):
            with pytest.raises(PersonalizationError):
                _required_regression_value(None, kind=kind, prediction=record)
            with pytest.raises(PersonalizationError):
                _required_regression_value(float("nan"), kind=kind, prediction=record)
            assert _required_regression_value(0.0, kind=kind, prediction=record) == 0.0


class TestColdStart:
    def test_zero_calibration_windows_makes_every_subject_a_cold_start(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "cold-start",
            calibration_windows=0,
            method=PersonalizationMethod.PERSONAL_BASELINE,
        )
        evaluation = result.personalization
        assert evaluation.personalized_subject_count == 0
        assert evaluation.cold_start_subject_count > 0
        assert evaluation.personalization_coverage == 0.0
        assert evaluation.total_calibration_window_count == 0

    def test_a_cold_start_run_reproduces_the_population_metrics(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "cold-start-metrics",
            calibration_windows=0,
            method=PersonalizationMethod.PERSONAL_BASELINE,
        )
        population = {
            a.name: a.mean for a in result.personalization.population_aggregate
        }
        personalized = {
            a.name: a.mean for a in result.personalization.personalized_aggregate
        }
        assert population == personalized

    def test_no_personal_baseline_is_borrowed_from_another_subject(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "no-borrowing",
            calibration_windows=0,
            method=PersonalizationMethod.PERSONAL_BASELINE,
        )
        assert (
            all(not record.normalized for record in result.baselines.statistics)
            or not result.baselines.statistics
        )

    def test_the_population_only_method_changes_nothing(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        result = _run(
            m5_dataset,
            tmp_path / "population-only",
            method=PersonalizationMethod.POPULATION_ONLY,
        )
        table = pq.read_table(result.directory / "predictions.parquet").to_pandas()
        assert not table["personalization_applied"].any()
        assert (
            table["population_predicted_class"] == table["personalized_predicted_class"]
        ).all()


class TestArtifacts:
    def test_every_required_artifact_is_written(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        directory = m6_personalization_classification_run.directory
        for name in (*PERSONALIZATION_REQUIRED_ARTIFACTS, "predictions.parquet"):
            assert (directory / name).exists(), name

    def test_every_checksum_verifies(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        assert verify_checksums(m6_personalization_classification_run.directory) == ()

    def test_the_checksums_cover_the_personalization_artifacts(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        directory = m6_personalization_classification_run.directory
        recorded = json.loads((directory / "checksums.json").read_text())
        for name in PERSONALIZATION_REQUIRED_ARTIFACTS:
            assert name in recorded

    def test_calibration_and_evaluation_window_ids_are_recorded(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        document = json.loads(
            (
                m6_personalization_classification_run.directory / "personalization.json"
            ).read_text()
        )
        splits = [s for fold in document["folds"] for s in fold["splits"]]
        assert splits
        assert any(s["calibration_window_ids"] for s in splits)
        assert all(s["evaluation_window_ids"] for s in splits if s["available"])

    def test_baseline_statistics_are_recorded(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        statistics = m6_personalization_classification_run.baselines.statistics
        assert statistics
        assert any(record.normalized for record in statistics)
        for record in statistics:
            assert record.column.startswith("feat__")
            assert record.scale > 0.0

    def test_correction_parameters_are_recorded(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        applied = [
            correction
            for fold in m6_personalization_classification_run.personalization.folds
            for correction in fold.corrections
            if correction.available
        ]
        assert applied
        for correction in applied:
            assert correction.log_odds_shift
            assert correction.shrinkage is not None
            assert correction.smoothing is not None

    def test_no_identifier_or_secret_reaches_a_document(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        directory = m6_personalization_classification_run.directory
        for path in sorted(directory.glob("*.json")):
            text = path.read_text().lower()
            for token in FORBIDDEN_IDENTIFIERS:
                assert token not in text, f"{path.name} contains {token!r}"

    def test_no_document_makes_a_forbidden_claim(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        directory = m6_personalization_classification_run.directory
        for path in sorted(directory.glob("*.json")):
            text = path.read_text().lower()
            for claim in FORBIDDEN_CLAIMS:
                assert claim not in text, f"{path.name} contains {claim!r}"

    def test_champion_appears_only_inside_a_denial(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        """The word is allowed only in the sentence that denies there is one."""
        directory = m6_personalization_classification_run.directory
        seen = 0
        for path in sorted(directory.glob("*.json")):
            text = path.read_text().lower()
            occurrences = text.count("champion")
            assert occurrences == text.count("none is a champion"), path.name
            seen += occurrences
        assert seen > 0

    def test_no_document_carries_a_champion_field(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        def _keys(node: object) -> set[str]:
            if isinstance(node, dict):
                found = set(node)
                for value in node.values():
                    found |= _keys(value)
                return found
            if isinstance(node, list):
                found = set()
                for value in node:
                    found |= _keys(value)
                return found
            return set()

        directory = m6_personalization_classification_run.directory
        for path in sorted(directory.glob("*.json")):
            names = {name.lower() for name in _keys(json.loads(path.read_text()))}
            assert not {
                name
                for name in names
                if "champion" in name or "best" in name or "winner" in name
            }, path.name

    def test_the_synthetic_disclaimer_is_persisted(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        document = json.loads(
            (
                m6_personalization_classification_run.directory / "personalization.json"
            ).read_text()
        )
        assert any(
            SOFTWARE_SELF_CHECK_BANNER in disclaimer
            for disclaimer in document["disclaimers"]
        )
        assert (
            "not evidence of a personalization benefit" in document["comparison_note"]
        )

    def test_every_row_is_synthetic_and_none_is_eligible(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        directory = m6_personalization_classification_run.directory
        table = pq.read_table(directory / "predictions.parquet").to_pandas()
        assert table["is_synthetic"].all()
        assert not table["scientific_evaluation_eligible"].any()
        metrics = m6_personalization_classification_run.metrics
        assert not metrics.scientific_evaluation_eligible
        assert metrics.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK

    def test_the_manifest_records_the_personalization_configuration(
        self, m6_personalization_classification_run: PersonalizationRunResult
    ) -> None:
        manifest = m6_personalization_classification_run.manifest
        assert manifest.configuration["kind"] == "personalization"
        assert "personalization" in manifest.configuration
        assert manifest.status.value == "completed"


class TestDeterminism:
    def test_a_repeat_run_reproduces_the_identifier_and_the_documents(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        first = _run(m5_dataset, tmp_path / "repeat-a")
        second = _run(m5_dataset, tmp_path / "repeat-b")
        assert first.run_id == second.run_id
        for name in (
            "personalization.json",
            "personal_baselines.json",
            "metrics.json",
            "splits.json",
            "personalization_config.json",
        ):
            assert (first.directory / name).read_text() == (
                second.directory / name
            ).read_text()


class TestRefusals:
    def test_scientific_mode_refuses_a_synthetic_dataset(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ScientificModeError, match="synthetic"):
            run_personalization(
                PersonalizationRunConfiguration(
                    dataset_path=m5_dataset,
                    target_name=TargetName.ENGAGEMENT_CLASS,
                    output_directory=tmp_path / "scientific",
                    personalization=_configuration(),
                    evaluation_mode=EvaluationMode.SCIENTIFIC,
                    n_splits=3,
                )
            )

    def test_a_session_grouped_dataset_is_refused(
        self, m5_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a subject there is no person to personalise to.

        The synthetic generator refuses to build a one-subject dataset, so
        the grouping decision is forced here instead. That is the condition
        the guard exists for: a dataset with no usable subject identifier
        falls back to session grouping, and personalising to a session is
        not personalization.
        """
        monkeypatch.setattr(
            "engagevr.training.personalization_runner.choose_group_field",
            lambda subject_ids, session_ids: (
                GroupField.SESSION_ID,
                "forced for this test",
            ),
        )
        with pytest.raises(
            PersonalizationConfigurationError, match="requires subject grouping"
        ):
            _run(m5_dataset, tmp_path / "session-grouped")

    def test_a_refused_run_writes_no_manifest_claiming_success(
        self, m5_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "engagevr.training.personalization_runner.choose_group_field",
            lambda subject_ids, session_ids: (
                GroupField.SESSION_ID,
                "forced for this test",
            ),
        )
        output = tmp_path / "refused"
        with pytest.raises(PersonalizationConfigurationError):
            _run(m5_dataset, output)
        assert not (output / "manifest.json").exists()

    def test_a_subject_with_too_few_windows_is_reported_not_dropped_silently(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        # The shared fixture gives each subject ten windows; asking for
        # twelve calibration windows leaves no evaluation region at all.
        result = _run(
            m5_dataset,
            tmp_path / "too-few",
            calibration_windows=12,
            minimum_calibration_windows=2,
        )
        evaluation = result.personalization
        assert evaluation.unavailable_personalization_count > 0
        assert evaluation.total_evaluation_window_count == 0
        reasons = {
            split.unavailable_reason
            for fold in evaluation.folds
            for split in fold.splits
            if not split.available
        }
        assert reasons and all(reason for reason in reasons)
