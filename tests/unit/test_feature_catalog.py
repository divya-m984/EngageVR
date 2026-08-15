"""Feature-catalog contract tests.

The catalog is the single declaration of what may enter a dataset and what
may reach a model. These tests assert that every entry is complete, that
the vocabulary stays free of psychological constructs, and that the
permitted-predictor rule is enforced rather than merely documented.
"""

from __future__ import annotations

import pytest

from engagevr.features.catalog import (
    FEATURE_CATALOG,
    FEATURE_CATALOG_VERSION,
    feature_names_for_modalities,
    get_catalog,
)
from engagevr.schemas.features import (
    FeatureCatalog,
    FeatureDtype,
    FeatureModality,
    FeatureSpec,
    MissingBehaviour,
)


class TestCatalogCompleteness:
    def test_every_entry_declares_the_full_contract(self) -> None:
        for entry in FEATURE_CATALOG.entries:
            assert entry.canonical_name
            assert entry.description
            assert entry.unit
            assert entry.aggregation_formula
            assert entry.minimum_evidence
            assert entry.quality_dependency
            assert isinstance(entry.modality, FeatureModality)
            assert isinstance(entry.missing_behaviour, MissingBehaviour)
            assert isinstance(entry.permitted_predictor, bool)

    def test_names_are_unique(self) -> None:
        names = FEATURE_CATALOG.names()
        assert len(set(names)) == len(names)

    def test_duplicate_names_are_rejected(self) -> None:
        entry = FEATURE_CATALOG.entries[0]
        with pytest.raises(ValueError, match="duplicate feature names"):
            FeatureCatalog(version="test", entries=(entry, entry))

    def test_every_modality_group_is_populated(self) -> None:
        for modality in FeatureModality:
            assert FEATURE_CATALOG.by_modality(modality), modality

    def test_catalog_order_is_stable(self) -> None:
        assert FEATURE_CATALOG.names() == get_catalog().names()

    def test_unknown_catalog_version_is_refused(self) -> None:
        with pytest.raises(KeyError, match="not implemented"):
            get_catalog("99.0")

    def test_version_constant_matches_catalog(self) -> None:
        assert FEATURE_CATALOG.version == FEATURE_CATALOG_VERSION


class TestUnitsAndRanges:
    def test_percentage_features_declare_a_percent_unit_and_range(self) -> None:
        for entry in FEATURE_CATALOG.entries:
            if entry.canonical_name.endswith("_pct"):
                assert entry.unit == "percent", entry.canonical_name
                assert entry.value_minimum == 0.0
                assert entry.value_maximum == 100.0

    def test_proportion_features_are_bounded_to_unit_interval(self) -> None:
        for entry in FEATURE_CATALOG.entries:
            if entry.canonical_name.endswith("_proportion"):
                assert entry.unit == "proportion"
                assert entry.value_minimum == 0.0
                assert entry.value_maximum == 1.0

    def test_reaction_time_features_are_milliseconds(self) -> None:
        for entry in FEATURE_CATALOG.by_modality(FeatureModality.TASK):
            if "reaction_time" in entry.canonical_name:
                assert entry.unit == "milliseconds"

    def test_head_pose_angles_are_degrees(self) -> None:
        for entry in FEATURE_CATALOG.by_modality(FeatureModality.HEAD_POSE):
            if entry.canonical_name.endswith("_deg"):
                assert entry.unit == "degrees"

    def test_inconsistent_bounds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="value_minimum exceeds"):
            FeatureSpec(
                canonical_name="bad",
                description="d",
                modality=FeatureModality.TASK,
                unit="count",
                aggregation_formula="f",
                minimum_evidence="e",
                missing_behaviour=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
                quality_dependency="none",
                permitted_predictor=True,
                value_minimum=5.0,
                value_maximum=1.0,
            )


class TestPredictorPermission:
    def test_rppg_method_is_not_a_permitted_predictor(self) -> None:
        entry = FEATURE_CATALOG.get("rppg_method")
        assert entry.dtype is FeatureDtype.CATEGORY
        assert entry.permitted_predictor is False

    def test_categorical_features_can_never_be_predictors(self) -> None:
        with pytest.raises(ValueError, match="categorical features are not"):
            FeatureSpec(
                canonical_name="bad_category",
                description="d",
                modality=FeatureModality.RPPG,
                unit="method name",
                aggregation_formula="f",
                minimum_evidence="e",
                missing_behaviour=MissingBehaviour.NULL_WHEN_UNAVAILABLE,
                quality_dependency="none",
                permitted_predictor=True,
                dtype=FeatureDtype.CATEGORY,
            )

    def test_predictor_names_exclude_non_permitted_entries(self) -> None:
        assert "rppg_method" not in FEATURE_CATALOG.predictor_names()
        assert set(FEATURE_CATALOG.predictor_names()) <= set(FEATURE_CATALOG.names())

    def test_feature_names_for_modalities_respects_catalog_order(self) -> None:
        names = feature_names_for_modalities((FeatureModality.TASK,))
        catalog_order = [
            e.canonical_name
            for e in FEATURE_CATALOG.by_modality(FeatureModality.TASK)
            if e.permitted_predictor
        ]
        assert list(names) == catalog_order

    def test_feature_names_for_modalities_can_include_non_predictors(self) -> None:
        names = feature_names_for_modalities(
            (FeatureModality.RPPG,), predictors_only=False
        )
        assert "rppg_method" in names


class TestNoPsychologicalConstructs:
    #: Words that would assert a psychological or clinical conclusion.
    FORBIDDEN = (
        "engagement",
        "engaged",
        "attention",
        "attentive",
        "cognitive_load",
        "fatigue",
        "drowsi",
        "stress",
        "anxiety",
        "emotion",
        "mood",
        "boredom",
        "alertness",
    )

    def test_no_feature_name_asserts_a_construct(self) -> None:
        for name in FEATURE_CATALOG.names():
            lowered = name.lower()
            for word in self.FORBIDDEN:
                assert word not in lowered, name

    def test_proxy_features_disclaim_the_construct_in_their_description(
        self,
    ) -> None:
        """A proxy's description must say what it is not.

        The names carry "proxy"; the descriptions must also state, in
        words, that the value is geometry rather than a psychological
        measurement, so a reader of the catalog cannot mistake one for the
        other.
        """
        disclaimers = ("not a", "not a measure", "geometric", "ratio", "does not")
        for entry in FEATURE_CATALOG.by_modality(FeatureModality.BEHAVIOURAL):
            if "proxy" not in entry.canonical_name:
                continue
            lowered = entry.description.lower()
            assert any(phrase in lowered for phrase in disclaimers), (
                entry.canonical_name
            )


class TestMissingBehaviour:
    def test_counts_use_zero_when_no_events(self) -> None:
        for name in (
            "blink_proxy_count",
            "task_attempted_trials",
            "task_correct_count",
            "task_timeout_count",
        ):
            entry = FEATURE_CATALOG.get(name)
            assert entry.missing_behaviour is MissingBehaviour.ZERO_WHEN_NO_EVENTS

    def test_physiological_estimates_are_null_when_unavailable(self) -> None:
        entry = FEATURE_CATALOG.get("rppg_heart_rate_bpm")
        assert entry.missing_behaviour is MissingBehaviour.NULL_WHEN_UNAVAILABLE
        assert "quality" in entry.quality_dependency

    def test_rppg_spectral_summaries_exclude_rejected_windows(self) -> None:
        for name in ("rppg_spectral_peak_ratio", "rppg_peak_prominence"):
            entry = FEATURE_CATALOG.get(name)
            assert "rejected" in entry.quality_dependency

    def test_unknown_feature_lookup_raises(self) -> None:
        with pytest.raises(KeyError, match="not in feature catalog"):
            FEATURE_CATALOG.get("no_such_feature")
