"""Fusion algebra tests: modality columns, weights, combination, disagreement.

Everything exercised here is pure: no estimator is fitted and no fold is
touched, so these tests pin the arithmetic that the runner depends on.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

import math

import pytest

from engagevr.features.catalog import FEATURE_CATALOG, feature_names_for_modalities
from engagevr.schemas.features import FeatureModality
from engagevr.schemas.fusion import (
    FusionModality,
    FusionStrategy,
    MissingQualityPolicy,
    ModalityPrediction,
    QualitySource,
    QualityWeightingConfiguration,
)
from engagevr.training.fusion import (
    FEATURE_MODALITY_OF,
    STRATEGY_DESCRIPTIONS,
    FusionError,
    align_probability_vector,
    build_fusion_weights,
    contributing_modalities,
    distinct_predicted_classes,
    early_fusion_columns,
    fuse_class_probabilities,
    fuse_regression_predictions,
    mean_pairwise_probability_distance,
    missing_modality_pattern,
    modality_expert_columns,
    parse_modality,
    parse_strategy,
    prediction_spread,
    probability_entropy,
    resolve_base_weights,
)

ALL = tuple(FusionModality)
LABELS = ("low", "medium", "high")


def _columns() -> tuple[str, ...]:
    """Predictor columns in catalogue order, as the loader produces them."""
    columns: list[str] = []
    for entry in FEATURE_CATALOG.entries:
        if not entry.permitted_predictor:
            continue
        columns.append(f"feat__{entry.canonical_name}")
        columns.append(f"avail__{entry.canonical_name}")
    for modality in FEATURE_CATALOG.modalities():
        columns.append(f"modality_available__{modality.value}")
        columns.append(f"modality_quality__{modality.value}")
    return tuple(columns)


def _available(modality: FusionModality, value: float = 0.5) -> ModalityPrediction:
    return ModalityPrediction(modality=modality, available=True, predicted_value=value)


def _unavailable(modality: FusionModality) -> ModalityPrediction:
    return ModalityPrediction(
        modality=modality,
        available=False,
        unavailable_reason="no evidence in this window",
    )


class TestParsing:
    def test_quality_is_refused_with_an_explanation(self) -> None:
        with pytest.raises(FusionError, match="support/context signals"):
            parse_modality("quality")

    def test_an_unknown_modality_is_refused(self) -> None:
        with pytest.raises(FusionError, match="unknown fusion modality"):
            parse_modality("eeg")

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("early", FusionStrategy.EARLY),
            ("uniform-late", FusionStrategy.UNIFORM_LATE),
            ("quality_late", FusionStrategy.QUALITY_LATE),
            ("validation-late", FusionStrategy.VALIDATION_WEIGHTED_LATE),
            ("stacked-late", FusionStrategy.STACKED_LATE),
        ],
    )
    def test_strategy_names_accept_hyphens(
        self, name: str, expected: FusionStrategy
    ) -> None:
        assert parse_strategy(name) is expected

    def test_an_unknown_strategy_is_refused(self) -> None:
        with pytest.raises(FusionError, match="No deep or neural fusion"):
            parse_strategy("transformer-attention")

    def test_every_strategy_is_described(self) -> None:
        assert set(STRATEGY_DESCRIPTIONS) == set(FusionStrategy)
        for description in STRATEGY_DESCRIPTIONS.values():
            assert "attention" not in description.lower()


class TestModalityColumns:
    def test_an_expert_sees_only_its_own_modality(self) -> None:
        columns = _columns()
        for modality in ALL:
            selected = modality_expert_columns(modality, columns, FEATURE_CATALOG)
            for column in selected:
                if column.startswith(("feat__", "avail__")):
                    name = column.split("__", 1)[1]
                    assert (
                        FEATURE_CATALOG.get(name).modality
                        is FEATURE_MODALITY_OF[modality]
                    )
                else:
                    assert column == f"modality_available__{modality.value}"

    def test_modality_quality_is_excluded_unless_configured(self) -> None:
        columns = _columns()
        without = modality_expert_columns(FusionModality.RPPG, columns, FEATURE_CATALOG)
        with_quality = modality_expert_columns(
            FusionModality.RPPG,
            columns,
            FEATURE_CATALOG,
            include_modality_quality=True,
        )
        assert "modality_quality__rppg" not in without
        assert "modality_quality__rppg" in with_quality

    def test_a_non_predictor_feature_never_appears(self) -> None:
        columns = _columns()
        selected = modality_expert_columns(
            FusionModality.RPPG, columns, FEATURE_CATALOG
        )
        assert "feat__rppg_method" not in selected

    def test_early_fusion_preserves_catalogue_order(self) -> None:
        columns = _columns()
        selected = early_fusion_columns(ALL, columns, FEATURE_CATALOG)
        positions = [columns.index(column) for column in selected]
        assert positions == sorted(positions)

    def test_early_fusion_covers_every_requested_group(self) -> None:
        columns = _columns()
        selected = early_fusion_columns(
            (FusionModality.TASK, FusionModality.RPPG), columns, FEATURE_CATALOG
        )
        task_features = set(
            feature_names_for_modalities(
                (FeatureModality.TASK,), catalog=FEATURE_CATALOG
            )
        )
        assert any(c.removeprefix("feat__") in task_features for c in selected)
        assert not any(c.startswith("feat__head_yaw") for c in selected)

    def test_early_fusion_excludes_targets_and_identifiers(self) -> None:
        columns = _columns()
        selected = early_fusion_columns(ALL, columns, FEATURE_CATALOG)
        for column in selected:
            assert not column.startswith(("target__", "target_meta__"))
            assert column not in {"window_id", "subject_id", "session_id"}

    def test_early_fusion_needs_two_groups(self) -> None:
        with pytest.raises(FusionError, match="at least two modality groups"):
            early_fusion_columns((FusionModality.TASK,), _columns(), FEATURE_CATALOG)

    def test_a_missing_group_is_reported_not_reduced(self) -> None:
        columns = tuple(c for c in _columns() if "rppg" not in c)
        with pytest.raises(FusionError, match="contribute no permitted predictor"):
            early_fusion_columns(ALL, columns, FEATURE_CATALOG)

    def test_missing_modality_pattern_is_stable(self) -> None:
        pattern = missing_modality_pattern(
            {m: m is not FusionModality.RPPG for m in ALL}, ALL
        )
        assert pattern == "rppg"
        assert missing_modality_pattern({m: True for m in ALL}, ALL) == "none"


class TestBaseWeights:
    def test_the_default_is_deterministic_equal_weighting(self) -> None:
        assert resolve_base_weights(ALL) == dict.fromkeys(ALL, 1.0)

    def test_a_configured_weight_is_honoured(self) -> None:
        weights = resolve_base_weights(ALL, {"task": 2.0})
        assert weights[FusionModality.TASK] == pytest.approx(2.0)
        assert weights[FusionModality.RPPG] == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
    def test_a_non_positive_weight_is_refused(self, bad: float) -> None:
        with pytest.raises(FusionError, match="finite and positive"):
            resolve_base_weights(ALL, {"task": bad})


class TestUniformWeights:
    def test_available_experts_share_equally(self) -> None:
        predictions = {m: _available(m) for m in ALL}
        weights = build_fusion_weights(
            modalities=ALL,
            predictions=predictions,
            base_weights=dict.fromkeys(ALL, 1.0),
        )
        assert all(w.contributed for w in weights)
        assert all(w.normalized_weight == pytest.approx(0.25) for w in weights)
        assert sum(w.normalized_weight for w in weights) == pytest.approx(1.0)

    def test_an_unavailable_expert_is_excluded_and_the_rest_renormalise(self) -> None:
        predictions = {
            m: (_unavailable(m) if m is FusionModality.RPPG else _available(m))
            for m in ALL
        }
        weights = build_fusion_weights(
            modalities=ALL,
            predictions=predictions,
            base_weights=dict.fromkeys(ALL, 1.0),
        )
        by_modality = {w.modality: w for w in weights}
        assert by_modality[FusionModality.RPPG].normalized_weight == 0.0
        assert by_modality[FusionModality.RPPG].contributed is False
        assert by_modality[FusionModality.RPPG].exclusion_reason
        assert sum(w.normalized_weight for w in weights) == pytest.approx(1.0)
        assert by_modality[FusionModality.TASK].normalized_weight == pytest.approx(
            1.0 / 3.0
        )

    def test_quality_is_not_used_when_no_configuration_is_supplied(self) -> None:
        weights = build_fusion_weights(
            modalities=ALL,
            predictions={m: _available(m) for m in ALL},
            base_weights=dict.fromkeys(ALL, 1.0),
        )
        assert all(w.quality_source is QualitySource.NOT_USED for w in weights)
        assert all(w.quality_used is None for w in weights)

    def test_no_available_expert_produces_no_contributor(self) -> None:
        weights = build_fusion_weights(
            modalities=ALL,
            predictions={m: _unavailable(m) for m in ALL},
            base_weights=dict.fromkeys(ALL, 1.0),
        )
        assert contributing_modalities(weights) == ()
        assert all(w.normalized_weight == 0.0 for w in weights)


class TestQualityAwareWeights:
    def test_weights_follow_the_documented_equation(self) -> None:
        quality = {FusionModality.TASK: 0.8, FusionModality.RPPG: 0.2}
        modalities = (FusionModality.TASK, FusionModality.RPPG)
        weights = build_fusion_weights(
            modalities=modalities,
            predictions={m: _available(m) for m in modalities},
            base_weights=dict.fromkeys(modalities, 1.0),
            quality=quality,
            quality_config=QualityWeightingConfiguration(),
        )
        by_modality = {w.modality: w for w in weights}
        assert by_modality[FusionModality.TASK].raw_effective_weight == pytest.approx(
            0.8
        )
        assert by_modality[FusionModality.TASK].normalized_weight == pytest.approx(0.8)
        assert by_modality[FusionModality.RPPG].normalized_weight == pytest.approx(0.2)
        assert sum(w.normalized_weight for w in weights) == pytest.approx(1.0)

    def test_a_low_quality_modality_contributes_less(self) -> None:
        modalities = (FusionModality.TASK, FusionModality.RPPG)
        weights = build_fusion_weights(
            modalities=modalities,
            predictions={m: _available(m) for m in modalities},
            base_weights=dict.fromkeys(modalities, 1.0),
            quality={FusionModality.TASK: 0.9, FusionModality.RPPG: 0.1},
            quality_config=QualityWeightingConfiguration(),
        )
        by_modality = {w.modality: w.normalized_weight for w in weights}
        assert by_modality[FusionModality.RPPG] < by_modality[FusionModality.TASK]

    def test_an_unavailable_modality_gets_zero_effective_weight(self) -> None:
        modalities = (FusionModality.TASK, FusionModality.RPPG)
        weights = build_fusion_weights(
            modalities=modalities,
            predictions={
                FusionModality.TASK: _available(FusionModality.TASK),
                FusionModality.RPPG: _unavailable(FusionModality.RPPG),
            },
            base_weights=dict.fromkeys(modalities, 1.0),
            quality={FusionModality.TASK: 0.9, FusionModality.RPPG: 1.0},
            quality_config=QualityWeightingConfiguration(),
        )
        by_modality = {w.modality: w for w in weights}
        assert by_modality[FusionModality.RPPG].raw_effective_weight == 0.0
        assert by_modality[FusionModality.RPPG].availability == 0.0
        assert by_modality[FusionModality.TASK].normalized_weight == pytest.approx(1.0)

    def test_missing_quality_uses_the_documented_fallback(self) -> None:
        modalities = (FusionModality.TASK, FusionModality.RPPG)
        weights = build_fusion_weights(
            modalities=modalities,
            predictions={m: _available(m) for m in modalities},
            base_weights=dict.fromkeys(modalities, 1.0),
            quality={FusionModality.TASK: None, FusionModality.RPPG: 0.5},
            quality_config=QualityWeightingConfiguration(),
        )
        by_modality = {w.modality: w for w in weights}
        task = by_modality[FusionModality.TASK]
        assert task.quality_source is QualitySource.DOCUMENTED_FALLBACK
        assert task.quality_used == pytest.approx(0.5)
        assert task.quality_used != 1.0

    def test_missing_quality_can_exclude_instead(self) -> None:
        modalities = (FusionModality.TASK, FusionModality.RPPG)
        weights = build_fusion_weights(
            modalities=modalities,
            predictions={m: _available(m) for m in modalities},
            base_weights=dict.fromkeys(modalities, 1.0),
            quality={FusionModality.TASK: None, FusionModality.RPPG: 0.5},
            quality_config=QualityWeightingConfiguration(
                missing_quality_policy=MissingQualityPolicy.EXCLUDE
            ),
        )
        by_modality = {w.modality: w for w in weights}
        assert by_modality[FusionModality.TASK].contributed is False
        assert by_modality[FusionModality.TASK].quality_source is (
            QualitySource.UNAVAILABLE
        )
        assert "never treated as perfect quality" in (
            by_modality[FusionModality.TASK].exclusion_reason or ""
        )
        assert by_modality[FusionModality.RPPG].normalized_weight == pytest.approx(1.0)

    def test_quality_below_the_minimum_is_excluded_with_a_reason(self) -> None:
        modalities = (FusionModality.TASK, FusionModality.RPPG)
        weights = build_fusion_weights(
            modalities=modalities,
            predictions={m: _available(m) for m in modalities},
            base_weights=dict.fromkeys(modalities, 1.0),
            quality={FusionModality.TASK: 0.9, FusionModality.RPPG: 0.05},
            quality_config=QualityWeightingConfiguration(minimum_quality=0.3),
        )
        by_modality = {w.modality: w for w in weights}
        excluded = by_modality[FusionModality.RPPG]
        assert excluded.contributed is False
        assert "below the configured minimum" in (excluded.exclusion_reason or "")
        assert "not about the person" in (excluded.exclusion_reason or "")

    def test_zero_quality_is_excluded_by_the_minimum_effective_weight(self) -> None:
        modalities = (FusionModality.TASK, FusionModality.RPPG)
        weights = build_fusion_weights(
            modalities=modalities,
            predictions={m: _available(m) for m in modalities},
            base_weights=dict.fromkeys(modalities, 1.0),
            quality={FusionModality.TASK: 0.9, FusionModality.RPPG: 0.0},
            quality_config=QualityWeightingConfiguration(),
        )
        by_modality = {w.modality: w for w in weights}
        assert by_modality[FusionModality.RPPG].contributed is False
        assert "minimum effective weight" in (
            by_modality[FusionModality.RPPG].exclusion_reason or ""
        )

    @pytest.mark.parametrize("bad", [-0.1, 1.5, math.nan])
    def test_an_invalid_quality_value_is_refused(self, bad: float) -> None:
        with pytest.raises(FusionError):
            build_fusion_weights(
                modalities=(FusionModality.TASK,),
                predictions={FusionModality.TASK: _available(FusionModality.TASK)},
                base_weights={FusionModality.TASK: 1.0},
                quality={FusionModality.TASK: bad},
                quality_config=QualityWeightingConfiguration(),
            )

    def test_equal_quality_reproduces_the_uniform_control(self) -> None:
        weights = build_fusion_weights(
            modalities=ALL,
            predictions={m: _available(m) for m in ALL},
            base_weights=dict.fromkeys(ALL, 1.0),
            quality=dict.fromkeys(ALL, 0.7),
            quality_config=QualityWeightingConfiguration(),
        )
        assert all(w.normalized_weight == pytest.approx(0.25) for w in weights)

    def test_a_quality_value_is_never_a_prediction(self) -> None:
        weights = build_fusion_weights(
            modalities=(FusionModality.RPPG,),
            predictions={FusionModality.RPPG: _available(FusionModality.RPPG, 0.9)},
            base_weights={FusionModality.RPPG: 1.0},
            quality={FusionModality.RPPG: 0.2},
            quality_config=QualityWeightingConfiguration(),
        )
        weight = weights[0]
        assert weight.quality_used == pytest.approx(0.2)
        assert weight.normalized_weight == pytest.approx(1.0)
        # The expert's own estimate is untouched by its signal quality.
        assert _available(FusionModality.RPPG, 0.9).predicted_value == pytest.approx(
            0.9
        )


class TestProbabilityFusion:
    def test_uniform_fusion_is_the_mean(self) -> None:
        fused = fuse_class_probabilities(
            [(0.5, (0.8, 0.1, 0.1)), (0.5, (0.2, 0.4, 0.4))], LABELS
        )
        assert fused == pytest.approx((0.5, 0.25, 0.25))
        assert sum(fused) == pytest.approx(1.0)

    def test_the_result_always_sums_to_one(self) -> None:
        fused = fuse_class_probabilities(
            [(0.3, (0.7, 0.2, 0.1)), (0.7, (0.1, 0.1, 0.8))], LABELS
        )
        assert sum(fused) == pytest.approx(1.0)

    def test_no_contribution_is_refused(self) -> None:
        with pytest.raises(FusionError, match="at least one available expert"):
            fuse_class_probabilities([], LABELS)

    def test_a_negative_weight_is_refused(self) -> None:
        with pytest.raises(FusionError, match="negative"):
            fuse_class_probabilities([(-1.0, (1.0, 0.0, 0.0))], LABELS)

    def test_a_non_finite_weight_is_refused(self) -> None:
        with pytest.raises(FusionError, match="not finite"):
            fuse_class_probabilities([(math.inf, (1.0, 0.0, 0.0))], LABELS)

    def test_all_zero_weights_are_refused(self) -> None:
        with pytest.raises(FusionError, match="every contributed fusion weight"):
            fuse_class_probabilities([(0.0, (1.0, 0.0, 0.0))], LABELS)

    def test_a_mismatched_vector_length_is_refused(self) -> None:
        with pytest.raises(FusionError, match="but 3 classes are declared"):
            fuse_class_probabilities([(1.0, (0.5, 0.5))], LABELS)


class TestVocabularyAlignment:
    def test_a_reordered_vocabulary_is_realigned(self) -> None:
        aligned = align_probability_vector(
            (0.1, 0.7, 0.2), ("high", "low", "medium"), LABELS
        )
        assert aligned == pytest.approx((0.7, 0.2, 0.1))

    def test_an_unseen_class_receives_zero_and_the_rest_renormalise(self) -> None:
        aligned = align_probability_vector((0.6, 0.4), ("low", "medium"), LABELS)
        assert aligned == pytest.approx((0.6, 0.4, 0.0))
        assert sum(aligned) == pytest.approx(1.0)

    def test_disjoint_vocabularies_are_refused(self) -> None:
        with pytest.raises(FusionError, match="do not overlap"):
            align_probability_vector((1.0,), ("other",), LABELS)

    def test_a_mismatched_source_length_is_refused(self) -> None:
        with pytest.raises(FusionError, match="source classes"):
            align_probability_vector((1.0,), ("low", "medium"), LABELS)


class TestRegressionFusion:
    def test_the_weighted_mean_is_returned(self) -> None:
        assert fuse_regression_predictions([(1.0, 0.2), (3.0, 0.6)]) == pytest.approx(
            0.5
        )

    def test_weights_are_normalised_over_contributors(self) -> None:
        assert fuse_regression_predictions([(0.25, 0.4), (0.25, 0.6)]) == pytest.approx(
            0.5
        )

    def test_an_absent_expert_is_never_replaced_with_zero(self) -> None:
        # Two experts, one absent: the fused value is the surviving estimate,
        # not the mean of that estimate and a fabricated zero.
        assert fuse_regression_predictions([(1.0, 0.8)]) == pytest.approx(0.8)

    def test_no_contribution_is_refused(self) -> None:
        with pytest.raises(FusionError, match="never replaced with zero"):
            fuse_regression_predictions([])

    def test_a_non_finite_expert_prediction_is_refused(self) -> None:
        with pytest.raises(FusionError, match="must report unavailable"):
            fuse_regression_predictions([(1.0, math.nan)])


class TestDisagreement:
    def test_unanimous_classification_has_one_distinct_class(self) -> None:
        assert distinct_predicted_classes(["low", "low", "low"]) == 1

    def test_disagreeing_classification_has_several(self) -> None:
        assert distinct_predicted_classes(["low", "high", "medium"]) == 3

    def test_pairwise_distance_is_zero_for_identical_vectors(self) -> None:
        distance = mean_pairwise_probability_distance(
            [(0.5, 0.3, 0.2), (0.5, 0.3, 0.2)]
        )
        assert distance == pytest.approx(0.0)

    def test_pairwise_distance_grows_with_disagreement(self) -> None:
        close = mean_pairwise_probability_distance([(0.5, 0.3, 0.2), (0.4, 0.4, 0.2)])
        far = mean_pairwise_probability_distance([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
        assert close is not None and far is not None
        assert far > close

    def test_a_single_expert_cannot_disagree(self) -> None:
        assert mean_pairwise_probability_distance([(1.0, 0.0, 0.0)]) is None
        assert prediction_spread([0.4]) == (None, None)

    def test_entropy_is_zero_for_a_certain_vector(self) -> None:
        assert probability_entropy((1.0, 0.0, 0.0)) == pytest.approx(0.0)

    def test_entropy_is_maximal_for_a_uniform_vector(self) -> None:
        assert probability_entropy((1 / 3, 1 / 3, 1 / 3)) == pytest.approx(math.log(3))

    def test_regression_spread_reports_deviation_and_range(self) -> None:
        deviation, spread = prediction_spread([0.2, 0.4, 0.6])
        assert deviation == pytest.approx(math.sqrt(0.02666666666), rel=1e-6)
        assert spread == pytest.approx(0.4)
