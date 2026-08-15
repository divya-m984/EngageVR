"""Feature-group ablation definitions.

An ablation here is a **feature-subset comparison**, nothing more.  Each
subset is evaluated on the *same* outer folds as every other, with
preprocessing refitted inside each fold, so the only thing that differs
between two ablations is which features the model could see.

This is not multimodal fusion.  There is no modality mask, no
quality-aware weighting, and no early-versus-late fusion architecture in
this milestone; ``all_available`` simply means "no feature group was
removed".  Fusion is Milestone 6.

No scientific conclusion may be drawn from an ablation over synthetic
data.  Which feature group helps a model recover a latent variable that
this repository itself inserted is a fact about the generator.
"""

from __future__ import annotations

from dataclasses import dataclass

from engagevr.features.catalog import feature_names_for_modalities
from engagevr.schemas.features import FeatureCatalog, FeatureModality

_ALL_MODALITIES: tuple[FeatureModality, ...] = (
    FeatureModality.TASK,
    FeatureModality.BEHAVIOURAL,
    FeatureModality.HEAD_POSE,
    FeatureModality.RPPG,
    FeatureModality.QUALITY,
)


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """One named feature subset."""

    name: str
    modalities: tuple[FeatureModality, ...]
    description: str


#: The deterministic ablation set evaluated by ``baseline-demo``.
ABLATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        "task_only",
        (FeatureModality.TASK,),
        "Task telemetry only. Task output is a software measurement.",
    ),
    AblationSpec(
        "behavioural_only",
        (FeatureModality.BEHAVIOURAL,),
        "Facial behavioural proxies only.",
    ),
    AblationSpec(
        "head_pose_only",
        (FeatureModality.HEAD_POSE,),
        "Head-pose geometry only.",
    ),
    AblationSpec(
        "rppg_only",
        (FeatureModality.RPPG,),
        "Camera-based pulse estimates and their diagnostics only.",
    ),
    AblationSpec(
        "quality_only",
        (FeatureModality.QUALITY,),
        (
            "Capture-quality diagnostics only. Included as a control: if "
            "quality alone scores well, the result is being driven by "
            "measurement conditions rather than by anything about a person."
        ),
    ),
    AblationSpec(
        "all_available",
        _ALL_MODALITIES,
        (
            "Every permitted feature. NOT a fusion architecture: no modality "
            "mask and no quality weighting is applied."
        ),
    ),
    AblationSpec(
        "all_except_task",
        tuple(m for m in _ALL_MODALITIES if m is not FeatureModality.TASK),
        "Everything except task telemetry.",
    ),
    AblationSpec(
        "all_except_rppg",
        tuple(m for m in _ALL_MODALITIES if m is not FeatureModality.RPPG),
        "Everything except camera-based pulse estimates.",
    ),
    AblationSpec(
        "all_except_behavioural",
        tuple(m for m in _ALL_MODALITIES if m is not FeatureModality.BEHAVIOURAL),
        "Everything except facial behavioural proxies.",
    ),
)


def resolve_ablation_features(
    spec: AblationSpec,
    catalog: FeatureCatalog,
    available_features: frozenset[str],
) -> tuple[tuple[str, ...], str | None]:
    """Feature names for ``spec`` that are actually present in a dataset.

    Returns the resolved names and, when the ablation cannot be evaluated,
    the reason.  A missing feature group is reported rather than silently
    reduced to whatever happened to survive.
    """
    declared = feature_names_for_modalities(
        spec.modalities, catalog=catalog, predictors_only=True
    )
    resolved = tuple(name for name in declared if name in available_features)
    if not declared:
        return (), (
            f"ablation {spec.name!r} names modalities with no permitted "
            "predictors in this catalog"
        )
    if not resolved:
        missing = ", ".join(m.value for m in spec.modalities)
        return (), (
            f"ablation {spec.name!r} is unavailable: the dataset carries no "
            f"permitted predictor from modality group(s) {missing}"
        )
    return resolved, None
