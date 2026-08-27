"""The dashboard's pages.

Ten pages, each of which either renders a run or says precisely why it
cannot.  Every page that shows a result calls
:func:`~engagevr.dashboard.components.provenance_banner` first; a test
checks that, because a page that renders a synthetic metric under no
banner is exactly the failure this milestone exists to prevent.

Pages are plain functions taking a :class:`PageContext`.  They read the
view models the loaders built and hand them to the components; no page
opens a file, computes a metric, or decides anything about provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from engagevr.dashboard import components as ui
from engagevr.dashboard import formatting as fmt
from engagevr.dashboard import presentation
from engagevr.dashboard.catalogue import build_catalogue
from engagevr.dashboard.presentation import (
    ADAPTATION_NOTE,
    LIMITATIONS,
    PERSONALIZATION_NOTE,
    SELECTIVE_PREDICTION_NOTE,
    SIGNAL_QUALITY_NOTE,
    SUBJECT_ID_NOTE,
    SYNTHETIC_PAGE_NOTE,
    TERMINOLOGY,
)
from engagevr.dashboard.views_adaptation import hold_reason_detail, load_adaptation
from engagevr.dashboard.views_dataset import (
    load_dataset_provenance,
    load_signal_quality,
)
from engagevr.dashboard.views_fusion import load_fusion, load_personalization
from engagevr.dashboard.views_models import load_classification, load_regression
from engagevr.dashboard.views_uncertainty import load_uncertainty
from engagevr.schemas.dashboard import (
    DASHBOARD_PURPOSE,
    ArtifactIntegrityStatus,
    DashboardCatalogue,
    DashboardRunFamily,
    DashboardRunStatus,
    DashboardRunSummary,
    MetricKind,
)


@dataclass(frozen=True)
class PageContext:
    """Everything a page needs: the catalogue, the selection, the settings."""

    catalogue: DashboardCatalogue
    run: DashboardRunSummary | None
    max_table_rows: int
    show_subject_ids: bool
    validate_checksums: bool


def load_catalogue(
    artifact_root: Path, *, validate_checksums: bool
) -> DashboardCatalogue:
    """Scan the artifact root.

    Wrapped here so the Streamlit cache decorator in
    :mod:`engagevr.dashboard.app` has one function to attach to, and so
    the cached thing is a pure read of files rather than any mutable
    state.
    """
    return build_catalogue(artifact_root, validate_checksums=validate_checksums)


def _require_run(context: PageContext) -> DashboardRunSummary | None:
    if context.run is None:
        ui.unavailable(
            "No run is selected. Choose one in the sidebar, or generate one "
            "with a documented CLI command such as `uv run python -m engagevr "
            "baseline-demo`."
        )
        return None
    if not context.run.is_inspectable:
        ui.provenance_banner(context.run.provenance)
        ui.unavailable(
            f"This run's status is {context.run.provenance.status.value!r}, so "
            "no result is displayed for it. Showing numbers from a run that "
            "did not conclude would present them as though it had."
        )
        return None
    return context.run


def _require_family(
    context: PageContext, family: DashboardRunFamily, label: str
) -> DashboardRunSummary | None:
    run = _require_run(context)
    if run is None:
        return None
    if run.provenance.family is not family:
        ui.provenance_banner(run.provenance)
        ui.unavailable(
            f"The selected run is a {run.provenance.family.value} run, not a "
            f"{label} run. Select a {label} run in the sidebar to use this "
            "page."
        )
        return None
    return run


# --- 1. Overview ---------------------------------------------------------


def overview_page(context: PageContext) -> None:
    """Orient a researcher to the selected experiment."""
    st.title("Overview")
    st.caption(DASHBOARD_PURPOSE)

    catalogue = context.catalogue
    if not catalogue.root_exists:
        ui.render_warnings(catalogue.warnings)
        ui.unavailable(
            f"The artifact root {catalogue.artifact_root} does not exist yet. "
            "The dashboard runs; there is simply nothing to show."
        )
        return
    if catalogue.is_empty:
        ui.render_warnings(catalogue.warnings)
        ui.unavailable(
            f"No candidate run directory was found under "
            f"{catalogue.artifact_root}. This is a fresh artifact root, not "
            "an error."
        )
        return

    ui.section("Runs discovered")
    by_family = {
        family.value: len(catalogue.by_family(family))
        for family in catalogue.families()
    }
    ui.render_table(
        fmt.counts_table(
            title="Runs by family",
            counts=by_family,
            key_column="run family",
            source_artifact=catalogue.artifact_root,
            caption=(
                "Families are detected from the artifacts each directory "
                "contains. A directory name takes no part in classification."
            ),
        )
    )

    statuses = {
        status.value: sum(
            1 for run in catalogue.runs if run.provenance.status is status
        )
        for status in DashboardRunStatus
    }
    ui.render_table(
        fmt.counts_table(
            title="Runs by status",
            counts={k: v for k, v in statuses.items() if v},
            key_column="status",
            source_artifact=catalogue.artifact_root,
            caption=(
                "A directory existing is not a successful run. Incomplete, "
                "corrupt, and unsupported runs are listed rather than hidden."
            ),
        )
    )
    ui.render_warnings(catalogue.warnings)

    run = context.run
    if run is None:
        ui.unavailable("Select a run in the sidebar to see its details.")
        return

    st.divider()
    ui.section("Selected run")
    ui.provenance_banner(run.provenance)
    ui.render_metrics(
        [
            fmt.count(
                "Independent groups",
                run.group_count,
                source_artifact="manifest.json",
            ),
            fmt.count("Folds", run.fold_count, source_artifact="manifest.json"),
            fmt.count(
                "Evaluated windows",
                run.evaluated_window_count,
                source_artifact="run summary",
                unavailable_reason=(
                    "this run family does not record a single evaluated-window "
                    "count in its summary document"
                ),
            ),
            fmt.count(
                "Sessions",
                run.session_count,
                source_artifact="run summary",
                unavailable_reason=("this run family does not record a session count"),
            ),
        ]
    )
    if run.detection_note:
        st.caption(run.detection_note)
    missing = run.missing_required_artifacts
    if missing:
        st.error(
            f"Required artifact(s) missing: {', '.join(missing)}. This run is "
            "not a completed run.",
            icon="🛑",
        )
    else:
        st.success(
            "Every required artifact for this run family is present.",
            icon="✅",
        )
    st.caption(
        "This page deliberately shows no headline engagement score. The "
        "dashboard orients a researcher to an experiment; it does not "
        "monitor a person."
    )


# --- 2. Dataset and provenance ------------------------------------------


def dataset_page(context: PageContext) -> None:
    """Where the rows came from and how they were split."""
    st.title("Dataset and provenance")
    run = _require_run(context)
    if run is None:
        return
    ui.provenance_banner(run.provenance)
    data = load_dataset_provenance(run, max_rows=context.max_table_rows)
    ui.render_warnings(data.warnings)
    if data.unavailable_reason:
        ui.unavailable(data.unavailable_reason)
        return
    ui.render_table(data.dataset_table)
    if data.data_source_counts:
        ui.render_table(
            fmt.build_table(
                title="Data source of every window",
                columns=("data source", "label", "windows", "what it establishes"),
                rows=[
                    (
                        name,
                        presentation.data_source_label(name),
                        str(value),
                        presentation.data_source_statement(name),
                    )
                    for name, value in data.data_source_counts
                ],
                source_artifact="dataset.json",
                caption=(
                    "Recorded per row by the dataset builder. Synthetic rows "
                    "are permanently labelled at the row level, not by "
                    "convention. A public or live source says where the rows "
                    "came from and never that they are scientifically "
                    "eligible."
                ),
            )
        )
    st.divider()
    ui.section("Targets")
    ui.render_table(
        data.target_table,
        empty_reason="This dataset records no target column.",
    )
    st.divider()
    ui.section("Split and leakage audit")
    ui.render_table(data.split_table)
    if data.split_audit_passed is False:
        st.error(
            "The recorded split audit did not pass. Any metric computed on "
            "these folds must be read with that in mind.",
            icon="🛑",
        )
    for note in data.split_audit_notes:
        st.caption(note)
    ui.render_table(data.fold_table)
    if context.show_subject_ids:
        st.caption(SUBJECT_ID_NOTE)


# --- 3. Signal and feature quality --------------------------------------


def quality_page(context: PageContext) -> None:
    """Whether the measurement could be taken at all."""
    st.title("Signal and feature quality")
    st.error(SIGNAL_QUALITY_NOTE, icon="⚠️")
    run = _require_run(context)
    if run is None:
        return
    ui.provenance_banner(run.provenance)
    data = load_signal_quality(run, max_rows=context.max_table_rows)
    ui.render_warnings(data.warnings)
    if data.unavailable_reason:
        ui.unavailable(data.unavailable_reason)
        return
    if data.overall_missing_percentage is not None:
        ui.render_metrics([data.overall_missing_percentage])
    ui.render_table(data.modality_availability_table)
    st.divider()
    ui.render_bar_chart(data.missingness_chart)
    st.divider()
    ui.render_table(data.missing_feature_table)
    st.caption(SIGNAL_QUALITY_NOTE)


# --- 4. Baseline models --------------------------------------------------


def baseline_page(context: PageContext) -> None:
    """Milestone 5 interpretable baselines."""
    st.title("Baseline models")
    run = _require_family(context, DashboardRunFamily.BASELINE, "baseline")
    if run is None:
        return
    ui.provenance_banner(run.provenance)
    if run.provenance.is_synthetic:
        st.caption(SYNTHETIC_PAGE_NOTE)
    _render_task_results(context, run)


def _render_task_results(
    context: PageContext,
    run: DashboardRunSummary,
    *,
    document_name: str = "metrics.json",
) -> None:
    """Render whichever of the two task views this run's target calls for."""
    task_type = run.provenance.task_type
    if task_type == "classification":
        _render_classification(context, run, document_name=document_name)
    elif task_type == "regression":
        _render_regression(context, run, document_name=document_name)
    else:
        ui.unavailable(
            f"This run records task type {fmt.text(task_type)}, which is "
            "neither classification nor regression, so no result view is "
            "offered for it."
        )


def _render_classification(
    context: PageContext, run: DashboardRunSummary, *, document_name: str
) -> None:
    sort_by = st.selectbox(
        "Sort models by (descriptive only — this selects nothing)",
        options=("(artifact order)", "macro_f1", "balanced_accuracy", "accuracy"),
        index=0,
    )
    data = load_classification(
        run,
        document_name=document_name,
        max_rows=context.max_table_rows,
        sort_by=None if sort_by == "(artifact order)" else sort_by,
    )
    ui.render_warnings(data.warnings)
    if data.unavailable_reason:
        ui.unavailable(data.unavailable_reason)
        return
    st.caption(f"Classes: {', '.join(data.class_labels) or 'Unavailable'}")
    ui.render_table(data.aggregate_table)
    st.divider()
    ui.section("Confusion matrices")
    if data.confusion_matrices:
        for matrix in data.confusion_matrices:
            ui.render_confusion_matrix(matrix)
    else:
        ui.unavailable(
            "This run recorded no aggregate confusion matrix, so none is "
            "shown. One has not been reconstructed."
        )
    st.divider()
    ui.section("Probability calibration")
    st.caption(f"Calibration method: {fmt.text(data.calibration_method)}")
    ui.render_table(data.calibration_table)
    ui.render_chart(data.reliability_chart)
    st.divider()
    with st.expander("Fold-level results"):
        ui.render_table(data.fold_table)
    with st.expander("Per-class results"):
        ui.render_table(data.per_class_table)


def _render_regression(
    context: PageContext, run: DashboardRunSummary, *, document_name: str
) -> None:
    data = load_regression(
        run, document_name=document_name, max_rows=context.max_table_rows
    )
    ui.render_warnings(data.warnings)
    if data.unavailable_reason:
        ui.unavailable(data.unavailable_reason)
        return
    ui.render_table(data.aggregate_table)
    st.divider()
    ui.render_scatter(data.observed_versus_predicted)
    st.caption(
        f"The horizontal axis is the {data.observed_axis_label} this run "
        "recorded. No statistical significance is implied by any pattern in "
        "this plot."
    )
    st.divider()
    ui.render_bar_chart(data.residual_histogram)
    st.divider()
    ui.render_scatter(data.residual_versus_predicted)
    st.divider()
    with st.expander("Fold-level results"):
        ui.render_table(data.fold_table)


# --- 5. Multimodal fusion ------------------------------------------------


def fusion_page(context: PageContext) -> None:
    """Milestone 6 fusion strategies, experts, and robustness."""
    st.title("Multimodal fusion")
    run = _require_family(context, DashboardRunFamily.FUSION, "fusion")
    if run is None:
        return
    ui.provenance_banner(run.provenance)
    data = load_fusion(run, max_rows=context.max_table_rows)
    ui.render_warnings(data.warnings)
    if data.unavailable_reason:
        ui.unavailable(data.unavailable_reason)
        return
    st.caption(f"Modalities: {', '.join(data.modalities) or 'Unavailable'}")
    ui.render_table(data.strategy_table)
    st.divider()
    ui.section(
        "Fusion support weights",
        "A support weight is not a probability of correctness.",
    )
    ui.render_table(data.fusion_support_weight_table)
    st.divider()
    ui.section(
        "Expert disagreement",
        "Expert disagreement is not calibrated uncertainty. Calibrated "
        "uncertainty is on the Uncertainty and abstention page and comes "
        "from a different milestone.",
    )
    ui.render_table(data.expert_disagreement_table)
    st.divider()
    ui.section(
        "Modality availability",
        "Availability is a measurement property, kept separate from signal "
        "quality and from model confidence.",
    )
    ui.render_table(data.modality_availability_table)
    st.divider()
    ui.section("Missing-modality robustness")
    ui.render_chart(data.robustness_chart)
    ui.render_table(data.robustness_table)
    st.caption(
        "The absence of a modality is the absence of a measurement. It is "
        "not the absence of engagement or cognitive load."
    )
    st.divider()
    with st.expander("Modality experts"):
        ui.render_table(data.expert_table)


# --- 6. Personalization --------------------------------------------------


def personalization_page(context: PageContext) -> None:
    """Milestone 6 personalization, reported without a benefit claim."""
    st.title("Personalization")
    st.warning(PERSONALIZATION_NOTE, icon="⚠️")
    run = _require_family(
        context, DashboardRunFamily.PERSONALIZATION, "personalization"
    )
    if run is None:
        return
    ui.provenance_banner(run.provenance)
    data = load_personalization(run, max_rows=context.max_table_rows)
    ui.render_warnings(data.warnings)
    if data.unavailable_reason:
        ui.unavailable(data.unavailable_reason)
        return
    ui.render_metrics(
        [
            fmt.count(
                "Evaluation windows (both arms)",
                data.population_evaluation_window_count,
                source_artifact="personalization.json",
            ),
            fmt.count(
                "Calibration windows",
                data.calibration_window_count,
                source_artifact="personalization.json",
            ),
            fmt.count(
                "Personalized subjects",
                data.personalized_subject_count,
                source_artifact="personalization.json",
            ),
            fmt.count(
                "Cold-start subjects",
                data.cold_start_subject_count,
                source_artifact="personalization.json",
            ),
        ]
    )
    ui.render_table(data.paired_metric_table)
    st.divider()
    ui.render_table(data.metric_delta_table)
    st.caption(
        "This column is a difference. A negative value means the "
        "personalized arm scored lower on this run's synthetic data, and it "
        "is shown exactly as recorded."
    )
    st.divider()
    ui.render_table(data.coverage_table)
    st.divider()
    with st.expander("Per-fold personalization"):
        ui.render_table(data.fold_table)
    if context.show_subject_ids:
        with st.expander("Subject-wise software evaluation"):
            st.caption(SUBJECT_ID_NOTE)
            ui.render_table(data.subject_diagnostic_table)


# --- 7. Uncertainty and abstention --------------------------------------


def uncertainty_page(context: PageContext) -> None:
    """Milestone 7 selective prediction, with task-aware controls."""
    st.title("Uncertainty and abstention")
    st.caption(SELECTIVE_PREDICTION_NOTE)
    run = _require_family(context, DashboardRunFamily.UNCERTAINTY, "uncertainty")
    if run is None:
        return
    ui.provenance_banner(run.provenance)
    data = load_uncertainty(run, max_rows=context.max_table_rows)
    ui.render_warnings(data.warnings)
    if data.unavailable_reason:
        ui.unavailable(data.unavailable_reason)
        return

    accounting = data.accounting
    if accounting is not None:
        ui.render_metrics(
            [
                fmt.count("Evaluated windows", accounting.evaluated_window_count),
                fmt.count("Accepted", accounting.accepted_count),
                fmt.count("Abstained", accounting.abstained_count),
                fmt.count("Unavailable", accounting.unavailable_count),
            ]
        )
        if accounting.reconciles:
            st.success(
                f"{accounting.accepted_count} accepted + "
                f"{accounting.abstained_count} abstained + "
                f"{accounting.unavailable_count} unavailable = "
                f"{accounting.evaluated_window_count} evaluated windows.",
                icon="✅",
            )
        else:
            st.error(accounting.reconciliation_error, icon="🛑")
        st.caption(
            "Abstained is not an error count. Unavailable is a separate "
            "state again: nothing was withheld, because nothing was produced."
        )
    else:
        ui.unavailable(
            "This run does not record the four counts needed for selective "
            "accounting, so no coverage figure is shown."
        )

    st.divider()
    ui.render_table(data.abstention_reason_table)
    st.divider()
    st.caption(
        f"Coverage axis for this task type: **{fmt.text(data.coverage_axis)}**. "
        f"{fmt.text(data.coverage_axis_units)}"
    )

    if data.task_type == "classification":
        ui.section("Classification selective prediction")
        ui.render_bar_chart(data.calibrated_confidence_histogram)
        ui.render_bar_chart(data.predictive_entropy_histogram)
        ui.render_bar_chart(data.probability_margin_histogram)
        st.caption(
            "Probability calibration status: "
            f"{fmt.text(data.probability_calibration_status)}"
        )
        st.divider()
        ui.render_chart(data.confidence_coverage_curve)
        ui.render_chart(data.risk_coverage_curve)
    elif data.task_type == "regression":
        ui.section("Regression selective prediction")
        st.caption(
            "No confidence control appears on this page. A regression target "
            "has no class probability, so it has no calibrated confidence, no "
            "probability margin, and no confidence threshold."
        )
        ui.render_metrics(
            [
                m
                for m in (
                    data.empirical_interval_coverage,
                    data.configured_maximum_interval_width,
                )
                if m is not None
            ]
        )
        ui.render_bar_chart(data.interval_width_histogram)
        ui.render_table(data.interval_table)
        st.divider()
        ui.render_chart(data.width_coverage_curve)
        ui.render_chart(data.risk_coverage_curve)
    else:
        ui.unavailable(
            f"Task type {data.task_type!r} has no selective view in this dashboard."
        )

    st.divider()
    with st.expander("Applied thresholds"):
        ui.render_table(data.threshold_table)


# --- 8. Adaptive environment --------------------------------------------


def adaptation_page(context: PageContext) -> None:
    """Milestone 8 controller behaviour. No effectiveness is reported."""
    st.title("Adaptive environment")
    st.warning(ADAPTATION_NOTE, icon="⚠️")
    run = _require_family(context, DashboardRunFamily.ADAPTATION, "adaptation")
    if run is None:
        return
    ui.provenance_banner(run.provenance)
    data = load_adaptation(run, max_rows=context.max_table_rows)
    ui.render_warnings(data.warnings)
    if data.unavailable_reason:
        ui.unavailable(data.unavailable_reason)
        return

    st.caption(
        f"Experiment mode: **{fmt.text(data.experiment_mode)}** · "
        f"policy mode: {fmt.text(data.policy_mode)} · "
        f"adaptation enabled: {fmt.text(data.adaptation_enabled)} · "
        f"configuration fingerprint: {fmt.text(data.configuration_fingerprint)}"
    )
    ui.render_metrics(
        [
            fmt.count("Evaluated windows", data.evaluated_windows),
            fmt.count("Gate eligible", data.gate_eligible_windows),
            fmt.count("Gate blocked", data.gate_blocked_windows),
            fmt.count("HOLD decisions", data.hold_decisions),
        ]
    )
    st.divider()
    ui.section(
        "Lifecycle",
        "Proposal, command built, dispatched, acknowledged, and applied are "
        "five separate states and are never added together.",
    )
    ui.render_table(data.lifecycle_table)
    if data.lifecycle is not None and data.lifecycle.commands_dispatched == 0:
        st.info(
            f"{data.lifecycle.proposals} proposal(s) and "
            f"{data.lifecycle.commands_built} built command(s), with 0 "
            "dispatched and 0 acknowledged. Nothing reached a running "
            "environment, so nothing in any environment changed.",
        )
    st.divider()
    ui.section("Why the controller held")
    ui.render_table(data.hold_reason_table)
    st.divider()
    ui.section("Controller behaviour")
    ui.render_table(data.spacing_table)
    st.divider()
    ui.section("Difficulty trace")
    ui.render_chart(data.difficulty_trace)
    st.divider()
    ui.section("Per-session behaviour")
    ui.render_table(data.session_table)
    if data.session_ids:
        selected = st.selectbox(
            "Inspect one session's per-window decisions",
            options=("(none)", *data.session_ids),
            index=0,
        )
        if selected != "(none)":
            ui.render_table(
                hold_reason_detail(run, selected, max_rows=context.max_table_rows)
            )
    st.divider()
    ui.section("Configured guards")
    ui.render_table(data.guard_table)
    st.divider()
    ui.section("Action-frequency comparison")
    ui.render_table(
        data.action_frequency_comparison_table,
        empty_reason=(
            "This run recorded no guard-free comparison controller, so no "
            "comparison is shown."
        ),
    )
    st.divider()
    with st.expander("Controller scenarios"):
        ui.render_table(data.scenario_table)


# --- 9. Run integrity ----------------------------------------------------


def integrity_page(context: PageContext) -> None:
    """Artifact presence, checksums, and pipeline status across all runs."""
    st.title("Run integrity")
    st.caption(
        "Artifact presence and checksum state for every discovered run. This "
        "page reads and reports; it never deletes, regenerates, or repairs "
        "anything."
    )
    catalogue = context.catalogue
    ui.render_warnings(catalogue.warnings)
    if catalogue.is_empty:
        ui.unavailable(
            f"No candidate run directory was found under {catalogue.artifact_root}."
        )
        return

    if not context.validate_checksums:
        st.info(
            "Checksum verification is switched off in the dashboard "
            "configuration, so every integrity status below reads 'not "
            "checked'. That is not the same as a passing check.",
        )

    rows = []
    for run in catalogue.runs:
        provenance = run.provenance
        rows.append(
            (
                run.directory_name,
                provenance.family.value,
                ui.STATUS_TEXT[provenance.status],
                ui.INTEGRITY_TEXT[provenance.integrity],
                fmt.text(provenance.is_synthetic),
                fmt.text(provenance.scientific_evaluation_eligible),
                fmt.text(provenance.dataset_fingerprint),
                fmt.text(", ".join(run.missing_required_artifacts) or None),
                fmt.text(provenance.failure_reason),
            )
        )
    ui.render_table(
        fmt.build_table(
            title="Every discovered run",
            columns=(
                "directory",
                "family",
                "status",
                "integrity",
                "synthetic",
                "scientifically eligible",
                "dataset fingerprint",
                "missing required artifacts",
                "failure reason",
            ),
            rows=rows,
            source_artifact=catalogue.artifact_root,
            max_rows=context.max_table_rows,
            caption=(
                "Run identity comes from recorded metadata. Filesystem "
                "modification time is not provenance and is not shown."
            ),
        )
    )

    mismatched = [
        run
        for run in catalogue.runs
        if run.provenance.integrity is ArtifactIntegrityStatus.MISMATCHED
    ]
    if mismatched:
        st.error(
            "Checksum mismatch in: "
            + ", ".join(run.directory_name for run in mismatched)
            + ". The displayed numbers for those runs may not be the numbers "
            "the run produced. Nothing has been deleted or regenerated.",
            icon="🛑",
        )

    selected = context.run
    if selected is None:
        return
    st.divider()
    ui.section(f"Artifacts of {selected.directory_name}")
    # The banner repeats here: this section shows one run's own state,
    # and a reader who scrolled straight to it must still see where that
    # run's numbers would come from.
    ui.provenance_banner(selected.provenance)
    ui.render_table(
        fmt.build_table(
            title="Artifact availability",
            columns=("artifact", "required", "present", "size (bytes)", "reason"),
            rows=[
                (
                    artifact.name,
                    fmt.text(artifact.required),
                    fmt.text(artifact.present),
                    fmt.text(artifact.size_bytes),
                    fmt.text(artifact.unavailable_reason),
                )
                for artifact in selected.artifacts
            ],
            source_artifact=selected.directory_name,
            max_rows=context.max_table_rows,
            caption=(
                "Model files are Python pickles and are never opened by this "
                "dashboard. Everything shown anywhere here comes from JSON "
                "and Parquet."
            ),
        )
    )


# --- 10. Limitations and scientific status -------------------------------


def limitations_page(context: PageContext) -> None:
    """The standing limitations, as a page rather than a footnote."""
    st.title("Limitations and scientific status")
    st.error(
        "No validated participant-labelled engagement or cognitive-load "
        "study exists in this repository. Nothing displayed anywhere in this "
        "dashboard is scientific evidence about any person.",
        icon="⚠️",
    )
    ui.section("Standing limitations")
    for record in LIMITATIONS:
        with st.expander(record.title, expanded=False):
            st.write(record.detail)
            if record.reference:
                st.caption(f"See {record.reference}")
    st.divider()
    ui.section(
        "Terminology",
        "These quantities are kept apart throughout the dashboard. None is "
        "a synonym for another and none is combined into a single "
        "'uncertainty score'.",
    )
    ui.render_table(
        fmt.build_table(
            title="Project vocabulary",
            columns=("quantity", "what it is", "what it is not"),
            rows=[
                (term.display_name, term.definition, "; ".join(term.is_not))
                for term in TERMINOLOGY
            ],
            source_artifact="engagevr.dashboard.presentation",
            max_rows=context.max_table_rows,
        )
    )
    st.divider()
    ui.section("Privacy")
    st.write(SUBJECT_ID_NOTE)
    st.write(
        "This dashboard reads no video, no image, and no webcam frame. It "
        "opens no model file. It contains no participant name, email "
        "address, or contact detail, because no such data exists in this "
        "repository."
    )


#: The pages, in navigation order.
PAGES: tuple[tuple[str, str], ...] = (
    ("Overview", "overview_page"),
    ("Dataset and provenance", "dataset_page"),
    ("Signal and feature quality", "quality_page"),
    ("Baseline models", "baseline_page"),
    ("Multimodal fusion", "fusion_page"),
    ("Personalization", "personalization_page"),
    ("Uncertainty and abstention", "uncertainty_page"),
    ("Adaptive environment", "adaptation_page"),
    ("Run integrity", "integrity_page"),
    ("Limitations and scientific status", "limitations_page"),
)

#: Pages that render a run result and must therefore show the banner.
RESULT_BEARING_PAGES: frozenset[str] = frozenset(
    {
        "overview_page",
        "dataset_page",
        "quality_page",
        "baseline_page",
        "fusion_page",
        "personalization_page",
        "uncertainty_page",
        "adaptation_page",
        "integrity_page",
    }
)

#: Metric kinds a page may pass to a card. Present so a reviewer can see
#: at a glance that no page invents its own formatting.
ALLOWED_METRIC_KINDS: frozenset[MetricKind] = frozenset(MetricKind)


__all__ = [
    "ALLOWED_METRIC_KINDS",
    "PAGES",
    "RESULT_BEARING_PAGES",
    "PageContext",
    "adaptation_page",
    "baseline_page",
    "dataset_page",
    "fusion_page",
    "integrity_page",
    "limitations_page",
    "load_catalogue",
    "overview_page",
    "personalization_page",
    "quality_page",
    "uncertainty_page",
]
