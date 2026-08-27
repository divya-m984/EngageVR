"""Read-only discovery of experiment run directories.

The catalogue answers three questions about every directory under the
artifact root, and refuses to guess at any of them.

**Which milestone produced this?**  From the artifacts the directory
actually contains, never from its name.  A directory called ``m7-trial``
that holds no Milestone 7 document is not a Milestone 7 run; it is
``unknown``.  Directory names are display metadata and nothing else.

**Did it succeed?**  A directory existing is not a successful run.  A
manifest with ``status=completed`` is, and a manifest with
``status=failed`` is a run that reached a conclusion.  A directory with
no conclusive document at all is *incomplete*, which is a different
statement from *failed* and leads the reader somewhere different.

**Are the bytes still the bytes?**  Checksums are compared on request.
A mismatch is surfaced, never swallowed, and nothing here deletes,
regenerates, or repairs anything.

Nothing in this module opens a model file.  ``models/*.joblib`` are
pickles and loading one executes code in it; every fact this module
needs is in the JSON documents beside them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engagevr.schemas.dashboard import (
    ArtifactIntegrityStatus,
    DashboardArtifactAvailability,
    DashboardCatalogue,
    DashboardError,
    DashboardProvenance,
    DashboardRunFamily,
    DashboardRunStatus,
    DashboardRunSummary,
    DashboardWarning,
    DashboardWarningLevel,
)

#: Dataset-schema versions this dashboard knows how to read.
SUPPORTED_DATASET_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})

#: Feature-catalog versions this dashboard knows how to read.
SUPPORTED_FEATURE_CATALOG_VERSIONS: frozenset[str] = frozenset({"1.0"})

#: The document that concludes a training-family run.
MANIFEST = "manifest.json"
#: The document that concludes a Milestone 8 adaptation run.
ADAPTATION_SUMMARY = "adaptation_summary.json"
#: Recorded SHA-256 digests, written beside every run.
CHECKSUMS = "checksums.json"


@dataclass(frozen=True)
class FamilySignature:
    """How one run family is recognised from its artifacts.

    ``distinguishing`` are the documents only this family writes.  All of
    them must be present, which is what stops a fusion run from being
    read as a baseline run merely because it also writes ``metrics.json``.
    """

    family: DashboardRunFamily
    distinguishing: tuple[str, ...]
    required: tuple[str, ...]
    optional: tuple[str, ...]
    #: Artifacts whose presence rules this family out. Baseline needs
    #: this: every training family writes ``manifest.json``,
    #: ``metrics.json``, and ``splits.json``, so "baseline" means "a
    #: training run carrying none of the later milestones' documents"
    #: rather than "a directory holding those three files".
    disqualifying: tuple[str, ...] = ()
    #: Value of ``manifest.configuration.kind`` or ``.milestone``, when
    #: the producing runner records one, used only as a cross-check.
    configuration_marker: tuple[str, object] | None = None


#: Ordered most-specific first. The first signature whose distinguishing
#: artifacts are all present wins; a directory matching none is unknown.
FAMILY_SIGNATURES: tuple[FamilySignature, ...] = (
    FamilySignature(
        family=DashboardRunFamily.ADAPTATION,
        distinguishing=("adaptation_summary.json", "adaptation_policy_config.json"),
        required=(
            "adaptation_policy_config.json",
            "adaptation_trace.parquet",
            "adaptation_summary.json",
        ),
        optional=("scenarios.json", "checksums.json"),
    ),
    FamilySignature(
        family=DashboardRunFamily.UNCERTAINTY,
        distinguishing=("uncertainty.json", "uncertainty_config.json"),
        required=(
            "dataset.json",
            "feature_catalog.json",
            "splits.json",
            "uncertainty_config.json",
            "uncertainty.json",
            "thresholds.json",
            "selective_metrics.json",
            "coverage_curve.json",
            "metrics.json",
        ),
        optional=(
            "manifest.json",
            "checksums.json",
            "calibration.json",
            "selective_predictions.parquet",
            "adaptation_gate.parquet",
            "predictions.parquet",
        ),
    ),
    FamilySignature(
        family=DashboardRunFamily.PERSONALIZATION,
        distinguishing=("personalization.json", "personalization_config.json"),
        required=(
            "dataset.json",
            "feature_catalog.json",
            "splits.json",
            "personalization_config.json",
            "personalization.json",
            "personal_baselines.json",
            "metrics.json",
        ),
        optional=(
            "manifest.json",
            "checksums.json",
            "calibration.json",
            "predictions.parquet",
        ),
        configuration_marker=("kind", "personalization"),
    ),
    FamilySignature(
        family=DashboardRunFamily.FUSION,
        distinguishing=("fusion_metrics.json", "fusion_config.json"),
        required=(
            "dataset.json",
            "feature_catalog.json",
            "splits.json",
            "fusion_config.json",
            "experts.json",
            "metrics.json",
            "fusion_metrics.json",
            "robustness.json",
        ),
        optional=(
            "manifest.json",
            "checksums.json",
            "calibration.json",
            "predictions.parquet",
            "expert_predictions.parquet",
            "fusion_weights.parquet",
            "feature_importance.parquet",
        ),
        configuration_marker=("milestone", 6),
    ),
    FamilySignature(
        family=DashboardRunFamily.BASELINE,
        distinguishing=("manifest.json", "metrics.json", "splits.json"),
        required=(
            "dataset.json",
            "feature_catalog.json",
            "splits.json",
            "metrics.json",
        ),
        optional=(
            "manifest.json",
            "checksums.json",
            "calibration.json",
            "ablations.json",
            "predictions.parquet",
            "feature_importance.parquet",
        ),
        disqualifying=(
            "adaptation_summary.json",
            "adaptation_policy_config.json",
            "uncertainty.json",
            "uncertainty_config.json",
            "personalization.json",
            "personalization_config.json",
            "fusion_metrics.json",
            "fusion_config.json",
        ),
    ),
)


class ArtifactReadError(OSError):
    """A run artifact could not be read or parsed for display."""


def read_json(path: Path) -> Any:
    """Read one JSON document, or raise :class:`ArtifactReadError`.

    Callers turn this into a *corrupt* status rather than letting it
    propagate: one unreadable run must not take the dashboard down.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ArtifactReadError(f"{path.name} is missing from {path.parent}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactReadError(
            f"{path.name} in {path.parent} is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:  # pragma: no cover - permissions, device errors
        raise ArtifactReadError(f"{path} could not be read: {exc}") from exc


def read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON document that must be an object."""
    document = read_json(path)
    if not isinstance(document, dict):
        raise ArtifactReadError(
            f"{path.name} in {path.parent} holds a "
            f"{type(document).__name__}, not a JSON object"
        )
    return document


def detect_family(directory: Path) -> tuple[DashboardRunFamily, str | None]:
    """Classify a run directory from the artifacts it contains.

    Two passes, because an interrupted run must still be recognisable.
    The first pass requires *every* distinguishing artifact and yields a
    confident classification.  The second accepts *any* of them, which
    is how a fusion run that died before writing ``fusion_metrics.json``
    is reported as an incomplete fusion run rather than as an
    unclassifiable directory — the reader needs to know which run failed.

    A directory matching neither pass stays ``UNKNOWN``.  Its name is
    never consulted: a folder called ``m7-trial`` that holds no
    Milestone 7 document is not a Milestone 7 run.

    Returns the family and a note describing the evidence used, so a page
    can show *why* a directory was classified as it was rather than
    asking the reader to trust it.
    """
    for signature in FAMILY_SIGNATURES:
        if any((directory / name).exists() for name in signature.disqualifying):
            continue
        if all((directory / name).exists() for name in signature.distinguishing):
            evidence = ", ".join(signature.distinguishing)
            return signature.family, (
                f"classified as {signature.family.value} from the presence of "
                f"{evidence}. The directory name took no part in this."
            )
    for signature in FAMILY_SIGNATURES:
        if any((directory / name).exists() for name in signature.disqualifying):
            continue
        found = [
            name for name in signature.distinguishing if (directory / name).exists()
        ]
        if found:
            missing = [name for name in signature.distinguishing if name not in found]
            return signature.family, (
                f"classified as an INCOMPLETE {signature.family.value} run: "
                f"{', '.join(found)} is/are present but {', '.join(missing)} "
                "is/are absent. The directory name took no part in this."
            )
    return DashboardRunFamily.UNKNOWN, (
        "no known artifact signature matched. The directory name is not used "
        "to classify a run, so this stays unknown rather than being guessed."
    )


def signature_for(family: DashboardRunFamily) -> FamilySignature | None:
    """The signature of one family, or ``None`` for ``UNKNOWN``."""
    for signature in FAMILY_SIGNATURES:
        if signature.family is family:
            return signature
    return None


def artifact_availability(
    directory: Path, signature: FamilySignature | None
) -> tuple[DashboardArtifactAvailability, ...]:
    """Presence, size, and requirement status of a run's artifacts."""
    if signature is None:
        return ()
    entries: list[DashboardArtifactAvailability] = []
    seen: set[str] = set()
    for name, required in [(n, True) for n in signature.required] + [
        (n, False) for n in signature.optional
    ]:
        if name in seen:
            continue
        seen.add(name)
        path = directory / name
        present = path.is_file()
        entries.append(
            DashboardArtifactAvailability(
                name=name,
                present=present,
                required=required,
                size_bytes=path.stat().st_size if present else None,
                unavailable_reason=(
                    None
                    if present
                    else (
                        f"{name} is not present in this run directory"
                        + (
                            ". A required artifact is missing, so the run is "
                            "incomplete."
                            if required
                            else ". Views that need it will say so."
                        )
                    )
                ),
            )
        )
    return tuple(entries)


def verify_integrity(
    directory: Path, *, validate: bool
) -> tuple[ArtifactIntegrityStatus, tuple[str, ...], str | None]:
    """Compare recorded checksums with the bytes on disk.

    Returns the status, the offending artifact names, and a message.
    Nothing is deleted, regenerated, or repaired: a mismatch is reported
    to the reader and the reader decides.
    """
    if not validate:
        return (
            ArtifactIntegrityStatus.NOT_CHECKED,
            (),
            "checksum verification is switched off in the dashboard configuration",
        )
    path = directory / CHECKSUMS
    if not path.is_file():
        return (
            ArtifactIntegrityStatus.CHECKSUM_FILE_UNAVAILABLE,
            (),
            f"{CHECKSUMS} is not present, so integrity cannot be checked. "
            "This is not the same as a failed check.",
        )
    try:
        recorded = read_json_mapping(path)
    except ArtifactReadError as exc:
        return (ArtifactIntegrityStatus.CHECKSUM_FILE_CORRUPT, (), str(exc))

    from engagevr.training.artifacts import sha256_file

    missing: list[str] = []
    mismatched: list[str] = []
    for name, digest in recorded.items():
        artifact = directory / name
        if not artifact.is_file():
            missing.append(name)
            continue
        if sha256_file(artifact) != str(digest):
            mismatched.append(name)
    if mismatched:
        return (
            ArtifactIntegrityStatus.MISMATCHED,
            tuple(sorted(mismatched)),
            f"{len(mismatched)} artifact(s) no longer match the SHA-256 "
            "recorded when the run finished. The displayed numbers may not be "
            "the numbers the run produced.",
        )
    if missing:
        return (
            ArtifactIntegrityStatus.REFERENCED_FILE_MISSING,
            tuple(sorted(missing)),
            f"{len(missing)} artifact(s) named in {CHECKSUMS} are no longer "
            "present. This is a missing file, not a checksum failure.",
        )
    return (
        ArtifactIntegrityStatus.VALID,
        (),
        f"every artifact matches the SHA-256 recorded in {CHECKSUMS}",
    )


def _unsupported_version(
    document: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Declared artifact version, and why it is unsupported if it is."""
    declared = document.get("dataset_schema_version")
    if isinstance(declared, str):
        if declared not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
            return declared, (
                f"dataset_schema_version {declared!r} is not one this "
                f"dashboard can interpret "
                f"({sorted(SUPPORTED_DATASET_SCHEMA_VERSIONS)}). The format is "
                "not guessed at."
            )
        return declared, None
    catalog = document.get("feature_catalog_version")
    if isinstance(catalog, str):
        if catalog not in SUPPORTED_FEATURE_CATALOG_VERSIONS:
            return catalog, (
                f"feature_catalog_version {catalog!r} is not one this "
                f"dashboard can interpret "
                f"({sorted(SUPPORTED_FEATURE_CATALOG_VERSIONS)})"
            )
        return catalog, None
    return None, None


def _warning(
    level: DashboardWarningLevel, message: str, subject: str
) -> DashboardWarning:
    return DashboardWarning(level=level, message=message, subject=subject)


def _training_provenance(
    directory: Path,
    family: DashboardRunFamily,
    signature: FamilySignature | None,
    artifacts: Sequence[DashboardArtifactAvailability],
    *,
    integrity: ArtifactIntegrityStatus,
    integrity_message: str | None,
    validate_checksums: bool,
) -> DashboardRunSummary:
    """Build the summary of a run concluded by ``manifest.json``."""
    name = directory.name
    warnings: list[DashboardWarning] = []
    manifest = read_json_mapping(directory / MANIFEST)

    run_id = str(manifest.get("run_id") or name)
    evaluation_mode = manifest.get("evaluation_mode")
    eligible = bool(manifest.get("scientific_evaluation_eligible", False))
    # ``is_synthetic`` is not a manifest field for the training families;
    # the equivalent recorded statement is the evaluation mode, and
    # dataset.json carries the data source counts.
    synthetic = evaluation_mode == "software_self_check"
    data_source, source_warning = _data_source(directory, name)
    if source_warning is not None:
        warnings.append(source_warning)
    if data_source == "synthetic":
        synthetic = True
    if synthetic and eligible:
        raise DashboardError(
            f"run {run_id!r} records evaluation_mode={evaluation_mode!r} and "
            "scientific_evaluation_eligible=true at the same time"
        )

    declared_status = manifest.get("status")
    failure_reason = manifest.get("failure_reason")
    missing_required = [a.name for a in artifacts if a.required and not a.present]
    version, version_problem = _unsupported_version(manifest)

    if declared_status == "failed":
        status = DashboardRunStatus.FAILED
        failure_reason = str(
            failure_reason or "the manifest records a failure with no reason"
        )
    elif version_problem is not None:
        status = DashboardRunStatus.UNSUPPORTED
        warnings.append(_warning(DashboardWarningLevel.ERROR, version_problem, name))
    elif missing_required:
        status = DashboardRunStatus.INCOMPLETE
        warnings.append(
            _warning(
                DashboardWarningLevel.ERROR,
                f"the manifest claims completion but {missing_required} "
                "is/are absent, so this run is shown as incomplete",
                name,
            )
        )
    elif declared_status == "completed":
        status = DashboardRunStatus.COMPLETED
    else:
        status = DashboardRunStatus.UNKNOWN
        warnings.append(
            _warning(
                DashboardWarningLevel.WARNING,
                f"the manifest records status {declared_status!r}, which this "
                "dashboard does not recognise",
                name,
            )
        )

    if signature is not None and signature.configuration_marker is not None:
        key, expected = signature.configuration_marker
        recorded = _configuration(manifest).get(key)
        if recorded is not None and recorded != expected:
            warnings.append(
                _warning(
                    DashboardWarningLevel.ERROR,
                    f"artifact signature says {family.value} but the manifest "
                    f"records configuration.{key}={recorded!r}. The two "
                    "disagree; the signature was used and this conflict is "
                    "not resolved silently.",
                    name,
                )
            )

    if integrity is ArtifactIntegrityStatus.MISMATCHED:
        warnings.append(
            _warning(DashboardWarningLevel.ERROR, str(integrity_message), name)
        )
    elif integrity is ArtifactIntegrityStatus.REFERENCED_FILE_MISSING:
        warnings.append(
            _warning(DashboardWarningLevel.WARNING, str(integrity_message), name)
        )
    elif (
        integrity is ArtifactIntegrityStatus.CHECKSUM_FILE_UNAVAILABLE
        and validate_checksums
    ):
        warnings.append(
            _warning(DashboardWarningLevel.INFORMATION, str(integrity_message), name)
        )

    provenance = DashboardProvenance(
        run_id=run_id,
        run_directory=name,
        family=family,
        status=status,
        data_source=data_source,
        is_synthetic=synthetic,
        scientific_evaluation_eligible=eligible,
        evaluation_mode=(str(evaluation_mode) if evaluation_mode is not None else None),
        target_name=_optional_str(manifest.get("target_name")),
        task_type=_optional_str(manifest.get("task_type")),
        dataset_fingerprint=_optional_str(manifest.get("dataset_fingerprint")),
        split_manifest_fingerprint=_optional_str(
            _configuration(manifest).get("split_manifest_fingerprint")
        ),
        model_source=_model_source(manifest),
        finished_at_utc=_optional_str(manifest.get("finished_at_utc")),
        integrity=integrity,
        failure_reason=(
            str(failure_reason) if status is DashboardRunStatus.FAILED else None
        ),
        warnings=tuple(warnings),
    )
    _family, note = detect_family(directory)
    return DashboardRunSummary(
        directory_name=name,
        absolute_path=str(directory.resolve()),
        provenance=provenance,
        group_field=_optional_str(manifest.get("group_field")),
        group_count=_optional_int(manifest.get("group_count")),
        fold_count=_optional_int(manifest.get("fold_count")),
        evaluated_window_count=None,
        session_count=None,
        artifacts=tuple(artifacts),
        artifact_schema_version=version,
        detection_note=note,
    )


def _adaptation_provenance(
    directory: Path,
    artifacts: Sequence[DashboardArtifactAvailability],
    *,
    integrity: ArtifactIntegrityStatus,
    integrity_message: str | None,
    validate_checksums: bool,
) -> DashboardRunSummary:
    """Build the summary of a run concluded by ``adaptation_summary.json``."""
    name = directory.name
    warnings: list[DashboardWarning] = []
    summary_document = read_json_mapping(directory / ADAPTATION_SUMMARY)

    run_id = str(summary_document.get("run_id") or name)
    synthetic = bool(summary_document.get("is_synthetic", True))
    eligible = bool(summary_document.get("scientific_evaluation_eligible", False))
    missing_required = [a.name for a in artifacts if a.required and not a.present]

    if missing_required:
        status = DashboardRunStatus.INCOMPLETE
        warnings.append(
            _warning(
                DashboardWarningLevel.ERROR,
                f"an adaptation summary is present but {missing_required} "
                "is/are absent, so this run is shown as incomplete",
                name,
            )
        )
    else:
        status = DashboardRunStatus.COMPLETED

    if integrity is ArtifactIntegrityStatus.MISMATCHED:
        warnings.append(
            _warning(DashboardWarningLevel.ERROR, str(integrity_message), name)
        )
    elif integrity is ArtifactIntegrityStatus.REFERENCED_FILE_MISSING:
        warnings.append(
            _warning(DashboardWarningLevel.WARNING, str(integrity_message), name)
        )
    elif (
        integrity is ArtifactIntegrityStatus.CHECKSUM_FILE_UNAVAILABLE
        and validate_checksums
    ):
        warnings.append(
            _warning(DashboardWarningLevel.INFORMATION, str(integrity_message), name)
        )

    configuration = summary_document.get("configuration")
    policy_mode = None
    if isinstance(configuration, dict):
        policy_mode = _optional_str(configuration.get("mode"))
    metrics = summary_document.get("metrics")
    evaluated = None
    if isinstance(metrics, dict):
        evaluated = _optional_int(metrics.get("evaluated_windows"))
    sessions = summary_document.get("session_ids")

    provenance = DashboardProvenance(
        run_id=run_id,
        run_directory=name,
        family=DashboardRunFamily.ADAPTATION,
        status=status,
        data_source=_optional_str(summary_document.get("data_source")),
        is_synthetic=synthetic,
        scientific_evaluation_eligible=eligible,
        evaluation_mode=_optional_str(summary_document.get("evaluation_mode")),
        target_name=None,
        task_type=None,
        dataset_fingerprint=None,
        split_manifest_fingerprint=None,
        model_source=policy_mode,
        finished_at_utc=_optional_str(summary_document.get("finished_at_utc")),
        integrity=integrity,
        failure_reason=None,
        warnings=tuple(warnings),
    )
    _family, note = detect_family(directory)
    return DashboardRunSummary(
        directory_name=name,
        absolute_path=str(directory.resolve()),
        provenance=provenance,
        group_field=None,
        group_count=None,
        fold_count=None,
        evaluated_window_count=evaluated,
        session_count=len(sessions) if isinstance(sessions, list) else None,
        artifacts=tuple(artifacts),
        artifact_schema_version=_optional_str(
            summary_document.get("configuration_fingerprint")
        ),
        detection_note=note,
    )


def _broken_summary(
    directory: Path,
    family: DashboardRunFamily,
    artifacts: Sequence[DashboardArtifactAvailability],
    *,
    status: DashboardRunStatus,
    reason: str,
    level: DashboardWarningLevel,
    detection_note: str | None,
    integrity: ArtifactIntegrityStatus = ArtifactIntegrityStatus.NOT_CHECKED,
) -> DashboardRunSummary:
    """A run the catalogue could not read, kept visible rather than dropped.

    A directory that cannot be parsed is listed as corrupt with an
    actionable reason.  Skipping it would leave the reader with a shorter
    list and no indication that anything was wrong.  The provenance is
    pinned to synthetic-and-ineligible, because an unreadable run has
    established nothing.
    """
    name = directory.name
    provenance = DashboardProvenance(
        run_id=name,
        run_directory=name,
        family=family,
        status=status,
        data_source=None,
        is_synthetic=True,
        scientific_evaluation_eligible=False,
        evaluation_mode=None,
        integrity=integrity,
        failure_reason=None,
        warnings=(_warning(level, reason, name),),
    )
    return DashboardRunSummary(
        directory_name=name,
        absolute_path=str(directory.resolve()),
        provenance=provenance,
        artifacts=tuple(artifacts),
        detection_note=detection_note,
    )


def inspect_run(directory: Path, *, validate_checksums: bool) -> DashboardRunSummary:
    """Classify and summarise one candidate run directory.

    Never raises for a bad run.  Every failure mode becomes a summary
    with an explicit status, because the catalogue's job is to make a bad
    run visible as a bad run.
    """
    family, note = detect_family(directory)
    signature = signature_for(family)
    artifacts = artifact_availability(directory, signature)

    if family is DashboardRunFamily.UNKNOWN:
        return _broken_summary(
            directory,
            family,
            artifacts,
            status=DashboardRunStatus.UNKNOWN,
            reason=(
                "no Milestone 5-8 artifact signature matched this directory. "
                "It may be an unrelated folder, or a run that was interrupted "
                "before it wrote anything identifying."
            ),
            level=DashboardWarningLevel.WARNING,
            detection_note=note,
        )

    integrity, _offenders, integrity_message = verify_integrity(
        directory, validate=validate_checksums
    )

    try:
        if family is DashboardRunFamily.ADAPTATION:
            return _adaptation_provenance(
                directory,
                artifacts,
                integrity=integrity,
                integrity_message=integrity_message,
                validate_checksums=validate_checksums,
            )
        if not (directory / MANIFEST).is_file():
            return _broken_summary(
                directory,
                family,
                artifacts,
                status=DashboardRunStatus.INCOMPLETE,
                reason=(
                    f"this directory holds {family.value} artifacts but no "
                    f"{MANIFEST}. The run was interrupted before it reached a "
                    "conclusion; it is not a successful run."
                ),
                level=DashboardWarningLevel.ERROR,
                detection_note=note,
                integrity=integrity,
            )
        return _training_provenance(
            directory,
            family,
            signature,
            artifacts,
            integrity=integrity,
            integrity_message=integrity_message,
            validate_checksums=validate_checksums,
        )
    except (ArtifactReadError, DashboardError, ValueError) as exc:
        return _broken_summary(
            directory,
            family,
            artifacts,
            status=DashboardRunStatus.CORRUPT,
            reason=(
                f"this {family.value} run could not be read: {exc}. Nothing "
                "from it is displayed, and nothing has been modified."
            ),
            level=DashboardWarningLevel.ERROR,
            detection_note=note,
            integrity=integrity,
        )


def build_catalogue(
    artifact_root: Path, *, validate_checksums: bool = True
) -> DashboardCatalogue:
    """Scan an artifact root and summarise every candidate run.

    Ordering is by directory name, which is deterministic and does not
    depend on the filesystem's iteration order or on modification times.
    A modification time is not provenance.
    """
    root = Path(artifact_root)
    if not root.exists():
        return DashboardCatalogue(
            artifact_root=str(root),
            root_exists=False,
            runs=(),
            warnings=(
                DashboardWarning(
                    level=DashboardWarningLevel.INFORMATION,
                    message=(
                        f"the artifact root {root} does not exist. Generate a "
                        "run with one of the documented CLI commands, or point "
                        "the dashboard at another root."
                    ),
                    subject=str(root),
                ),
            ),
        )
    if not root.is_dir():
        return DashboardCatalogue(
            artifact_root=str(root),
            root_exists=False,
            runs=(),
            warnings=(
                DashboardWarning(
                    level=DashboardWarningLevel.ERROR,
                    message=f"the artifact root {root} is not a directory",
                    subject=str(root),
                ),
            ),
        )

    runs: list[DashboardRunSummary] = []
    warnings: list[DashboardWarning] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        try:
            runs.append(inspect_run(child, validate_checksums=validate_checksums))
        except OSError as exc:  # pragma: no cover - permissions, device errors
            warnings.append(
                DashboardWarning(
                    level=DashboardWarningLevel.ERROR,
                    message=f"{child.name} could not be inspected: {exc}",
                    subject=child.name,
                )
            )
    if not runs:
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.INFORMATION,
                message=(
                    f"no candidate run directory was found under {root}. This "
                    "is a fresh artifact root, not an error."
                ),
                subject=str(root),
            )
        )
    return DashboardCatalogue(
        artifact_root=str(root),
        root_exists=True,
        runs=tuple(runs),
        warnings=tuple(warnings),
    )


def _configuration(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    configuration = manifest.get("configuration")
    return configuration if isinstance(configuration, dict) else {}


def _model_source(manifest: Mapping[str, Any]) -> str | None:
    names = manifest.get("model_names")
    if isinstance(names, list) and names:
        return ", ".join(str(n) for n in names)
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = int(value)
    return number if number >= 0 else None


def _data_source(
    directory: Path, name: str
) -> tuple[str | None, DashboardWarning | None]:
    """Data source recorded in ``dataset.json``, if it is readable.

    Reading it here rather than inferring one keeps the synthetic flag
    tied to what the dataset document actually says.
    """
    path = directory / "dataset.json"
    if not path.is_file():
        return None, None
    try:
        document = read_json_mapping(path)
    except ArtifactReadError as exc:
        return None, DashboardWarning(
            level=DashboardWarningLevel.WARNING,
            message=(
                f"dataset.json could not be read, so the data source is "
                f"unknown for this run: {exc}"
            ),
            subject=name,
        )
    counts = document.get("data_source_counts")
    if not isinstance(counts, dict) or not counts:
        return None, None
    sources = sorted(str(k) for k, v in counts.items() if isinstance(v, int) and v > 0)
    if not sources:
        return None, None
    if len(sources) == 1:
        return sources[0], None
    return "mixed", DashboardWarning(
        level=DashboardWarningLevel.WARNING,
        message=(
            f"this run's dataset mixes data sources {sources}. Provenance is "
            "reported as mixed and the strictest reading applies."
        ),
        subject=name,
    )


__all__ = [
    "ADAPTATION_SUMMARY",
    "CHECKSUMS",
    "FAMILY_SIGNATURES",
    "MANIFEST",
    "SUPPORTED_DATASET_SCHEMA_VERSIONS",
    "SUPPORTED_FEATURE_CATALOG_VERSIONS",
    "ArtifactReadError",
    "FamilySignature",
    "artifact_availability",
    "build_catalogue",
    "detect_family",
    "inspect_run",
    "read_json",
    "read_json_mapping",
    "signature_for",
    "verify_integrity",
]
