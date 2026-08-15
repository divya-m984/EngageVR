"""Dataset validation, leakage checks, and privacy assertions.

Everything here fails loudly.  A dataset that cannot be validated is not
quietly repaired: a silently corrected leak produces a number that looks
fine and means nothing, which is worse than an error.

The leakage checks cover four distinct failure modes, which are often
conflated:

1. **Target leakage** — the answer, or a re-encoding of it, in the
   predictor matrix.
2. **Identifier leakage** — a subject or session id, or a timestamp
   ordering, letting a model recover group membership.
3. **Post-window leakage** — evidence from after the window being
   predicted, including session-completion facts and adaptation outcomes.
4. **Preprocessing leakage** — statistics fitted on data outside the
   training fold.  That one is structural and lives in
   :mod:`engagevr.training.preprocessing`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from engagevr.features.windowing import SessionInterval
from engagevr.schemas.features import (
    AVAILABILITY_PREFIX,
    FEATURE_PREFIX,
    MODALITY_AVAILABLE_PREFIX,
    MODALITY_QUALITY_PREFIX,
    NON_PREDICTOR_COLUMNS,
    FeatureCatalog,
    FeatureWindow,
    is_target_related_column,
)
from engagevr.schemas.targets import TargetName


class DatasetValidationError(ValueError):
    """A dataset violates a structural invariant."""


class LeakageError(ValueError):
    """A predictor matrix would leak the target, a group, or the future."""


#: Column-name fragments that must never appear in a modelling dataset.
#: This is a defence in depth: the row schema has no field that could hold
#: any of them, and this check asserts that no future change introduces one.
FORBIDDEN_COLUMN_TOKENS: tuple[str, ...] = (
    "first_name",
    "last_name",
    "full_name",
    "given_name",
    "surname",
    "email",
    "e_mail",
    "phone",
    "postal",
    "address",
    "date_of_birth",
    "birth_date",
    "national_id",
    "password",
    "passwd",
    "secret",
    "api_key",
    "token",
    "credential",
    "raw_frame",
    "frame_bytes",
    "image",
    "video",
    "pixel_array",
    "landmark",
    "thumbnail",
    "photo",
)

#: Name fragments that indicate evidence from after the predicted window.
#: A feature named for a whole session, a final outcome, or a subsequent
#: adaptation cannot have been observable at prediction time.
POST_WINDOW_TOKENS: tuple[str, ...] = (
    "session_total",
    "session_final",
    "session_complete",
    "session_completed",
    "session_outcome",
    "session_score",
    "final_",
    "_final",
    "future_",
    "next_",
    "subsequent_",
    "post_window",
    "postwindow",
    "after_window",
    "end_of_session",
    "outcome_after",
    "adaptation_applied_after",
    "adaptation_outcome",
    "retrospective",
    "whole_session",
)

_EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class DatasetValidationReport(BaseModel):
    """Outcome of a full dataset audit."""

    model_config = {"frozen": True}

    row_count: int = Field(ge=0)
    session_count: int = Field(ge=0)
    subject_count: int = Field(ge=0)
    windows_overlap: bool
    checks_passed: tuple[str, ...] = ()


def validate_feature_windows(
    rows: Sequence[FeatureWindow],
    catalog: FeatureCatalog,
    *,
    intervals: Mapping[str, SessionInterval] | None = None,
) -> DatasetValidationReport:
    """Validate a set of rows before they become a dataset.

    Raises
    ------
    DatasetValidationError
        On duplicate window ids, reversed or invalid time ranges,
        non-finite feature values, undeclared or missing features, rows
        escaping their source session, or inconsistent window geometry.
    """
    if not rows:
        raise DatasetValidationError("a dataset must contain at least one window")

    known = set(catalog.names())
    seen: dict[str, str] = {}
    overlaps: set[bool] = set()

    for row in rows:
        if row.window_id in seen:
            raise DatasetValidationError(
                f"duplicate window id {row.window_id!r}: it appears in session "
                f"{seen[row.window_id]!r} and again in {row.session_id!r}"
            )
        seen[row.window_id] = row.session_id

        if row.window_end_utc <= row.window_start_utc:
            raise DatasetValidationError(
                f"window {row.window_id!r} has a reversed or zero-length UTC range"
            )
        start = row.window_start_monotonic_seconds
        end = row.window_end_monotonic_seconds
        if start is not None and end is not None and end <= start:
            raise DatasetValidationError(
                f"window {row.window_id!r} has a reversed or zero-length "
                "monotonic range"
            )

        declared = set(row.features) | set(row.categorical_features)
        undeclared = declared - known
        if undeclared:
            raise DatasetValidationError(
                f"window {row.window_id!r} carries features that are not in "
                f"feature catalog {catalog.version}: {sorted(undeclared)}"
            )
        absent = known - declared
        if absent:
            raise DatasetValidationError(
                f"window {row.window_id!r} is missing catalog features: "
                f"{sorted(absent)}"
            )

        for name, value in row.features.items():
            if value is not None and not math.isfinite(value):
                raise DatasetValidationError(
                    f"window {row.window_id!r}: feature {name!r} is {value!r}. "
                    "A non-finite value is not a measurement; report it as "
                    "unavailable instead."
                )

        overlaps.add(row.windows_overlap)

        if intervals is not None:
            interval = intervals.get(row.session_id)
            if interval is None:
                raise DatasetValidationError(
                    f"window {row.window_id!r} references session "
                    f"{row.session_id!r}, which has no recorded interval"
                )
            if row.window_start_utc < interval.start_utc:
                raise DatasetValidationError(
                    f"window {row.window_id!r} starts before session "
                    f"{row.session_id!r} began"
                )
            if row.window_end_utc > interval.end_utc:
                raise DatasetValidationError(
                    f"window {row.window_id!r} ends after session "
                    f"{row.session_id!r} ended; it would summarise a period "
                    "for which the session holds no evidence"
                )

    if len(overlaps) > 1:
        raise DatasetValidationError(
            "rows disagree about whether windows overlap; one dataset must "
            "use one window geometry"
        )

    return DatasetValidationReport(
        row_count=len(rows),
        session_count=len({row.session_id for row in rows}),
        subject_count=len({row.subject_id for row in rows}),
        windows_overlap=next(iter(overlaps)),
        checks_passed=(
            "unique_window_ids",
            "forward_time_ranges",
            "finite_feature_values",
            "catalog_conformance",
            "session_containment" if intervals is not None else "geometry_consistency",
        ),
    )


def assert_no_identity_columns(columns: Iterable[str]) -> None:
    """Assert no column name suggests personal or raw-media content.

    Raises
    ------
    DatasetValidationError
        On the first offending column.
    """
    for column in columns:
        lowered = column.lower()
        for token in FORBIDDEN_COLUMN_TOKENS:
            if token in lowered:
                raise DatasetValidationError(
                    f"column {column!r} contains the forbidden fragment "
                    f"{token!r}. A modelling dataset must not carry personal "
                    "identifiers or raw media."
                )


def assert_no_identifier_values(values: Iterable[object]) -> None:
    """Assert no cell value looks like an email address.

    Raises
    ------
    DatasetValidationError
        On the first value matching an email pattern.
    """
    for value in values:
        if isinstance(value, str) and _EMAIL_PATTERN.search(value):
            raise DatasetValidationError(
                "a dataset value looks like an email address; participants are "
                "identified by pseudonymous ids only"
            )


def is_post_window_column(column: str) -> bool:
    """Whether ``column``'s name indicates evidence from after the window."""
    lowered = column.lower()
    return any(token in lowered for token in POST_WINDOW_TOKENS)


def select_predictor_columns(
    columns: Sequence[str],
    catalog: FeatureCatalog,
) -> tuple[str, ...]:
    """Predictor columns available in ``columns``, in catalog order.

    A feature is a predictor only if it is in the catalog **and** the
    catalog permits it.  Availability flags and modality-quality columns
    are included: whether a measurement succeeded, and how good it was,
    are legitimate inputs and are kept structurally distinct from the
    measurement itself.
    """
    present = set(columns)
    selected: list[str] = []
    for entry in catalog.entries:
        if not entry.permitted_predictor:
            continue
        column = f"{FEATURE_PREFIX}{entry.canonical_name}"
        if column in present:
            selected.append(column)
        availability = f"{AVAILABILITY_PREFIX}{entry.canonical_name}"
        if availability in present:
            selected.append(availability)
    for modality in catalog.modalities():
        for prefix in (MODALITY_AVAILABLE_PREFIX, MODALITY_QUALITY_PREFIX):
            column = f"{prefix}{modality.value}"
            if column in present:
                selected.append(column)
    return tuple(selected)


def assert_no_leakage(
    predictor_columns: Sequence[str],
    catalog: FeatureCatalog,
    *,
    target_name: TargetName | str | None = None,
) -> None:
    """Assert a predictor matrix leaks neither the target, a group, nor the future.

    Raises
    ------
    LeakageError
        On the first offending column, naming which of the four leakage
        modes it triggers.
    """
    known_features = set(catalog.names())
    permitted = set(catalog.predictor_names())
    target_value = (
        target_name.value if isinstance(target_name, TargetName) else target_name
    )

    for column in predictor_columns:
        if is_target_related_column(column):
            raise LeakageError(
                f"predictor column {column!r} carries the target or its "
                "provenance. The answer cannot be an input."
            )
        if column in NON_PREDICTOR_COLUMNS:
            raise LeakageError(
                f"predictor column {column!r} is an identifier, timestamp, or "
                "schema column. Including it lets a model recover group "
                "membership or session ordering instead of learning a signal."
            )
        if is_post_window_column(column):
            raise LeakageError(
                f"predictor column {column!r} names evidence from after the "
                "predicted window. Session-completion facts and adaptation "
                "outcomes were not observable at prediction time."
            )
        if column.startswith(FEATURE_PREFIX):
            name = column[len(FEATURE_PREFIX) :]
            if name not in known_features:
                raise LeakageError(
                    f"predictor column {column!r} is not declared in feature "
                    f"catalog {catalog.version}; undeclared columns cannot be "
                    "audited and are refused"
                )
            if name not in permitted:
                raise LeakageError(
                    f"predictor column {column!r} names feature {name!r}, which "
                    "the catalog marks as not permitted as a predictor"
                )
        elif column.startswith(AVAILABILITY_PREFIX):
            name = column[len(AVAILABILITY_PREFIX) :]
            if name not in known_features:
                raise LeakageError(
                    f"availability column {column!r} refers to feature {name!r}, "
                    f"which is not in feature catalog {catalog.version}"
                )
        elif not column.startswith(
            (MODALITY_AVAILABLE_PREFIX, MODALITY_QUALITY_PREFIX)
        ):
            raise LeakageError(
                f"predictor column {column!r} does not follow any recognised "
                "dataset column convention and cannot be audited"
            )

    if target_value is not None:
        forbidden = f"target__{target_value}"
        if forbidden in set(predictor_columns):  # pragma: no cover - caught above
            raise LeakageError(
                f"predictor column {forbidden!r} is the evaluated target"
            )
