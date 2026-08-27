"""The exportable session report.

A **presentation artifact**, not an experiment result.  It restates what
one recording contains, with the provenance that recording carried, in a
form a researcher can attach to a lab notebook and a reviewer can audit
later.

Purity
------
:func:`build_report` is a pure function of an already-completed read.  It
opens nothing, writes nothing, and derives nothing that is not already in
the recording.  The dashboard hands the returned text to Streamlit's
download control; the source recording is untouched, and a digest of
every source file travels with the report so that can be checked
afterwards rather than taken on trust.

Reproducibility
---------------
The report's identity is its content.  :data:`FINGERPRINT_EXCLUDED`
names the two fields that take no part in it — the fingerprint itself
and ``exported_at_utc`` — so the same recording inspected twice a week
apart yields byte-identical JSON and the same fingerprint.  Wall-clock
time is never part of a report's identity; it would make every report
unique and every comparison useless.

Provenance cannot be exported away
----------------------------------
There is no "clean" variant of this report.  ``is_synthetic``,
``scientific_evaluation_eligible``, the eligibility reason, the standing
disclaimer, and — for a synthetic recording — the software-self-check
banner are required fields, and
:class:`~engagevr.schemas.dashboard_session.DashboardSessionReport`
refuses to be constructed without them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from engagevr.dashboard.session_catalogue import SessionRead
from engagevr.dashboard.session_reader import file_checksums, session_paths
from engagevr.schemas.dashboard import (
    DASHBOARD_DISCLAIMER,
    SYNTHETIC_BANNER,
    DashboardError,
)
from engagevr.schemas.dashboard_session import (
    SESSION_CONTENT_NOTE,
    SESSION_REPORT_SCHEMA_VERSION,
    DashboardSessionMode,
    DashboardSessionReport,
)

#: Fields deliberately outside the report's logical identity.
FINGERPRINT_EXCLUDED: frozenset[str] = frozenset(
    {"report_fingerprint", "exported_at_utc"}
)


class SessionReportError(DashboardError):
    """A session report could not be built from the given read."""


def build_report(
    read: SessionRead,
    *,
    mode: DashboardSessionMode,
    exported_at_utc: str | None = None,
) -> DashboardSessionReport:
    """Build the report for one fully-read recording.

    ``read`` must be a complete pass: every complete record decoded or
    accounted for as malformed.  A partial read is refused rather than
    reported, because a count taken from half a file reads exactly like a
    count taken from all of it.

    ``exported_at_utc`` is optional display metadata.  Passing it changes
    the serialized document but not the fingerprint, so two reports of
    one recording remain comparable.
    """
    summary = read.summary
    if not summary.fully_parsed:
        raise SessionReportError(
            f"session {summary.provenance.session_id!r} was read from line "
            f"{summary.parse_start_line} with {summary.unparsed_record_count} "
            "record(s) left unread. A report is built from a complete pass "
            "only; counts taken from part of a recording read exactly like "
            "counts taken from all of it."
        )
    if mode is DashboardSessionMode.ARTIFACT:
        raise SessionReportError(
            "a session report records a live or replay observation of a "
            "recording; experiment runs are reported by the artifact views"
        )

    provenance = summary.provenance
    synthetic = provenance.is_synthetic
    fields: dict[str, Any] = {
        "report_schema_version": SESSION_REPORT_SCHEMA_VERSION,
        "source_mode": mode,
        "session_id": provenance.session_id,
        "session_directory": provenance.session_directory,
        "session_format_version": provenance.session_format_version,
        "protocol_version": provenance.protocol_version,
        "engagevr_version": provenance.engagevr_version,
        "data_sources": provenance.data_sources,
        "is_synthetic": synthetic,
        "scientific_evaluation_eligible": False,
        "eligibility_reason": provenance.eligibility_reason,
        "synthetic_message_count": provenance.synthetic_message_count,
        "non_synthetic_message_count": provenance.non_synthetic_message_count,
        "replayed_message_count": provenance.replayed_message_count,
        "disclaimer": DASHBOARD_DISCLAIMER,
        "synthetic_banner": SYNTHETIC_BANNER if synthetic else None,
        "content_note": SESSION_CONTENT_NOTE,
        "status": provenance.status,
        "status_reason": provenance.status_reason,
        "session_end_recorded": summary.session_end_recorded,
        "disconnect_reason": summary.disconnect_reason,
        "summary_recovered": summary.summary_recovered,
        "created_at_utc": provenance.created_at_utc,
        "first_sent_at_utc": summary.first_sent_at_utc,
        "last_sent_at_utc": summary.last_sent_at_utc,
        "first_received_at_utc": summary.first_received_at_utc,
        "last_received_at_utc": summary.last_received_at_utc,
        "complete_record_count": summary.complete_record_count,
        "decoded_record_count": summary.decoded_record_count,
        "malformed_record_count": summary.malformed_record_count,
        "malformed_line_numbers": summary.malformed_line_numbers,
        "partial_trailing_line": summary.partial_trailing_line,
        "dropped_message_count": summary.dropped_message_count,
        "message_type_counts": summary.message_type_counts,
        "source_counts": summary.source_counts,
        "recorded_anomaly_counts": summary.recorded_anomaly_counts,
        "sequence_observation_count": len(summary.sequence_observations),
        "sequence_observation_reasons": tuple(
            f"{observation.kind} at line {observation.line_number} "
            f"(source {observation.source})"
            for observation in summary.sequence_observations
        ),
        "task_state": summary.task_state,
        "current_difficulty_level": summary.current_difficulty_level,
        "adaptation": summary.adaptation,
        "unavailable_statements": summary.unavailable_statements,
        "source_paths": tuple(
            str(path) for path in session_paths(read.directory) if path.is_file()
        ),
        "source_checksums": file_checksums(read.directory),
    }
    fingerprint = _fingerprint(fields)
    return DashboardSessionReport(
        **fields, report_fingerprint=fingerprint, exported_at_utc=exported_at_utc
    )


def _fingerprint(fields: dict[str, Any]) -> str:
    """SHA-256 over the canonical content, excluding the non-identity fields."""
    payload = {
        name: _canonical(value)
        for name, value in sorted(fields.items())
        if name not in FINGERPRINT_EXCLUDED
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    """Turn a report field into something JSON can encode deterministically."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        return [_canonical(entry) for entry in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def report_to_dict(report: DashboardSessionReport) -> dict[str, Any]:
    """The report as a plain JSON-compatible mapping."""
    return report.model_dump(mode="json")


def report_to_json(report: DashboardSessionReport) -> str:
    """The report as canonical JSON text.

    Keys are sorted and the indentation is fixed, so two reports of one
    recording compare byte for byte rather than merely field for field.
    """
    return json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n"


def report_to_markdown(report: DashboardSessionReport) -> str:
    """The report as readable Markdown, with the same content.

    Generated from the same fields as the JSON so the two cannot drift,
    and produced with the standard library alone: no template engine, no
    HTML toolkit, and certainly no PDF dependency.
    """
    lines: list[str] = [
        f"# EngageVR session report: {report.session_id}",
        "",
        f"> {report.disclaimer}",
        "",
    ]
    if report.synthetic_banner:
        lines += [f"> **{report.synthetic_banner}**", ""]
    lines += [
        f"> {report.eligibility_reason}",
        "",
        f"> {report.content_note}",
        "",
        "## Identity and provenance",
        "",
        *_rows(
            (
                ("report schema version", report.report_schema_version),
                ("generated by", report.generated_by),
                ("source mode", report.source_mode.value),
                ("session id", report.session_id),
                ("session directory", report.session_directory),
                ("session format version", report.session_format_version),
                ("protocol version", report.protocol_version),
                ("engagevr version", report.engagevr_version),
                ("recorded data sources", ", ".join(report.data_sources) or None),
                ("is_synthetic", report.is_synthetic),
                (
                    "scientific_evaluation_eligible",
                    report.scientific_evaluation_eligible,
                ),
                ("synthetic records", report.synthetic_message_count),
                ("non-synthetic records", report.non_synthetic_message_count),
                ("already-replayed records", report.replayed_message_count),
                ("report fingerprint", report.report_fingerprint),
            )
        ),
        "",
        "## Session state",
        "",
        *_rows(
            (
                ("status", report.status.value),
                ("status reason", report.status_reason),
                ("session_end recorded", report.session_end_recorded),
                ("disconnect reason", report.disconnect_reason),
                ("summary rebuilt from a partial stream", report.summary_recovered),
                ("created (recorder clock)", report.created_at_utc),
                ("first sent (sender clock)", report.first_sent_at_utc),
                ("last sent (sender clock)", report.last_sent_at_utc),
                ("first received (receiver clock)", report.first_received_at_utc),
                ("last received (receiver clock)", report.last_received_at_utc),
                ("task state last recorded", report.task_state),
                ("difficulty level last recorded", report.current_difficulty_level),
            )
        ),
        "",
        "## Records",
        "",
        *_rows(
            (
                ("complete records", report.complete_record_count),
                ("decoded records", report.decoded_record_count),
                ("malformed records", report.malformed_record_count),
                (
                    "malformed line numbers",
                    ", ".join(str(n) for n in report.malformed_line_numbers) or None,
                ),
                ("partial trailing line", report.partial_trailing_line),
                ("dropped under backpressure", report.dropped_message_count),
            )
        ),
        "",
        "### Message types",
        "",
        *_rows(report.message_type_counts),
        "",
        "### Sources",
        "",
        *_rows(report.source_counts),
        "",
        "## Ordering anomalies",
        "",
        *_rows(report.recorded_anomaly_counts),
        "",
        f"Derived sequence observations: {report.sequence_observation_count}",
        "",
        *(f"- {reason}" for reason in report.sequence_observation_reasons),
        "",
        "## Adaptation messages",
        "",
        *_rows(
            (
                ("commands recorded", report.adaptation.commands_recorded),
                (
                    "acknowledgements recorded",
                    report.adaptation.acknowledgements_recorded,
                ),
                ("acknowledgements accepting", report.adaptation.accepted_recorded),
                ("acknowledgements rejecting", report.adaptation.rejected_recorded),
                (
                    "acknowledgements with an applied timestamp",
                    report.adaptation.applied_timestamp_recorded,
                ),
                ("policy proposals", None),
                ("commands built but not sent", None),
            )
        ),
        "",
        report.adaptation.note,
        "",
        "## Not present in a session recording",
        "",
        *(f"- {statement}" for statement in report.unavailable_statements),
        "",
        "## Audit",
        "",
        *(f"- source: `{path}`" for path in report.source_paths),
        "",
        *(f"- sha256 `{name}`: `{digest}`" for name, digest in report.source_checksums),
        "",
    ]
    if report.exported_at_utc:
        lines += [
            f"Exported at {report.exported_at_utc} (display metadata only; "
            "not part of this report's identity or fingerprint).",
            "",
        ]
    return "\n".join(lines)


def _rows(pairs: tuple[tuple[str, Any], ...]) -> list[str]:
    """Markdown definition rows, with absence rendered as *Unavailable*."""
    rendered: list[str] = []
    for name, value in pairs:
        if value is None or value == "":
            text = "Unavailable"
        elif isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value)
        rendered.append(f"- **{name}**: {text}")
    return rendered


def report_field_names(report: DashboardSessionReport) -> tuple[str, ...]:
    """Every key in the serialized report, for a privacy scan."""
    return tuple(sorted(report_to_dict(report)))


__all__ = [
    "FINGERPRINT_EXCLUDED",
    "SessionReportError",
    "build_report",
    "report_field_names",
    "report_to_dict",
    "report_to_json",
    "report_to_markdown",
]
