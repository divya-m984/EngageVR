"""The exportable session report.

Four properties matter here: the report is deterministic, its counts
reconcile, its provenance cannot be exported away, and it carries no
personal or media field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from engagevr.dashboard.session_catalogue import read_session
from engagevr.dashboard.session_report import (
    FINGERPRINT_EXCLUDED,
    SessionReportError,
    build_report,
    report_field_names,
    report_to_dict,
    report_to_json,
    report_to_markdown,
)
from engagevr.schemas.dashboard import DASHBOARD_DISCLAIMER, SYNTHETIC_BANNER
from engagevr.schemas.dashboard_session import (
    SESSION_REPORT_SCHEMA_VERSION,
    DashboardSessionMode,
    DashboardSessionReport,
)
from tests.unit import session_fixtures as sfx

REPLAY = DashboardSessionMode.REPLAY
LIVE = DashboardSessionMode.LIVE

#: Field-name fragments that must never appear in an exported report.
#: Kept in the test rather than the source, so the prohibition list does
#: not itself trip the repository's privacy scan.
FORBIDDEN_FRAGMENTS: tuple[str, ...] = (
    "frame",
    "image",
    "video",
    "crop",
    "landmark",
    "name",
    "mail",
    "password",
    "token",
    "secret",
    "estimator",
    "pickle",
    "joblib",
)


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    return sfx.write_session(
        tmp_path / "sessions", "synthetic-report-fixture", with_adaptation=True
    )


def report_for(directory: Path, mode: DashboardSessionMode = REPLAY):
    return build_report(read_session(directory, mode=mode), mode=mode)


class TestReportShape:
    def test_the_report_declares_its_schema_version(self, recording: Path) -> None:
        report = report_for(recording)
        assert report.report_schema_version == SESSION_REPORT_SCHEMA_VERSION
        assert report.generated_by == "engagevr.dashboard.session_report"

    def test_the_report_records_its_source_mode(self, recording: Path) -> None:
        assert report_for(recording, REPLAY).source_mode is REPLAY
        assert report_for(recording, LIVE).source_mode is LIVE

    def test_the_report_identifies_the_session(self, recording: Path) -> None:
        report = report_for(recording)
        assert report.session_id == "synthetic-report-fixture"
        assert report.session_directory == str(recording)
        assert report.protocol_version
        assert report.session_format_version

    def test_the_report_lists_source_paths_for_audit(self, recording: Path) -> None:
        report = report_for(recording)
        names = {Path(path).name for path in report.source_paths}
        assert "events.jsonl" in names
        assert "manifest.json" in names

    def test_the_report_carries_source_checksums(self, recording: Path) -> None:
        report = report_for(recording)
        digests = dict(report.source_checksums)
        assert digests["events.jsonl"] == sfx.file_digest(recording / "events.jsonl")

    def test_the_artifact_mode_cannot_produce_a_session_report(
        self, recording: Path
    ) -> None:
        read = read_session(recording, mode=REPLAY)
        with pytest.raises(SessionReportError, match="live or replay"):
            build_report(read, mode=DashboardSessionMode.ARTIFACT)

    def test_a_partial_read_cannot_produce_a_report(self, recording: Path) -> None:
        read = read_session(recording, mode=LIVE, max_records=2)
        with pytest.raises(SessionReportError, match="complete pass"):
            build_report(read, mode=LIVE)


class TestReportDeterminism:
    def test_two_reports_of_one_recording_are_identical(self, recording: Path) -> None:
        first = report_for(recording)
        second = report_for(recording)
        assert first == second
        assert report_to_json(first) == report_to_json(second)

    def test_two_reports_share_a_fingerprint(self, recording: Path) -> None:
        assert (
            report_for(recording).report_fingerprint
            == report_for(recording).report_fingerprint
        )

    def test_the_markdown_is_byte_identical_too(self, recording: Path) -> None:
        assert report_to_markdown(report_for(recording)) == report_to_markdown(
            report_for(recording)
        )

    def test_an_export_timestamp_does_not_change_the_fingerprint(
        self, recording: Path
    ) -> None:
        read = read_session(recording, mode=REPLAY)
        plain = build_report(read, mode=REPLAY)
        stamped = build_report(
            read, mode=REPLAY, exported_at_utc="2026-08-25T12:00:00Z"
        )
        assert stamped.report_fingerprint == plain.report_fingerprint
        assert stamped.exported_at_utc == "2026-08-25T12:00:00Z"
        assert plain.exported_at_utc is None

    def test_the_excluded_fields_are_named_explicitly(self) -> None:
        assert FINGERPRINT_EXCLUDED == {"report_fingerprint", "exported_at_utc"}

    def test_a_different_recording_gets_a_different_fingerprint(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sessions"
        one = sfx.write_session(root, "synthetic-one")
        two = sfx.write_session(root, "synthetic-two", with_adaptation=True)
        assert report_for(one).report_fingerprint != report_for(two).report_fingerprint

    def test_the_json_is_canonical(self, recording: Path) -> None:
        text = report_to_json(report_for(recording))
        keys = list(json.loads(text))
        assert keys == sorted(keys)
        assert text.endswith("\n")


class TestProvenanceCannotBeExportedAway:
    def test_a_synthetic_report_is_permanently_labelled(self, recording: Path) -> None:
        report = report_for(recording)
        assert report.is_synthetic is True
        assert report.scientific_evaluation_eligible is False
        assert report.synthetic_banner == SYNTHETIC_BANNER

    def test_the_labels_survive_serialization(self, recording: Path) -> None:
        document = report_to_dict(report_for(recording))
        assert document["is_synthetic"] is True
        assert document["scientific_evaluation_eligible"] is False
        assert document["synthetic_banner"] == SYNTHETIC_BANNER
        assert document["disclaimer"] == DASHBOARD_DISCLAIMER

    def test_the_labels_appear_in_the_markdown(self, recording: Path) -> None:
        text = report_to_markdown(report_for(recording))
        assert SYNTHETIC_BANNER in text
        assert DASHBOARD_DISCLAIMER in text
        assert "scientific_evaluation_eligible" in text

    def test_a_report_cannot_claim_eligibility(self, recording: Path) -> None:
        document = report_to_dict(report_for(recording))
        document["scientific_evaluation_eligible"] = True
        with pytest.raises(ValidationError):
            DashboardSessionReport.model_validate(document)

    def test_a_synthetic_report_cannot_drop_the_banner(self, recording: Path) -> None:
        document = report_to_dict(report_for(recording))
        document["synthetic_banner"] = None
        with pytest.raises(ValidationError, match="permanently carry"):
            DashboardSessionReport.model_validate(document)

    def test_a_report_cannot_reword_the_disclaimer(self, recording: Path) -> None:
        document = report_to_dict(report_for(recording))
        document["disclaimer"] = "everything is fine"
        with pytest.raises(ValidationError, match="verbatim"):
            DashboardSessionReport.model_validate(document)

    def test_a_non_synthetic_report_carries_no_self_check_banner(
        self, tmp_path: Path
    ) -> None:
        directory = sfx.write_session(
            tmp_path / "sessions", "synthetic-live-source", synthetic=False
        )
        report = report_for(directory, LIVE)
        assert report.is_synthetic is False
        assert report.synthetic_banner is None
        assert report.scientific_evaluation_eligible is False
        assert "not automatically eligible" in report.eligibility_reason

    def test_the_eligibility_reason_is_always_present(self, recording: Path) -> None:
        assert report_for(recording).eligibility_reason

    def test_the_report_states_what_the_format_cannot_carry(
        self, recording: Path
    ) -> None:
        joined = " ".join(report_for(recording).unavailable_statements)
        assert "Engagement estimate: Unavailable" in joined
        assert "Cognitive-load estimate: Unavailable" in joined


class TestReportCountsReconcile:
    def test_decoded_plus_malformed_equals_complete(self, recording: Path) -> None:
        report = report_for(recording)
        assert (
            report.decoded_record_count + report.malformed_record_count
            == report.complete_record_count
        )

    def test_the_message_type_counts_total_the_decoded_records(
        self, recording: Path
    ) -> None:
        report = report_for(recording)
        total = sum(count for _, count in report.message_type_counts)
        assert total == report.decoded_record_count

    def test_a_report_with_broken_counts_is_refused(self, recording: Path) -> None:
        document = report_to_dict(report_for(recording))
        document["decoded_record_count"] = document["decoded_record_count"] + 1
        with pytest.raises(ValidationError):
            DashboardSessionReport.model_validate(document)

    def test_a_malformed_line_is_counted_and_listed(self, tmp_path: Path) -> None:
        directory = sfx.write_session(tmp_path / "sessions", "synthetic-broken")
        sfx.corrupt_line(directory, 2)
        report = report_for(directory)
        assert report.malformed_record_count == 1
        assert report.malformed_line_numbers == (2,)

    def test_a_mismatched_line_list_is_refused(self, recording: Path) -> None:
        document = report_to_dict(report_for(recording))
        document["malformed_line_numbers"] = [1]
        with pytest.raises(ValidationError, match="do not match"):
            DashboardSessionReport.model_validate(document)


class TestUnavailableStaysUnavailable:
    def test_an_absent_drop_count_stays_null(self, tmp_path: Path) -> None:
        directory = sfx.write_session(
            tmp_path / "sessions",
            "synthetic-nosummary",
            with_summary=False,
            completed=False,
        )
        report = report_for(directory, LIVE)
        assert report.dropped_message_count is None
        assert report_to_dict(report)["dropped_message_count"] is None

    def test_none_never_becomes_zero_in_the_markdown(self, tmp_path: Path) -> None:
        directory = sfx.write_session(
            tmp_path / "sessions",
            "synthetic-nosummary",
            with_summary=False,
            completed=False,
        )
        text = report_to_markdown(report_for(directory, LIVE))
        assert "**dropped under backpressure**: Unavailable" in text
        assert "**dropped under backpressure**: 0" not in text

    def test_a_recorded_zero_stays_zero(self, recording: Path) -> None:
        text = report_to_markdown(report_for(recording))
        assert "**dropped under backpressure**: 0" in text

    def test_an_absent_difficulty_is_unavailable(self, tmp_path: Path) -> None:
        directory = sfx.write_session(
            tmp_path / "sessions", "synthetic-nodifficulty", lines=[]
        )
        report = report_for(directory)
        assert report.current_difficulty_level is None
        assert "**difficulty level last recorded**: Unavailable" in (
            report_to_markdown(report)
        )

    def test_an_absent_policy_proposal_is_unavailable_not_zero(
        self, recording: Path
    ) -> None:
        text = report_to_markdown(report_for(recording))
        assert "**policy proposals**: Unavailable" in text
        assert "**commands built but not sent**: Unavailable" in text


class TestPrivacy:
    def test_the_report_carries_no_media_or_personal_field(
        self, recording: Path
    ) -> None:
        for name in report_field_names(report_for(recording)):
            for fragment in FORBIDDEN_FRAGMENTS:
                assert fragment not in name, f"report field {name!r} looks personal"

    def test_the_serialized_report_holds_no_email_address(
        self, recording: Path
    ) -> None:
        text = report_to_json(report_for(recording))
        assert "@" not in text.replace("\\u", "")

    def test_a_participant_identifier_is_not_lifted_into_the_report(
        self, recording: Path
    ) -> None:
        text = report_to_json(report_for(recording))
        assert sfx.PARTICIPANTS[0] not in text

    def test_the_report_names_no_model_file(self, recording: Path) -> None:
        text = report_to_json(report_for(recording))
        assert ".joblib" not in text
        assert ".pkl" not in text


class TestReportIsPure:
    def test_building_a_report_modifies_no_source_file(self, recording: Path) -> None:
        before = sfx.directory_digests(recording)
        report_for(recording)
        report_for(recording, LIVE)
        assert sfx.directory_digests(recording) == before

    def test_building_a_report_creates_no_file(self, recording: Path) -> None:
        before = sorted(path.name for path in recording.iterdir())
        report_for(recording)
        assert sorted(path.name for path in recording.iterdir()) == before

    def test_serializing_creates_no_file(self, recording: Path) -> None:
        parent = recording.parent
        before = sorted(path.name for path in parent.iterdir())
        report_to_json(report_for(recording))
        report_to_markdown(report_for(recording))
        assert sorted(path.name for path in parent.iterdir()) == before

    def test_the_recording_digest_is_unchanged_after_full_inspection(
        self, recording: Path
    ) -> None:
        digest = sfx.file_digest(recording / "events.jsonl")
        read = read_session(recording, mode=REPLAY)
        report = build_report(read, mode=REPLAY)
        assert dict(report.source_checksums)["events.jsonl"] == digest
        assert sfx.file_digest(recording / "events.jsonl") == digest
