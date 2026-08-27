"""Replay navigation, mode distinction, and the session view models.

Replay here is *presentation*: a cursor over records already on disk.
The tests below check that stepping cannot leave the recording, that
navigating produces no new record, and that the two session modes cannot
be mistaken for one another.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from engagevr.dashboard import views_session as views
from engagevr.dashboard.session_catalogue import build_session_catalogue, read_session
from engagevr.schemas.dashboard_session import (
    LIVE_OBSERVATION_NOTE,
    REPLAY_PRESENTATION_NOTE,
    DashboardReplayState,
    DashboardSessionMode,
    DashboardSessionProvenance,
    DashboardSessionStatus,
    SessionAdaptationCounts,
)
from tests.unit import session_fixtures as sfx

REPLAY = DashboardSessionMode.REPLAY
LIVE = DashboardSessionMode.LIVE


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    root = tmp_path / "sessions"
    return sfx.write_session(root, "synthetic-replay-fixture", with_adaptation=True)


class TestReplayStateNavigation:
    def test_a_new_state_starts_at_the_first_position(self) -> None:
        state = DashboardReplayState(total=5)
        assert state.position == 0
        assert state.at_first is True
        assert state.at_last is False
        assert state.human_position == "1 of 5"

    def test_step_forward_advances_by_one(self) -> None:
        state = DashboardReplayState(total=5).step_forward()
        assert state.position == 1

    def test_step_backward_returns_by_one(self) -> None:
        state = DashboardReplayState(total=5, position=3).step_backward()
        assert state.position == 2

    def test_last_jumps_to_the_final_record(self) -> None:
        state = DashboardReplayState(total=5).last()
        assert state.position == 4
        assert state.at_last is True

    def test_first_jumps_to_the_opening_record(self) -> None:
        state = DashboardReplayState(total=5, position=4).first()
        assert state.position == 0

    def test_stepping_past_the_end_clamps(self) -> None:
        state = DashboardReplayState(total=3, position=2).step_forward()
        assert state.position == 2

    def test_stepping_before_the_start_clamps(self) -> None:
        state = DashboardReplayState(total=3, position=0).step_backward()
        assert state.position == 0

    def test_a_large_step_clamps_at_both_ends(self) -> None:
        state = DashboardReplayState(total=4)
        assert state.step_forward(100).position == 3
        assert state.last().step_backward(100).position == 0

    def test_an_empty_recording_has_one_position(self) -> None:
        state = DashboardReplayState(total=0)
        assert state.is_empty is True
        assert state.at_last is True
        assert state.human_position == "0 of 0"
        assert state.first().position == 0
        assert state.last().position == 0
        assert state.step_forward().position == 0

    def test_a_position_outside_the_recording_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            DashboardReplayState(total=3, position=3)

    def test_an_empty_recording_refuses_a_non_zero_position(self) -> None:
        with pytest.raises(ValidationError):
            DashboardReplayState(total=0, position=1)

    def test_navigation_returns_a_new_state_and_leaves_the_old_one(self) -> None:
        original = DashboardReplayState(total=5, position=2)
        moved = original.step_forward()
        assert original.position == 2
        assert moved.position == 3
        assert moved is not original

    def test_resizing_keeps_the_place_where_possible(self) -> None:
        state = DashboardReplayState(total=10, position=7)
        assert state.resized(10).position == 7
        assert state.resized(5).position == 4
        assert state.resized(0).is_empty is True


class TestReplayShowsOnlyRecordedEvidence:
    def test_the_cursor_is_sized_to_the_records_read(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        state = views.replay_state_for(read)
        assert state.total == len(read.records)

    def test_navigating_produces_no_new_record(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        before = read.records
        state = views.replay_state_for(read)
        for _ in range(len(before) + 5):
            state = state.step_forward()
        assert read.records == before
        assert state.position == len(before) - 1

    def test_navigating_modifies_no_source_file(self, recording: Path) -> None:
        before = sfx.directory_digests(recording)
        read = read_session(recording, mode=REPLAY)
        state = views.replay_state_for(read)
        state.last().first().step_forward().step_backward()
        assert sfx.directory_digests(recording) == before

    def test_an_incomplete_recording_can_still_be_replayed(
        self, tmp_path: Path
    ) -> None:
        directory = sfx.write_session(
            tmp_path / "sessions",
            "synthetic-incomplete",
            with_summary=False,
            completed=False,
        )
        read = read_session(directory, mode=REPLAY)
        state = views.replay_state_for(read)
        assert state.total > 0
        assert read.summary.provenance.status is (
            DashboardSessionStatus.ACTIVE_OR_INCOMPLETE
        )

    def test_replaying_preserves_the_synthetic_label(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        assert all(record.is_synthetic for record in read.records if record.decoded)
        assert read.summary.provenance.is_synthetic is True

    def test_an_already_replayed_record_says_where_it_came_from(
        self, tmp_path: Path
    ) -> None:
        directory = sfx.write_session(
            tmp_path / "sessions", "synthetic-nested-replay", replayed=True
        )
        read = read_session(directory, mode=REPLAY)
        record = read.records[0]
        assert record.is_replayed is True
        assert record.replay_source_session_id == "synthetic-nested-replay-origin"


class TestModeDistinction:
    def test_the_three_modes_are_distinct_values(self) -> None:
        assert len({m.value for m in DashboardSessionMode}) == 3

    def test_each_mode_states_its_own_evidence_source(self) -> None:
        statements = {mode: views.mode_statement(mode) for mode in DashboardSessionMode}
        assert len(set(statements.values())) == 3
        assert "experiment artifacts" in statements[DashboardSessionMode.ARTIFACT]
        assert "appended to" in statements[LIVE]
        assert "not a re-emission" in statements[REPLAY]

    def test_a_session_provenance_cannot_claim_the_artifact_mode(self) -> None:
        with pytest.raises(ValidationError):
            DashboardSessionProvenance(
                session_id="synthetic-session",
                directory_name="synthetic-session",
                session_directory="/tmp/synthetic-session",
                mode=DashboardSessionMode.ARTIFACT,
                status=DashboardSessionStatus.COMPLETED,
            )

    def test_a_live_read_is_not_labelled_replay(self, recording: Path) -> None:
        read = read_session(recording, mode=LIVE)
        assert read.summary.provenance.mode is LIVE
        assert read.summary.provenance.mode is not REPLAY

    def test_a_replay_read_is_not_labelled_live(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        assert read.summary.provenance.mode is REPLAY
        assert read.summary.provenance.mode is not LIVE

    def test_the_two_mode_notes_say_different_things(self) -> None:
        assert "LIVE OBSERVATION" in LIVE_OBSERVATION_NOTE
        assert "MODE: REPLAY" in REPLAY_PRESENTATION_NOTE
        assert "runs no model" in LIVE_OBSERVATION_NOTE
        assert "Nothing is re-emitted" in REPLAY_PRESENTATION_NOTE


class TestSessionProvenanceRules:
    def test_a_session_can_never_be_marked_eligible(self) -> None:
        with pytest.raises(ValidationError):
            DashboardSessionProvenance(
                session_id="synthetic-session",
                directory_name="synthetic-session",
                session_directory="/tmp/synthetic-session",
                mode=REPLAY,
                status=DashboardSessionStatus.COMPLETED,
                scientific_evaluation_eligible=True,
            )

    def test_a_failure_status_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError):
            DashboardSessionProvenance(
                session_id="synthetic-session",
                directory_name="synthetic-session",
                session_directory="/tmp/synthetic-session",
                mode=REPLAY,
                status=DashboardSessionStatus.FAILED,
            )

    def test_the_synthetic_banner_follows_the_record_composition(self) -> None:
        provenance = DashboardSessionProvenance(
            session_id="synthetic-session",
            directory_name="synthetic-session",
            session_directory="/tmp/synthetic-session",
            mode=REPLAY,
            status=DashboardSessionStatus.COMPLETED,
            synthetic_message_count=3,
            data_sources=("synthetic",),
        )
        assert provenance.is_synthetic is True
        assert len(provenance.banners) == 2


class TestSessionAdaptationCounts:
    def test_a_default_is_all_zero(self) -> None:
        counts = SessionAdaptationCounts()
        assert counts.commands_recorded == 0
        assert counts.acknowledgements_recorded == 0

    def test_more_acknowledgements_than_commands_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            SessionAdaptationCounts(commands_recorded=1, acknowledgements_recorded=2)

    def test_more_outcomes_than_acknowledgements_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            SessionAdaptationCounts(
                commands_recorded=2,
                acknowledgements_recorded=1,
                accepted_recorded=1,
                rejected_recorded=1,
            )

    def test_an_applied_timestamp_requires_an_acceptance(self) -> None:
        with pytest.raises(ValidationError):
            SessionAdaptationCounts(
                commands_recorded=1,
                acknowledgements_recorded=1,
                accepted_recorded=0,
                applied_timestamp_recorded=1,
            )

    def test_the_note_denies_a_policy_proposal(self) -> None:
        note = SessionAdaptationCounts().note
        assert "records no policy proposal" in note

    def test_a_recorded_command_is_counted(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        counts = read.summary.adaptation
        assert counts.commands_recorded == 1
        assert counts.acknowledgements_recorded == 1
        assert counts.accepted_recorded == 1


class TestSessionViewModels:
    def test_the_record_table_keeps_file_order(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        table = views.record_table(read.records, title="Records", max_rows=100)
        lines = [int(row[0]) for row in table.rows]
        assert lines == sorted(lines)

    def test_the_record_table_names_its_source(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        table = views.record_table(read.records, title="Records", max_rows=100)
        assert table.source_artifact == "events.jsonl"
        assert "never re-sorted" in (table.caption or "")

    def test_the_record_table_states_a_display_limit(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        table = views.record_table(read.records, title="Records", max_rows=2)
        assert table.truncated_row_count == len(read.records) - 2

    def test_an_absent_drop_count_is_unavailable_not_zero(self, tmp_path: Path) -> None:
        directory = sfx.write_session(
            tmp_path / "sessions",
            "synthetic-nosummary",
            with_summary=False,
            completed=False,
        )
        read = read_session(directory, mode=LIVE)
        dropped = views.session_metrics(read.summary)[3]
        assert dropped.value is None
        assert dropped.unavailable_reason

    def test_a_recorded_drop_count_of_zero_stays_zero(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        dropped = views.session_metrics(read.summary)[3]
        assert dropped.value == 0.0

    def test_the_provenance_table_states_the_eligibility_reason(
        self, recording: Path
    ) -> None:
        read = read_session(recording, mode=REPLAY)
        table = views.provenance_table(read.summary)
        cells = {row[0]: row[1] for row in table.rows}
        assert cells["scientifically eligible"] == "No"
        assert "not automatically eligible" in cells["eligibility reason"]

    def test_the_anomaly_table_separates_recorded_from_derived(
        self, tmp_path: Path
    ) -> None:
        session_id = "synthetic-mixed-anomaly"
        lines = sfx.default_lines(session_id)
        lines.append(lines[-1])
        directory = sfx.write_session(tmp_path / "sessions", session_id, lines=lines)
        read = read_session(directory, mode=REPLAY)
        table = views.anomaly_table(read.summary)
        origins = {row[0] for row in table.rows}
        assert "derived from recorded sequence numbers" in origins
        assert "never corrected" in (table.caption or "")

    def test_the_adaptation_table_has_no_effectiveness_row(
        self, recording: Path
    ) -> None:
        read = read_session(recording, mode=REPLAY)
        table = views.adaptation_table(read.summary)
        text = " ".join(cell for row in table.rows for cell in row).lower()
        assert "effectiveness" not in text
        assert "benefit" not in text
        assert "effectiveness" in (table.caption or "").lower()

    def test_the_adaptation_table_states_what_is_not_recorded(
        self, recording: Path
    ) -> None:
        read = read_session(recording, mode=REPLAY)
        cells = {row[0]: row[1] for row in views.adaptation_table(read.summary).rows}
        assert cells["policy proposals"].startswith("Unavailable")
        assert cells["commands built but not sent"].startswith("Unavailable")

    def test_the_unavailable_table_never_renders_a_zero(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        table = views.unavailable_table(read.summary)
        for row in table.rows:
            assert row[1] != "0"
            assert "Unavailable" in row[1]

    def test_the_catalogue_table_marks_every_session_ineligible(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sessions"
        sfx.write_session(root, "synthetic-one")
        sfx.write_session(root, "synthetic-two")
        catalogue = build_session_catalogue(root, mode=REPLAY)
        table = views.catalogue_table(catalogue, max_rows=100)
        eligible_column = table.columns.index("scientifically eligible")
        assert {row[eligible_column] for row in table.rows} == {"No"}

    def test_the_status_statement_carries_a_stated_reason(self, tmp_path: Path) -> None:
        directory = sfx.write_session(
            tmp_path / "sessions", "synthetic-broken", with_manifest=False
        )
        read = read_session(directory, mode=REPLAY)
        statement = views.status_statement(read.summary)
        assert "UNREADABLE" in statement
        assert "manifest.json" in statement

    def test_an_active_session_is_not_described_as_a_failure(self) -> None:
        text = views.SESSION_STATUS_TEXT[DashboardSessionStatus.ACTIVE_OR_INCOMPLETE]
        assert "neither is a failure" in text
        assert "FAILED" not in text

    def test_the_quality_note_keeps_the_separation(self) -> None:
        assert "no engagement or cognitive-load value" in views.NO_ESTIMATOR_NOTE
        assert "not" in views.TASK_EVENT_NOTE
        assert "engagement" in views.TASK_EVENT_NOTE

    def test_the_record_detail_lists_the_payload_fields(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        start = next(
            record for record in read.records if record.message_type == "session_start"
        )
        table = views.record_detail_table(start)
        keys = {row[0] for row in table.rows}
        assert "payload.participant_id" in keys
        assert "payload.difficulty_level" in keys

    def test_the_timing_table_keeps_the_clocks_apart(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        table = views.timing_table(read.summary)
        assert "never subtracted" in (table.caption or "")
        labels = {row[0] for row in table.rows}
        assert "first sent (sender clock)" in labels
        assert "first received (receiver clock)" in labels


class TestSummaryReconciliation:
    def test_decoded_plus_malformed_equals_parsed(self, tmp_path: Path) -> None:
        directory = sfx.write_session(tmp_path / "sessions", "synthetic-reconcile")
        sfx.corrupt_line(directory, 2)
        read = read_session(directory, mode=REPLAY)
        summary = read.summary
        assert (
            summary.decoded_record_count + summary.malformed_record_count
            == summary.parsed_record_count
        )

    def test_a_summary_that_does_not_reconcile_is_refused(
        self, recording: Path
    ) -> None:
        read = read_session(recording, mode=REPLAY)
        document = read.summary.model_dump()
        document["decoded_record_count"] = document["decoded_record_count"] - 1
        with pytest.raises(ValidationError, match="exactly one of the two"):
            type(read.summary).model_validate(document)

    def test_a_pass_cannot_account_for_more_than_the_file_holds(
        self, recording: Path
    ) -> None:
        read = read_session(recording, mode=REPLAY)
        document = read.summary.model_dump()
        document["parse_start_line"] = document["complete_record_count"]
        with pytest.raises(ValidationError, match="complete records exist"):
            type(read.summary).model_validate(document)

    def test_a_partial_trailing_line_must_be_explained(self, recording: Path) -> None:
        read = read_session(recording, mode=REPLAY)
        document = read.summary.model_dump()
        document["partial_trailing_line"] = True
        document["partial_trailing_note"] = None
        with pytest.raises(ValidationError, match="not corruption"):
            type(read.summary).model_validate(document)

    def test_fully_parsed_is_true_for_a_whole_pass(self, recording: Path) -> None:
        assert read_session(recording, mode=REPLAY).summary.fully_parsed is True


class TestRecordingsAreAddressedByDirectory:
    """A recording is addressed by its folder, never by its recorded id.

    Copying a recording for comparison keeps its recorded ``session_id``
    under a new folder name. Resolving the id would then read the wrong
    file, or none at all, and the page would render an empty session with
    no synthetic banner — which is exactly what it must never do.
    """

    def test_a_copied_recording_keeps_its_recorded_identifier(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sessions"
        original = sfx.write_session(root, "synthetic-original")
        copy = root / "synthetic-copy"
        copy.mkdir()
        for path in original.iterdir():
            (copy / path.name).write_bytes(path.read_bytes())

        read = read_session(copy, mode=REPLAY)
        provenance = read.summary.provenance
        assert provenance.session_id == "synthetic-original"
        assert provenance.directory_name == "synthetic-copy"
        assert provenance.identifier_matches_directory is False

    def test_the_mismatch_is_warned_about_not_reconciled(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        original = sfx.write_session(root, "synthetic-original")
        copy = root / "synthetic-copy"
        copy.mkdir()
        for path in original.iterdir():
            (copy / path.name).write_bytes(path.read_bytes())

        read = read_session(copy, mode=REPLAY)
        messages = " ".join(w.message for w in read.summary.provenance.warnings)
        assert "declares session_id" in messages
        assert "neither has been rewritten" in messages

    def test_a_copied_recording_is_still_read_and_labelled(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sessions"
        original = sfx.write_session(root, "synthetic-original")
        copy = root / "synthetic-copy"
        copy.mkdir()
        for path in original.iterdir():
            (copy / path.name).write_bytes(path.read_bytes())

        read = read_session(copy, mode=REPLAY)
        assert read.summary.decoded_record_count > 0
        assert read.summary.provenance.is_synthetic is True

    def test_the_catalogue_lists_both_copies_separately(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        original = sfx.write_session(root, "synthetic-original")
        copy = root / "synthetic-copy"
        copy.mkdir()
        for path in original.iterdir():
            (copy / path.name).write_bytes(path.read_bytes())

        catalogue = build_session_catalogue(root, mode=REPLAY)
        assert catalogue.directory_names() == (
            "synthetic-copy",
            "synthetic-original",
        )
        assert catalogue.session_ids() == (
            "synthetic-original",
            "synthetic-original",
        )

    def test_find_addresses_the_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        original = sfx.write_session(root, "synthetic-original")
        copy = root / "synthetic-copy"
        copy.mkdir()
        for path in original.iterdir():
            (copy / path.name).write_bytes(path.read_bytes())

        catalogue = build_session_catalogue(root, mode=REPLAY)
        found = catalogue.find("synthetic-copy")
        assert found is not None
        assert found.provenance.directory_name == "synthetic-copy"
        assert found.provenance.session_directory == str(copy)

    def test_the_catalogue_table_shows_both_names(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        original = sfx.write_session(root, "synthetic-original")
        copy = root / "synthetic-copy"
        copy.mkdir()
        for path in original.iterdir():
            (copy / path.name).write_bytes(path.read_bytes())

        table = views.catalogue_table(
            build_session_catalogue(root, mode=REPLAY), max_rows=100
        )
        assert table.columns[0] == "directory"
        assert table.columns[1] == "recorded session id"
        assert ("synthetic-copy", "synthetic-original") == table.rows[0][:2]

    def test_the_provenance_table_shows_both_names(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        directory = sfx.write_session(root, "synthetic-both-names")
        cells = {
            row[0]: row[1]
            for row in views.provenance_table(
                read_session(directory, mode=REPLAY).summary
            ).rows
        }
        assert cells["recorded session id"] == "synthetic-both-names"
        assert cells["directory name"] == "synthetic-both-names"
