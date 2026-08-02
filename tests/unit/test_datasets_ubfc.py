"""Tests for the UBFC-rPPG adapter and evaluation metrics.

These tests use temporary deterministic fixtures only.  They never
require the real UBFC-rPPG dataset, never touch the network, and assert
that the adapter has no download path.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from engagevr.datasets import ADAPTERS, DatasetError, UbfcRppgAdapter
from engagevr.datasets import ubfc_rppg as ubfc_module
from engagevr.rppg.evaluation import aggregate_metrics, per_subject_metrics
from engagevr.schemas.rppg import DatasetEvaluationRecord, RppgMethod

# --- fixture layout -------------------------------------------------------


def test_adapter_is_registered() -> None:
    assert ADAPTERS["ubfc-rppg"] is UbfcRppgAdapter


def test_valid_fixture_layout_passes_validation(
    ubfc_fixture_root: Path,
) -> None:
    adapter = UbfcRppgAdapter(ubfc_fixture_root)

    adapter.validate()

    assert adapter.list_subjects() == ["subject1", "subject2"]


def test_load_recording_returns_video_path_and_reference(
    ubfc_fixture_root: Path,
) -> None:
    adapter = UbfcRppgAdapter(ubfc_fixture_root)

    recording = adapter.load_recording("subject1")

    assert recording.subject_id == "subject1"
    assert recording.video_path.name == "vid.avi"
    assert recording.video_path.exists()
    assert recording.reference.waveform.size > 0
    assert np.all(np.isfinite(recording.reference.waveform))


def test_load_recording_reads_optional_rows(ubfc_fixture_root: Path) -> None:
    adapter = UbfcRppgAdapter(ubfc_fixture_root)

    reference = adapter.load_recording("subject1").reference

    assert reference.heart_rate_bpm is not None
    assert reference.timestamps_s is not None
    assert reference.heart_rate_bpm.size == reference.waveform.size


def test_sampling_rate_is_not_guessed(ubfc_fixture_root: Path) -> None:
    """The official page does not state it unambiguously; do not invent it."""
    adapter = UbfcRppgAdapter(ubfc_fixture_root)

    assert adapter.load_recording("subject1").reference.sampling_rate_hz is None


# --- missing and malformed data -------------------------------------------


def test_missing_root_raises_actionable_error(tmp_path: Path) -> None:
    adapter = UbfcRppgAdapter(tmp_path / "absent")

    with pytest.raises(DatasetError) as exc:
        adapter.validate()

    message = str(exc.value)
    assert "not found" in message
    assert "sites.google.com" in message
    assert "does not download" in message


def test_root_that_is_a_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "afile"
    path.write_text("x")

    with pytest.raises(DatasetError, match="not a directory"):
        UbfcRppgAdapter(path).validate()


def test_root_without_subjects_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "unrelated").mkdir()

    with pytest.raises(DatasetError) as exc:
        UbfcRppgAdapter(tmp_path).validate()

    assert "subject*" in str(exc.value)


def test_missing_video_file_is_reported(tmp_path: Path) -> None:
    subject = tmp_path / "subject1"
    subject.mkdir()
    (subject / "ground_truth.txt").write_text("1 2 3\n")

    with pytest.raises(DatasetError) as exc:
        UbfcRppgAdapter(tmp_path).validate()

    assert "subject1/vid.avi" in str(exc.value)


def test_missing_ground_truth_is_reported(tmp_path: Path) -> None:
    subject = tmp_path / "subject1"
    subject.mkdir()
    (subject / "vid.avi").write_bytes(b"\x00")

    with pytest.raises(DatasetError) as exc:
        UbfcRppgAdapter(tmp_path).validate()

    assert "subject1/ground_truth.txt" in str(exc.value)


def test_unknown_subject_lists_available_ones(
    ubfc_fixture_root: Path,
) -> None:
    adapter = UbfcRppgAdapter(ubfc_fixture_root)

    with pytest.raises(DatasetError) as exc:
        adapter.load_recording("subject99")

    assert "subject1" in str(exc.value)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("", "empty"),
        ("not numbers here\n", "Malformed"),
        ("1.0\n", "at least"),
        ("1.0 nan 3.0\n", "non-finite"),
    ],
)
def test_malformed_ground_truth_is_rejected(
    tmp_path: Path, content: str, match: str
) -> None:
    path = tmp_path / "ground_truth.txt"
    path.write_text(content)

    with pytest.raises(DatasetError, match=match):
        ubfc_module.parse_ground_truth(path)


def test_missing_ground_truth_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        ubfc_module.parse_ground_truth(tmp_path / "absent.txt")


def test_mismatched_optional_rows_are_dropped_not_truncated(
    tmp_path: Path,
) -> None:
    """A length mismatch means 'unusable', never 'silently trimmed'."""
    path = tmp_path / "ground_truth.txt"
    path.write_text("1.0 2.0 3.0 4.0\n60.0 61.0\n")

    waveform, heart_rate, timestamps = ubfc_module.parse_ground_truth(path)

    assert waveform.size == 4
    assert heart_rate is None
    assert timestamps is None


def test_ground_truth_with_only_the_waveform_row_is_valid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ground_truth.txt"
    path.write_text("1.0 2.0 3.0 4.0\n")

    waveform, heart_rate, timestamps = ubfc_module.parse_ground_truth(path)

    assert waveform.size == 4
    assert heart_rate is None
    assert timestamps is None


# --- metadata and licensing -----------------------------------------------


def test_licensing_is_marked_as_requiring_manual_verification() -> None:
    status = UbfcRppgAdapter.license_status

    assert "MANUAL VERIFICATION" in status
    assert "No explicit licence" in status


def test_describe_carries_full_provenance(ubfc_fixture_root: Path) -> None:
    described = UbfcRppgAdapter(ubfc_fixture_root).describe()

    for key in (
        "dataset",
        "root",
        "official_source",
        "citation",
        "license_status",
        "reference_device",
    ):
        assert described[key]


def test_adapter_has_no_download_capability() -> None:
    """No network access may exist anywhere in the adapter module."""
    source = inspect.getsource(ubfc_module)

    for forbidden in (
        "urllib",
        "requests",
        "urlretrieve",
        "http.client",
        "socket",
        "wget",
        "curl",
        "download(",
    ):
        assert forbidden not in source


# --- evaluation metrics ---------------------------------------------------


def record(
    *,
    estimated: float | None,
    reference: float | None,
    abstained: bool = False,
    subject: str = "subject1",
    index: int = 0,
) -> DatasetEvaluationRecord:
    return DatasetEvaluationRecord(
        dataset="ubfc-rppg",
        subject_id=subject,
        window_index=index,
        method=RppgMethod.POS,
        estimated_bpm=estimated,
        reference_bpm=reference,
        abstained=abstained,
    )


def test_metrics_are_none_without_a_reference() -> None:
    """Coverage is meaningful without a reference; error metrics are not."""
    records = [record(estimated=72.0, reference=None) for _ in range(4)]

    metrics = aggregate_metrics(records, dataset="ubfc-rppg", method=RppgMethod.POS)

    assert metrics.reference_available is False
    assert metrics.mae_bpm is None
    assert metrics.rmse_bpm is None
    assert metrics.bias_bpm is None
    assert metrics.error_std_bpm is None
    assert metrics.coverage == pytest.approx(1.0)


def test_error_metrics_are_computed_against_real_references() -> None:
    records = [
        record(estimated=72.0, reference=70.0, index=0),
        record(estimated=68.0, reference=70.0, index=1),
        record(estimated=74.0, reference=70.0, index=2),
    ]

    metrics = aggregate_metrics(records, dataset="ubfc-rppg", method=RppgMethod.POS)

    assert metrics.reference_available is True
    assert metrics.mae_bpm == pytest.approx((2.0 + 2.0 + 4.0) / 3.0)
    assert metrics.rmse_bpm == pytest.approx(float(np.sqrt((4.0 + 4.0 + 16.0) / 3.0)))
    assert metrics.bias_bpm == pytest.approx((2.0 - 2.0 + 4.0) / 3.0)
    assert metrics.error_std_bpm is not None


def test_abstained_windows_reduce_coverage_but_not_error() -> None:
    records = [
        record(estimated=72.0, reference=70.0, index=0),
        record(estimated=None, reference=70.0, abstained=True, index=1),
        record(estimated=None, reference=70.0, abstained=True, index=2),
        record(estimated=71.0, reference=70.0, index=3),
    ]

    metrics = aggregate_metrics(records, dataset="ubfc-rppg", method=RppgMethod.POS)

    assert metrics.n_windows_attempted == 4
    assert metrics.n_windows_valid == 2
    assert metrics.n_windows_abstained == 2
    assert metrics.coverage == pytest.approx(0.5)
    assert metrics.valid_window_pct == pytest.approx(50.0)
    assert metrics.mae_bpm == pytest.approx(1.5)


def test_empty_record_set_yields_zero_coverage_and_no_metrics() -> None:
    metrics = aggregate_metrics([], dataset="ubfc-rppg", method=RppgMethod.POS)

    assert metrics.n_windows_attempted == 0
    assert metrics.coverage == 0.0
    assert metrics.mae_bpm is None
    assert metrics.reference_available is False


def test_per_subject_metrics_keep_subjects_separate() -> None:
    records = [
        record(estimated=72.0, reference=70.0, subject="subject1", index=0),
        record(estimated=90.0, reference=70.0, subject="subject2", index=0),
    ]

    results = per_subject_metrics(records, dataset="ubfc-rppg", method=RppgMethod.POS)

    assert [m.subject_id for m in results] == ["subject1", "subject2"]
    assert results[0].mae_bpm == pytest.approx(2.0)
    assert results[1].mae_bpm == pytest.approx(20.0)


def test_provenance_is_preserved_on_metrics() -> None:
    metrics = aggregate_metrics(
        [record(estimated=72.0, reference=70.0)],
        dataset="ubfc-rppg",
        method=RppgMethod.POS,
        provenance="root=/data/UBFC-rPPG; adapter=ubfc-rppg",
    )

    assert "UBFC-rPPG" in metrics.provenance
