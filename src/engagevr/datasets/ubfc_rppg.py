"""UBFC-rPPG dataset adapter.

Source
------
Official page: https://sites.google.com/view/ybenezeth/ubfcrppg

Citation
--------
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois (2017).
"Unsupervised skin tissue segmentation for remote photoplethysmography."
Pattern Recognition Letters.

Access and licensing
--------------------
The dataset is obtained by following the download link on the official
page.  **This adapter never downloads anything.**  At the time this
adapter was written the official page carried no explicit licence or
permitted-use statement, so licensing status is recorded as
**requiring manual verification** -- see ``docs/DATASETS.md``.  Do not
assume permission that the source does not grant.

Expected layout
---------------
The layout below matches the official page's description and the loader
used by the widely-used ``rPPG-Toolbox`` reference implementation::

    <root>/
      subject1/
        vid.avi
        ground_truth.txt
      subject2/
        ...

``ground_truth.txt`` is whitespace-separated numeric text.  Row 0 is the
reference PPG waveform.  Rows 1 and 2 are, by community convention,
heart rate and timestamps respectively; the official page does not
document the row ordering, so this adapter reads them **optionally** and
treats their absence as normal rather than as an error.  Confirm the row
semantics against the copy of the dataset actually in use before relying
on rows 1 or 2.

Reference device
----------------
A CMS50E transmissive pulse oximeter, per the official page.  The page
does not state the oximeter's sampling rate unambiguously, so this
adapter does **not** assume one; ``sampling_rate_hz`` is left ``None``
unless a caller supplies a value it has verified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from engagevr.datasets.base import (
    DatasetAdapter,
    DatasetError,
    DatasetRecording,
    ReferencePhysiology,
)

#: Filenames expected inside each subject directory.
VIDEO_FILENAME = "vid.avi"
GROUND_TRUTH_FILENAME = "ground_truth.txt"

#: Subject directories match this glob.
SUBJECT_GLOB = "subject*"

#: Row index of the reference PPG waveform in ground_truth.txt.
PPG_ROW = 0
#: Row indices read opportunistically; see the module docstring.
HEART_RATE_ROW = 1
TIMESTAMP_ROW = 2

#: Minimum samples for a ground-truth file to be considered usable.
_MIN_REFERENCE_SAMPLES = 2


def parse_ground_truth(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Parse a UBFC-rPPG ``ground_truth.txt``.

    Returns
    -------
    (waveform, heart_rate, timestamps)
        ``heart_rate`` and ``timestamps`` are ``None`` when the file does
        not contain those rows, or when their length does not match the
        waveform.  Length mismatch is treated as "not usable" rather than
        being silently truncated to fit.

    Raises
    ------
    DatasetError
        When the file is missing, empty, non-numeric, or contains too few
        waveform samples to analyse.
    """
    if not path.exists():
        raise DatasetError(
            f"Ground-truth file not found: {path}. Expected "
            f"'{GROUND_TRUTH_FILENAME}' inside the subject directory."
        )

    text = path.read_text().strip()
    if not text:
        raise DatasetError(f"Ground-truth file is empty: {path}")

    rows: list[np.ndarray] = []
    for line_no, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(np.asarray([float(v) for v in stripped.split()]))
        except ValueError as exc:
            raise DatasetError(
                f"Malformed ground truth in {path} at line {line_no + 1}: "
                f"expected whitespace-separated numbers, got {stripped[:60]!r}"
            ) from exc

    if len(rows) <= PPG_ROW:
        raise DatasetError(
            f"Ground-truth file {path} has no PPG waveform row "
            f"(expected row {PPG_ROW})."
        )

    waveform = rows[PPG_ROW]
    if waveform.size < _MIN_REFERENCE_SAMPLES:
        raise DatasetError(
            f"Ground-truth waveform in {path} has {waveform.size} samples; "
            f"at least {_MIN_REFERENCE_SAMPLES} are required."
        )
    if not np.all(np.isfinite(waveform)):
        raise DatasetError(
            f"Ground-truth waveform in {path} contains non-finite values."
        )

    def _optional(index: int) -> np.ndarray | None:
        if len(rows) <= index:
            return None
        row = rows[index]
        if row.size != waveform.size or not np.all(np.isfinite(row)):
            return None
        return row

    return waveform, _optional(HEART_RATE_ROW), _optional(TIMESTAMP_ROW)


class UbfcRppgAdapter(DatasetAdapter):
    """Adapter for a locally obtained UBFC-rPPG dataset root."""

    name = "ubfc-rppg"

    citation = (
        "S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois (2017). "
        "Unsupervised skin tissue segmentation for remote photoplethysmography. "
        "Pattern Recognition Letters."
    )
    official_source = "https://sites.google.com/view/ybenezeth/ubfcrppg"
    license_status = (
        "No explicit licence or permitted-use statement was found on the "
        "official page. Licensing REQUIRES MANUAL VERIFICATION with the "
        "dataset authors before use. See docs/DATASETS.md."
    )
    reference_device = "CMS50E transmissive pulse oximeter (per official page)"

    def validate(self) -> None:
        """Check that the root exists and contains usable subject folders."""
        if not self.root.exists():
            raise DatasetError(
                f"UBFC-rPPG root not found: {self.root}. Obtain the dataset "
                f"from {self.official_source} and pass its path with --root "
                "or set rppg.datasets.ubfc_rppg_root in configs/defaults.yaml. "
                "This software does not download datasets."
            )
        if not self.root.is_dir():
            raise DatasetError(f"UBFC-rPPG root is not a directory: {self.root}")

        subject_dirs = sorted(p for p in self.root.glob(SUBJECT_GLOB) if p.is_dir())
        if not subject_dirs:
            raise DatasetError(
                f"No subject directories matching '{SUBJECT_GLOB}' under "
                f"{self.root}. Expected a layout of "
                f"<root>/subject1/{VIDEO_FILENAME} and "
                f"<root>/subject1/{GROUND_TRUTH_FILENAME}."
            )

        problems: list[str] = []
        for subject_dir in subject_dirs:
            for filename in (VIDEO_FILENAME, GROUND_TRUTH_FILENAME):
                if not (subject_dir / filename).exists():
                    problems.append(f"  missing {subject_dir.name}/{filename}")
        if problems:
            raise DatasetError(
                f"UBFC-rPPG root {self.root} is incomplete:\n" + "\n".join(problems)
            )

    def list_subjects(self) -> list[str]:
        """Return sorted subject directory names."""
        if not self.root.is_dir():
            raise DatasetError(f"UBFC-rPPG root not found: {self.root}")
        return sorted(p.name for p in self.root.glob(SUBJECT_GLOB) if p.is_dir())

    def subject_dir(self, subject_id: str) -> Path:
        """Return one subject's directory, checking that it exists."""
        path = self.root / subject_id
        if not path.is_dir():
            raise DatasetError(
                f"Subject directory not found: {path}. "
                f"Available subjects: {', '.join(self.list_subjects()) or 'none'}"
            )
        return path

    def load_recording(self, subject_id: str) -> DatasetRecording:
        """Load one subject's video path and reference PPG signal.

        The video is **not** decoded here -- only its path is returned, so
        that callers can decode lazily and so that this method needs no
        video codec to run.
        """
        directory = self.subject_dir(subject_id)
        video_path = directory / VIDEO_FILENAME
        if not video_path.exists():
            raise DatasetError(
                f"Video not found: {video_path}. Expected "
                f"'{VIDEO_FILENAME}' inside {directory}."
            )

        waveform, heart_rate, timestamps = parse_ground_truth(
            directory / GROUND_TRUTH_FILENAME
        )

        reference = ReferencePhysiology(
            subject_id=subject_id,
            waveform=waveform,
            sampling_rate_hz=None,  # not documented unambiguously; do not guess
            heart_rate_bpm=heart_rate,
            timestamps_s=timestamps,
            provenance=(
                f"{self.name}: {self.reference_device}; "
                f"root={self.root}; citation={self.citation}"
            ),
        )
        return DatasetRecording(
            subject_id=subject_id,
            video_path=video_path,
            reference=reference,
            metadata={
                "dataset": self.name,
                "official_source": self.official_source,
                "license_status": self.license_status,
                "reference_device": self.reference_device,
            },
        )

    def describe(self) -> dict[str, str]:
        return {
            "dataset": self.name,
            "root": str(self.root),
            "official_source": self.official_source,
            "citation": self.citation,
            "license_status": self.license_status,
            "reference_device": self.reference_device,
        }
