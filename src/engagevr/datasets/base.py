"""Abstract public-dataset adapter interface.

Every dataset gets its own adapter.  Dataset-specific quirks -- directory
layout, file naming, ground-truth encoding, sampling rate -- stay inside
that adapter and never leak into the signal-processing pipeline.
Datasets are never merged with one another: their recording conditions,
reference devices, and populations differ, so pooling them would produce
a metric that describes nothing in particular.

Nothing in this module downloads data.  Dataset roots are supplied by
configuration or on the command line, and the user is responsible for
obtaining the data through the official channel and accepting whatever
terms the provider states.  See ``docs/DATASETS.md``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt


class DatasetError(Exception):
    """Raised when a dataset root or file is missing, or malformed."""


@dataclass(frozen=True)
class ReferencePhysiology:
    """Reference physiological signal accompanying a dataset recording.

    This is a genuine measurement from the dataset's reference device.
    It is never synthesised, and it is the only thing an error metric may
    be computed against.
    """

    subject_id: str
    #: Reference PPG/BVP waveform samples.
    waveform: npt.NDArray[np.float64]
    #: Reference sampling rate in hertz, when the dataset documents one.
    sampling_rate_hz: float | None = None
    #: Reference heart rate per sample, in BPM, when the dataset provides it.
    heart_rate_bpm: npt.NDArray[np.float64] | None = None
    #: Reference timestamps in seconds, when the dataset provides them.
    timestamps_s: npt.NDArray[np.float64] | None = None
    #: Free-form provenance, e.g. the reference device named by the source.
    provenance: str = ""


@dataclass(frozen=True)
class DatasetRecording:
    """One subject recording: a video path plus its reference signal."""

    subject_id: str
    video_path: Path
    reference: ReferencePhysiology
    metadata: dict[str, str] = field(default_factory=dict)


class DatasetAdapter(abc.ABC):
    """Base class for public-dataset adapters."""

    #: Short machine-readable dataset name, used in CLI arguments.
    name: str = ""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @abc.abstractmethod
    def validate(self) -> None:
        """Raise :class:`DatasetError` unless the root looks correct.

        Error messages must be actionable: they must say which path was
        checked and what was expected there.
        """

    @abc.abstractmethod
    def list_subjects(self) -> list[str]:
        """Return sorted subject identifiers found under the root."""

    @abc.abstractmethod
    def load_recording(self, subject_id: str) -> DatasetRecording:
        """Load one subject's video path and reference physiology."""

    def describe(self) -> dict[str, str]:
        """Return provenance metadata for inclusion in every result."""
        return {"dataset": self.name, "root": str(self.root)}
