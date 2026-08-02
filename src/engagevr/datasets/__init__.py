"""Public-dataset adapters.

No adapter downloads data. Dataset roots are supplied by configuration
or on the command line, and obtaining the data through the official
channel -- including accepting whatever terms the provider states -- is
the user's responsibility. See ``docs/DATASETS.md``.
"""

from engagevr.datasets.base import (
    DatasetAdapter,
    DatasetError,
    DatasetRecording,
    ReferencePhysiology,
)
from engagevr.datasets.ubfc_rppg import UbfcRppgAdapter

#: Adapters available to the CLI, keyed by their ``--dataset`` name.
ADAPTERS: dict[str, type[DatasetAdapter]] = {
    UbfcRppgAdapter.name: UbfcRppgAdapter,
}

__all__ = [
    "ADAPTERS",
    "DatasetAdapter",
    "DatasetError",
    "DatasetRecording",
    "ReferencePhysiology",
    "UbfcRppgAdapter",
]
