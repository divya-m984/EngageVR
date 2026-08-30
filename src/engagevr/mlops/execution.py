"""The volatile half of every deterministic document.

The split
---------
A Milestone 10 document that is a DVC-declared output is part of the
pipeline's identity: its bytes are hashed into ``dvc.lock``.  A creation
timestamp inside one would make every reproduction rewrite the lock,
which turns "the lock changed" from a signal into noise.

So the timestamp is not written into the document.  It is written beside
it, as ``<name>.execution.json``, and that sidecar is never declared as a
DVC output.  Nothing is discarded: provenance is kept where it cannot
make a reproducible pipeline look irreproducible.

What this module does *not* touch
---------------------------------
The Milestone 5--8 artifacts.  A run manifest records ``started_at_utc``
and ``finished_at_utc`` because a run did happen at a time, and rewriting
those semantics to please a build tool would be the wrong repair.  Those
documents are simply never DVC-declared; see
:mod:`engagevr.mlops.stage_record`.
"""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from pathlib import Path

from engagevr.schemas.mlops import ExecutionMetadata
from engagevr.training.artifacts import engagevr_version, write_json_atomic

#: Suffix of every execution sidecar.
EXECUTION_SUFFIX = ".execution.json"


def sidecar_path(output: Path) -> Path:
    """Where the execution sidecar for ``output`` lives.

    Beside it, never inside it: a sidecar written into a directory output
    would be hashed along with the directory and defeat the whole point.
    """
    target = Path(output)
    return target.with_name(f"{target.stem}{EXECUTION_SUFFIX}")


def build_execution_metadata(*, describes: str, produced_by: str) -> ExecutionMetadata:
    """The volatile record of one execution."""
    return ExecutionMetadata(
        describes=describes,
        produced_by=produced_by,
        created_at_utc=datetime.now(UTC),
        engagevr_version=engagevr_version(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
    )


def write_execution_sidecar(output: Path, *, describes: str, produced_by: str) -> Path:
    """Write ``<output stem>.execution.json`` beside ``output``.

    ``describes`` is a pipeline-relative path, so the sidecar names the
    document it belongs to without embedding a machine-specific location.
    """
    path = sidecar_path(output)
    write_json_atomic(
        path,
        build_execution_metadata(
            describes=describes, produced_by=produced_by
        ).model_dump(mode="json"),
    )
    return path


__all__ = [
    "EXECUTION_SUFFIX",
    "build_execution_metadata",
    "sidecar_path",
    "write_execution_sidecar",
]
