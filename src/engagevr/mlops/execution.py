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

The same boundary, for the same reason, holds a second kind of fact.  A
serialized estimator's SHA-256 is worth keeping — it is how anybody
notices that a model file changed after it was written — but a
``.joblib`` embeds uninitialised C struct padding and reflects the
interpreter, library, and CPU that produced it, so its digest describes
one execution rather than the experiment.  It is written to
``<name>.artifact-integrity.execution.json``, also beside the document
and also never declared.  See DEC-105.

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
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from engagevr.schemas.mlops import (
    ArtifactIntegrityEntry,
    ArtifactIntegrityRecord,
    ExecutionMetadata,
)
from engagevr.training.artifacts import engagevr_version, write_json_atomic

#: Suffix of every execution sidecar.
EXECUTION_SUFFIX = ".execution.json"

#: Suffix of the sidecar holding execution-specific artifact checksums.
#:
#: It ends with :data:`EXECUTION_SUFFIX` on purpose: every rule that keeps
#: an execution sidecar out of ``dvc.yaml``, out of the Docker images, and
#: out of any identity applies to this record unchanged, because it *is*
#: one.  See DEC-105.
INTEGRITY_SUFFIX = ".artifact-integrity.execution.json"


def sidecar_path(output: Path) -> Path:
    """Where the execution sidecar for ``output`` lives.

    Beside it, never inside it: a sidecar written into a directory output
    would be hashed along with the directory and defeat the whole point.
    """
    target = Path(output)
    return target.with_name(f"{target.stem}{EXECUTION_SUFFIX}")


def integrity_sidecar_path(output: Path) -> Path:
    """Where ``output``'s artifact-integrity record lives.

    Beside the deterministic document, for the same reason as
    :func:`sidecar_path`: this record holds the checksums that may not
    reach ``dvc.lock``, so it must not sit inside anything declared.
    """
    target = Path(output)
    return target.with_name(f"{target.stem}{INTEGRITY_SUFFIX}")


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


def build_integrity_record(
    entries: Sequence[ArtifactIntegrityEntry],
    *,
    describes: str,
    produced_by: str,
    paths_relative_to: str,
) -> ArtifactIntegrityRecord:
    """The execution-specific checksums of one stage's generated artifacts.

    ``paths_relative_to`` states what the entries' paths resolve against —
    the pipeline root for a stage record, the producing run directory for a
    model version. It is required rather than inferred, because a checksum
    whose file nobody can locate verifies nothing.

    The environment is recorded with the digests because it is the reason
    they are not portable: a different interpreter build, library build, or
    CPU legitimately produces different bytes for the same model.
    """
    from engagevr.training.artifacts import dependency_versions

    return ArtifactIntegrityRecord(
        describes=describes,
        produced_by=produced_by,
        created_at_utc=datetime.now(UTC),
        paths_relative_to=paths_relative_to,
        artifacts=tuple(sorted(entries, key=lambda entry: entry.path)),
        engagevr_version=engagevr_version(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        dependency_versions=dependency_versions(),
    )


def write_integrity_sidecar(
    output: Path,
    entries: Sequence[ArtifactIntegrityEntry],
    *,
    describes: str,
    produced_by: str,
    paths_relative_to: str,
) -> Path:
    """Write ``<output stem>.artifact-integrity.execution.json`` beside ``output``.

    Written even when ``entries`` is empty, so that "this stage produced no
    execution-specific artifact" is an assertion on disk rather than a
    missing file that could equally mean the step never ran.
    """
    path = integrity_sidecar_path(output)
    write_json_atomic(
        path,
        build_integrity_record(
            entries,
            describes=describes,
            produced_by=produced_by,
            paths_relative_to=paths_relative_to,
        ).model_dump(mode="json"),
    )
    return path


__all__ = [
    "EXECUTION_SUFFIX",
    "INTEGRITY_SUFFIX",
    "build_execution_metadata",
    "build_integrity_record",
    "integrity_sidecar_path",
    "sidecar_path",
    "write_execution_sidecar",
    "write_integrity_sidecar",
]
