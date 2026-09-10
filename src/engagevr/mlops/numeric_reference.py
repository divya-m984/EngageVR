"""The accepted numerical result, committed to the repository.

The hole this closes
--------------------
DEC-106 removed floating-point *values* from portable identity, which was
right: they follow the CPU's BLAS kernel, and hashing them made a correct
reproduction on another machine look like a changed pipeline.

But removing them from identity also removed them from detection, and the
first version of the check compounded that by comparing two executions of
the **same** CI runner against each other.  Both sides then come from one
CPU, so both can be wrong in the same way:

    accepted result   0.72
    this runner       0.91   (twice, identically)

Measured on this repository before the repair: mutating all 1,463 floats
in ``metrics.json`` — structure, keys, dtypes, ordering and labels
untouched — left ``dvc.lock`` **byte-identical** and the same-runner
comparison **exited zero**.  Every gate passed on a grossly different
result.  That is fail-open, and it is what this module fixes.

The repair
----------
The accepted numbers are committed to the repository, under
``references/numeric/``, and a fresh execution anywhere is compared
against **them**:

    committed reference        accepted numbers, version-controlled,
                               checksummed in MANIFEST.json
              |
    fresh execution            reproduced on any machine
              |
    numeric-check --accepted   structure exact, floats within tolerance

So the thing CI trusts is a deterministic property of the repository
revision, not an artifact of the runner it happens to be scheduled on.

What is still not claimed
-------------------------
Reproducing a reference means the software is numerically portable. It
does not mean any number in it is correct, and nothing in this repository
has been evaluated against a participant-provided label. The references
describe SYNTHETIC output and are ``scientific_evaluation_eligible=false``
by construction.

See DEC-107.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from engagevr.mlops.numeric_contract import (
    DEFAULT_TOLERANCE,
    NumericContractError,
    NumericTolerance,
    iter_floats,
    structure_digest,
)
from engagevr.schemas.experiments import SELF_CHECK_DISCLAIMER
from engagevr.schemas.mlops import (
    MLOPS_DISCLAIMER,
    CpuDependentNumericArtifact,
    NumericReference,
    NumericReferenceManifest,
)
from engagevr.training.artifacts import sha256_file, write_json_atomic

#: Where the accepted references live, relative to the repository root.
#:
#: Committed, reviewed, and changed only on purpose.  A directory rather
#: than one file because a reader comparing one artifact should not have
#: to diff half a megabyte of unrelated numbers.
REFERENCE_DIRECTORY = Path("references/numeric")

#: The manifest naming every reference and its exact digest.
MANIFEST_NAME = "MANIFEST.json"

#: Suffix of one accepted reference document.
REFERENCE_SUFFIX = ".reference.json"


class NumericReferenceError(ValueError):
    """An accepted numerical reference is missing, unreadable, or wrong."""


def reference_file_name(artifact_path: str) -> str:
    """The reference file name for a pipeline-relative artifact path.

    Flattened rather than nested so the manifest can name a bare file and
    a reader can find it without walking a tree.
    """
    return artifact_path.replace("/", "__") + REFERENCE_SUFFIX


def build_reference(
    artifact: Path,
    artifact_path: str,
    *,
    tolerance: NumericTolerance = DEFAULT_TOLERANCE,
) -> NumericReference:
    """Read one produced artifact and record the numbers it contains."""
    target = Path(artifact)
    if not target.is_file():
        raise NumericReferenceError(
            f"{target} does not exist, so there is no accepted result to record"
        )
    values = iter_floats(target)
    return NumericReference(
        artifact_path=artifact_path,
        structure_sha256=structure_digest(target),
        value_count=len(values),
        values=tuple(values),
        atol=tolerance.atol,
        rtol=tolerance.rtol,
        numeric_tolerance=tolerance.describe(),
        disclaimers=(SELF_CHECK_DISCLAIMER, MLOPS_DISCLAIMER),
    )


def write_reference(reference: NumericReference, directory: Path) -> Path:
    """Write one reference document, atomically, and return its path."""
    path = Path(directory) / reference_file_name(reference.artifact_path)
    return write_json_atomic(path, reference.model_dump(mode="json"))


def read_reference(path: Path) -> NumericReference:
    """Read and validate one accepted reference."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NumericReferenceError(f"{path} is not readable JSON: {exc}") from exc
    try:
        return NumericReference.model_validate(document)
    except ValueError as exc:
        raise NumericReferenceError(f"{path} is not a valid reference: {exc}") from exc


def write_manifest(
    directory: Path,
    *,
    target: str,
    tolerance: NumericTolerance = DEFAULT_TOLERANCE,
) -> Path:
    """Checksum every reference in ``directory`` and write the manifest.

    Written last, from the files as they landed on disk, so the digests
    describe the committed bytes rather than the in-memory objects that
    produced them.
    """
    root = Path(directory)
    references = {
        path.name: sha256_file(path)
        for path in sorted(root.glob(f"*{REFERENCE_SUFFIX}"))
    }
    manifest = NumericReferenceManifest(
        target=target,
        references=references,
        atol=tolerance.atol,
        rtol=tolerance.rtol,
        numeric_tolerance=tolerance.describe(),
    )
    return write_json_atomic(root / MANIFEST_NAME, manifest.model_dump(mode="json"))


def read_manifest(directory: Path) -> NumericReferenceManifest:
    """Read and validate the manifest beside a set of references."""
    path = Path(directory) / MANIFEST_NAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NumericReferenceError(
            f"{path} is missing or unreadable ({exc}). Without it the "
            "references cannot be trusted, so the check refuses rather than "
            "reading them anyway."
        ) from exc
    try:
        return NumericReferenceManifest.model_validate(document)
    except ValueError as exc:
        raise NumericReferenceError(f"{path} is not a valid manifest: {exc}") from exc


def verify_manifest(directory: Path) -> list[str]:
    """Differences between the manifest and the reference files on disk.

    Tamper-evidence for the thing CI trusts.  A reference edited without
    updating the manifest, a reference listed and missing, and a reference
    present but unlisted are all reported — the last one because an
    unlisted file is exactly how an unreviewed number would arrive.
    """
    root = Path(directory)
    manifest = read_manifest(root)
    problems: list[str] = []
    on_disk = {path.name for path in root.glob(f"*{REFERENCE_SUFFIX}")}
    for name, expected in sorted(manifest.references.items()):
        path = root / name
        if not path.is_file():
            problems.append(f"{name}: listed in {MANIFEST_NAME} but missing on disk")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(
                f"{name}: SHA-256 {actual} does not match the {expected} "
                f"recorded in {MANIFEST_NAME}. The accepted reference was "
                "modified without being re-accepted."
            )
    for name in sorted(on_disk - set(manifest.references)):
        problems.append(
            f"{name}: present on disk but absent from {MANIFEST_NAME}; an "
            "unlisted reference has not been reviewed"
        )
    return problems


def manifest_tolerance(manifest: NumericReferenceManifest) -> NumericTolerance:
    """The tolerance the accepted reference set declares.

    The single authority.  Nothing else — not the running code's default,
    not a command-line flag — may widen it, because the accepted numbers
    and the tolerance they were accepted under are one contract and
    reading half of it from the repository and half from the caller would
    let the gate be loosened without the diff showing it.
    """
    return NumericTolerance(atol=manifest.atol, rtol=manifest.rtol)


def reference_tolerance(reference: NumericReference) -> NumericTolerance:
    """The tolerance one accepted reference was recorded under."""
    return NumericTolerance(atol=reference.atol, rtol=reference.rtol)


def verify_tolerance_agreement(
    directory: Path,
    *,
    declared: Sequence[CpuDependentNumericArtifact] = (),
) -> list[str]:
    """Whether the repository states one tolerance or several.

    Three places record it and all three must agree: the accepted
    manifest, every accepted reference beside it, and every stage record
    the pipeline produces.  The running code is checked against the
    manifest too, so widening :data:`DEFAULT_TOLERANCE` without
    re-accepting the references is caught rather than silently applied to
    a set of numbers accepted under a stricter contract.

    Changing the tolerance is a deliberate repository contract change; it
    means editing the code, re-running ``numeric-reference --update``, and
    reviewing the diff.
    """
    root = Path(directory)
    manifest = read_manifest(root)
    accepted = manifest_tolerance(manifest)
    problems: list[str] = []
    if (accepted.atol, accepted.rtol) != (
        DEFAULT_TOLERANCE.atol,
        DEFAULT_TOLERANCE.rtol,
    ):
        problems.append(
            f"{MANIFEST_NAME} declares atol={accepted.atol:g} "
            f"rtol={accepted.rtol:g} but this build's DEFAULT_TOLERANCE is "
            f"atol={DEFAULT_TOLERANCE.atol:g} rtol={DEFAULT_TOLERANCE.rtol:g}. "
            "The repository would be claiming two different tolerances. "
            "Changing one is a deliberate contract change: update the code "
            "and re-run 'engagevr numeric-reference --update', then review "
            "the diff."
        )
    for name in sorted(manifest.references):
        path = root / name
        if not path.is_file():
            continue
        try:
            reference = read_reference(path)
        except NumericReferenceError:
            continue  # verify_manifest reports unreadable references
        if (reference.atol, reference.rtol) != (accepted.atol, accepted.rtol):
            problems.append(
                f"{name}: recorded under atol={reference.atol:g} "
                f"rtol={reference.rtol:g}, but {MANIFEST_NAME} declares "
                f"atol={accepted.atol:g} rtol={accepted.rtol:g}"
            )
    for artifact in declared:
        if (artifact.atol, artifact.rtol) != (accepted.atol, accepted.rtol):
            problems.append(
                f"{artifact.path}: the stage record declares "
                f"atol={artifact.atol:g} rtol={artifact.rtol:g}, but the "
                f"accepted contract declares atol={accepted.atol:g} "
                f"rtol={accepted.rtol:g}"
            )
    return problems


def compare_to_reference(
    artifact: Path,
    reference: NumericReference,
    *,
    tolerance: NumericTolerance | None = None,
) -> list[str]:
    """Differences between a produced artifact and the accepted result.

    Structure first, and if it differs the values are not compared at
    all: a reordered table's floats would line up against the wrong
    reference entries and the report would be noise.  Structure covers
    the schema, the column and row order, the dtypes, every non-float
    value, every predicted label, and every missing and non-finite
    position, and no tolerance touches any of it.
    """
    contract = reference_tolerance(reference)
    if tolerance is not None and (tolerance.atol, tolerance.rtol) != (
        contract.atol,
        contract.rtol,
    ):
        return [
            f"{reference.artifact_path}: refused — the caller asked for "
            f"atol={tolerance.atol:g} rtol={tolerance.rtol:g} but this "
            f"reference was accepted under atol={contract.atol:g} "
            f"rtol={contract.rtol:g}. The accepted reference set is "
            "authoritative; a comparison may not widen it."
        ]
    target = Path(artifact)
    if not target.is_file():
        return [f"{reference.artifact_path}: produced no artifact to compare"]
    try:
        actual_structure = structure_digest(target)
    except NumericContractError as exc:
        return [f"{reference.artifact_path}: unreadable ({exc})"]
    if actual_structure != reference.structure_sha256:
        return [
            f"{reference.artifact_path}: structure differs from the accepted "
            "reference (schema, column or row order, dtypes, non-float "
            "values, labels, or missing-value positions). No tolerance "
            "covers this."
        ]
    values = iter_floats(target)
    if len(values) != reference.value_count:
        # Defensive: a matching structure digest should already guarantee
        # this. If it ever does not, refusing beats comparing misaligned
        # sequences.
        return [
            f"{reference.artifact_path}: {len(values)} finite values against "
            f"{reference.value_count} in the accepted reference, despite a "
            "matching structure digest"
        ]
    differences: list[str] = []
    worst = 0.0
    for index, (produced, accepted) in enumerate(
        zip(values, reference.values, strict=True)
    ):
        if not contract.holds(produced, accepted):
            worst = max(worst, abs(produced - accepted))
            if len(differences) < 5:
                differences.append(
                    f"value[{index}]: {produced!r} against accepted "
                    f"{accepted!r} (delta {abs(produced - accepted):.3e})"
                )
    if differences:
        count = sum(
            1
            for produced, accepted in zip(values, reference.values, strict=True)
            if not contract.holds(produced, accepted)
        )
        return [
            f"{reference.artifact_path}: {count} of {len(values)} values "
            f"exceed the tolerance (worst delta {worst:.3e})",
            *differences,
        ]
    return []


def check_against_references(
    pipeline_root: Path,
    reference_directory: Path,
    declared: Sequence[CpuDependentNumericArtifact],
    *,
    tolerance: NumericTolerance | None = None,
) -> tuple[list[str], int]:
    """Hold a produced pipeline to the accepted references.

    ``declared`` is every artifact the pipeline's stage records classify
    as cpu-dependent numeric.  Driving the check from the stage records
    rather than from the reference directory is what makes it **fail
    closed**: a newly classified artifact with no accepted reference is an
    error, not an absence.

    The tolerance comes from the accepted manifest, never from the
    caller.  A caller-supplied tolerance is permitted only if it equals
    the accepted one, so ``--atol``/``--rtol`` cannot loosen the gate.

    Returns the problems found and how many artifacts were compared.
    """
    root = Path(reference_directory)
    problems = verify_manifest(root)
    manifest = read_manifest(root)
    problems.extend(verify_tolerance_agreement(root, declared=declared))
    accepted = manifest_tolerance(manifest)
    if tolerance is not None and (tolerance.atol, tolerance.rtol) != (
        accepted.atol,
        accepted.rtol,
    ):
        problems.append(
            f"refused: --atol/--rtol asked for atol={tolerance.atol:g} "
            f"rtol={tolerance.rtol:g}, but the accepted references were "
            f"recorded under atol={accepted.atol:g} rtol={accepted.rtol:g}. "
            "The accepted reference set is authoritative and a check may "
            "not widen it ad hoc."
        )
        return problems, 0
    compared = 0
    expected_names: set[str] = set()
    for artifact in declared:
        artifact_path = artifact.path
        name = reference_file_name(artifact_path)
        expected_names.add(name)
        path = root / name
        if name not in manifest.references or not path.is_file():
            problems.append(
                f"{artifact_path}: classified cpu-dependent numeric but has "
                f"no accepted reference at {root / name}. Record one with "
                "'engagevr numeric-reference --update' and review the "
                "numbers before accepting them."
            )
            continue
        try:
            reference = read_reference(path)
        except NumericReferenceError as exc:
            problems.append(str(exc))
            continue
        if reference.artifact_path != artifact_path:
            problems.append(
                f"{name}: describes {reference.artifact_path!r} rather than "
                f"{artifact_path!r}"
            )
            continue
        compared += 1
        problems.extend(
            compare_to_reference(Path(pipeline_root) / artifact_path, reference)
        )
    for name in sorted(set(manifest.references) - expected_names):
        problems.append(
            f"{name}: an accepted reference the pipeline no longer declares. "
            "A stale reference checks nothing and hides that it stopped."
        )
    return problems, compared


__all__ = [
    "MANIFEST_NAME",
    "REFERENCE_DIRECTORY",
    "REFERENCE_SUFFIX",
    "NumericReferenceError",
    "build_reference",
    "check_against_references",
    "compare_to_reference",
    "manifest_tolerance",
    "read_manifest",
    "read_reference",
    "reference_file_name",
    "reference_tolerance",
    "verify_manifest",
    "verify_tolerance_agreement",
    "write_manifest",
    "write_reference",
]
