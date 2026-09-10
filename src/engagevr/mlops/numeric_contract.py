"""Portable numerical identity for artifacts whose floats follow the CPU.

The problem this solves
-----------------------
DEC-104 separated byte-stable outputs from timestamped provenance.
DEC-105 separated them again from serialized estimator bytes, and said
plainly what it could not yet prove:

    Reproducing the baseline stage with only the OpenBLAS kernel changed
    changed ``metrics.json``, ``predictions.parquet``, and
    ``feature_importance.parquet`` as well as the model files. [...] if a
    runner's CPU ever produces different numbers the lock check fails
    loudly and says so.

It did.  numpy and scipy ship OpenBLAS built ``DYNAMIC_ARCH``, which
selects a kernel from the CPU it finds at run time.  A different kernel
sums a dot product in a different order, so the last bits of a fitted
coefficient differ, and every number derived from it differs too.  The
measured spread is small — see :data:`MEASURED_CROSS_KERNEL_DEVIATION` —
but a SHA-256 has no notion of small.

Why this is not a fourth exclusion list
---------------------------------------
The temptation is to stop checksumming these files, which would delete
the evidence rather than classify it.  What actually differs between two
CPUs is *only the floating-point values*.  Everything else — the schema,
the column order, the row order, the dtypes, the predicted labels, the
integer counts, the identifiers, the positions of missing and non-finite
values — is identical, and can still be pinned exactly.

So an artifact in this class is split rather than dropped:

``structure_digest``
    SHA-256 over the artifact's exact non-floating content: schema,
    ordering, dtypes, every non-float value, and the positions of nulls
    and non-finite floats.  Floats are replaced by a token naming their
    *kind*, never their value.  This is portable, so it goes in the
    deterministic record and reaches ``dvc.lock``.

raw SHA-256
    The real digest of the real bytes, unchanged and undiminished.  It
    goes to the ``<name>.artifact-integrity.execution.json`` sidecar, the
    same mechanism DEC-105 already uses, where it still detects
    corruption and tampering.

numerical equivalence
    A *comparison*, run against a reference artifact, that holds every
    float to :data:`NUMERIC_PORTABILITY_ATOL` and
    :data:`NUMERIC_PORTABILITY_RTOL`.

Why numerical identity is a comparison and not a digest
-------------------------------------------------------
It is tempting to quantise the floats and hash the result, which would
give a single tolerance-stable fingerprint.  That construction does not
work, and it is worth saying why so nobody re-invents it.  Rounding maps
values onto a grid, and two values a single ULP apart can still fall on
opposite sides of a grid boundary.  With tens of thousands of floats per
artifact, a straddled boundary is not unlikely but nearly certain, and
the fingerprint would differ across machines for a subset of runs — the
same intermittent, inexplicable failure this module exists to remove.

A hash has no metric; tolerance is a metric. The two cannot be combined.
Numerical agreement is therefore expressed as a checked relation between
two artifacts, and the thing that is hashed is the part that really is
exact.

What the tolerance is, and is not
---------------------------------
It is an **engineering portability tolerance**: the amount of last-bit
disagreement this repository accepts between two CPUs before calling the
results different. It is not a scientific uncertainty interval, not a
confidence bound, not a measurement error, and not evidence about any
model's validity. Nothing in this repository has been evaluated against
a participant-provided label. See DEC-106.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engagevr.mlops.fingerprints import sha256_payload

#: The largest cross-kernel disagreement measured in this repository.
#:
#: Measured, not assumed.  The ``baseline`` and ``uncertainty`` stages
#: were reproduced on one machine with only ``OPENBLAS_CORETYPE``
#: changed — ``SKYLAKEX`` (this CPU's own choice), ``HASWELL``, and
#: ``NEHALEM``, standing in for three different CPUs — and every float in
#: every produced artifact was compared across all three pairings:
#:
#: ==================================  ==================
#: quantity                            value
#: ==================================  ==================
#: kernel pairings compared            3
#: floating-point values compared      76,896
#: **worst absolute deviation**        **1.0712e-08**
#: structure-digest mismatches         0
#: ==================================  ==================
#:
#: Per artifact, the largest movers were ``predictions.parquet``
#: (1.07e-08), ``feature_importance.parquet`` (4.65e-09), and
#: ``uncertainty.json`` (2.38e-09).
#:
#: The mechanism is not inferred: the same procedure reproduced a GitHub
#: runner's ``feature_importance.parquet`` byte for byte — SHA-256
#: ``6017d8f7…``, 23,827 bytes — under ``OPENBLAS_CORETYPE=HASWELL``.
#: See DEC-106.
MEASURED_CROSS_KERNEL_DEVIATION = 1.0712e-08

#: Absolute half of the portability tolerance.
#:
#: ``1e-6`` is 93x the worst deviation measured across three OpenBLAS
#: kernels and 76,896 values, which leaves room for a CPU this repository
#: has not seen while staying far below any difference a reader could
#: act on.  These artifacts carry probabilities, importances, and scores
#: reported to a handful of decimals; a genuine disagreement of 1e-6
#: would be visible in the reported value, and this contract refuses it.
#:
#: The margin is deliberately not larger.  A tolerance chosen to make a
#: check pass is not a contract, and a wide one would hide exactly the
#: numerical regression this check exists to catch — a changed
#: convergence criterion, a changed random draw, a changed formula.
NUMERIC_PORTABILITY_ATOL = 1e-6

#: Relative half of the portability tolerance.
#:
#: Present because an absolute bound alone is the wrong test for a value
#: far from zero — a log loss of 40 and a probability of 0.4 do not
#: deserve the same absolute slack.
NUMERIC_PORTABILITY_RTOL = 1e-6

#: Why a float token never carries a value.
FLOAT_TOKEN = "<float>"
NAN_TOKEN = "<nan>"
POSITIVE_INFINITY_TOKEN = "<+inf>"
NEGATIVE_INFINITY_TOKEN = "<-inf>"
NULL_TOKEN = "<null>"


class NumericContractError(ValueError):
    """An artifact could not be read under the numerical contract."""


@dataclass(frozen=True)
class NumericTolerance:
    """The portability tolerance applied to one comparison.

    ``|a - b| <= atol + rtol * |b|``, the combined form, because neither
    half is right alone: ``atol`` governs values near zero and ``rtol``
    governs values far from it.
    """

    atol: float = NUMERIC_PORTABILITY_ATOL
    rtol: float = NUMERIC_PORTABILITY_RTOL

    def holds(self, left: float, right: float) -> bool:
        """Whether two floats agree within this tolerance.

        Non-finite values are compared exactly: a NaN that became a
        number, or an infinity that changed sign, is a structural change
        rather than a rounding difference, and no tolerance covers it.
        """
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        if math.isinf(left) or math.isinf(right):
            return left == right
        return abs(left - right) <= self.atol + self.rtol * abs(right)

    def describe(self) -> str:
        """One line naming the tolerance, for a record or a report."""
        return (
            f"|a-b| <= {self.atol:g} + {self.rtol:g}*|b|; non-finite values "
            "compared exactly. Engineering portability tolerance, not a "
            "scientific uncertainty interval."
        )


DEFAULT_TOLERANCE = NumericTolerance()


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _float_token(value: float) -> str:
    """Name a float's *kind* without naming its value."""
    if math.isnan(value):
        return NAN_TOKEN
    if math.isinf(value):
        return POSITIVE_INFINITY_TOKEN if value > 0 else NEGATIVE_INFINITY_TOKEN
    return FLOAT_TOKEN


def iter_json_floats(document: Any) -> list[float]:
    """Every finite float in a JSON document, in canonical order.

    Dictionaries are traversed in **sorted key order** and lists in their
    own order, which is the same convention
    :func:`json_structure` digests — so if two documents agree on their
    structure digest, these sequences correspond element for element and
    may be compared positionally.

    Non-finite values are deliberately absent.  A NaN or an infinity is
    pinned *exactly* by the structure digest, which records where each one
    sits; carrying it here as well would only force a reference document
    to hold a value JSON cannot represent portably.
    """
    values: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, float):
            if math.isfinite(node):
                values.append(node)
            return
        if isinstance(node, Mapping):
            for key in sorted(node, key=str):
                walk(node[key])
            return
        if isinstance(node, str):
            return
        if isinstance(node, Sequence):
            for item in node:
                walk(item)

    walk(document)
    return values


def json_structure(document: Any) -> Any:
    """Replace every float leaf with a token naming its kind.

    Booleans are checked before numbers because ``bool`` is a subclass of
    ``int`` in Python, and an ``isinstance(value, int)`` test that ran
    first would silently reclassify ``True``.  Integers stay exact: a
    count, a fold index, and a seed are not floating-point results and
    must never be compared with a tolerance.
    """
    if isinstance(document, bool):
        return document
    if isinstance(document, float):
        return _float_token(document)
    if isinstance(document, int) or document is None or isinstance(document, str):
        return document
    if isinstance(document, Mapping):
        return {str(key): json_structure(value) for key, value in document.items()}
    if isinstance(document, Sequence):
        return [json_structure(value) for value in document]
    return document


def read_json(path: Path) -> Any:
    """Read a JSON artifact, refusing rather than guessing on failure."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NumericContractError(f"{path} is not readable JSON: {exc}") from exc


def compare_json(
    left: Any,
    right: Any,
    *,
    tolerance: NumericTolerance = DEFAULT_TOLERANCE,
    path: str = "",
) -> list[str]:
    """Differences between two JSON documents under the numeric contract.

    Structure and every non-floating leaf are compared **exactly**.  Only
    a leaf that is a float in both documents is compared with a
    tolerance, so an identifier, a count, a seed, a label, or a version
    string can never be waved through by a numerical allowance.
    """
    where = path or "$"
    if isinstance(left, bool) or isinstance(right, bool):
        if left != right or type(left) is not type(right):
            return [f"{where}: {left!r} != {right!r}"]
        return []
    if isinstance(left, float) and isinstance(right, float):
        if not tolerance.holds(left, right):
            return [
                f"{where}: {left!r} != {right!r} "
                f"(delta {abs(left - right):.3e} exceeds tolerance)"
            ]
        return []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        missing = set(right) - set(left)
        added = set(left) - set(right)
        for key in sorted(missing):
            differences.append(f"{where}.{key}: missing on the left")
        for key in sorted(added):
            differences.append(f"{where}.{key}: missing on the right")
        for key in sorted(set(left) & set(right)):
            differences.extend(
                compare_json(
                    left[key],
                    right[key],
                    tolerance=tolerance,
                    path=f"{where}.{key}",
                )
            )
        return differences
    if isinstance(left, str) or isinstance(right, str):
        return [] if left == right else [f"{where}: {left!r} != {right!r}"]
    if isinstance(left, Sequence) and isinstance(right, Sequence):
        if len(left) != len(right):
            return [f"{where}: length {len(left)} != {len(right)}"]
        differences = []
        for index, (one, other) in enumerate(zip(left, right, strict=True)):
            differences.extend(
                compare_json(one, other, tolerance=tolerance, path=f"{where}[{index}]")
            )
        return differences
    if type(left) is not type(right):
        return [f"{where}: type {type(left).__name__} != {type(right).__name__}"]
    return [] if left == right else [f"{where}: {left!r} != {right!r}"]


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------


def _load_frame(path: Path) -> Any:
    import pandas as pd

    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise NumericContractError(f"{path} is not readable parquet: {exc}") from exc


def _is_missing(value: Any) -> bool:
    """Whether one cell is missing, without tripping over array cells.

    ``pd.isna`` of a list or an array is *elementwise*, so its truth
    value is ambiguous and raises.  A container is never itself missing,
    which is the distinction this makes explicit: an empty ``reasons``
    array means "no reasons", not "no value".
    """
    import numpy as np
    import pandas as pd

    if isinstance(value, (list, tuple, set, np.ndarray)):
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):  # pragma: no cover - exotic cell types
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _is_float(value: Any) -> bool:
    """Whether one cell is a floating-point number.

    Decided by the **value's** type rather than the column's dtype,
    because an ``object`` column can hold floats — this repository's
    ``adaptation_gate.parquet`` has a ``maximum_interval_width`` column
    of ``None`` and ``float`` — and comparing those exactly would put
    CPU-dependent values back under an exact test.  ``bool`` is excluded
    first: ``numpy.bool_`` and ``bool`` are not measurements.
    """
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (float, np.floating))


def _cell_tokens(value: Any) -> Any:
    """Canonical form of one cell: exact, except floats become tokens.

    Recursive, so a list- or array-valued cell keeps its length, its
    order, and every non-float element exactly.
    """
    import numpy as np

    if _is_missing(value):
        return NULL_TOKEN
    if _is_float(value):
        return _float_token(float(value))
    if isinstance(value, np.ndarray):
        return [_cell_tokens(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_cell_tokens(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compare_cell(left: Any, right: Any, tolerance: NumericTolerance) -> str | None:
    """Whether two cells agree; a message naming the disagreement if not."""
    import numpy as np

    left_missing = _is_missing(left)
    right_missing = _is_missing(right)
    if left_missing or right_missing:
        if left_missing and right_missing:
            return None
        return "missing on one side only"
    if _is_float(left) and _is_float(right):
        if tolerance.holds(float(left), float(right)):
            return None
        return f"{float(left)!r} != {float(right)!r}"
    left_list = isinstance(left, (list, tuple, np.ndarray))
    right_list = isinstance(right, (list, tuple, np.ndarray))
    if left_list or right_list:
        if not (left_list and right_list):
            return "one side is a container and the other is not"
        one = list(left)
        other = list(right)
        if len(one) != len(other):
            return f"length {len(one)} != {len(other)}"
        for index, (a, b) in enumerate(zip(one, other, strict=True)):
            message = _compare_cell(a, b, tolerance)
            if message is not None:
                return f"[{index}] {message}"
        return None
    if _cell_tokens(left) != _cell_tokens(right):
        return f"{left!r} != {right!r}"
    return None


def _column_tokens(series: Any) -> list[Any]:
    """Exact values for every cell, with float values replaced by tokens.

    A float contributes only its *kind* — finite, NaN, or an infinity —
    and never its value.  Everything else contributes itself, so a
    changed label, a reordered row, or a moved NaN still changes the
    digest.
    """
    return [_cell_tokens(value) for value in series.to_list()]


def parquet_structure(path: Path) -> dict[str, Any]:
    """The exact, CPU-independent content of a parquet artifact.

    Column names, column order, dtypes, row count, every non-float value,
    and the null and non-finite positions of every float column.  No
    float value appears.
    """
    frame = _load_frame(path)
    return {
        "columns": [str(name) for name in frame.columns],
        "dtypes": [str(frame[name].dtype) for name in frame.columns],
        "row_count": len(frame),
        "values": {str(name): _column_tokens(frame[name]) for name in frame.columns},
    }


def iter_parquet_floats(path: Path) -> list[float]:
    """Every finite float in a parquet artifact, in canonical order.

    Column order first, then row order, descending into list-valued cells
    — the same traversal :func:`parquet_structure` digests, so a matching
    structure digest makes the sequences positionally comparable.
    """
    frame = _load_frame(path)
    values: list[float] = []

    def walk(cell: Any) -> None:
        import numpy as np

        if _is_missing(cell):
            return
        if _is_float(cell):
            number = float(cell)
            if math.isfinite(number):
                values.append(number)
            return
        if isinstance(cell, np.ndarray):
            for item in cell.tolist():
                walk(item)
            return
        if isinstance(cell, (list, tuple)):
            for item in cell:
                walk(item)

    for name in frame.columns:
        for cell in frame[name].to_list():
            walk(cell)
    return values


def iter_floats(path: Path) -> list[float]:
    """Every finite float in an artifact, in the canonical order."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".json":
        return iter_json_floats(read_json(target))
    if suffix == ".parquet":
        return iter_parquet_floats(target)
    raise NumericContractError(
        f"{target} has suffix {suffix!r}, which the numerical contract cannot read"
    )


def compare_parquet(
    left: Path,
    right: Path,
    *,
    tolerance: NumericTolerance = DEFAULT_TOLERANCE,
) -> list[str]:
    """Differences between two parquet artifacts under the numeric contract.

    Schema, column order, dtypes, row count, row order, every non-float
    column, and every null and non-finite position are compared exactly.
    Only the finite values of a float column are given a tolerance.
    """
    one = _load_frame(left)
    other = _load_frame(right)
    differences: list[str] = []
    if list(one.columns) != list(other.columns):
        return [f"columns {list(one.columns)} != {list(other.columns)}"]
    if len(one) != len(other):
        return [f"row count {len(one)} != {len(other)}"]
    for name in one.columns:
        a = one[name]
        b = other[name]
        if str(a.dtype) != str(b.dtype):
            differences.append(f"{name}: dtype {a.dtype} != {b.dtype}")
            continue
        offenders: list[str] = []
        for index, (x, y) in enumerate(zip(a.to_list(), b.to_list(), strict=True)):
            message = _compare_cell(x, y, tolerance)
            if message is not None:
                offenders.append(f"row {index}: {message}")
        if offenders:
            differences.append(
                f"{name}: {len(offenders)} of {len(a)} cells differ "
                f"({'; '.join(offenders[:3])})"
            )
    return differences


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

#: Suffixes this module knows how to open.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".json", ".parquet")


def is_supported(path: str) -> bool:
    """Whether the numerical contract can read this kind of artifact."""
    return path.lower().endswith(SUPPORTED_SUFFIXES)


def structure_digest(path: Path) -> str:
    """SHA-256 over an artifact's exact, CPU-independent content.

    This is what a deterministic record stores for a CPU-dependent
    numerical artifact, and it is derived **separately** from the raw
    bytes: it never replaces the raw SHA-256, which stays in the
    artifact-integrity sidecar doing the job it has always done.
    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".json":
        return sha256_payload(
            {"kind": "json", "structure": json_structure(read_json(target))}
        )
    if suffix == ".parquet":
        return sha256_payload(
            {"kind": "parquet", "structure": parquet_structure(target)}
        )
    raise NumericContractError(
        f"{target} has suffix {suffix!r}, which the numerical contract cannot "
        f"read; supported: {list(SUPPORTED_SUFFIXES)}. An artifact that cannot "
        "be compared numerically must not be classified cpu_dependent_numeric."
    )


def compare_artifacts(
    left: Path,
    right: Path,
    *,
    tolerance: NumericTolerance = DEFAULT_TOLERANCE,
) -> list[str]:
    """Differences between two artifacts of the same kind."""
    one = Path(left)
    other = Path(right)
    if one.suffix.lower() != other.suffix.lower():
        return [f"kind {one.suffix} != {other.suffix}"]
    if one.suffix.lower() == ".json":
        return compare_json(read_json(one), read_json(other), tolerance=tolerance)
    if one.suffix.lower() == ".parquet":
        return compare_parquet(one, other, tolerance=tolerance)
    raise NumericContractError(
        f"{one} has suffix {one.suffix!r}, which the numerical contract cannot read"
    )


__all__ = [
    "DEFAULT_TOLERANCE",
    "FLOAT_TOKEN",
    "MEASURED_CROSS_KERNEL_DEVIATION",
    "NAN_TOKEN",
    "NEGATIVE_INFINITY_TOKEN",
    "NULL_TOKEN",
    "NUMERIC_PORTABILITY_ATOL",
    "NUMERIC_PORTABILITY_RTOL",
    "POSITIVE_INFINITY_TOKEN",
    "SUPPORTED_SUFFIXES",
    "NumericContractError",
    "NumericTolerance",
    "compare_artifacts",
    "compare_json",
    "compare_parquet",
    "is_supported",
    "iter_floats",
    "iter_json_floats",
    "iter_parquet_floats",
    "json_structure",
    "parquet_structure",
    "read_json",
    "structure_digest",
]
