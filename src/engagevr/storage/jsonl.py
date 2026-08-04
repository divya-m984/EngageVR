"""Append-only JSON Lines reading and writing.

Only the Python standard library is used.  A JSONL event stream is
append-only by construction, which is what makes an interrupted session
recoverable: every line that was flushed before the interruption is a
complete, independently parseable record, and a torn final line affects
only itself.

Read errors are reported with a 1-based line number and the offending
text, because "the recording is corrupt" is not an actionable message.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

#: How much of an offending line is echoed back in an error message.
_EXCERPT_LIMIT = 200


class JsonlFormatError(ValueError):
    """A line in a JSONL file could not be parsed.

    Attributes
    ----------
    path:
        File the error came from.
    line_number:
        1-based line number, so it matches what an editor shows.
    excerpt:
        The start of the offending line.
    """

    def __init__(self, path: Path, line_number: int, detail: str, excerpt: str) -> None:
        super().__init__(f"{path}:{line_number}: {detail}: {excerpt!r}")
        self.path = path
        self.line_number = line_number
        self.detail = detail
        self.excerpt = excerpt


@dataclass(frozen=True, slots=True)
class JsonlLine:
    """One successfully parsed line, with its position in the file."""

    line_number: int
    record: dict[str, Any]


class JsonlWriter:
    """Append-only JSONL writer with a predictable flush policy.

    Records are flushed to the OS every ``flush_every`` writes, and on
    close.  A caller that needs a record durable immediately can set
    ``flush_every=1``; the trade-off is one write syscall per record.

    ``fsync`` is *not* called per record.  That is a deliberate choice:
    a per-record fsync would dominate the cost of a high-rate task
    stream, and the recovery guarantee this store offers is "every line
    the OS accepted is readable", not "every line survives a power
    cut". :meth:`sync` is available for callers that need the stronger
    guarantee at a checkpoint.
    """

    def __init__(self, path: Path, *, flush_every: int = 1) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self._path = path
        self._flush_every = flush_every
        self._since_flush = 0
        self._count = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def record_count(self) -> int:
        """How many records this writer has written."""
        return self._count

    def write(self, record: dict[str, Any]) -> None:
        """Append one record as a single line of compact JSON."""
        line = json.dumps(record, separators=(",", ":"), sort_keys=False)
        if "\n" in line:  # pragma: no cover - json.dumps never emits raw newlines
            raise ValueError("encoded record contains a newline")
        self._handle.write(line + "\n")
        self._count += 1
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        """Flush buffered records to the operating system."""
        self._handle.flush()
        self._since_flush = 0

    def sync(self) -> None:
        """Flush and force the OS to write the file to storage."""
        self.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        """Flush and close the underlying file."""
        if not self._handle.closed:
            self.flush()
            self._handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_jsonl(path: Path, *, skip_blank: bool = True) -> Iterator[JsonlLine]:
    """Yield each line of a JSONL file as a parsed object.

    Raises
    ------
    JsonlFormatError
        On the first line that is not a JSON object, reporting its
        1-based line number.
    FileNotFoundError
        If ``path`` does not exist.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                if skip_blank:
                    continue
                raise JsonlFormatError(path, line_number, "line is empty", "")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise JsonlFormatError(
                    path,
                    line_number,
                    f"invalid JSON ({exc.msg})",
                    text[:_EXCERPT_LIMIT],
                ) from exc
            if not isinstance(parsed, dict):
                raise JsonlFormatError(
                    path,
                    line_number,
                    f"expected a JSON object, got {type(parsed).__name__}",
                    text[:_EXCERPT_LIMIT],
                )
            yield JsonlLine(line_number=line_number, record=parsed)


def read_jsonl_tolerant(path: Path) -> tuple[list[JsonlLine], list[int]]:
    """Read a JSONL file, collecting malformed line numbers instead of raising.

    Used for recovering an interrupted session, where the final line may
    be torn.  Returns the successfully parsed lines and the 1-based line
    numbers that could not be parsed.
    """
    good: list[JsonlLine] = []
    bad: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                bad.append(line_number)
                continue
            if not isinstance(parsed, dict):
                bad.append(line_number)
                continue
            good.append(JsonlLine(line_number=line_number, record=parsed))
    return good, bad


def write_json_atomic(path: Path, data: dict[str, Any]) -> Path:
    """Write a JSON document atomically.

    The document is written to a temporary file in the same directory,
    flushed and fsynced, then renamed over the target.  ``os.replace``
    is atomic within a filesystem, so a reader sees either the previous
    document or the complete new one, never a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(data, indent=2, sort_keys=False, default=str) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path
