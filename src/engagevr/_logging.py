"""Structured local logging for EngageVR.

Provides a JSON-formatted or plain-text logger configured from the
project settings.  All log records include an ISO-8601 UTC timestamp.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(
    *,
    level: str = "INFO",
    fmt: str = "json",
    log_file: str | None = None,
) -> logging.Logger:
    """Configure and return the root ``engagevr`` logger.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"DEBUG"``, ``"INFO"``).
    fmt:
        ``"json"`` for structured JSON output, ``"text"`` for human-readable.
    log_file:
        Optional file path.  When *None*, logs go to *stderr* only.
    """
    logger = logging.getLogger("engagevr")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    if fmt == "json":
        formatter: logging.Formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``engagevr`` namespace."""
    base = "engagevr"
    if name:
        return logging.getLogger(f"{base}.{name}")
    return logging.getLogger(base)
