"""Typed failure signalling for the rPPG pipeline.

Signal-processing helpers raise :class:`RppgUnavailable` carrying a
machine-readable :class:`~engagevr.schemas.rppg.UnavailableReason`.  The
orchestration layer catches it and converts it into an ``available=False``
result with that reason attached.

This keeps the numeric helpers total and easy to test in isolation while
guaranteeing that no NaN-filled, zero-filled, or silently-clamped value
ever escapes into a schema.
"""

from __future__ import annotations

from engagevr.schemas.rppg import UnavailableReason


class RppgError(Exception):
    """Base class for rPPG pipeline errors."""


class RppgUnavailable(RppgError):
    """Raised when an rPPG result cannot be produced.

    Parameters
    ----------
    reason:
        Machine-readable cause, surfaced in the resulting schema.
    detail:
        Optional human-readable elaboration for logs and warnings.
    """

    def __init__(self, reason: UnavailableReason, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        message = f"{reason.value}: {detail}" if detail else reason.value
        super().__init__(message)
