"""Replay pacing.

Three modes, all producing the **same ordered messages** and differing
only in how long the player waits between them:

``immediate`` (``speed == 0``)
    No sleeping at all.  Deterministic and instant; this is what tests
    use.

``original`` (``speed == 1``)
    Waits the recorded gap between consecutive messages.

``accelerated`` (``speed > 1``, or any other positive speed)
    Waits the recorded gap divided by ``speed``.

Which recorded timestamp defines "the gap" matters.  The player uses
``ingestion.server_received_at_utc`` — the *receiver's* clock — because
it is the only timeline on which messages from different sources are
comparable.  Using each sender's own ``sent_at_utc`` would mix
independent clocks and could produce negative gaps.  A negative gap
from a clock adjustment is clamped to zero rather than rewinding.
"""

from __future__ import annotations

import asyncio
import enum
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


class ReplayMode(enum.StrEnum):
    """How replay relates to wall-clock time."""

    IMMEDIATE = "immediate"
    ORIGINAL = "original"
    ACCELERATED = "accelerated"


class InvalidReplaySpeedError(ValueError):
    """A replay speed was negative, non-finite, or above the maximum."""


def validate_speed(speed: float, *, maximum_speed: float) -> float:
    """Return ``speed`` if it is usable, else raise.

    Raises
    ------
    InvalidReplaySpeedError
        On a negative, NaN, infinite, or too-large speed.  Zero is
        valid and means immediate.
    """
    if speed != speed:  # NaN
        raise InvalidReplaySpeedError("replay speed must not be NaN")
    if speed in (float("inf"), float("-inf")):
        raise InvalidReplaySpeedError(
            "replay speed must be finite; use --speed 0 for immediate replay"
        )
    if speed < 0:
        raise InvalidReplaySpeedError(
            f"replay speed must not be negative, got {speed}; "
            "0 means immediate, 1 means original timing"
        )
    if speed > maximum_speed:
        raise InvalidReplaySpeedError(
            f"replay speed {speed} exceeds replay.maximum_speed {maximum_speed}"
        )
    return speed


def mode_for_speed(speed: float) -> ReplayMode:
    """Which pacing mode a validated speed selects."""
    if speed == 0.0:
        return ReplayMode.IMMEDIATE
    if speed == 1.0:
        return ReplayMode.ORIGINAL
    return ReplayMode.ACCELERATED


@dataclass
class ReplayPacer:
    """Turns recorded inter-message gaps into waits.

    ``sleep`` is injected so a test can replay original timing without
    real time passing, and assert on the exact sequence of requested
    delays.
    """

    speed: float
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    preserve_original_timing: bool = True
    fixed_interval_seconds: float = 0.0

    #: Every delay this pacer has requested, in order. Useful in tests.
    requested_delays: list[float] | None = None

    @property
    def mode(self) -> ReplayMode:
        return mode_for_speed(self.speed)

    def delay_for(self, recorded_gap_seconds: float) -> float:
        """The wait this pacer would apply for a recorded gap."""
        if self.mode is ReplayMode.IMMEDIATE:
            return 0.0
        if not self.preserve_original_timing:
            return self.fixed_interval_seconds / self.speed
        gap = max(recorded_gap_seconds, 0.0)
        return gap / self.speed

    async def wait(self, recorded_gap_seconds: float) -> None:
        """Wait for one recorded gap, scaled by the speed."""
        delay = self.delay_for(recorded_gap_seconds)
        if self.requested_delays is not None:
            self.requested_delays.append(delay)
        if delay > 0.0:
            await self.sleep(delay)
