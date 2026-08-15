"""Deterministic fixed-duration windowing over a session.

Boundaries are half-open, ``[start, end)``.  A sample exactly on a
boundary belongs to the later window and to that window only, so no
observation is counted twice and none falls through the gap between
consecutive windows.

Two properties matter more than convenience here:

- **Determinism.**  Window ``k`` starts at ``session_start + k * step``,
  computed from the session start every time rather than accumulated, so
  a long session does not drift and two builds of the same session
  produce byte-identical boundaries.
- **Containment.**  Every window is verified to lie inside its source
  session.  A window that extends past the recorded end of a session
  would summarise a period for which no evidence exists.

Nothing here looks forward.  A window's evidence is drawn only from
``[start, end)``; :func:`select_in_window` is the single selection
primitive, and it cannot return an item outside those bounds.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Self

from pydantic import BaseModel, Field, model_validator

#: Tolerance for float second arithmetic on window boundaries.
_EPSILON_SECONDS = 1e-9


class WindowingError(ValueError):
    """A session or window specification is not analysable."""


class PartialWindowPolicy(enum.StrEnum):
    """What to do with the trailing fragment of a session.

    ``DROP`` is the default.  A shorter window is summarised from less
    evidence than every other row, and mixing the two silently changes
    what a feature means from row to row.
    """

    DROP = "drop"
    KEEP_IF_MINIMUM = "keep_if_minimum"


class WindowSpec(BaseModel):
    """Window geometry shared by every session in a dataset build."""

    model_config = {"frozen": True, "extra": "forbid"}

    duration_seconds: float = Field(gt=0.0)
    step_seconds: float = Field(gt=0.0)
    partial_window_policy: PartialWindowPolicy = PartialWindowPolicy.DROP
    minimum_partial_duration_seconds: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.step_seconds > self.duration_seconds:
            raise ValueError(
                "window step_seconds must not exceed duration_seconds, "
                "otherwise evidence between windows is silently discarded"
            )
        if self.partial_window_policy is PartialWindowPolicy.KEEP_IF_MINIMUM:
            if self.minimum_partial_duration_seconds <= 0.0:
                raise ValueError(
                    "keep_if_minimum requires a positive "
                    "minimum_partial_duration_seconds"
                )
            if self.minimum_partial_duration_seconds > self.duration_seconds:
                raise ValueError(
                    "minimum_partial_duration_seconds must not exceed duration_seconds"
                )
        return self

    @property
    def overlapping(self) -> bool:
        """Whether consecutive windows share evidence."""
        return self.step_seconds < self.duration_seconds - _EPSILON_SECONDS


class WindowBounds(BaseModel):
    """One window's boundaries on both the wall clock and the monotonic clock."""

    model_config = {"frozen": True, "extra": "forbid"}

    index: int = Field(ge=0)
    start_utc: datetime
    end_utc: datetime
    start_monotonic_seconds: float | None = None
    end_monotonic_seconds: float | None = None
    duration_seconds: float = Field(gt=0.0)
    is_partial: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.end_utc <= self.start_utc:
            raise ValueError(
                f"window {self.index}: end_utc must be strictly after start_utc"
            )
        start = self.start_monotonic_seconds
        end = self.end_monotonic_seconds
        if (start is None) != (end is None):
            raise ValueError(
                f"window {self.index}: monotonic bounds must be supplied "
                "together or not at all"
            )
        if start is not None and end is not None and end <= start:
            raise ValueError(
                f"window {self.index}: monotonic end must be strictly after "
                "monotonic start"
            )
        return self

    def contains_utc(self, moment: datetime) -> bool:
        """Whether ``moment`` falls in ``[start_utc, end_utc)``."""
        return self.start_utc <= moment < self.end_utc

    def contains_monotonic(self, seconds: float) -> bool:
        """Whether ``seconds`` falls in the half-open monotonic interval."""
        if self.start_monotonic_seconds is None:
            raise WindowingError(
                f"window {self.index} has no monotonic bounds to test against"
            )
        end = self.end_monotonic_seconds
        assert end is not None  # guaranteed by the model validator
        return self.start_monotonic_seconds <= seconds < end


class SessionInterval(BaseModel):
    """The analysable extent of one session on both clocks."""

    model_config = {"frozen": True, "extra": "forbid"}

    session_id: str = Field(min_length=1)
    start_utc: datetime
    end_utc: datetime
    start_monotonic_seconds: float | None = None
    end_monotonic_seconds: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name, moment in (
            ("start_utc", self.start_utc),
            ("end_utc", self.end_utc),
        ):
            if moment.tzinfo is None:
                raise ValueError(
                    f"session {self.session_id!r}: {name} must be timezone-aware "
                    "UTC; a naive timestamp cannot be compared across sessions"
                )
        if self.end_utc <= self.start_utc:
            raise ValueError(
                f"session {self.session_id!r}: end_utc must be strictly after "
                "start_utc (reversed or zero-length sessions are not analysable)"
            )
        start = self.start_monotonic_seconds
        end = self.end_monotonic_seconds
        if (start is None) != (end is None):
            raise ValueError(
                f"session {self.session_id!r}: monotonic bounds must be supplied "
                "together or not at all"
            )
        if start is not None and end is not None and end <= start:
            raise ValueError(
                f"session {self.session_id!r}: monotonic end must be strictly "
                "after monotonic start"
            )
        return self

    @property
    def duration_seconds(self) -> float:
        """Session length on the wall clock, in seconds."""
        return (self.end_utc - self.start_utc).total_seconds()


def build_windows(
    interval: SessionInterval,
    spec: WindowSpec,
) -> tuple[WindowBounds, ...]:
    """Return the deterministic window boundaries for one session.

    Windows are generated from the session start, not accumulated, so
    boundary ``k`` is exactly ``start + k * step`` regardless of session
    length.

    Raises
    ------
    WindowingError
        If the session is shorter than one window and the partial-window
        policy does not permit a shorter row.
    """
    total = interval.duration_seconds
    monotonic_start = interval.start_monotonic_seconds

    bounds: list[WindowBounds] = []
    if total + _EPSILON_SECONDS >= spec.duration_seconds:
        full_count = (
            math.floor(
                (total - spec.duration_seconds + _EPSILON_SECONDS) / spec.step_seconds
            )
            + 1
        )
    else:
        full_count = 0

    for index in range(full_count):
        offset = index * spec.step_seconds
        bounds.append(
            _make_bounds(
                interval,
                index=index,
                offset_seconds=offset,
                duration_seconds=spec.duration_seconds,
                monotonic_start=monotonic_start,
                is_partial=False,
            )
        )

    consumed = full_count * spec.step_seconds
    remainder = total - consumed
    if (
        spec.partial_window_policy is PartialWindowPolicy.KEEP_IF_MINIMUM
        and remainder > _EPSILON_SECONDS
        and remainder + _EPSILON_SECONDS >= spec.minimum_partial_duration_seconds
        and remainder < spec.duration_seconds - _EPSILON_SECONDS
    ):
        bounds.append(
            _make_bounds(
                interval,
                index=full_count,
                offset_seconds=consumed,
                duration_seconds=remainder,
                monotonic_start=monotonic_start,
                is_partial=True,
            )
        )

    if not bounds:
        raise WindowingError(
            f"session {interval.session_id!r} is {total:.3f} s long, which is "
            f"shorter than the {spec.duration_seconds:.3f} s window and the "
            f"partial-window policy is {spec.partial_window_policy.value!r}; "
            "no window can be built"
        )

    assert_windows_within_session(bounds, interval)
    return tuple(bounds)


def _make_bounds(
    interval: SessionInterval,
    *,
    index: int,
    offset_seconds: float,
    duration_seconds: float,
    monotonic_start: float | None,
    is_partial: bool,
) -> WindowBounds:
    start_utc = interval.start_utc + timedelta(seconds=offset_seconds)
    end_utc = start_utc + timedelta(seconds=duration_seconds)
    start_monotonic: float | None = None
    end_monotonic: float | None = None
    if monotonic_start is not None:
        start_monotonic = monotonic_start + offset_seconds
        end_monotonic = start_monotonic + duration_seconds
    return WindowBounds(
        index=index,
        start_utc=start_utc,
        end_utc=end_utc,
        start_monotonic_seconds=start_monotonic,
        end_monotonic_seconds=end_monotonic,
        duration_seconds=duration_seconds,
        is_partial=is_partial,
    )


def assert_windows_within_session(
    windows: Sequence[WindowBounds],
    interval: SessionInterval,
) -> None:
    """Verify every window lies inside its source session.

    Raises
    ------
    WindowingError
        On the first window that starts before the session or ends after it.
    """
    tolerance = timedelta(seconds=_EPSILON_SECONDS)
    for window in windows:
        if window.start_utc + tolerance < interval.start_utc:
            raise WindowingError(
                f"session {interval.session_id!r}: window {window.index} starts "
                f"at {window.start_utc.isoformat()}, before the session start "
                f"{interval.start_utc.isoformat()}"
            )
        if window.end_utc > interval.end_utc + tolerance:
            raise WindowingError(
                f"session {interval.session_id!r}: window {window.index} ends at "
                f"{window.end_utc.isoformat()}, after the session end "
                f"{interval.end_utc.isoformat()}; a window may not summarise a "
                "period for which the session holds no evidence"
            )


def select_in_window[T](
    items: Iterable[T],
    window: WindowBounds,
    *,
    timestamp: Callable[[T], float],
) -> tuple[T, ...]:
    """Items whose monotonic timestamp lies in ``[start, end)``.

    This is the only way evidence enters an aggregation, which is what
    makes "no future events" a structural property rather than a habit:
    an item at or after ``end`` is not returned, so it cannot influence
    the window's features.
    """
    return tuple(item for item in items if window.contains_monotonic(timestamp(item)))


def select_in_window_utc[T](
    items: Iterable[T],
    window: WindowBounds,
    *,
    timestamp: Callable[[T], datetime],
) -> tuple[T, ...]:
    """Items whose UTC timestamp lies in ``[start_utc, end_utc)``."""
    return tuple(item for item in items if window.contains_utc(timestamp(item)))


def window_id(session_id: str, index: int) -> str:
    """Deterministic identifier for window ``index`` of ``session_id``."""
    return f"{session_id}:w{index:06d}"


def utc_now() -> datetime:
    """Current UTC instant.

    Isolated behind a function so that everything which must be
    deterministic can avoid it and everything that legitimately records a
    creation time uses one implementation.
    """
    return datetime.now(UTC)
