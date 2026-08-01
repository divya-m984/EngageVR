"""Head-motion features: angular velocity and variability."""

from __future__ import annotations

import math
from collections import deque


class HeadMotionTracker:
    """Track head angular velocity and variability over a window.

    Angular velocity is computed as the Euclidean distance between
    consecutive (yaw, pitch, roll) readings divided by the time delta.

    Variability is the standard deviation of angular velocity over a
    configurable time window.
    """

    def __init__(self, *, window_seconds: float = 1.0) -> None:
        self._window_seconds = window_seconds
        self._prev_angles: tuple[float, float, float] | None = None
        self._prev_time: float | None = None
        self._velocities: deque[tuple[float, float]] = deque()

    def update(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        monotonic_ts: float,
    ) -> tuple[float | None, float | None]:
        """Update with new head-pose reading.

        Returns
        -------
        (angular_velocity_deg_s, motion_variability_deg_s)
        Both are None on the first frame.
        """
        if self._prev_angles is None or self._prev_time is None:
            self._prev_angles = (yaw, pitch, roll)
            self._prev_time = monotonic_ts
            return None, None

        dt = monotonic_ts - self._prev_time
        if dt <= 0:
            return None, None

        dy = yaw - self._prev_angles[0]
        dp = pitch - self._prev_angles[1]
        dr = roll - self._prev_angles[2]
        angular_dist = math.sqrt(dy**2 + dp**2 + dr**2)
        velocity = angular_dist / dt

        self._prev_angles = (yaw, pitch, roll)
        self._prev_time = monotonic_ts

        self._velocities.append((monotonic_ts, velocity))

        # Trim window
        cutoff = monotonic_ts - self._window_seconds
        while self._velocities and self._velocities[0][0] < cutoff:
            self._velocities.popleft()

        variability = self._compute_variability()
        return velocity, variability

    def _compute_variability(self) -> float | None:
        """Std dev of angular velocity in the current window."""
        if len(self._velocities) < 2:
            return None
        vals = [v for _, v in self._velocities]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        return math.sqrt(variance)

    def reset(self) -> None:
        self._prev_angles = None
        self._prev_time = None
        self._velocities.clear()
