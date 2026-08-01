"""Frame conversion utilities."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt


def bgr_to_rgb(frame: npt.NDArray[np.uint8]) -> npt.NDArray[Any]:
    """Convert a BGR frame (OpenCV default) to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def bgr_to_gray(frame: npt.NDArray[np.uint8]) -> npt.NDArray[Any]:
    """Convert a BGR frame to single-channel grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
