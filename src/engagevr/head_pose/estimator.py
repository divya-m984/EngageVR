"""Head-pose estimation from facial landmarks using PnP.

Method
------
Uses cv2.solvePnP with a generic 3D face model and six corresponding
2D landmark projections to estimate yaw, pitch, and roll.

This is an approximation using a canonical face geometry.
The model is not calibrated per individual.

MediaPipe landmark indices used:
- Nose tip:   1
- Chin:       152
- Left eye outer corner:  263
- Right eye outer corner: 33
- Left mouth corner:  61
- Right mouth corner: 291

Reference: OpenCV Perspective-n-Point (PnP) algorithm.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import numpy.typing as npt

from engagevr.schemas.capture import NormalizedLandmark

# 3D model points in a canonical coordinate system (arbitrary scale)
_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],  # Nose tip
        [0.0, -330.0, -65.0],  # Chin
        [-225.0, 170.0, -135.0],  # Left eye outer
        [225.0, 170.0, -135.0],  # Right eye outer
        [-150.0, -150.0, -125.0],  # Left mouth
        [150.0, -150.0, -125.0],  # Right mouth
    ],
    dtype=np.float64,
)

# Corresponding MediaPipe landmark indices
_LANDMARK_INDICES = [1, 152, 263, 33, 61, 291]

_MIN_LANDMARKS = 468


def estimate_head_pose(
    landmarks: list[NormalizedLandmark],
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float] | None:
    """Estimate yaw, pitch, roll from facial landmarks.

    Parameters
    ----------
    landmarks:
        Full set of MediaPipe normalized landmarks (>=468).
    frame_width, frame_height:
        Pixel dimensions for de-normalizing landmark coordinates.

    Returns
    -------
    (yaw_deg, pitch_deg, roll_deg) or None if estimation fails.
    Yaw: + = turning right. Pitch: + = looking up. Roll: + = CW tilt.
    """
    if len(landmarks) < _MIN_LANDMARKS:
        return None

    image_points = np.zeros((6, 2), dtype=np.float64)
    for i, idx in enumerate(_LANDMARK_INDICES):
        lm = landmarks[idx]
        image_points[i] = [lm.x * frame_width, lm.y * frame_height]

    focal_length = float(frame_width)
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    camera_matrix = np.array(
        [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs: npt.NDArray[np.float64] = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _tvec = cv2.solvePnP(
        _MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    return _rotation_matrix_to_euler(np.asarray(rmat, dtype=np.float64))


def _rotation_matrix_to_euler(
    rmat: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Convert 3x3 rotation matrix to (yaw, pitch, roll) in degrees."""
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = math.atan2(rmat[1, 0], rmat[0, 0])
        roll = math.atan2(rmat[2, 1], rmat[2, 2])
    else:
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = math.atan2(-rmat[1, 2], rmat[1, 1])
        roll = 0.0

    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )
