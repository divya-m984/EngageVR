"""MediaPipe FaceLandmarker wrapper.

Uses the MediaPipe Tasks Vision API (FaceLandmarker) in VIDEO mode.
Returns implementation-independent typed results.

Model asset
-----------
- Name:    face_landmarker.task (float16)
- Source:  Google MediaPipe
- License: Apache 2.0
- Setup:   python scripts/download_models.py

Does NOT identify people or classify emotion.
Landmarks are observable geometry, not indicators of mental state.
"""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
import numpy.typing as npt

from engagevr.config import FaceConfig
from engagevr.schemas.capture import LandmarkObservation, NormalizedLandmark

_BaseOptions = mp.tasks.BaseOptions
_FaceLandmarker = mp.tasks.vision.FaceLandmarker
_FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode


class FaceLandmarkerError(Exception):
    """Raised when the FaceLandmarker cannot be initialised."""


class FaceLandmarkerWrapper:
    """Typed wrapper around MediaPipe FaceLandmarker.

    Keeps MediaPipe-specific objects internal.
    """

    def __init__(self, cfg: FaceConfig) -> None:
        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FaceLandmarkerError(
                f"FaceLandmarker model not found at {model_path}. "
                "Run: python scripts/download_models.py"
            )

        options = _FaceLandmarkerOptions(
            base_options=_BaseOptions(
                model_asset_path=str(model_path),
            ),
            running_mode=_RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=cfg.min_detection_confidence,
            min_face_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = _FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(
        self,
        rgb_frame: npt.NDArray[np.uint8],
        timestamp_ms: int,
        *,
        session_id: str = "",
        frame_index: int = 0,
        monotonic_timestamp: float = 0.0,
    ) -> LandmarkObservation:
        """Detect face landmarks in an RGB frame.

        Parameters
        ----------
        rgb_frame:
            RGB uint8 image (H, W, 3).
        timestamp_ms:
            Monotonically increasing timestamp in milliseconds.
            Must be strictly greater than the previous call.
        """
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return LandmarkObservation(
                session_id=session_id,
                frame_index=frame_index,
                monotonic_timestamp=monotonic_timestamp,
                face_detected=False,
            )

        face = result.face_landmarks[0]
        landmarks = [NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face]
        return LandmarkObservation(
            session_id=session_id,
            frame_index=frame_index,
            monotonic_timestamp=monotonic_timestamp,
            face_detected=True,
            landmark_count=len(landmarks),
            landmarks=landmarks,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarkerWrapper:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
