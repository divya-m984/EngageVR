"""Webcam acquisition with dependency-injectable backends.

Privacy: raw video storage is DISABLED by default.
Frames are processed in memory and only extracted features are persisted.
Frames never leave the local process.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from engagevr.schemas.capture import FrameMetadata


class CaptureBackend(Protocol):
    """Protocol for capture backends (real or fake)."""

    def open(self, index: int, width: int, height: int, fps: int) -> bool: ...
    def read(self) -> tuple[bool, npt.NDArray[np.uint8] | None]: ...
    def release(self) -> None: ...
    def is_opened(self) -> bool: ...
    def get_actual_fps(self) -> float | None: ...
    def get_actual_width(self) -> int | None: ...
    def get_actual_height(self) -> int | None: ...


class OpenCVBackend:
    """Real OpenCV webcam backend."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None

    def open(self, index: int, width: int, height: int, fps: int) -> bool:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        return True

    def read(self) -> tuple[bool, npt.NDArray[np.uint8] | None]:
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        arr: npt.NDArray[np.uint8] = np.asarray(frame, dtype=np.uint8)
        return True, arr

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_actual_fps(self) -> float | None:
        if self._cap is None:
            return None
        val = self._cap.get(cv2.CAP_PROP_FPS)
        return float(val) if val > 0 else None

    def get_actual_width(self) -> int | None:
        if self._cap is None:
            return None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_actual_height(self) -> int | None:
        if self._cap is None:
            return None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


class FakeCaptureBackend:
    """Deterministic fake backend for testing without a webcam."""

    def __init__(
        self,
        frames: list[npt.NDArray[np.uint8]] | None = None,
        *,
        fail_open: bool = False,
        fail_after: int | None = None,
    ) -> None:
        self._frames = frames or []
        self._fail_open = fail_open
        self._fail_after = fail_after
        self._index = 0
        self._opened = False
        self._width = 640
        self._height = 480

    def open(self, index: int, width: int, height: int, fps: int) -> bool:
        if self._fail_open:
            return False
        self._opened = True
        self._width = width
        self._height = height
        return True

    def read(self) -> tuple[bool, npt.NDArray[np.uint8] | None]:
        if not self._opened:
            return False, None
        if self._fail_after is not None and self._index >= self._fail_after:
            return False, None
        if self._index < len(self._frames):
            frame = self._frames[self._index]
        else:
            frame = np.full((self._height, self._width, 3), 128, dtype=np.uint8)
        self._index += 1
        return True, frame

    def release(self) -> None:
        self._opened = False

    def is_opened(self) -> bool:
        return self._opened

    def get_actual_fps(self) -> float | None:
        return 30.0

    def get_actual_width(self) -> int | None:
        return self._width if self._opened else None

    def get_actual_height(self) -> int | None:
        return self._height if self._opened else None


class WebcamCapture:
    """Webcam capture with timestamping and resource management."""

    def __init__(
        self,
        backend: CaptureBackend | None = None,
        *,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        session_id: str = "",
    ) -> None:
        self._backend = backend or OpenCVBackend()
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._fps = fps
        self._session_id = session_id
        self._frame_index = 0
        self._opened = False

    def open(self) -> bool:
        self._opened = self._backend.open(
            self._camera_index, self._width, self._height, self._fps
        )
        self._frame_index = 0
        return self._opened

    def read_frame(
        self,
    ) -> tuple[FrameMetadata | None, npt.NDArray[np.uint8] | None]:
        """Read one frame with timestamps.

        Returns (metadata, bgr_frame) or (None, None) on failure.
        """
        if not self._opened:
            return None, None

        mono_ts = time.monotonic()
        utc_ts = datetime.now(UTC)
        ok, frame = self._backend.read()

        if not ok or frame is None:
            return None, None

        h, w = frame.shape[:2]
        meta = FrameMetadata(
            session_id=self._session_id,
            frame_index=self._frame_index,
            monotonic_timestamp=mono_ts,
            utc_timestamp=utc_ts,
            width=w,
            height=h,
        )
        self._frame_index += 1
        return meta, frame

    def release(self) -> None:
        self._backend.release()
        self._opened = False

    def is_opened(self) -> bool:
        return self._opened and self._backend.is_opened()

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def actual_fps(self) -> float | None:
        return self._backend.get_actual_fps()

    def actual_width(self) -> int | None:
        return self._backend.get_actual_width()

    def actual_height(self) -> int | None:
        return self._backend.get_actual_height()

    def __enter__(self) -> WebcamCapture:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
