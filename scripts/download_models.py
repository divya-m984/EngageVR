#!/usr/bin/env python3
"""Download required model assets for EngageVR.

FaceLandmarker model
--------------------
- Name:    face_landmarker.task (float16)
- Source:  Google MediaPipe model garden
- URL:     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- License: Apache 2.0
- Dest:    models/face_landmarker.task

Usage::

    python scripts/download_models.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

_FACE_LANDMARKER = {
    "name": "face_landmarker.task",
    "url": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/"
        "face_landmarker.task"
    ),
    "license": "Apache-2.0",
    "source": "Google MediaPipe",
}


def download_face_landmarker() -> Path:
    """Download the FaceLandmarker model if not already present."""
    dest = _MODELS_DIR / _FACE_LANDMARKER["name"]
    if dest.exists():
        print(f"Model already exists: {dest}")
        return dest

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = _FACE_LANDMARKER["url"]
    print(f"Downloading {_FACE_LANDMARKER['name']} ...")
    print(f"  Source:  {_FACE_LANDMARKER['source']}")
    print(f"  License: {_FACE_LANDMARKER['license']}")
    print(f"  URL:     {url}")

    urllib.request.urlretrieve(url, dest)

    size_mb = dest.stat().st_size / (1024 * 1024)
    sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"  Saved:   {dest} ({size_mb:.1f} MB)")
    print(f"  SHA-256: {sha256}")
    return dest


def main() -> int:
    try:
        download_face_landmarker()
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
