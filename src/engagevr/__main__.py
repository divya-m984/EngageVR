"""CLI entry point for EngageVR.

Usage::

    uv run python -m engagevr demo --seed 42 --output artifacts/demo-session.json
    uv run python -m engagevr capture --camera 0 --duration 30

All outputs from the ``demo`` command are deterministic SYNTHETIC data.
Capture outputs are behavioural proxies, NOT engagement, psychological,
clinical, or diagnostic conclusions.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from engagevr.simulator.synthetic import generate_synthetic_session


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engagevr",
        description="EngageVR research prototype CLI.",
    )
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser(
        "demo",
        help="Generate a deterministic SYNTHETIC session.",
    )
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--trials", type=int, default=10)
    demo.add_argument(
        "--output",
        type=str,
        default="artifacts/demo-session.json",
    )

    capture = sub.add_parser(
        "capture",
        help="Capture webcam behavioural features (proxies only).",
    )
    capture.add_argument("--camera", type=int, default=0)
    capture.add_argument("--duration", type=float, default=30.0)
    capture.add_argument(
        "--output",
        type=str,
        default="artifacts/webcam-session.json",
    )
    capture.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview window (press q/Esc to quit).",
    )
    return parser


def _run_demo(args: argparse.Namespace) -> int:
    session = generate_synthetic_session(
        seed=args.seed,
        n_trials=args.trials,
    )
    data = session.to_dict()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2) + "\n")

    n_events = len(session.events)
    n_predictions = len(session.predictions)
    n_abstentions = sum(1 for p in session.predictions if p.abstain)

    print(f"Session ID:       {session.session.session_id}")
    print(f"Events:           {n_events}")
    print(f"Predictions:      {n_predictions}")
    print(f"Abstentions:      {n_abstentions}")
    print(f"Output:           {out_path}")
    print("Data source:      SYNTHETIC")
    print()
    print(
        "This is SYNTHETIC data for software testing only. "
        "It is not experimental evidence."
    )
    return 0


def _run_capture(args: argparse.Namespace) -> int:
    # Lazy imports to avoid loading heavy deps for other commands
    from engagevr.capture.frame import bgr_to_gray, bgr_to_rgb
    from engagevr.capture.quality import (
        assess_blur,
        assess_brightness,
        assess_motion,
        compute_blur_score,
        compute_brightness,
        compute_motion_score,
    )
    from engagevr.capture.webcam import WebcamCapture
    from engagevr.config import load_config
    from engagevr.face.features import (
        BlinkTracker,
        compute_mean_ear,
        compute_mouth_aspect_ratio,
    )
    from engagevr.face.landmarker import FaceLandmarkerError, FaceLandmarkerWrapper
    from engagevr.head_pose.estimator import estimate_head_pose
    from engagevr.head_pose.features import HeadMotionTracker
    from engagevr.schemas.session import Session

    cfg = load_config()

    # Check model availability
    try:
        landmarker = FaceLandmarkerWrapper(cfg.face)
    except FaceLandmarkerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    session = Session(participant_id="capture_user")
    cam = WebcamCapture(
        camera_index=args.camera,
        width=cfg.capture.width,
        height=cfg.capture.height,
        fps=cfg.capture.webcam_fps_target,
        session_id=session.session_id,
    )

    if not cam.open():
        landmarker.close()
        print(
            f"Error: cannot open camera {args.camera}.",
            file=sys.stderr,
        )
        return 1

    blink_tracker = BlinkTracker(
        ear_threshold=cfg.face.blink_ear_threshold,
        min_frames=cfg.face.eye_closure_min_frames,
        fps=float(cfg.capture.webcam_fps_target),
    )
    motion_tracker = HeadMotionTracker(
        window_seconds=cfg.head_pose.velocity_window_seconds,
    )

    interrupted = False

    def _handle_sigint(_sig: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _handle_sigint)

    results: list[dict[str, object]] = []
    start = time.monotonic()
    prev_gray = None
    face_count = 0
    dropped = 0

    print(
        "Capturing behavioural proxies. "
        "These are NOT engagement or diagnostic conclusions."
    )
    print(f"Duration: {args.duration}s | Camera: {args.camera}")
    if args.preview:
        print("Preview: press q or Esc to quit.")
    print()

    try:
        while not interrupted:
            elapsed = time.monotonic() - start
            if elapsed >= args.duration:
                break

            meta, frame = cam.read_frame()
            if meta is None or frame is None:
                dropped += 1
                continue

            gray = bgr_to_gray(frame)
            rgb = bgr_to_rgb(frame)

            brightness = compute_brightness(gray)
            blur = compute_blur_score(gray)
            motion = compute_motion_score(prev_gray, gray)
            prev_gray = gray

            under, over = assess_brightness(brightness, cfg.quality)
            blurry = assess_blur(blur, cfg.quality)
            excess_motion = assess_motion(motion, cfg.quality)

            ts_ms = int(meta.monotonic_timestamp * 1000)
            obs = landmarker.detect(
                rgb,
                ts_ms,
                session_id=session.session_id,
                frame_index=meta.frame_index,
                monotonic_timestamp=meta.monotonic_timestamp,
            )

            ear = None
            mar = None
            blink = None
            closure = None
            pose: tuple[float, float, float] | None = None
            vel = None

            if obs.face_detected and obs.landmarks:
                face_count += 1
                ear = compute_mean_ear(obs.landmarks)
                mar = compute_mouth_aspect_ratio(obs.landmarks)
                blink, closure = blink_tracker.update(ear)
                pose = estimate_head_pose(obs.landmarks, meta.width, meta.height)
                if pose:
                    vel, _var = motion_tracker.update(
                        pose[0],
                        pose[1],
                        pose[2],
                        meta.monotonic_timestamp,
                    )

            record: dict[str, object] = {
                "frame_index": meta.frame_index,
                "monotonic_timestamp": meta.monotonic_timestamp,
                "face_detected": obs.face_detected,
                "mean_ear": round(ear, 4) if ear else None,
                "blink_detected": blink,
                "eye_closure_s": (round(closure, 3) if closure else None),
                "mouth_ar": round(mar, 4) if mar else None,
                "yaw_deg": round(pose[0], 2) if pose else None,
                "pitch_deg": round(pose[1], 2) if pose else None,
                "roll_deg": round(pose[2], 2) if pose else None,
                "angular_velocity": (round(vel, 2) if vel is not None else None),
                "brightness": round(brightness, 1),
                "blur_score": round(blur, 1),
                "underexposed": under,
                "overexposed": over,
                "is_blurry": blurry,
                "excessive_motion": excess_motion,
            }
            results.append(record)

            # Live status every 30 frames
            if meta.frame_index % 30 == 0 and meta.frame_index > 0:
                fps_est = meta.frame_index / elapsed if elapsed > 0 else 0
                face_str = "YES" if obs.face_detected else "NO"
                warn = ""
                if under:
                    warn += " [UNDEREXPOSED]"
                if over:
                    warn += " [OVEREXPOSED]"
                if blurry:
                    warn += " [BLURRY]"
                print(
                    f"  frame={meta.frame_index} "
                    f"fps={fps_est:.1f} "
                    f"face={face_str} "
                    f"EAR={ear:.3f if ear else 'N/A'}"
                    f"{warn}"
                )

            if args.preview:
                import cv2

                cv2.imshow("EngageVR Capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    finally:
        cam.release()
        landmarker.close()
        if args.preview:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = cam.frame_index
    output = {
        "session": session.model_dump(mode="json"),
        "summary": {
            "total_frames": total,
            "dropped_frames": dropped,
            "face_present_frames": face_count,
            "face_present_pct": (
                round(100.0 * face_count / total, 1) if total > 0 else 0
            ),
            "data_source": "live",
        },
        "frames": results,
        "_disclaimer": (
            "Behavioural proxies only. NOT engagement, psychological, "
            "clinical, or diagnostic conclusions."
        ),
    }
    out_path.write_text(json.dumps(output, indent=2, default=str) + "\n")

    print()
    print(f"Frames captured:  {total}")
    print(f"Dropped frames:   {dropped}")
    print(f"Face present:     {face_count}/{total}")
    print(f"Output:           {out_path}")
    print()
    print(
        "These are behavioural proxies only. NOT engagement or diagnostic conclusions."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        return _run_demo(args)
    if args.command == "capture":
        return _run_capture(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
