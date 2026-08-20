"""CLI entry point for EngageVR.

Usage::

    uv run python -m engagevr demo --seed 42 --output artifacts/demo-session.json
    uv run python -m engagevr capture --camera 0 --duration 30
    uv run python -m engagevr rppg-demo --bpm 72 --duration 30 --method pos
    uv run python -m engagevr rppg-evaluate --dataset ubfc-rppg --root /path/to/data
    uv run python -m engagevr serve --host 127.0.0.1 --port 8000
    uv run python -m engagevr task-sim --seed 42 --blocks 2 --trials-per-block 10
    uv run python -m engagevr session-inspect artifacts/sessions/SESSION_ID
    uv run python -m engagevr session-replay artifacts/sessions/SESSION_ID --speed 0
    uv run python -m engagevr features-demo --seed 42 --subjects 30
    uv run python -m engagevr baseline-demo --target engagement_class --folds 5
    uv run python -m engagevr baseline-train --mode scientific --target engagement_class
    uv run python -m engagevr fusion-demo --target engagement_class --folds 5
    uv run python -m engagevr fusion-train --mode scientific --target engagement_class
    uv run python -m engagevr personalization-demo --target engagement_class --folds 5
    uv run python -m engagevr personalization-train --mode scientific --folds 5
    uv run python -m engagevr uncertainty-demo --target engagement_class --folds 5
    uv run python -m engagevr uncertainty-train --mode scientific --folds 5
    uv run python -m engagevr adaptation-demo --output artifacts/experiments/m8

All outputs from the ``demo``, ``rppg-demo``, ``task-sim``, ``features-demo``,
``baseline-demo``, ``fusion-demo``, ``personalization-demo``,
``uncertainty-demo``, and ``adaptation-demo`` commands are
deterministic SYNTHETIC data.  Capture
outputs are behavioural proxies, NOT engagement, psychological, clinical, or
diagnostic conclusions.  rPPG heart rates are signal-processing estimates, NOT
medical measurements.  Task telemetry is a software measurement, NOT an
engagement, attention, cognitive-load, or fatigue measurement.  Baseline
metrics computed from synthetic data are software self-checks, NOT model
accuracy and NOT experimental evidence.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from engagevr.datasets import ADAPTERS
from engagevr.schemas.rppg import RppgMethod
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

    rppg_demo = sub.add_parser(
        "rppg-demo",
        help="Run the rPPG pipeline on a deterministic SYNTHETIC RGB trace.",
    )
    rppg_demo.add_argument(
        "--bpm",
        type=float,
        default=72.0,
        help="Ground-truth pulse rate of the SYNTHETIC trace (must be > 0).",
    )
    rppg_demo.add_argument("--duration", type=float, default=30.0)
    rppg_demo.add_argument("--fps", type=float, default=30.0)
    rppg_demo.add_argument(
        "--method",
        type=str,
        default="pos",
        choices=[m.value for m in RppgMethod],
    )
    rppg_demo.add_argument("--seed", type=int, default=42)
    rppg_demo.add_argument(
        "--noise-std",
        type=float,
        default=0.4,
        help="Sensor-noise standard deviation of the SYNTHETIC trace.",
    )
    rppg_demo.add_argument(
        "--dropout-rate",
        type=float,
        default=0.0,
        help="Fraction of SYNTHETIC frames emitted with no usable ROI.",
    )
    rppg_demo.add_argument(
        "--output",
        type=str,
        default="artifacts/rppg-demo.json",
    )

    rppg_eval = sub.add_parser(
        "rppg-evaluate",
        help="Evaluate rPPG against a locally obtained public dataset.",
    )
    rppg_eval.add_argument(
        "--dataset",
        type=str,
        default="ubfc-rppg",
        choices=sorted(ADAPTERS),
    )
    rppg_eval.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to the locally obtained dataset root. Never downloaded.",
    )
    rppg_eval.add_argument(
        "--method",
        type=str,
        default="pos",
        choices=[m.value for m in RppgMethod],
    )
    rppg_eval.add_argument(
        "--output",
        type=str,
        default="artifacts/dataset-evaluation.json",
    )

    from engagevr.cli_milestone4 import add_parsers
    from engagevr.cli_milestone5 import add_parsers as add_milestone5_parsers
    from engagevr.cli_milestone6 import add_parsers as add_milestone6_parsers
    from engagevr.cli_milestone7 import add_parsers as add_milestone7_parsers
    from engagevr.cli_milestone8 import add_parsers as add_milestone8_parsers

    add_parsers(sub)
    add_milestone5_parsers(sub)
    add_milestone6_parsers(sub)
    add_milestone7_parsers(sub)
    add_milestone8_parsers(sub)
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


_SYNTHETIC_BANNER = (
    "=== SYNTHETIC DATA ===\n"
    "This trace was generated by software. It is NOT a measurement, NOT a\n"
    "recording of any person, and NOT evidence. The estimation error shown\n"
    "below describes how well the pipeline recovers a frequency that this\n"
    "program itself inserted. It is a software self-check, NOT scientific\n"
    "validation of rPPG accuracy and NOT a performance claim."
)


def _run_rppg_demo(args: argparse.Namespace) -> int:
    from engagevr.config import load_config
    from engagevr.rppg.trace import (
        build_synthetic_window,
        generate_synthetic_rgb_trace,
    )
    from engagevr.rppg.window import process_window
    from engagevr.schemas.rppg import SYNTHETIC_LABEL, RppgSessionSummary
    from engagevr.schemas.session import DataSource

    if args.bpm <= 0.0:
        print("Error: --bpm must be positive.", file=sys.stderr)
        return 2
    if args.fps <= 0.0:
        print("Error: --fps must be positive.", file=sys.stderr)
        return 2
    if args.duration <= 0.0:
        print("Error: --duration must be positive.", file=sys.stderr)
        return 2
    if not 0.0 <= args.dropout_rate < 1.0:
        print("Error: --dropout-rate must be in [0, 1).", file=sys.stderr)
        return 2

    method = RppgMethod(args.method)
    cfg = load_config()
    rppg_cfg = cfg.rppg

    # The demo generates its own trace, so the demo's own fps and duration
    # define the window rather than the configured capture rate.
    rppg_cfg = rppg_cfg.model_copy(deep=True)
    rppg_cfg.trace = rppg_cfg.trace.model_copy(update={"expected_fps": args.fps})
    rppg_cfg.window = rppg_cfg.window.model_copy(
        update={
            "duration_seconds": max(
                args.duration, rppg_cfg.window.min_duration_seconds
            ),
        }
    )
    try:
        rppg_cfg = type(rppg_cfg).model_validate(rppg_cfg.model_dump())
    except Exception as exc:  # invalid band/Nyquist combination for this fps
        print(
            f"Error: invalid rPPG configuration for fps={args.fps}: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        samples = generate_synthetic_rgb_trace(
            bpm=args.bpm,
            duration_seconds=args.duration,
            fps=args.fps,
            seed=args.seed,
            noise_std=args.noise_std,
            dropout_rate=args.dropout_rate,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    window = build_synthetic_window(samples)
    result = process_window(window, rppg_cfg, method=method)

    hr = result.heart_rate
    quality = result.quality
    estimated = hr.bpm
    abs_error = abs(estimated - args.bpm) if estimated is not None else None

    summary = RppgSessionSummary(
        data_source=DataSource.SYNTHETIC,
        synthetic_label=SYNTHETIC_LABEL,
        method=method,
        n_windows=1,
        n_windows_available=1 if hr.available else 0,
        n_windows_unavailable=0 if hr.available else 1,
        mean_quality=quality.overall_quality,
        mean_bpm=estimated,
        rejection_reasons=quality.rejection_reasons,
    )

    output: dict[str, object] = {
        "data_source": DataSource.SYNTHETIC.value,
        "synthetic_label": SYNTHETIC_LABEL,
        "_synthetic_disclaimer": _SYNTHETIC_BANNER,
        "input_parameters": {
            "requested_bpm": args.bpm,
            "duration_seconds": args.duration,
            "fps": args.fps,
            "seed": args.seed,
            "noise_std": args.noise_std,
            "dropout_rate": args.dropout_rate,
            "method": method.value,
        },
        "method_parameters": result.method_params,
        "trace_summary": window.model_dump(mode="json", exclude={"samples"}),
        "waveform_summary": result.waveform.model_dump(
            mode="json", exclude={"values", "timestamps"}
        ),
        "quality": quality.model_dump(mode="json"),
        "heart_rate": hr.model_dump(mode="json"),
        "synthetic_recovery_check": {
            "requested_synthetic_bpm": args.bpm,
            "estimated_bpm": estimated,
            "absolute_error_bpm": abs_error,
            "frequency_resolution_bpm": (
                hr.frequency_resolution_hz * 60.0
                if hr.frequency_resolution_hz is not None
                else None
            ),
            "note": (
                "Recovery of a self-inserted frequency. NOT scientific "
                "validation and NOT an accuracy claim."
            ),
        },
        "session_summary": summary.model_dump(mode="json"),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str) + "\n")

    print(_SYNTHETIC_BANNER)
    print()
    print(f"Method:                 {method.value}")
    print(f"Requested SYNTHETIC BPM:{args.bpm:>8.2f}")
    if estimated is not None:
        print(f"Estimated BPM:          {estimated:>8.2f}")
        print(f"Absolute error (BPM):   {abs_error:>8.2f}")
    else:
        print("Estimated BPM:          unavailable")
        print("Absolute error (BPM):   unavailable")
    print(
        f"Signal quality:         {quality.overall_quality:>8.3f} "
        f"(threshold {quality.quality_threshold:.2f})"
    )
    print(f"Quality acceptable:     {quality.acceptable}")
    print()
    print("Spectral peak:")
    if hr.peak_frequency_hz is None:
        print("  no spectral peak was accepted")
    else:

        def _fmt(value: float | None, spec: str = ".6g") -> str:
            return "unavailable" if value is None else format(value, spec)

        print(f"  peak frequency:       {_fmt(hr.peak_frequency_hz, '.4f')} Hz")
        print(f"  peak power:           {_fmt(hr.peak_power)}")
        print(f"  peak prominence:      {_fmt(hr.peak_prominence)}")
        print(f"  in-band power:        {_fmt(hr.in_band_power)}")
        print(f"  out-of-band power:    {_fmt(hr.out_of_band_power)}")
        print(f"  spectral peak ratio:  {_fmt(hr.spectral_peak_ratio, '.4f')}")
        resolution = hr.frequency_resolution_hz
        if resolution is None:
            print("  freq resolution:      unavailable")
        else:
            print(
                f"  freq resolution:      {resolution:.4f} Hz "
                f"({resolution * 60.0:.2f} BPM)"
            )
    if hr.reason is not None:
        print()
        print(f"Rejection reason:       {hr.reason.value}")
    if quality.rejection_reasons:
        print(
            "Quality rejections:     "
            + ", ".join(r.value for r in quality.rejection_reasons)
        )
    for warning in quality.warnings:
        print(f"  warning: {warning}")
    print()
    print(f"Output:                 {out_path}")
    print()
    print(
        "SYNTHETIC DATA. Not a measurement, not evidence, not validation. "
        "rPPG heart rate is a signal-processing estimate, not a medical "
        "measurement, and is not engagement or cognitive load."
    )
    return 0


def _run_rppg_evaluate(args: argparse.Namespace) -> int:
    from engagevr.config import load_config
    from engagevr.datasets import DatasetError

    cfg = load_config()
    adapter_cls = ADAPTERS[args.dataset]

    root = args.root
    if root is None and args.dataset == "ubfc-rppg":
        root = cfg.rppg.datasets.ubfc_rppg_root
    if root is None:
        print(
            f"Error: no dataset root for '{args.dataset}'. Pass --root, or set "
            "rppg.datasets.ubfc_rppg_root in configs/defaults.yaml. "
            "This software never downloads datasets; obtain the data through "
            "the official source documented in docs/DATASETS.md.",
            file=sys.stderr,
        )
        return 2

    adapter = adapter_cls(root)
    try:
        adapter.validate()
        subjects = adapter.list_subjects()
    except DatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset:   {args.dataset}")
    print(f"Root:      {adapter.root}")
    print(f"Subjects:  {len(subjects)}")
    for key, value in adapter.describe().items():
        print(f"  {key}: {value}")
    print()
    print(
        "Dataset evaluation requires decoding each subject video and running "
        "the full ROI -> trace -> method -> heart-rate pipeline against the "
        "reference PPG. That path has NOT been executed in this repository "
        "because the dataset is not present locally. No metrics are reported."
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "method": args.method,
                "provenance": adapter.describe(),
                "subjects_found": subjects,
                "metrics": None,
                "status": "adapter_validated_only",
                "_note": (
                    "No metrics computed. Public-dataset validation is "
                    "PENDING. No synthetic value is ever reported here as "
                    "dataset performance."
                ),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print()
    print(f"Output:    {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        return _run_demo(args)
    if args.command == "capture":
        return _run_capture(args)
    if args.command == "rppg-demo":
        return _run_rppg_demo(args)
    if args.command == "rppg-evaluate":
        return _run_rppg_evaluate(args)

    if args.command in ("features-demo", "baseline-demo", "baseline-train"):
        from engagevr import cli_milestone5

        milestone5 = {
            "features-demo": cli_milestone5.run_features_demo,
            "baseline-demo": cli_milestone5.run_baseline_demo,
            "baseline-train": cli_milestone5.run_baseline_train,
        }
        return milestone5[args.command](args)

    if args.command in (
        "fusion-demo",
        "fusion-train",
        "personalization-demo",
        "personalization-train",
    ):
        from engagevr import cli_milestone6

        milestone6 = {
            "fusion-demo": cli_milestone6.run_fusion_demo,
            "fusion-train": cli_milestone6.run_fusion_train,
            "personalization-demo": cli_milestone6.run_personalization_demo,
            "personalization-train": cli_milestone6.run_personalization_train,
        }
        return milestone6[args.command](args)

    if args.command in ("uncertainty-demo", "uncertainty-train"):
        from engagevr import cli_milestone7

        milestone7 = {
            "uncertainty-demo": cli_milestone7.run_uncertainty_demo,
            "uncertainty-train": cli_milestone7.run_uncertainty_train,
        }
        return milestone7[args.command](args)

    if args.command == "adaptation-demo":
        from engagevr import cli_milestone8

        return cli_milestone8.run_adaptation_demo(args)

    if args.command in ("serve", "task-sim", "session-inspect", "session-replay"):
        from engagevr import cli_milestone4

        dispatch = {
            "serve": cli_milestone4.run_serve,
            "task-sim": cli_milestone4.run_task_sim,
            "session-inspect": cli_milestone4.run_session_inspect,
            "session-replay": cli_milestone4.run_session_replay,
        }
        return dispatch[args.command](args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
