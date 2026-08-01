"""CLI entry point for EngageVR.

Usage::

    uv run python -m engagevr demo --seed 42 --output artifacts/demo-session.json

All outputs from the ``demo`` command are deterministic SYNTHETIC data
generated for software testing.  They must NEVER be presented as
experimental evidence or used to make engagement or cognitive-load
validity claims.
"""

from __future__ import annotations

import argparse
import json
import sys
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
        help="Generate a deterministic SYNTHETIC session and save it to JSON.",
    )
    demo.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    demo.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of synthetic trials (default: 10).",
    )
    demo.add_argument(
        "--output",
        type=str,
        default="artifacts/demo-session.json",
        help="Destination JSON path (default: artifacts/demo-session.json).",
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

    # Summary
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        return _run_demo(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
