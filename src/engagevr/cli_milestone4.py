"""Milestone 4 CLI commands: serve, task-sim, session-inspect, session-replay.

Kept in its own module so ``__main__`` stays a thin dispatcher and the
Milestone 4 commands can be tested without importing the webcam or rPPG
code paths.

Every command that emits or reports simulated data prints a permanent
SYNTHETIC disclaimer, and every command that replays prints a permanent
REPLAY disclaimer.  No command in this module produces an engagement
estimate, a cognitive-load estimate, or any scientific, psychological,
or clinical conclusion.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import sys
import uuid
from pathlib import Path

from engagevr.config import EngageVRConfig, TaskConfig, load_config
from engagevr.protocol.version import PROTOCOL_VERSION
from engagevr.replay.clock import InvalidReplaySpeedError, validate_speed
from engagevr.replay.player import (
    REPLAY_DISCLAIMER,
    ReplayPlayer,
    ReplayResult,
    parse_message_type_filter,
    parse_source_filter,
)
from engagevr.replay.reader import (
    RecordedSession,
    ReplayFilter,
    read_recorded_session,
)
from engagevr.storage.jsonl import JsonlFormatError
from engagevr.storage.session_store import (
    InvalidSessionIdError,
    SessionStore,
    SessionStoreError,
    validate_session_id,
)
from engagevr.task.config import SYNTHETIC_DISCLAIMER, SimulatorConfig
from engagevr.task.simulator import TaskSimulator
from engagevr.transport import (
    InProcessTransport,
    JsonlFileTransport,
    MessageTransport,
    TransportError,
    WebSocketTransport,
)

_NO_AUTH_WARNING = (
    "WARNING: this prototype has NO authentication, NO authorization, and "
    "NO transport encryption. Do not expose it to a network."
)


# --------------------------------------------------------------------------
# Argument parsers
# --------------------------------------------------------------------------


def add_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the Milestone 4 subcommands on an existing subparser set."""
    serve = sub.add_parser(
        "serve",
        help="Run the local FastAPI backend (loopback only by default).",
    )
    serve.add_argument("--host", type=str, default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--session-root", type=str, default=None)
    serve.add_argument(
        "--allow-public-bind",
        action="store_true",
        help="Required to bind a non-loopback address. No authentication exists.",
    )
    serve.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
    )

    task_sim = sub.add_parser(
        "task-sim",
        help="Run the deterministic SYNTHETIC task simulator.",
    )
    task_sim.add_argument("--seed", type=int, default=42)
    task_sim.add_argument("--blocks", type=int, default=None)
    task_sim.add_argument("--trials-per-block", type=int, default=None)
    task_sim.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Time multiplier. 0 = immediate (no sleeping), 1 = real time.",
    )
    task_sim.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session identifier. Defaults to a fresh pseudonymous id.",
    )
    task_sim.add_argument(
        "--participant-id",
        type=str,
        default="synthetic_participant",
        help="Pseudonymous label for a SYNTHETIC participant that does not exist.",
    )
    task_sim.add_argument(
        "--output",
        type=str,
        default=None,
        help="Session-root directory for an offline run.",
    )
    task_sim.add_argument(
        "--connect",
        type=str,
        default=None,
        help="WebSocket URL of a running backend, e.g. "
        "ws://127.0.0.1:8000/ws/v1/sessions/demo-session",
    )

    inspect = sub.add_parser(
        "session-inspect",
        help="Summarize a recorded session directory.",
    )
    inspect.add_argument("session", type=str, help="Path to a session directory.")
    inspect.add_argument(
        "--json", action="store_true", help="Emit the summary as JSON."
    )

    replay = sub.add_parser(
        "session-replay",
        help="Replay a recorded session. Output is permanently labelled REPLAY.",
    )
    replay.add_argument("session", type=str, help="Path to a session directory.")
    replay.add_argument(
        "--speed",
        type=float,
        default=None,
        help="0 = immediate, 1 = original timing, >1 = accelerated.",
    )
    replay.add_argument(
        "--immediate",
        action="store_true",
        help="Equivalent to --speed 0: replay without sleeping.",
    )
    replay.add_argument("--connect", type=str, default=None)
    replay.add_argument(
        "--replay-session-id",
        type=str,
        default=None,
        help="Session id this replay publishes under. Defaults to the source id.",
    )
    replay.add_argument(
        "--message-type",
        action="append",
        default=None,
        help="Include only these message types. Repeatable.",
    )
    replay.add_argument(
        "--source",
        action="append",
        default=None,
        help="Include only these sources. Repeatable.",
    )
    replay.add_argument("--json", action="store_true")


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------


def run_serve(args: argparse.Namespace) -> int:
    """Start the local backend. Returns non-zero on invalid configuration."""
    import uvicorn
    from pydantic import ValidationError

    from engagevr.api.app import create_app

    config = load_config()
    overrides: dict[str, object] = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.allow_public_bind:
        overrides["allow_public_bind"] = True

    try:
        server = type(config.server).model_validate(
            {**config.server.model_dump(), **overrides}
        )
    except ValidationError as exc:
        print("Error: invalid server configuration.", file=sys.stderr)
        for error in exc.errors():
            print(f"  {error['msg']}", file=sys.stderr)
        return 2
    config = config.model_copy(update={"server": server})

    session_root = Path(
        args.session_root
        if args.session_root is not None
        else config.sessions.root_directory
    )
    try:
        session_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"Error: cannot create session root {session_root}: {exc}", file=sys.stderr
        )
        return 2

    app = create_app(config, session_root=session_root)

    print("EngageVR backend")
    print(f"  Protocol version:   {PROTOCOL_VERSION}")
    print(f"  Session root:       {session_root.resolve()}")
    print(f"  HTTP:               http://{server.host}:{server.port}")
    print(
        f"  WebSocket:          ws://{server.host}:{server.port}"
        f"/ws/v1/sessions/{{session_id}}"
    )
    print(f"  Max message bytes:  {server.maximum_message_bytes}")
    print(f"  Heartbeat interval: {server.heartbeat_interval_seconds}s")
    print(
        "  Workers:            1 (single-process registry; multi-worker "
        "operation is not supported)"
    )
    print()
    print(_NO_AUTH_WARNING)
    if server.allow_public_bind:
        print(
            "WARNING: public binding was explicitly permitted. Anyone who can "
            "reach this port can read and inject session data."
        )
    print()
    print("Press Ctrl+C to stop.")
    # Flush explicitly: stdout is block-buffered when redirected to a file
    # or a pipe, and this banner must appear before the server starts
    # accepting connections, not whenever the buffer happens to fill.
    sys.stdout.flush()

    uvicorn_config = uvicorn.Config(
        app,
        host=server.host,
        port=server.port,
        log_level=args.log_level,
        reload=False,  # auto-reload is never enabled by default
        ws_max_size=server.maximum_message_bytes,
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    try:
        uvicorn_server.run()
    except KeyboardInterrupt:  # pragma: no cover - uvicorn handles SIGINT itself
        print("\nInterrupted; sessions closed.")
    return 0


# --------------------------------------------------------------------------
# task-sim
# --------------------------------------------------------------------------


def _build_simulator_config(
    args: argparse.Namespace, config: EngageVRConfig
) -> SimulatorConfig:
    task = config.task
    updates: dict[str, object] = {}
    if args.blocks is not None:
        updates["blocks"] = args.blocks
    if args.trials_per_block is not None:
        updates["trials_per_block"] = args.trials_per_block
    task = TaskConfig.model_validate({**task.model_dump(), **updates})
    return SimulatorConfig(
        task=task,
        seed=args.seed,
        speed=args.speed,
        participant_id=args.participant_id,
    )


def run_task_sim(args: argparse.Namespace) -> int:
    """Run one simulated session, offline or over a WebSocket."""
    from pydantic import ValidationError

    config = load_config()

    if args.speed < 0:
        print("Error: --speed must not be negative (0 = immediate).", file=sys.stderr)
        return 2
    if args.connect is not None and args.output is not None:
        print(
            "Error: choose either --output (offline) or --connect (WebSocket), "
            "not both.",
            file=sys.stderr,
        )
        return 2

    session_id = args.session_id
    if session_id is None:
        session_id = (
            _session_id_from_url(args.connect)
            if args.connect is not None
            else f"sim-{uuid.uuid4().hex[:12]}"
        )
    try:
        validate_session_id(session_id)
    except InvalidSessionIdError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        simulator_config = _build_simulator_config(args, config)
    except ValidationError as exc:
        print("Error: invalid task configuration.", file=sys.stderr)
        for error in exc.errors():
            print(
                f"  {'.'.join(str(p) for p in error['loc'])}: {error['msg']}",
                file=sys.stderr,
            )
        return 2

    try:
        return asyncio.run(
            _run_task_sim_async(args, config, simulator_config, session_id)
        )
    except KeyboardInterrupt:
        print("\nInterrupted. The partial session recording is still readable.")
        return 130
    except TransportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


async def _run_task_sim_async(
    args: argparse.Namespace,
    config: EngageVRConfig,
    simulator_config: SimulatorConfig,
    session_id: str,
) -> int:
    destination: str
    recorder = None
    transport: MessageTransport

    if args.connect is not None:
        transport = WebSocketTransport(
            args.connect,
            maximum_message_bytes=config.server.maximum_message_bytes,
        )
        destination = args.connect
    else:
        root = Path(
            args.output if args.output is not None else config.sessions.root_directory
        )
        store = SessionStore(root)
        recorder = store.open_recorder(
            session_id,
            configuration={
                "task": simulator_config.task.model_dump(mode="json"),
                "seed": simulator_config.seed,
                "speed": simulator_config.speed,
                "data_source": "synthetic",
                "synthetic_label": "SYNTHETIC",
            },
            flush_every=config.sessions.flush_every_events,
            engagevr_version=config.project.version,
        )
        transport = JsonlFileTransport(recorder)
        destination = str(recorder.directory.resolve())

    simulator = TaskSimulator(
        session_id=session_id,
        config=simulator_config,
        transport=transport,
    )

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(simulator.run())
    with contextlib.suppress(NotImplementedError, ValueError):
        loop.add_signal_handler(signal.SIGINT, task.cancel)

    try:
        result = await task
    except asyncio.CancelledError:
        print("\nCancelled. A task_aborted event and session_end were recorded.")
        if recorder is not None:
            recorder.close()
        await transport.close()
        return 130
    finally:
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.remove_signal_handler(signal.SIGINT)

    if args.connect is not None:
        # Give the backend a moment to answer the final messages so the
        # acknowledgements are received before the socket closes.
        for _ in range(10):
            if await transport.receive(timeout=0.05) is None:
                break
    await transport.close()
    if recorder is not None:
        recorder.close()

    print("=== SYNTHETIC TASK SIMULATION ===")
    print(SYNTHETIC_DISCLAIMER)
    print()
    print(f"Session ID:            {result.session_id}")
    print(f"Protocol version:      {result.protocol_version}")
    print(f"Blocks:                {result.blocks}")
    print(f"Trials:                {result.trials}")
    print(f"Emitted events:        {result.emitted_message_count}")
    print(f"  task events:         {result.task_event_count}")
    print(f"Synthetic responses:   {result.synthetic_response_count}")
    print(f"  correct:             {result.correct_response_count}")
    print(f"  incorrect:           {result.incorrect_response_count}")
    print(f"Timeouts:              {result.timeout_count}")
    print(f"Adaptation commands:   {result.adaptation_commands_received}")
    print(f"Completed:             {result.completed}")
    if args.connect is not None:
        print(f"WebSocket destination: {destination}")
    else:
        print(f"Output:                {destination}")
    print(f"Data source:           {result.data_source.value}")
    print(f"Synthetic label:       {result.synthetic_label}")
    print()
    print(
        "Task accuracy, reaction time, and timeout counts above are SOFTWARE "
        "telemetry from fabricated responses. They are not engagement, "
        "attention, cognitive-load, or fatigue measurements, and they are not "
        "participant data."
    )
    return 0


def _session_id_from_url(url: str) -> str:
    """Extract the session id from a ``/ws/v1/sessions/<id>`` URL."""
    tail = url.rstrip("/").rsplit("/", 1)
    return tail[-1] if tail else url


# --------------------------------------------------------------------------
# session-inspect
# --------------------------------------------------------------------------


def _split_session_path(raw: str) -> tuple[Path, str]:
    path = Path(raw)
    return path.parent if str(path.parent) != "" else Path("."), path.name


def run_session_inspect(args: argparse.Namespace) -> int:
    """Print a recorded session's manifest, counts, and status."""
    root, session_id = _split_session_path(args.session)
    store = SessionStore(root)
    try:
        recording = read_recorded_session(store, session_id)
    except (InvalidSessionIdError, SessionStoreError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except JsonlFormatError as exc:
        print(f"Error: malformed recording. {exc}", file=sys.stderr)
        return 1

    summary = recording.summary
    if args.json:
        print(
            json.dumps(
                {
                    "manifest": recording.manifest.model_dump(mode="json"),
                    "summary": summary.model_dump(mode="json"),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print(f"Session:             {recording.session_id}")
    print(f"Directory:           {recording.directory}")
    print(f"Session format:      {recording.manifest.session_format_version}")
    print(f"Protocol version:    {recording.manifest.protocol_version}")
    print(f"Created (UTC):       {recording.manifest.created_at_utc.isoformat()}")
    print(f"Events:              {summary.event_count}")
    print(f"Dropped messages:    {summary.dropped_message_count}")
    print(f"Completed:           {summary.completed}")
    print(f"Recovered summary:   {summary.recovered}")
    if summary.malformed_line_numbers:
        print(f"Malformed lines:     {summary.malformed_line_numbers}")
    print(f"Synthetic messages:  {summary.synthetic_message_count}")
    print(f"Replay messages:     {summary.replay_message_count}")
    if summary.first_received_at_utc is not None:
        print(f"First received (UTC):{summary.first_received_at_utc.isoformat()}")
    if summary.last_received_at_utc is not None:
        print(f"Last received (UTC): {summary.last_received_at_utc.isoformat()}")
    print()
    print("Message types:")
    for name, count in sorted(summary.message_type_counts.items()):
        print(f"  {name:<32} {count}")
    print("Sources:")
    for name, count in sorted(summary.source_counts.items()):
        print(f"  {name:<32} {count}")
    if summary.anomaly_counts:
        print("Ordering anomalies:")
        for name, count in sorted(summary.anomaly_counts.items()):
            print(f"  {name:<32} {count}")
    if summary.dropped_message_types:
        print("Dropped message types:")
        for name, count in sorted(summary.dropped_message_types.items()):
            print(f"  {name:<32} {count}")
    print()
    print(summary.disclaimer)
    return 0


# --------------------------------------------------------------------------
# session-replay
# --------------------------------------------------------------------------


def run_session_replay(args: argparse.Namespace) -> int:
    """Replay a recorded session to stdout counts or to a WebSocket."""
    config = load_config()
    root, session_id = _split_session_path(args.session)
    store = SessionStore(root)

    speed = (
        0.0
        if args.immediate
        else (args.speed if args.speed is not None else config.replay.default_speed)
    )
    try:
        speed = validate_speed(speed, maximum_speed=config.replay.maximum_speed)
    except InvalidReplaySpeedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        message_types = parse_message_type_filter(args.message_type)
        sources = parse_source_filter(args.source)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        recording = read_recorded_session(store, session_id)
    except (InvalidSessionIdError, SessionStoreError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except JsonlFormatError as exc:
        print(f"Error: malformed recording. {exc}", file=sys.stderr)
        return 1

    replay_filter = ReplayFilter(
        message_types=message_types,
        sources=sources,  # type: ignore[arg-type]
    )

    try:
        result = asyncio.run(
            _run_replay_async(args, config, recording, replay_filter, speed, session_id)
        )
    except KeyboardInterrupt:
        print("\nInterrupted. The source recording was not modified.")
        return 130
    except TransportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {k: v for k, v in result.__dict__.items() if k != "envelopes"},
                indent=2,
                default=str,
            )
        )
        return 0

    print("=== REPLAY ===")
    print(REPLAY_DISCLAIMER)
    print()
    print(f"Source session:      {result.source_session_id}")
    print(f"Replay session:      {result.replay_session_id}")
    print(f"Protocol version:    {recording.manifest.protocol_version}")
    print(f"Speed:               {result.speed} ({result.mode.value})")
    print(f"Filter:              {result.filter_description}")
    print(f"Available messages:  {result.available_message_count}")
    print(f"Emitted messages:    {result.emitted_message_count}")
    print(f"Skipped by filter:   {result.skipped_by_filter_count}")
    print(f"Synthetic messages:  {result.synthetic_message_count}")
    print(f"Replay label:        {result.replay_label}")
    if args.connect is not None:
        print(f"WebSocket:           {args.connect}")
    print()
    print("Message types emitted:")
    for name, count in sorted(result.message_type_counts.items()):
        print(f"  {name:<32} {count}")
    print()
    print("The source recording was opened read-only and was not modified.")
    return 0


async def _run_replay_async(
    args: argparse.Namespace,
    config: EngageVRConfig,
    recording: RecordedSession,
    replay_filter: ReplayFilter,
    speed: float,
    session_id: str,
) -> ReplayResult:
    transport: MessageTransport
    if args.connect is not None:
        transport = WebSocketTransport(
            args.connect,
            maximum_message_bytes=config.server.maximum_message_bytes,
        )
        replay_session_id = args.replay_session_id or _session_id_from_url(args.connect)
    else:
        transport = InProcessTransport()
        replay_session_id = args.replay_session_id or session_id

    player = ReplayPlayer(
        recording,
        transport=transport,
        replay_session_id=replay_session_id,
        speed=speed,
        replay_filter=replay_filter,
        preserve_original_timing=config.replay.preserve_original_timing,
    )
    try:
        return await player.run()
    finally:
        await transport.close()
