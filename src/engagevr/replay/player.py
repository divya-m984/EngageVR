"""The replay player.

What replay does
----------------
Emits the recorded messages again, in recorded arrival order, at a
chosen pace, over any transport.

What replay does **not** do
---------------------------
It does not rewrite the messages.  Each replayed envelope keeps its
original ``source``, ``sequence_number``, ``sent_at_utc``,
``sent_at_monotonic_seconds``, ``message_id``, and provenance.  The only
change is an **added** ``replay`` block
(:class:`~engagevr.protocol.envelope.ReplayMetadata`) recording that
this emission is a replay, which recording it came from, where it sits
in the replay stream, and at what speed.

Consequently:

- a replayed message is never presentable as a live message;
- a replayed **synthetic** message carries ``SYNTHETIC`` in its
  provenance *and* ``REPLAY`` in its replay block;
- a replay is never a new participant session, and the added metadata
  makes that inspectable rather than merely documented.

The source recording is opened read-only and is never modified.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from engagevr.protocol.envelope import (
    REPLAY_LABEL,
    MessageEnvelope,
    ReplayMetadata,
)
from engagevr.protocol.messages import MessageType
from engagevr.replay.clock import ReplayMode, ReplayPacer, mode_for_speed
from engagevr.replay.reader import RecordedSession, ReplayFilter
from engagevr.storage.manifest import StoredMessage
from engagevr.synchronization.clock import Clock, SystemClock
from engagevr.transport import MessageTransport

#: Printed by the CLI and attached to every replay result.
REPLAY_DISCLAIMER = (
    "REPLAY. These messages are a re-emission of a previously recorded "
    "session. They are not live, are not a new session, and are not new "
    "participant data. Messages that were SYNTHETIC when recorded remain "
    "SYNTHETIC on replay."
)


@dataclass
class ReplayResult:
    """What one replay run emitted."""

    source_session_id: str
    replay_session_id: str
    speed: float
    mode: ReplayMode
    filter_description: str
    available_message_count: int = 0
    emitted_message_count: int = 0
    skipped_by_filter_count: int = 0
    synthetic_message_count: int = 0
    stopped_early: bool = False
    replay_label: str = REPLAY_LABEL
    disclaimer: str = REPLAY_DISCLAIMER
    envelopes: list[MessageEnvelope] = field(default_factory=list)
    message_type_counts: dict[str, int] = field(default_factory=dict)


class ReplayPlayer:
    """Replays one recorded session over a transport."""

    def __init__(
        self,
        recording: RecordedSession,
        *,
        transport: MessageTransport,
        replay_session_id: str | None = None,
        speed: float = 0.0,
        replay_filter: ReplayFilter | None = None,
        clock: Clock | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        preserve_original_timing: bool = True,
        collect_envelopes: bool = False,
        step_mode: bool = False,
    ) -> None:
        self._recording = recording
        self._transport = transport
        self._replay_session_id = (
            replay_session_id if replay_session_id is not None else recording.session_id
        )
        self._speed = speed
        self._filter = replay_filter if replay_filter is not None else ReplayFilter()
        self._clock = clock if clock is not None else SystemClock()
        self._pacer = ReplayPacer(
            speed=speed,
            sleep=sleep if sleep is not None else asyncio.sleep,
            preserve_original_timing=preserve_original_timing,
        )
        self._collect = collect_envelopes
        self._step_mode = step_mode
        self._step_gate: asyncio.Event | None = asyncio.Event() if step_mode else None

    @property
    def pacer(self) -> ReplayPacer:
        return self._pacer

    def step(self) -> None:
        """Release one message in step-by-step mode."""
        if self._step_gate is None:
            raise RuntimeError("this player was not created in step mode")
        self._step_gate.set()

    def build_replay_metadata(
        self, stored: StoredMessage, replay_index: int
    ) -> ReplayMetadata:
        """Metadata added to one replayed message."""
        return ReplayMetadata(
            replay_label=REPLAY_LABEL,
            source_session_id=self._recording.session_id,
            replay_session_id=self._replay_session_id,
            replay_index=replay_index,
            replay_speed=self._speed,
            replayed_at_utc=self._clock.utc_now(),
            original_arrival_index=stored.ingestion.arrival_index,
        )

    def prepare(self) -> tuple[tuple[StoredMessage, ...], tuple[float, ...]]:
        """Select and pace the messages this run will emit."""
        selected = self._recording.filtered(self._filter)
        gaps = self._recording.gaps_seconds(selected)
        return selected, gaps

    async def run(self) -> ReplayResult:
        """Replay the recording. Deterministic in order for a given input."""
        selected, gaps = self.prepare()
        result = ReplayResult(
            source_session_id=self._recording.session_id,
            replay_session_id=self._replay_session_id,
            speed=self._speed,
            mode=mode_for_speed(self._speed),
            filter_description=self._filter.describe(),
            available_message_count=self._recording.message_count,
            skipped_by_filter_count=self._recording.message_count - len(selected),
        )

        await self._transport.connect()
        try:
            for index, (stored, gap) in enumerate(zip(selected, gaps, strict=True)):
                if self._step_gate is not None:
                    await self._step_gate.wait()
                    self._step_gate.clear()
                else:
                    await self._pacer.wait(gap)

                envelope = stored.envelope.with_replay_metadata(
                    self.build_replay_metadata(stored, index)
                )
                await self._transport.send(envelope)

                result.emitted_message_count += 1
                key = envelope.message_type.value
                result.message_type_counts[key] = (
                    result.message_type_counts.get(key, 0) + 1
                )
                if envelope.provenance.synthetic_label is not None:
                    result.synthetic_message_count += 1
                if self._collect:
                    result.envelopes.append(envelope)
        except asyncio.CancelledError:
            result.stopped_early = True
            raise
        return result


def parse_message_type_filter(values: list[str] | None) -> frozenset[MessageType]:
    """Turn CLI ``--message-type`` values into a filter set.

    Raises
    ------
    ValueError
        Naming the unknown value and listing the valid ones.
    """
    if not values:
        return frozenset()
    known = {t.value: t for t in MessageType}
    selected: set[MessageType] = set()
    for value in values:
        if value not in known:
            valid = ", ".join(sorted(known))
            raise ValueError(f"unknown message type {value!r}; valid types: {valid}")
        selected.add(known[value])
    return frozenset(selected)


def parse_source_filter(values: list[str] | None) -> frozenset[object]:
    """Turn CLI ``--source`` values into a filter set."""
    from engagevr.protocol.messages import MessageSource

    if not values:
        return frozenset()
    known = {s.value: s for s in MessageSource}
    selected: set[MessageSource] = set()
    for value in values:
        if value not in known:
            valid = ", ".join(sorted(known))
            raise ValueError(f"unknown source {value!r}; valid sources: {valid}")
        selected.add(known[value])
    return frozenset(selected)
