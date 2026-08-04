"""Sequence and message-identity ordering diagnostics.

Two orders are tracked and kept separate, permanently:

**Arrival order**
    The order the receiver actually saw messages in.  This is what the
    JSONL recording preserves, byte for byte, append-only.  It is never
    re-sorted, because doing so would destroy the evidence of what
    actually happened.

**Source sequence order**
    Each source's own ``sequence_number`` within a session.  Anomalies
    between the two — reversal, duplication, gaps — are *recorded as
    anomalies on the affected message* rather than repaired.

The diagnostics here classify; they do not correct.
"""

from __future__ import annotations

import enum
from collections import OrderedDict
from dataclasses import dataclass, field

from engagevr.protocol.envelope import MessageEnvelope
from engagevr.protocol.messages import MessageSource


class OrderingAnomaly(enum.StrEnum):
    """A detected irregularity in a source's message stream."""

    DUPLICATE_MESSAGE_ID = "duplicate_message_id"
    DUPLICATE_SEQUENCE_NUMBER = "duplicate_sequence_number"
    SEQUENCE_REVERSAL = "sequence_reversal"
    MISSING_SEQUENCE_RANGE = "missing_sequence_range"
    EXCESSIVE_SEQUENCE_GAP = "excessive_sequence_gap"
    FUTURE_TIMESTAMP = "future_timestamp"
    EXCESSIVE_TRANSPORT_DELAY = "excessive_transport_delay"


@dataclass(frozen=True, slots=True)
class OrderingObservation:
    """The result of checking one message against its source's history."""

    anomalies: tuple[OrderingAnomaly, ...]
    expected_sequence_number: int | None
    missing_sequence_numbers: tuple[int, ...] = ()
    detail: str | None = None

    @property
    def ok(self) -> bool:
        """True when no anomaly was detected."""
        return not self.anomalies


@dataclass
class _SourceState:
    last_sequence_number: int | None = None
    highest_sequence_number: int | None = None
    seen_sequence_numbers: set[int] = field(default_factory=set)
    message_ids: OrderedDict[str, None] = field(default_factory=OrderedDict)
    message_count: int = 0


class SequenceTracker:
    """Per-``(session, source)`` sequence and message-id bookkeeping.

    Memory is bounded: only the most recent ``message_id_history``
    identifiers are retained per source, so a long session cannot grow
    this structure without limit.  The bound is documented rather than
    hidden, because it means duplicate detection has a finite horizon:
    a duplicate that arrives after ``message_id_history`` intervening
    messages from the same source will not be detected.
    """

    def __init__(
        self,
        *,
        maximum_sequence_gap: int = 1000,
        message_id_history: int = 4096,
    ) -> None:
        if maximum_sequence_gap < 0:
            raise ValueError("maximum_sequence_gap must be non-negative")
        if message_id_history < 1:
            raise ValueError("message_id_history must be at least 1")
        self._maximum_gap = maximum_sequence_gap
        self._history = message_id_history
        self._states: dict[MessageSource, _SourceState] = {}

    @property
    def message_id_history(self) -> int:
        """How many recent message ids are retained per source."""
        return self._history

    def observe(self, envelope: MessageEnvelope) -> OrderingObservation:
        """Record ``envelope`` and report any ordering anomaly it shows.

        The message is always recorded, whatever anomalies it carries.
        Nothing is dropped, reordered, or renumbered here.
        """
        state = self._states.setdefault(envelope.source, _SourceState())
        anomalies: list[OrderingAnomaly] = []
        details: list[str] = []
        missing: tuple[int, ...] = ()

        if envelope.message_id in state.message_ids:
            anomalies.append(OrderingAnomaly.DUPLICATE_MESSAGE_ID)
            details.append(f"message_id {envelope.message_id!r} was already seen")

        sequence = envelope.sequence_number
        expected = (
            None
            if state.last_sequence_number is None
            else state.last_sequence_number + 1
        )

        if sequence in state.seen_sequence_numbers:
            anomalies.append(OrderingAnomaly.DUPLICATE_SEQUENCE_NUMBER)
            details.append(f"sequence_number {sequence} was already seen")
        elif state.highest_sequence_number is not None:
            if sequence < state.highest_sequence_number:
                anomalies.append(OrderingAnomaly.SEQUENCE_REVERSAL)
                details.append(
                    f"sequence_number {sequence} arrived after "
                    f"{state.highest_sequence_number}"
                )
            elif sequence > state.highest_sequence_number + 1:
                gap_start = state.highest_sequence_number + 1
                gap_size = sequence - gap_start
                anomalies.append(OrderingAnomaly.MISSING_SEQUENCE_RANGE)
                details.append(
                    f"sequence numbers {gap_start}..{sequence - 1} were never seen"
                )
                if gap_size <= self._maximum_gap:
                    missing = tuple(range(gap_start, sequence))
                else:
                    anomalies.append(OrderingAnomaly.EXCESSIVE_SEQUENCE_GAP)
                    details.append(
                        f"gap of {gap_size} exceeds maximum_sequence_gap "
                        f"{self._maximum_gap}; the missing range is not enumerated"
                    )

        state.seen_sequence_numbers.add(sequence)
        state.last_sequence_number = sequence
        state.highest_sequence_number = (
            sequence
            if state.highest_sequence_number is None
            else max(state.highest_sequence_number, sequence)
        )
        state.message_ids[envelope.message_id] = None
        state.message_count += 1
        while len(state.message_ids) > self._history:
            state.message_ids.popitem(last=False)

        return OrderingObservation(
            anomalies=tuple(anomalies),
            expected_sequence_number=expected,
            missing_sequence_numbers=missing,
            detail="; ".join(details) if details else None,
        )

    def message_count(self, source: MessageSource) -> int:
        """How many messages this tracker has observed from ``source``."""
        state = self._states.get(source)
        return 0 if state is None else state.message_count

    def sources(self) -> tuple[MessageSource, ...]:
        """Every source seen so far, in first-seen order."""
        return tuple(self._states)
