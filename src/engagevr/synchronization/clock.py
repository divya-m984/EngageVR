"""Clock abstraction and cross-machine timing diagnostics.

What this module does and does not claim
----------------------------------------
It does **not** synchronize clocks.  Two independent machines running
this software have independent wall clocks and completely unrelated
monotonic clocks.  Nothing here corrects a timestamp, and nothing here
asserts that a client timestamp and a server timestamp are on a common
timeline.

What it does provide is a round-trip-time bound and, from that, an
*estimate* of the clock offset with an explicit uncertainty.  Every
offset value produced here is accompanied by the RTT it was derived
from, and every consumer is expected to treat the offset as
``estimate ± uncertainty``, never as a correction to apply silently.

The classical bound (as used by NTP and SNTP, RFC 5905 §8) is: for a
request sent at client time ``t0``, received at server time ``t1``, and
answered at server time ``t2`` back to the client at ``t3``,

    rtt    = (t3 - t0) - (t2 - t1)
    offset = ((t1 - t0) + (t2 - t3)) / 2

with ``|true offset - offset| <= rtt / 2`` under the assumption that
path delay is symmetric.  That assumption is not verifiable here, so
``rtt / 2`` is reported as a *bound under an unverified assumption*
rather than as an error bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from engagevr.utils.timestamps import monotonic_seconds, utc_now


class Clock(Protocol):
    """The clock interface every timing-sensitive component depends on.

    Injected rather than called directly so that tests and accelerated
    or immediate replay modes can run without real time passing.
    """

    def utc_now(self) -> datetime:
        """Return the current timezone-aware UTC wall-clock time."""
        ...

    def monotonic(self) -> float:
        """Return this process's monotonic clock in fractional seconds."""
        ...


class SystemClock:
    """The real clock: wall time from the OS, monotonic from the OS."""

    def utc_now(self) -> datetime:
        return utc_now()

    def monotonic(self) -> float:
        return monotonic_seconds()


class ManualClock:
    """A fully controlled clock for deterministic tests and replay.

    Time advances only when :meth:`advance` is called, so a test can
    exercise heartbeat timeouts, replay pacing, and timing diagnostics
    without sleeping.
    """

    def __init__(
        self,
        *,
        start_utc: datetime | None = None,
        start_monotonic: float = 0.0,
    ) -> None:
        self._utc = (
            start_utc if start_utc is not None else datetime(2026, 1, 1, tzinfo=UTC)
        )
        self._monotonic = start_monotonic

    def utc_now(self) -> datetime:
        return self._utc

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        """Advance both the wall and monotonic readings by ``seconds``."""
        if seconds < 0:
            raise ValueError("ManualClock cannot move backwards")
        self._monotonic += seconds
        self._utc = self._utc.fromtimestamp(self._utc.timestamp() + seconds, tz=UTC)


@dataclass(frozen=True, slots=True)
class RoundTripEstimate:
    """One heartbeat round trip and the offset estimate it supports.

    Attributes
    ----------
    heartbeat_id:
        Correlates the probe with its acknowledgement.
    round_trip_seconds:
        Measured on the client's own monotonic clock, minus the server's
        self-reported handling time.  This value is trustworthy: it is a
        single-clock difference.
    server_handling_seconds:
        How long the server said it held the probe.
    clock_offset_estimate_seconds:
        Estimated ``server_utc - client_utc``.  An **estimate**.
    offset_uncertainty_seconds:
        ``round_trip_seconds / 2``, valid only under the unverified
        assumption of a symmetric path delay.
    symmetric_delay_assumed:
        Always True, recorded explicitly so the assumption travels with
        the number rather than living only in documentation.
    """

    heartbeat_id: str
    round_trip_seconds: float
    server_handling_seconds: float
    clock_offset_estimate_seconds: float
    offset_uncertainty_seconds: float
    symmetric_delay_assumed: bool = True

    note: str = (
        "Clock offset is an ESTIMATE with an uncertainty of rtt/2 under an "
        "unverified symmetric-delay assumption. The clocks of independent "
        "machines are NOT synchronized by this software."
    )


def estimate_round_trip(
    *,
    heartbeat_id: str,
    client_sent_monotonic: float,
    client_received_monotonic: float,
    client_sent_utc: datetime,
    client_received_utc: datetime,
    server_received_utc: datetime,
    server_sent_utc: datetime,
) -> RoundTripEstimate:
    """Compute an RTT and a bounded clock-offset estimate from one probe.

    Raises
    ------
    ValueError
        If the client's monotonic readings are out of order, or if the
        server reports handling the probe for longer than the whole
        round trip took (which would make the result meaningless).
    """
    total = client_received_monotonic - client_sent_monotonic
    if total < 0:
        raise ValueError(
            "client monotonic clock moved backwards across a heartbeat; "
            "no round-trip time can be derived"
        )
    handling = (server_sent_utc - server_received_utc).total_seconds()
    if handling < 0:
        raise ValueError(
            "server reported sending a heartbeat reply before receiving it"
        )
    rtt = total - handling
    if rtt < 0:
        raise ValueError(
            "reported server handling time exceeds the measured round trip; "
            "the two clocks disagree too much for a usable estimate"
        )
    offset = (
        (server_received_utc - client_sent_utc).total_seconds()
        + (server_sent_utc - client_received_utc).total_seconds()
    ) / 2.0
    return RoundTripEstimate(
        heartbeat_id=heartbeat_id,
        round_trip_seconds=rtt,
        server_handling_seconds=handling,
        clock_offset_estimate_seconds=offset,
        offset_uncertainty_seconds=rtt / 2.0,
    )


@dataclass(frozen=True, slots=True)
class ArrivalTiming:
    """Timing diagnostics for one ingested message.

    ``apparent_transport_delay_seconds`` is populated **only** when the
    sender and receiver are the same process, where both timestamps come
    from one clock and the subtraction is meaningful.  Across processes
    it is left ``None`` with a stated reason, rather than reporting a
    number that is really a clock-offset artefact.
    """

    server_received_at_utc: datetime
    server_monotonic_seconds: float
    apparent_transport_delay_seconds: float | None
    delay_unavailable_reason: str | None
    future_timestamp: bool
    future_by_seconds: float | None
    excessive_delay: bool


def assess_arrival(
    *,
    sent_at_utc: datetime,
    server_received_at_utc: datetime,
    server_monotonic_seconds: float,
    same_process: bool,
    maximum_clock_skew_seconds: float,
    maximum_transport_delay_seconds: float,
) -> ArrivalTiming:
    """Diagnose one message's arrival timing.

    Detects a timestamp from the future beyond tolerance, and — when the
    comparison is meaningful — an excessive transport delay.

    A client whose wall clock is merely offset from the server's will
    trip ``future_timestamp``; that is reported as a *clock* observation,
    not as a message defect, and it never causes the message to be
    rejected.
    """
    skew = (sent_at_utc - server_received_at_utc).total_seconds()
    future = skew > maximum_clock_skew_seconds

    delay: float | None = None
    reason: str | None = None
    excessive = False
    if same_process:
        delay = -skew
        excessive = delay > maximum_transport_delay_seconds
    else:
        reason = (
            "sender and receiver are different processes; their wall clocks "
            "are independent, so send-to-receive subtraction would measure "
            "clock offset rather than transport delay. Use heartbeat "
            "round-trip diagnostics instead."
        )

    return ArrivalTiming(
        server_received_at_utc=server_received_at_utc,
        server_monotonic_seconds=server_monotonic_seconds,
        apparent_transport_delay_seconds=delay,
        delay_unavailable_reason=reason,
        future_timestamp=future,
        future_by_seconds=skew if future else None,
        excessive_delay=excessive,
    )
