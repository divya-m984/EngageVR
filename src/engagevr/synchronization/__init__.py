"""Clock abstraction, timing diagnostics, and ordering diagnostics.

This package records what the clocks and the sequence numbers actually
said.  It does not synchronize clocks, does not correct timestamps, and
does not reorder events.  Irregularities are reported as typed
anomalies attached to the affected message and preserved in the session
recording.
"""

from engagevr.synchronization.clock import (
    ArrivalTiming,
    Clock,
    ManualClock,
    RoundTripEstimate,
    SystemClock,
    assess_arrival,
    estimate_round_trip,
)
from engagevr.synchronization.ordering import (
    OrderingAnomaly,
    OrderingObservation,
    SequenceTracker,
)

__all__ = [
    "ArrivalTiming",
    "Clock",
    "ManualClock",
    "OrderingAnomaly",
    "OrderingObservation",
    "RoundTripEstimate",
    "SequenceTracker",
    "SystemClock",
    "assess_arrival",
    "estimate_round_trip",
]
