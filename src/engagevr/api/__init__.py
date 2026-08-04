"""The local FastAPI backend: HTTP endpoints and the WebSocket bridge.

The backend transports task telemetry and adaptation commands, records
them, and serves them back.  It produces no engagement estimate, no
cognitive-load estimate, and no adaptation decision of its own.

Deployment scope: one process, bound to loopback, with no
authentication, authorization, or transport encryption.
"""

from engagevr.api.app import WEBSOCKET_PATH, create_app
from engagevr.api.broker import (
    OBSERVABLE_MESSAGE_TYPES,
    BrokerSettings,
    IngestionResult,
    QueueFullError,
    SessionBroker,
)
from engagevr.api.connections import (
    ClientConnection,
    ConnectionClosedError,
    ConnectionRegistry,
)
from engagevr.api.state import ApplicationState

__all__ = [
    "OBSERVABLE_MESSAGE_TYPES",
    "WEBSOCKET_PATH",
    "ApplicationState",
    "BrokerSettings",
    "ClientConnection",
    "ConnectionClosedError",
    "ConnectionRegistry",
    "IngestionResult",
    "QueueFullError",
    "SessionBroker",
    "create_app",
]
