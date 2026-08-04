"""FastAPI application factory.

Nothing is constructed at import time.  :func:`create_app` builds the
application, and the lifespan context owns every resource: it creates
the :class:`~engagevr.api.state.ApplicationState`, and on shutdown it
stops every session broker, flushes and closes every open recording,
writes the summaries, and clears the connection registry.

Deployment scope
----------------
Single process, loopback only, no authentication.  See
:mod:`engagevr.api.connections` for why multi-worker operation is not
supported.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, WebSocket

from engagevr._logging import get_logger
from engagevr.api.errors import register_exception_handlers
from engagevr.api.routes import SCOPE_NOTE, SECURITY_NOTE, router
from engagevr.api.state import ApplicationState
from engagevr.api.websocket import session_websocket
from engagevr.config import EngageVRConfig, load_config
from engagevr.protocol.version import PROTOCOL_VERSION
from engagevr.synchronization.clock import Clock

logger = get_logger("api.app")

WEBSOCKET_PATH = "/ws/v1/sessions/{session_id}"


def create_app(
    config: EngageVRConfig | None = None,
    *,
    session_root: Path | str | None = None,
    clock: Clock | None = None,
) -> FastAPI:
    """Build the EngageVR local backend application."""
    settings = config if config is not None else load_config()
    root = Path(session_root) if session_root is not None else None

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = ApplicationState(settings, session_root=root, clock=clock)
        app.state.engagevr = state
        logger.info(
            "EngageVR backend starting: protocol %s, session root %s",
            PROTOCOL_VERSION,
            state.store.root,
        )
        logger.warning(
            "This prototype has NO authentication and NO transport encryption. "
            "It is intended for localhost development only."
        )
        try:
            yield
        finally:
            await state.shutdown()
            logger.info("EngageVR backend stopped; all sessions closed")

    app = FastAPI(
        title="EngageVR local backend",
        version=settings.project.version,
        description=(
            f"Local real-time bridge for the EngageVR task environment. "
            f"Protocol version {PROTOCOL_VERSION}.\n\n{SCOPE_NOTE}\n\n"
            f"{SECURITY_NOTE}"
        ),
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(router)

    @app.websocket(WEBSOCKET_PATH)
    async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
        state: ApplicationState = websocket.app.state.engagevr
        await session_websocket(websocket, session_id, state)

    return app
