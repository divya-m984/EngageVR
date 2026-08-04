"""Backend HTTP and WebSocket tests, run fully in-process.

No external server is started.  ``fastapi.testclient.TestClient`` runs
the ASGI app directly, including the WebSocket endpoint, so these tests
need no network, no port, and no Unity.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from engagevr.api import create_app
from engagevr.config import EngageVRConfig, load_config
from engagevr.protocol.envelope import (
    MessageProvenance,
    ReplayMetadata,
    build_envelope,
)
from engagevr.protocol.messages import (
    AdaptationCommandName,
    AdaptationCommandPayload,
    ClientHelloPayload,
    ClientRole,
    MessageSource,
    MessageType,
    ProtocolErrorCode,
    TaskEventPayload,
)
from engagevr.protocol.version import PROTOCOL_VERSION
from engagevr.schemas.events import EventType, TaskEventDetail
from engagevr.schemas.session import DataSource
from tests.unit.test_protocol import default_payload

SESSION = "test-session"


@pytest.fixture
def config() -> EngageVRConfig:
    return load_config()


@pytest.fixture
def session_root(tmp_path: Path) -> Path:
    root = tmp_path / "sessions"
    root.mkdir()
    return root


@pytest.fixture
def client(config: EngageVRConfig, session_root: Path) -> Iterator[TestClient]:
    app = create_app(config, session_root=session_root)
    with TestClient(app) as test_client:
        yield test_client


def envelope_json(
    message_type: MessageType,
    *,
    sequence_number: int,
    session_id: str = SESSION,
    source: MessageSource = MessageSource.PYTHON_SIMULATOR,
    payload: object | None = None,
    provenance: MessageProvenance | None = None,
) -> dict[str, Any]:
    envelope = build_envelope(
        message_type=message_type,
        session_id=session_id,
        source=source,
        sequence_number=sequence_number,
        payload=payload if payload is not None else default_payload(message_type),  # type: ignore[arg-type]
        provenance=provenance,
    )
    return envelope.to_json_dict()


def hello_json(
    *,
    role: ClientRole = ClientRole.SIMULATOR,
    source: MessageSource = MessageSource.PYTHON_SIMULATOR,
    session_id: str = SESSION,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    return envelope_json(
        MessageType.CLIENT_HELLO,
        sequence_number=0,
        session_id=session_id,
        source=source,
        payload=ClientHelloPayload(
            role=role,
            client_name="test-client",
            client_version="1.0",
            protocol_version=protocol_version,
        ),
    )


def handshake(websocket: Any, **kwargs: Any) -> dict[str, Any]:
    """Complete a handshake and return the server_hello message."""
    websocket.send_text(json.dumps(hello_json(**kwargs)))
    reply: dict[str, Any] = websocket.receive_json()
    return reply


# --- HTTP ------------------------------------------------------------------


class TestHttpEndpoints:
    def test_health(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["live_sessions"] == []
        assert body["connections"] == 0

    def test_version_reports_the_protocol_and_scope(self, client: TestClient) -> None:
        body = client.get("/version").json()
        assert body["protocol_version"] == PROTOCOL_VERSION
        assert "no engagement estimate" in body["scope"]
        assert "No authentication" in body["security"]

    def test_protocol_metadata_lists_every_message_type(
        self, client: TestClient
    ) -> None:
        body = client.get("/protocol").json()
        assert body["protocol_version"] == PROTOCOL_VERSION
        assert set(body["message_types"]) == {t.value for t in MessageType}
        assert set(body["sources"]) == {s.value for s in MessageSource}
        assert "json_schema" not in body

    def test_protocol_schema_is_served_on_request(self, client: TestClient) -> None:
        body = client.get("/protocol", params={"include_schema": True}).json()
        assert body["json_schema"]["x-protocol-version"] == PROTOCOL_VERSION

    def test_session_list_is_empty_initially(self, client: TestClient) -> None:
        assert client.get("/sessions").json()["sessions"] == []

    def test_missing_session_returns_404(self, client: TestClient) -> None:
        response = client.get("/sessions/never-existed")
        assert response.status_code == 404
        assert response.json()["error_code"] == "session_unavailable"

    def test_invalid_session_path_is_rejected(self, client: TestClient) -> None:
        response = client.get("/sessions/..%2F..%2Fetc")
        assert response.status_code in (400, 404)
        if response.status_code == 400:
            assert response.json()["error_code"] == "invalid_session_id"

    def test_session_endpoints_after_a_recorded_session(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            for n in range(1, 4):
                websocket.send_text(
                    json.dumps(envelope_json(MessageType.TASK_EVENT, sequence_number=n))
                )
                websocket.receive_json()

        listing = client.get("/sessions").json()
        assert SESSION in listing["sessions"]

        detail = client.get(f"/sessions/{SESSION}").json()
        assert detail["manifest"]["protocol_version"] == PROTOCOL_VERSION
        assert detail["summary"]["event_count"] == 4

        events = client.get(f"/sessions/{SESSION}/events").json()
        assert events["matched"] == 4
        assert events["returned"] == 4
        assert "never re-sorted" in events["ordering"]
        assert events["events"][0]["envelope"]["message_type"] == "client_hello"

        summary = client.get(f"/sessions/{SESSION}/summary").json()
        assert summary["summary"]["message_type_counts"]["task_event"] == 3
        assert summary["recovered_from_incomplete_session"] is False

    def test_event_filtering_and_paging(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            for n in range(1, 6):
                websocket.send_text(
                    json.dumps(envelope_json(MessageType.TASK_EVENT, sequence_number=n))
                )
                websocket.receive_json()

        filtered = client.get(
            f"/sessions/{SESSION}/events", params={"message_type": "task_event"}
        ).json()
        assert filtered["matched"] == 5

        paged = client.get(
            f"/sessions/{SESSION}/events", params={"limit": 2, "offset": 1}
        ).json()
        assert paged["returned"] == 2
        assert paged["matched"] == 6

        by_source = client.get(
            f"/sessions/{SESSION}/events", params={"source": "backend"}
        ).json()
        assert by_source["matched"] == 0


# --- WebSocket handshake ---------------------------------------------------


class TestHandshake:
    def test_successful_handshake(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            reply = handshake(websocket)
            assert reply["message_type"] == "server_hello"
            assert reply["payload"]["accepted"] is True
            assert reply["payload"]["protocol_version"] == PROTOCOL_VERSION
            assert reply["payload"]["session_id"] == SESSION
            assert reply["payload"]["assigned_client_id"]
            assert reply["source"] == "backend"

    def test_first_message_must_be_client_hello(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            websocket.send_text(
                json.dumps(envelope_json(MessageType.TASK_EVENT, sequence_number=0))
            )
            reply = websocket.receive_json()
            assert reply["message_type"] == "protocol_error"
            assert reply["payload"]["error_code"] == "handshake_required"
            assert reply["payload"]["fatal"] is True

    def test_unsupported_protocol_version_is_rejected(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            raw = hello_json()
            raw["protocol_version"] = "2.0"
            websocket.send_text(json.dumps(raw))
            reply = websocket.receive_json()
            assert (
                reply["payload"]["error_code"]
                == ProtocolErrorCode.UNSUPPORTED_PROTOCOL_VERSION.value
            )

    def test_mismatched_payload_version_is_rejected(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            reply = handshake(websocket, protocol_version="1.5")
            assert reply["payload"]["error_code"] == "handshake_rejected"
            assert "envelope declares" in reply["payload"]["detail"]

    def test_session_mismatch_is_rejected(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            websocket.send_text(json.dumps(hello_json(session_id="other-session")))
            reply = websocket.receive_json()
            assert reply["payload"]["error_code"] == "session_mismatch"

    def test_role_may_not_claim_an_unrelated_source(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            websocket.send_text(
                json.dumps(
                    hello_json(
                        role=ClientRole.UNITY, source=MessageSource.PYTHON_SIMULATOR
                    )
                )
            )
            reply = websocket.receive_json()
            assert reply["payload"]["error_code"] == "handshake_rejected"
            assert "may not declare source" in reply["payload"]["detail"]

    def test_invalid_session_id_closes_the_socket(self, client: TestClient) -> None:
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/v1/sessions/bad%20id") as websocket:
                websocket.receive_json()

    def test_binary_frame_is_rejected(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            websocket.send_bytes(b"\x00\x01\x02")
            reply = websocket.receive_json()
            assert reply["payload"]["error_code"] == "invalid_json"
            assert "JSON text frames" in reply["payload"]["detail"]


# --- WebSocket message handling --------------------------------------------


class TestMessageHandling:
    def test_task_event_is_acknowledged_and_stored(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            raw = envelope_json(MessageType.TASK_EVENT, sequence_number=1)
            websocket.send_text(json.dumps(raw))
            reply = websocket.receive_json()

            assert reply["message_type"] == "acknowledgement"
            assert reply["payload"]["acknowledged_message_id"] == raw["message_id"]
            assert reply["payload"]["acknowledged_message_type"] == "task_event"
            assert reply["payload"]["stored"] is True
            assert reply["payload"]["dropped"] is False
            assert reply["correlation_id"] == raw["message_id"]
            assert reply["payload"]["server_received_at_utc"].endswith("Z")

    def test_malformed_json_produces_a_protocol_error(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            websocket.send_text("{definitely not json")
            reply = websocket.receive_json()
            assert reply["payload"]["error_code"] == "invalid_json"

    def test_unknown_message_type_is_rejected_but_survivable(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            raw = envelope_json(MessageType.TASK_EVENT, sequence_number=1)
            raw["message_type"] = "telepathy"
            websocket.send_text(json.dumps(raw))
            reply = websocket.receive_json()
            assert reply["payload"]["error_code"] == "unknown_message_type"
            assert reply["payload"]["fatal"] is False

            # The connection still works afterwards.
            websocket.send_text(
                json.dumps(envelope_json(MessageType.TASK_EVENT, sequence_number=1))
            )
            assert websocket.receive_json()["message_type"] == "acknowledgement"

    def test_invalid_payload_is_rejected(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            raw = envelope_json(MessageType.TASK_EVENT, sequence_number=1)
            raw["payload"] = {
                "event": {
                    "event_type": "response_registered",
                    "reaction_time_ms": -3.0,
                }
            }
            websocket.send_text(json.dumps(raw))
            reply = websocket.receive_json()
            assert reply["payload"]["error_code"] == "invalid_payload"

    def test_oversized_message_is_rejected(
        self, config: EngageVRConfig, session_root: Path
    ) -> None:
        small = config.model_copy(
            update={
                "server": config.server.model_copy(
                    update={"maximum_message_bytes": 1024}
                )
            }
        )
        app = create_app(small, session_root=session_root)
        with TestClient(app) as test_client:
            with test_client.websocket_connect(
                f"/ws/v1/sessions/{SESSION}"
            ) as websocket:
                handshake(websocket)
                raw = envelope_json(MessageType.TASK_EVENT, sequence_number=1)
                raw["payload"] = {
                    "event": {
                        "event_type": "trial_started",
                        "stimulus_id": "x" * 4000,
                    }
                }
                websocket.send_text(json.dumps(raw))
                reply = websocket.receive_json()
                assert reply["payload"]["error_code"] == "message_too_large"

    def test_session_mismatch_after_handshake_is_rejected(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            raw = envelope_json(
                MessageType.TASK_EVENT, sequence_number=1, session_id="elsewhere"
            )
            websocket.send_text(json.dumps(raw))
            reply = websocket.receive_json()
            assert reply["payload"]["error_code"] == "session_mismatch"

    def test_heartbeat_is_answered_and_echoes_client_monotonic(
        self, client: TestClient
    ) -> None:
        from engagevr.protocol.messages import HeartbeatPayload

        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            websocket.send_text(
                json.dumps(
                    envelope_json(
                        MessageType.HEARTBEAT,
                        sequence_number=1,
                        payload=HeartbeatPayload(
                            heartbeat_id="hb-1", client_monotonic_seconds=123.75
                        ),
                    )
                )
            )
            reply = websocket.receive_json()
            assert reply["message_type"] == "heartbeat_acknowledgement"
            assert reply["payload"]["heartbeat_id"] == "hb-1"
            assert reply["payload"]["client_monotonic_seconds"] == 123.75
            assert reply["payload"]["server_received_at_utc"]
            assert reply["payload"]["server_sent_at_utc"]

    def test_ordering_anomalies_are_recorded_not_repaired(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            for sequence in (1, 5, 3):
                websocket.send_text(
                    json.dumps(
                        envelope_json(MessageType.TASK_EVENT, sequence_number=sequence)
                    )
                )
                websocket.receive_json()

        events = client.get(f"/sessions/{SESSION}/events").json()["events"]
        sequences = [e["envelope"]["sequence_number"] for e in events]
        assert sequences == [0, 1, 5, 3], "arrival order is preserved as-is"

        anomalies = [a for e in events for a in e["ingestion"]["anomalies"]]
        assert "missing_sequence_range" in anomalies
        assert "sequence_reversal" in anomalies

        summary = client.get(f"/sessions/{SESSION}/summary").json()["summary"]
        assert summary["anomaly_counts"]["sequence_reversal"] == 1

    def test_ingestion_records_server_timestamps_and_no_cross_process_delay(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)

        events = client.get(f"/sessions/{SESSION}/events").json()["events"]
        ingestion = events[0]["ingestion"]
        assert ingestion["server_received_at_utc"]
        assert ingestion["transport"] == "websocket"
        assert ingestion["apparent_transport_delay_seconds"] is None
        assert "clock offset" in ingestion["delay_unavailable_reason"]


# --- connections -----------------------------------------------------------


class TestConnections:
    def test_multiple_clients_share_one_session(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as first:
            handshake(first)
            with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as second:
                handshake(
                    second, source=MessageSource.UNITY_CLIENT, role=ClientRole.UNITY
                )
                detail = client.get(f"/sessions/{SESSION}").json()
                assert len(detail["connected_clients"]) == 2
                assert detail["live"] is True

    def test_observers_receive_broadcast_task_events(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as observer:
            handshake(
                observer,
                role=ClientRole.OBSERVER,
                source=MessageSource.TEST_FIXTURE,
            )
            with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as simulator:
                handshake(simulator)
                simulator.send_text(
                    json.dumps(envelope_json(MessageType.TASK_EVENT, sequence_number=1))
                )
                simulator.receive_json()  # the simulator's own acknowledgement

                # The observer sees the task event. It does not see the
                # simulator's client_hello: handshake traffic is not in
                # OBSERVABLE_MESSAGE_TYPES.
                broadcast = observer.receive_json()
                assert broadcast["message_type"] == "task_event"
                assert broadcast["source"] == "python_simulator"
                assert broadcast["provenance"]["synthetic_label"] == "SYNTHETIC"

    def test_an_observer_does_not_see_handshake_or_acknowledgement_traffic(
        self, client: TestClient
    ) -> None:
        from engagevr.api.broker import OBSERVABLE_MESSAGE_TYPES

        assert MessageType.CLIENT_HELLO not in OBSERVABLE_MESSAGE_TYPES
        assert MessageType.SERVER_HELLO not in OBSERVABLE_MESSAGE_TYPES
        assert MessageType.ACKNOWLEDGEMENT not in OBSERVABLE_MESSAGE_TYPES
        assert MessageType.HEARTBEAT not in OBSERVABLE_MESSAGE_TYPES

    def test_an_observer_may_not_send_task_events(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as observer:
            handshake(
                observer,
                role=ClientRole.OBSERVER,
                source=MessageSource.TEST_FIXTURE,
            )
            observer.send_text(
                json.dumps(
                    envelope_json(
                        MessageType.TASK_EVENT,
                        sequence_number=1,
                        source=MessageSource.TEST_FIXTURE,
                    )
                )
            )
            reply = observer.receive_json()
            assert reply["payload"]["error_code"] == "role_not_permitted"

    def test_clean_disconnect_closes_the_session(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
        assert client.get("/health").json()["connections"] == 0
        assert client.get("/health").json()["live_sessions"] == []

        summary = client.get(f"/sessions/{SESSION}/summary").json()["summary"]
        assert summary["disconnect_reason"] == "client_disconnect"

    def test_the_registry_is_emptied_on_disconnect(self, client: TestClient) -> None:
        for _ in range(3):
            with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
                handshake(websocket)
        assert client.get("/health").json()["connections"] == 0

    def test_a_silent_client_is_timed_out_and_told_why(
        self, config: EngageVRConfig, session_root: Path
    ) -> None:
        impatient = config.model_copy(
            update={
                "server": config.server.model_copy(
                    update={
                        "heartbeat_interval_seconds": 0.05,
                        "connection_timeout_seconds": 0.2,
                    }
                )
            }
        )
        app = create_app(impatient, session_root=session_root)
        with TestClient(app) as test_client:
            with test_client.websocket_connect(
                f"/ws/v1/sessions/{SESSION}"
            ) as websocket:
                handshake(websocket)
                reply = websocket.receive_json()
                assert reply["message_type"] == "protocol_error"
                assert reply["payload"]["fatal"] is True
                assert "no message received" in reply["payload"]["detail"]

            summary = test_client.get(f"/sessions/{SESSION}/summary").json()["summary"]
            assert summary["disconnect_reason"] == "timeout"

    def test_a_handshake_that_never_arrives_times_out(
        self, config: EngageVRConfig, session_root: Path
    ) -> None:
        impatient = config.model_copy(
            update={
                "server": config.server.model_copy(
                    update={
                        "heartbeat_interval_seconds": 0.05,
                        "connection_timeout_seconds": 0.2,
                    }
                )
            }
        )
        app = create_app(impatient, session_root=session_root)
        with TestClient(app) as test_client:
            with test_client.websocket_connect(
                f"/ws/v1/sessions/{SESSION}"
            ) as websocket:
                reply = websocket.receive_json()
                assert reply["payload"]["error_code"] == "handshake_required"
                assert "no client_hello arrived" in reply["payload"]["detail"]

    def test_a_session_stays_live_while_a_client_remains(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as first:
            handshake(first)
            with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as second:
                handshake(
                    second, source=MessageSource.UNITY_CLIENT, role=ClientRole.UNITY
                )
            assert client.get("/health").json()["live_sessions"] == [SESSION]


# --- adaptation transport ---------------------------------------------------


def command_body(
    *,
    command_id: str = "cmd-1",
    command: AdaptationCommandName = AdaptationCommandName.SET_DIFFICULTY,
    value: object = 3,
    target_role: ClientRole = ClientRole.SIMULATOR,
    expires_at_utc: datetime | None = None,
    target_client_id: str | None = None,
) -> dict[str, Any]:
    payload = AdaptationCommandPayload(
        command_id=command_id,
        command=command,
        value=value,  # type: ignore[arg-type]
        reason="manual test command",
        issued_at_utc=datetime.now(UTC),
        expires_at_utc=expires_at_utc,
        target_role=target_role,
        target_client_id=target_client_id,
    )
    return {"command": json.loads(payload.model_dump_json())}


class TestAdaptationTransport:
    def test_a_manual_command_reaches_the_task_client(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            response = client.post(f"/sessions/{SESSION}/commands", json=command_body())
            assert response.status_code == 200
            body = response.json()
            assert body["command_id"] == "cmd-1"
            assert len(body["delivered_to"]) == 1
            assert "No policy generated this command" in body["note"]

            received = websocket.receive_json()
            assert received["message_type"] == "adaptation_command"
            assert received["payload"]["command"] == "set_difficulty"
            assert received["payload"]["value"] == 3
            assert received["payload"]["is_manual"] is True
            assert received["source"] == "backend"

    def test_the_command_and_its_acknowledgement_are_persisted(
        self, client: TestClient
    ) -> None:
        from engagevr.protocol.messages import AdaptationAcknowledgementPayload

        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            client.post(f"/sessions/{SESSION}/commands", json=command_body())
            websocket.receive_json()

            websocket.send_text(
                json.dumps(
                    envelope_json(
                        MessageType.ADAPTATION_ACKNOWLEDGEMENT,
                        sequence_number=1,
                        payload=AdaptationAcknowledgementPayload(
                            command_id="cmd-1",
                            accepted=True,
                            applied_at_utc=datetime.now(UTC),
                        ),
                    )
                )
            )
            websocket.receive_json()

        counts = client.get(f"/sessions/{SESSION}/summary").json()["summary"][
            "message_type_counts"
        ]
        assert counts["adaptation_command"] == 1
        assert counts["adaptation_acknowledgement"] == 1

    def test_a_command_for_a_session_with_no_broker_is_rejected(
        self, client: TestClient
    ) -> None:
        response = client.post("/sessions/nobody-here/commands", json=command_body())
        assert response.status_code == 409
        assert "no live broker" in response.json()["detail"]

    def test_an_expired_command_is_rejected(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            body = command_body()
            body["command"]["issued_at_utc"] = (
                datetime.now(UTC) - timedelta(minutes=10)
            ).isoformat()
            body["command"]["expires_at_utc"] = (
                datetime.now(UTC) - timedelta(minutes=5)
            ).isoformat()
            response = client.post(f"/sessions/{SESSION}/commands", json=body)
            assert response.status_code == 400
            assert "expired" in response.json()["detail"]

    def test_a_command_with_no_matching_target_is_rejected(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            response = client.post(
                f"/sessions/{SESSION}/commands",
                json=command_body(target_role=ClientRole.UNITY),
            )
            assert response.status_code == 404
            assert "no live task client" in response.json()["detail"]

    def test_a_command_is_not_routed_to_an_observer(self, client: TestClient) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as observer:
            handshake(
                observer, role=ClientRole.OBSERVER, source=MessageSource.TEST_FIXTURE
            )
            response = client.post(f"/sessions/{SESSION}/commands", json=command_body())
            assert response.status_code == 404

    def test_a_malformed_command_is_rejected_by_the_schema(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            body = command_body()
            body["command"]["value"] = "not-an-integer"
            response = client.post(f"/sessions/{SESSION}/commands", json=body)
            assert response.status_code == 422

    def test_a_targeted_command_reaches_only_that_client(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as first:
            first_hello = handshake(first)
            first_id = first_hello["payload"]["assigned_client_id"]
            with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as second:
                handshake(
                    second, source=MessageSource.UNITY_CLIENT, role=ClientRole.UNITY
                )
                response = client.post(
                    f"/sessions/{SESSION}/commands",
                    json=command_body(target_client_id=first_id),
                )
                assert response.json()["delivered_to"] == [first_id]

    def test_the_backend_never_generates_a_command_on_its_own(
        self, client: TestClient
    ) -> None:
        """No policy exists: task events must not produce commands."""
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            for n in range(1, 11):
                websocket.send_text(
                    json.dumps(
                        envelope_json(
                            MessageType.TASK_EVENT,
                            sequence_number=n,
                            payload=TaskEventPayload(
                                event=TaskEventDetail(
                                    event_type=EventType.RESPONSE_TIMEOUT,
                                    trial_id=n,
                                )
                            ),
                        )
                    )
                )
                reply = websocket.receive_json()
                assert reply["message_type"] == "acknowledgement", (
                    "the backend must not answer task performance with a command"
                )

        counts = client.get(f"/sessions/{SESSION}/summary").json()["summary"][
            "message_type_counts"
        ]
        assert "adaptation_command" not in counts


# --- replay over the bridge -------------------------------------------------


class TestReplayOverWebSocket:
    def test_a_replayed_message_is_accepted_under_the_replay_session(
        self, client: TestClient
    ) -> None:
        original = build_envelope(
            message_type=MessageType.CLIENT_HELLO,
            session_id="recorded-session",
            source=MessageSource.PYTHON_SIMULATOR,
            sequence_number=0,
            payload=ClientHelloPayload(
                role=ClientRole.SIMULATOR,
                client_name="c",
                client_version="1",
                protocol_version=PROTOCOL_VERSION,
            ),
        )
        replayed = original.with_replay_metadata(
            ReplayMetadata(
                source_session_id="recorded-session",
                replay_session_id="replay-target",
                replay_index=0,
                replay_speed=0.0,
                replayed_at_utc=datetime.now(UTC),
            )
        )
        with client.websocket_connect("/ws/v1/sessions/replay-target") as websocket:
            websocket.send_text(json.dumps(replayed.to_json_dict()))
            reply = websocket.receive_json()
            assert reply["message_type"] == "server_hello"

        events = client.get("/sessions/replay-target/events").json()["events"]
        stored = events[0]["envelope"]
        assert stored["session_id"] == "recorded-session", "original id preserved"
        assert stored["replay"]["replay_label"] == "REPLAY"
        assert stored["provenance"]["synthetic_label"] == "SYNTHETIC"

    def test_a_live_message_for_the_wrong_session_is_still_rejected(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect("/ws/v1/sessions/replay-target") as websocket:
            websocket.send_text(json.dumps(hello_json(session_id="somewhere-else")))
            reply = websocket.receive_json()
            assert reply["payload"]["error_code"] == "session_mismatch"


# --- privacy ---------------------------------------------------------------


class TestBackendPrivacy:
    def test_the_backend_produces_no_estimate_of_any_kind(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            handshake(websocket)
            websocket.send_text(
                json.dumps(envelope_json(MessageType.TASK_EVENT, sequence_number=1))
            )
            websocket.receive_json()

        raw = client.get(f"/sessions/{SESSION}/events").json()
        text = json.dumps(raw["events"]).lower()
        for token in ("engagement", "cognitive_load", "attention", "fatigue", "score"):
            assert token not in text

    def test_backend_messages_are_marked_live_not_synthetic(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect(f"/ws/v1/sessions/{SESSION}") as websocket:
            reply = handshake(websocket)
            assert reply["provenance"]["data_source"] == DataSource.LIVE.value
            assert reply["provenance"]["synthetic_label"] is None
