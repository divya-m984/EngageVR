"""Deterministic JSON Schema generation for protocol version 1.

The generated document is checked into the repository at
``protocol/engagevr-protocol-v1.schema.json`` and is the artefact the
Unity client is written against.  A test regenerates it and compares,
so the checked-in copy cannot drift from the Pydantic models.

The document is a single schema whose root is the envelope, with a
``oneOf`` branch per message type binding ``message_type`` to its
payload schema.  That shape lets a generic validator check both the
envelope and the payload in one pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic.json_schema import models_json_schema

from engagevr.protocol.envelope import MessageEnvelope
from engagevr.protocol.messages import PAYLOAD_MODELS, MessageType
from engagevr.protocol.version import ACCEPTED_MAJOR_VERSIONS, PROTOCOL_VERSION

SCHEMA_ID = "https://engagevr.local/schemas/engagevr-protocol-v1.schema.json"

#: Path of the checked-in schema, relative to the repository root.
SCHEMA_RELATIVE_PATH = Path("protocol") / "engagevr-protocol-v1.schema.json"


def build_protocol_json_schema() -> dict[str, Any]:
    """Build the complete JSON Schema document for protocol version 1."""
    ordered_types = sorted(MessageType, key=lambda t: t.value)
    key_models: list[tuple[type[Any], str]] = [(MessageEnvelope, "validation")]
    key_models += [(PAYLOAD_MODELS[t], "validation") for t in ordered_types]

    _keys, combined = models_json_schema(
        key_models,  # type: ignore[arg-type]
        ref_template="#/$defs/{model}",
        title="EngageVR real-time protocol",
    )
    defs: dict[str, Any] = dict(combined.get("$defs", {}))

    branches: list[dict[str, Any]] = []
    for message_type in ordered_types:
        payload_model = PAYLOAD_MODELS[message_type]
        branches.append(
            {
                "title": message_type.value,
                "properties": {
                    "message_type": {"const": message_type.value},
                    "payload": {"$ref": f"#/$defs/{payload_model.__name__}"},
                },
                "required": ["message_type", "payload"],
            }
        )

    envelope_ref = f"#/$defs/{MessageEnvelope.__name__}"
    document: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "EngageVR protocol message",
        "description": (
            "One EngageVR real-time protocol message. Protocol version "
            f"{PROTOCOL_VERSION}; accepted major versions: "
            f"{list(ACCEPTED_MAJOR_VERSIONS)}. Payloads carry task and "
            "transport telemetry only: no engagement value, no "
            "cognitive-load value, no behavioural or physiological "
            "measurement, and no raw frame data is representable."
        ),
        "x-protocol-version": PROTOCOL_VERSION,
        "x-accepted-major-versions": list(ACCEPTED_MAJOR_VERSIONS),
        "x-message-types": [t.value for t in ordered_types],
        "allOf": [{"$ref": envelope_ref}],
        "oneOf": branches,
        "$defs": defs,
    }
    return document


def render_protocol_json_schema() -> str:
    """Render the schema document exactly as it is stored on disk."""
    return json.dumps(build_protocol_json_schema(), indent=2, sort_keys=True) + "\n"


def write_protocol_json_schema(path: Path) -> Path:
    """Write the schema document to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_protocol_json_schema(), encoding="utf-8")
    return path
