"""Protocol version constants and parsing.

The EngageVR real-time protocol is versioned as ``MAJOR.MINOR``.

Compatibility rule
------------------
A receiver accepts a message when the message's **major** version is in
:data:`ACCEPTED_MAJOR_VERSIONS`.  Minor versions are additive: a
receiver running ``1.0`` accepts ``1.1`` and ignores fields it does not
know.  A different major version is rejected outright with a
``unsupported_protocol_version`` protocol error rather than being
parsed on a best-effort basis.

This module is deliberately free of any dependency on the rest of the
package so that it can be read by tooling and mirrored in the Unity
client without pulling in Pydantic.
"""

from __future__ import annotations

import re
from typing import Final

#: The protocol version produced by this build.
PROTOCOL_VERSION: Final[str] = "1.0"

#: Major versions this build is able to parse.
ACCEPTED_MAJOR_VERSIONS: Final[tuple[int, ...]] = (1,)

_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(\d+)\.(\d+)$")


class ProtocolVersionError(ValueError):
    """Raised when a protocol version string is malformed or unsupported."""


def parse_protocol_version(value: str) -> tuple[int, int]:
    """Parse ``"MAJOR.MINOR"`` into an integer pair.

    Parameters
    ----------
    value:
        Version string exactly as it appeared on the wire.

    Raises
    ------
    ProtocolVersionError
        If ``value`` is not two dot-separated non-negative integers.
    """
    match = _VERSION_PATTERN.match(value.strip()) if isinstance(value, str) else None
    if match is None:
        raise ProtocolVersionError(
            f"malformed protocol version {value!r}; expected 'MAJOR.MINOR'"
        )
    return int(match.group(1)), int(match.group(2))


def is_supported_version(value: str) -> bool:
    """Return True when ``value``'s major version is accepted by this build."""
    try:
        major, _minor = parse_protocol_version(value)
    except ProtocolVersionError:
        return False
    return major in ACCEPTED_MAJOR_VERSIONS


def require_supported_version(value: str) -> tuple[int, int]:
    """Parse and check a version, raising on an unsupported major version.

    Raises
    ------
    ProtocolVersionError
        If the version is malformed or its major version is not accepted.
    """
    major, minor = parse_protocol_version(value)
    if major not in ACCEPTED_MAJOR_VERSIONS:
        accepted = ", ".join(str(v) for v in ACCEPTED_MAJOR_VERSIONS)
        raise ProtocolVersionError(
            f"unsupported protocol major version {major} (from {value!r}); "
            f"this build accepts major version(s): {accepted}"
        )
    return major, minor
