"""Typed, log-safe failures for the probe.

Error messages must never contain secrets, raw provider response bodies, or
signed URLs. Errors carry a category, a safe message, and optional remediation.
"""

from __future__ import annotations


class ProbeError(Exception):
    """Base class for probe failures. Safe to print and to serialize."""

    category = "probe_error"
    remediation: str | None = None

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if remediation is not None:
            self.remediation = remediation

    def to_dict(self) -> dict[str, str | None]:
        return {
            "category": self.category,
            "message": self.message,
            "remediation": self.remediation,
        }


class ScenarioError(ProbeError):
    """Invalid or inconsistent scenario script."""

    category = "scenario_invalid"


class ProtocolMapError(ProbeError):
    """Invalid or unsupported protocol map."""

    category = "protocol_map_invalid"


class CapabilityError(ProbeError):
    """The transport or protocol cannot support a requested behavior."""

    category = "capability_unsupported"


class CallerVoiceError(ProbeError):
    """Caller TTS failure (synthesis, format, quota)."""

    category = "caller_voice_error"


class CallerAuthError(CallerVoiceError):
    """Authentication or quota rejected; not retried."""

    category = "caller_auth"


class CallerFormatError(CallerVoiceError):
    """Caller audio is not the requested native PCM."""

    category = "caller_format"


class TransportError(ProbeError):
    """Connection, handshake, or socket failure."""

    category = "transport_error"


class ProtocolError(ProbeError):
    """Malformed or uncorrelatable events on the wire."""

    category = "protocol_failure"


class RecordingError(ProbeError):
    """Journal/audio write failure or recorder overflow."""

    category = "recording_error"


class ConfigError(ProbeError):
    """Invalid or contradictory configuration."""

    category = "config_invalid"


class TimeoutKind(str):
    """Namespace for timeout categories recorded in results."""


class ResponseTimeout(ProbeError):
    """A bounded wait for agent activity expired."""

    category = "response_timeout"

    def __init__(self, what: str, timeout_ms: float) -> None:
        super().__init__(f"timeout waiting for {what} after {timeout_ms:.0f}ms")
        self.what = what
        self.timeout_ms = timeout_ms


class ScenarioUnmet(ProbeError):
    """The agent did not satisfy a required script behavior."""

    category = "scenario_unmet"


class BargeInMissed(ProbeError):
    """A required interrupt window was missed."""

    category = "barge_in_missed"
