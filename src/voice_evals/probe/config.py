"""Typed probe configuration, env resolution, and safe public configuration.

Secrets are referenced by environment variable name and resolved only at
execution time. ``public_dict()`` outputs never contain endpoint URLs with
credentials, API keys, or signed query strings: they record env var names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio import AudioFormat
from .errors import ConfigError

DEFAULT_FRAME_MS = 20
DEFAULT_CONNECT_TIMEOUT_MS = 10_000
DEFAULT_READY_TIMEOUT_MS = 10_000
DEFAULT_FIRST_RESPONSE_TIMEOUT_MS = 15_000
DEFAULT_RESPONSE_TIMEOUT_MS = 30_000
DEFAULT_CALL_TIMEOUT_MS = 120_000
DEFAULT_DRAIN_TIMEOUT_MS = 2_000
DEFAULT_AGENT_QUIET_MS = 300
DEFAULT_MAX_BARGE_IN_STOP_MS = 500
DEFAULT_BARGE_OBSERVATION_TIMEOUT_MS = 5_000
DEFAULT_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_EVENT_JOURNAL_ENTRIES = 100_000
DEFAULT_MAX_AUDIO_BYTES_PER_DIRECTION = 200 * 1024 * 1024
DEFAULT_MAX_UTTERANCE_CHARS = 600
DEFAULT_MAX_TOTAL_SYNTHESIS_CHARS = 10_000


def resolve_env(name: str, *, required: bool = False) -> str:
    """Read an environment variable; empty/absent behaves like missing."""
    value = os.environ.get(name, "").strip()
    if required and not value:
        raise ConfigError(f"environment variable {name!r} is not set", remediation=f"export {name}=...")
    return value


@dataclass
class CallerConfig:
    """Caller voice selection. Only env names are serialized, never values."""

    kind: str = "mock"  # "mock" | "elevenlabs" | "http" | "fixture"
    voice_id: str | None = None
    voice_id_env: str = "ELEVENLABS_VOICE_ID"
    api_key_env: str = "ELEVENLABS_API_KEY"
    model_id: str = "eleven_multilingual_v2"
    base_url: str = "https://api.elevenlabs.io"
    # http caller
    url: str | None = None
    url_env: str = "CALLER_TTS_URL"
    auth_env: str | None = None
    language: str = "en"
    # fixture caller
    fixture_manifest: str | None = None
    retries: int = 2

    def resolved_voice_id(self) -> str:
        vid = self.voice_id or os.environ.get(self.voice_id_env, "").strip()
        if not vid:
            raise ConfigError(
                "caller voice id missing",
                remediation=(
                    f"pass --voice-id or set {self.voice_id_env}; an API key alone "
                    "does not identify a voice"
                ),
            )
        return vid

    def public_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "voice_id": self.voice_id,
            "voice_id_env": self.voice_id_env,
            "api_key_env": self.api_key_env,
            "model_id": self.model_id,
            "base_url": self.base_url if self.kind == "elevenlabs" else None,
            "url_env": self.url_env,
            "auth_env": self.auth_env,
            "language": self.language,
            "fixture_manifest": self.fixture_manifest,
            "retries": self.retries,
        }


@dataclass
class TransportConfig:
    """Where and how to reach the agent. Secrets excluded from serialization."""

    endpoint: str | None = None
    endpoint_env: str = "PROBE_TRANSPORT_URL"
    api_key_env: str | None = None  # e.g. PROBE_AGENT_API_KEY
    protocol_map_path: str | None = None
    protocol_map: dict[str, Any] | None = None
    frame_ms: int = DEFAULT_FRAME_MS
    handshake_timeout_ms: int = DEFAULT_READY_TIMEOUT_MS
    heartbeat_ms: int | None = None
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES

    def resolved_endpoint(self) -> str:
        endpoint = self.endpoint or os.environ.get(self.endpoint_env, "").strip()
        if not endpoint:
            raise ConfigError(
                "transport endpoint missing",
                remediation=f"pass --transport or set {self.endpoint_env}",
            )
        if endpoint.startswith("wss://"):
            return endpoint
        if endpoint.startswith("ws://"):
            return endpoint  # allowed for local development
        raise ConfigError(
            "transport endpoint must be a ws:// or wss:// URL",
            remediation="WSS uses certificate verification; WS is allowed for local dev",
        )

    def resolved_api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        value = os.environ.get(self.api_key_env, "").strip()
        return value or None

    def public_dict(self) -> dict[str, Any]:
        return {
            "endpoint": _redact_url(self.endpoint),
            "endpoint_env": self.endpoint_env,
            "api_key_env": self.api_key_env,
            "protocol_map_path": self.protocol_map_path,
            "frame_ms": self.frame_ms,
            "handshake_timeout_ms": self.handshake_timeout_ms,
            "heartbeat_ms": self.heartbeat_ms,
            "max_message_bytes": self.max_message_bytes,
        }


def _redact_url(url: str | None) -> str | None:
    """Keep scheme+host, drop userinfo and query (may carry signatures)."""
    if not url:
        return None
    kept = url
    for sep in ("?", ";"):
        kept = kept.split(sep, 1)[0]
    if "://" in kept:
        rest = kept.split("://", 1)[1]
        if "@" in rest:
            host = rest.split("@", 1)[1]
            kept = kept.split("://", 1)[0] + "://" + host
    return kept


@dataclass
class ProbeConfig:
    """Timeless knobs for one probe run. Every value is recorded in the manifest."""

    environment: str = "local"  # "local" | "kaggle" | "mock"
    output_dir: Path = field(default_factory=lambda: Path("out/probe"))
    session_id: str | None = None
    seed_override: int | None = None
    input_format: AudioFormat = field(default_factory=lambda: AudioFormat(sample_rate=16000))
    output_format: AudioFormat = field(default_factory=lambda: AudioFormat(sample_rate=16000))
    # timeouts (ms)
    connect_timeout_ms: int = DEFAULT_CONNECT_TIMEOUT_MS
    ready_timeout_ms: int = DEFAULT_READY_TIMEOUT_MS
    first_response_timeout_ms: int = DEFAULT_FIRST_RESPONSE_TIMEOUT_MS
    response_timeout_ms: int = DEFAULT_RESPONSE_TIMEOUT_MS
    call_timeout_ms: int = DEFAULT_CALL_TIMEOUT_MS
    drain_timeout_ms: int = DEFAULT_DRAIN_TIMEOUT_MS
    # barge-in
    agent_quiet_ms: int = DEFAULT_AGENT_QUIET_MS
    max_barge_in_stop_ms: int = DEFAULT_MAX_BARGE_IN_STOP_MS
    barge_observation_timeout_ms: int = DEFAULT_BARGE_OBSERVATION_TIMEOUT_MS
    # output bounds
    max_event_journal_entries: int = DEFAULT_MAX_EVENT_JOURNAL_ENTRIES
    max_audio_bytes_per_direction: int = DEFAULT_MAX_AUDIO_BYTES_PER_DIRECTION
    max_utterance_chars: int = DEFAULT_MAX_UTTERANCE_CHARS
    max_total_synthesis_chars: int = DEFAULT_MAX_TOTAL_SYNTHESIS_CHARS
    # gate requests (None = not requested)
    max_wer: float | None = None
    min_task_completion: float | None = None
    min_fact_coverage: float | None = None
    max_e2e_p95_ms: float | None = None
    max_hallucination_rate: float | None = None
    request_barge_in_gate: bool = False

    transport: TransportConfig = field(default_factory=TransportConfig)
    caller: CallerConfig = field(default_factory=CallerConfig)

    def effective_seed(self, scenario_seed: int) -> int:
        return scenario_seed if self.seed_override is None else self.seed_override

    def validate(self) -> None:
        if self.transport.frame_ms <= 0:
            raise ConfigError("frame_ms must be positive")
        self.input_format.validate()
        self.output_format.validate()
        if self.caller.kind == "mock" and (self.transport.endpoint or os.environ.get(self.transport.endpoint_env)):
            # mock together with a live endpoint is contradictory
            raise ConfigError(
                "contradictory configuration: --mock selects MockTransport; "
                "unset the transport endpoint to run mock"
            )

    def public_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "session_id": self.session_id,
            "seed_override": self.seed_override,
            "input_format": self.input_format.public_dict(),
            "output_format": self.output_format.public_dict(),
            "timeouts_ms": {
                "connect": self.connect_timeout_ms,
                "ready": self.ready_timeout_ms,
                "first_response": self.first_response_timeout_ms,
                "response": self.response_timeout_ms,
                "call": self.call_timeout_ms,
                "drain": self.drain_timeout_ms,
            },
            "barge_in": {
                "agent_quiet_ms": self.agent_quiet_ms,
                "max_barge_in_stop_ms": self.max_barge_in_stop_ms,
                "barge_observation_timeout_ms": self.barge_observation_timeout_ms,
            },
            "limits": {
                "max_event_journal_entries": self.max_event_journal_entries,
                "max_audio_bytes_per_direction": self.max_audio_bytes_per_direction,
                "max_utterance_chars": self.max_utterance_chars,
                "max_total_synthesis_chars": self.max_total_synthesis_chars,
            },
            "gates": {
                "max_wer": self.max_wer,
                "min_task_completion": self.min_task_completion,
                "min_fact_coverage": self.min_fact_coverage,
                "max_e2e_p95_ms": self.max_e2e_p95_ms,
                "max_hallucination_rate": self.max_hallucination_rate,
                "barge_in": self.request_barge_in_gate,
            },
            "transport": self.transport.public_dict(),
            "caller": self.caller.public_dict(),
        }
