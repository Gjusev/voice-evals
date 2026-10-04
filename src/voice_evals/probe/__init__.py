"""Public probe interfaces. No network imports at package import time.

Replay APIs, the runner, interfaces, mocks, and fixtures import with the base
installation (jiwer only). Network adapters (ElevenLabs/HTTP callers, the
WebSocket transport) import their libraries inside their modules and are
reached only through the factories below.
"""

from __future__ import annotations

import json
from pathlib import Path

from .audio import AudioChunk, AudioFormat, frame_pcm, pcm_to_wav_bytes
from .clock import MonotonicClock, VirtualClock
from .config import CallerConfig, ProbeConfig, TransportConfig, resolve_env
from .errors import (
    BargeInMissed,
    CallerAuthError,
    CallerFormatError,
    CallerVoiceError,
    CapabilityError,
    ConfigError,
    ProbeError,
    ProtocolError,
    ProtocolMapError,
    RecordingError,
    ResponseTimeout,
    ScenarioError,
    ScenarioUnmet,
    TransportError,
)
from .models import (
    AgentEvent,
    AgentObservation,
    AudioPayload,
    CallerAudioFrame,
    ControlReceipt,
    EventKind,
    FrameSendLog,
    InterruptionObservation,
    SendReceipt,
    SessionRecord,
    SessionStatus,
    SynthesizedAudio,
    TranscriptSemantics,
    TransportCapabilities,
    TurnRecord,
)
from .recording import SessionRecorder, read_journal
from .reporting import ProbeReport, score_session
from .runner import SessionRunner
from .scenario import (
    Condition,
    Cue,
    OutcomeRule,
    ScenarioScript,
    ScriptStep,
    SelectedUtterance,
    Utterance,
)
from .testing import MockAgentPolicy, MockCallerVoice, MockReplyRule, MockTransport
from .timing import ResponseTracker, ordinary_turn_means
from .transports.base import AgentTransport
from .voices.base import CallerVoice

__all__ = [
    "AgentEvent",
    "AgentObservation",
    "AgentTransport",
    "AudioChunk",
    "AudioFormat",
    "AudioPayload",
    "BargeInMissed",
    "CallerAudioFrame",
    "CallerAuthError",
    "CallerConfig",
    "CallerFormatError",
    "CallerVoice",
    "CallerVoiceError",
    "CapabilityError",
    "Condition",
    "ConfigError",
    "ControlReceipt",
    "Cue",
    "EventKind",
    "FrameSendLog",
    "InterruptionObservation",
    "MockAgentPolicy",
    "MockCallerVoice",
    "MockReplyRule",
    "MockTransport",
    "MonotonicClock",
    "OutcomeRule",
    "ProbeConfig",
    "ProbeError",
    "ProbeReport",
    "ProtocolError",
    "ProtocolMapError",
    "RecordingError",
    "ResponseTimeout",
    "ResponseTracker",
    "ScenarioError",
    "ScenarioScript",
    "ScenarioUnmet",
    "ScriptStep",
    "SelectedUtterance",
    "SendReceipt",
    "SessionRecord",
    "SessionRecorder",
    "SessionRunner",
    "SessionStatus",
    "SynthesizedAudio",
    "TranscriptSemantics",
    "TransportCapabilities",
    "TransportConfig",
    "TransportError",
    "TurnRecord",
    "Utterance",
    "VirtualClock",
    "build_caller",
    "build_transport",
    "frame_pcm",
    "load_protocol_map",
    "ordinary_turn_means",
    "pcm_to_wav_bytes",
    "read_journal",
    "resolve_env",
    "score_session",
]


def build_caller(config: ProbeConfig, clock=None):
    """Factory for caller voices; network adapters import lazily."""
    caller = config.caller
    if caller.kind == "mock":
        from .testing import MockCallerVoice as _Mock

        return _Mock(clock=clock)
    if caller.kind == "elevenlabs":
        from .voices.elevenlabs import ElevenLabsCallerVoice

        return ElevenLabsCallerVoice(caller, clock=clock)
    if caller.kind == "http":
        from .voices.http import HttpCallerVoice

        return HttpCallerVoice(caller, clock=clock)
    if caller.kind == "fixture":
        from .voices.fixture import FixtureCallerVoice

        return FixtureCallerVoice(caller.fixture_manifest or "fixtures/manifest.json")
    raise ConfigError(f"unknown caller kind {caller.kind!r}")


def build_transport(config: ProbeConfig, clock=None):
    """Factory for transports: mock pair or the real WebSocket client."""
    if config.caller.kind == "mock":
        from .testing import MockTransport as _MockT

        return _MockT(clock=clock)
    from .transports.websocket import WebSocketAgentTransport

    return WebSocketAgentTransport(load_protocol_map(config), clock=clock)


def load_protocol_map(config: ProbeConfig):
    """Load the protocol map from path/env or the bundled default."""
    from .transports.protocol import ProtocolMap

    path = config.transport.protocol_map_path
    if path is None:
        from .transports.protocol import load_default_protocol_map

        return load_default_protocol_map()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProtocolMap(raw)
