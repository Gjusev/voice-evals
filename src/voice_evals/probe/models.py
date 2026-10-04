"""Formats, frames, events, capabilities and session records.

All local timestamps are integer nanoseconds relative to the session origin
(the shared monotonic clock started when the transport became ready). Provider
timestamps stay in their own clock domain and are never mixed into local
arithmetic.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from .audio import AudioFormat


class EventKind(str, enum.Enum):
    """Normalized agent-session event kinds."""

    SESSION_READY = "session_ready"
    RESPONSE_START = "response_start"
    RESPONSE_DONE = "response_done"
    AGENT_AUDIO = "agent_audio"
    AGENT_TEXT_DELTA = "agent_text_delta"
    AGENT_TEXT_FINAL = "agent_text_final"
    CALLER_TRANSCRIPT_DELTA = "caller_transcript_delta"
    CALLER_TRANSCRIPT_FINAL = "caller_transcript_final"
    OUTCOME = "outcome"
    STAGE = "stage"
    INTERRUPTED = "interrupted_ack"
    ERROR = "error"


@dataclass
class AudioPayload:
    """Audio carried by one application media message."""

    data: bytes
    samples: int
    audio_format: AudioFormat
    sequence: int | None = None


@dataclass
class AgentEvent:
    """One normalized event observed from the agent session."""

    kind: EventKind
    receipt_ns: int  # local monotonic, session-relative, stamped before processing
    raw_type: str | None = None
    response_id: str | None = None
    utterance_id: str | None = None
    segment_id: str | None = None
    direction: str | None = None  # "agent" | "caller" | "system"
    text: str | None = None
    audio: AudioPayload | None = None
    stage: str | None = None  # e.g. "llm_start" | "tts_start", provider-named
    provider_time: str | None = None
    provider_clock: str | None = None
    payload: dict[str, Any] | None = None  # allowlisted provider extras

    def to_journal_dict(self) -> dict[str, Any]:
        import base64

        out: dict[str, Any] = {
            "kind": self.kind.value,
            "receipt_ns": self.receipt_ns,
            "raw_type": self.raw_type,
            "response_id": self.response_id,
            "utterance_id": self.utterance_id,
            "direction": self.direction,
        }
        if self.text is not None:
            out["text"] = self.text
        if self.audio is not None:
            out["audio"] = {
                "b64": base64.b64encode(self.audio.data).decode("ascii"),
                "samples": self.audio.samples,
                "format": self.audio.audio_format.public_dict(),
                "sequence": self.audio.sequence,
            }
        if self.stage is not None:
            out["stage"] = self.stage
        if self.provider_time is not None:
            out["provider_time"] = {"value": self.provider_time, "clock": self.provider_clock}
        if self.payload:
            out["payload"] = self.payload
        return out


@dataclass(frozen=True)
class SynthesizedAudio:
    """A complete synthesized caller clip.

    Duration derives from the sample count, never from HTTP chunk counts.
    """

    text: str
    pcm: bytes
    audio_format: AudioFormat
    samples: int
    provenance: dict[str, Any]

    @property
    def duration_ns(self) -> int:
        return self.audio_format.duration_ns(self.samples)

    @property
    def content_hash(self) -> str:
        import hashlib

        return hashlib.sha256(self.pcm).hexdigest()


@dataclass(frozen=True)
class CallerAudioFrame:
    """One paced frame of caller audio handed to the transport."""

    utterance_id: str
    sequence: int
    payload: bytes
    sample_offset: int  # samples before this frame within the utterance
    sample_count: int
    audio_format: AudioFormat
    planned_send_ns: int  # session-relative deadline for paced send


@dataclass(frozen=True)
class SendReceipt:
    """Local timestamps around the actual socket send.

    Explicitly a client-side send observation, not a remote playback ack.
    """

    utterance_id: str
    sequence: int
    send_start_ns: int  # immediately before the socket send
    send_complete_ns: int  # immediately after
    bytes_sent: int


@dataclass(frozen=True)
class ControlReceipt:
    kind: str  # "end_utterance" | "cancel_response"
    sent_ns: int
    ok: bool
    detail: str | None = None


class TranscriptSemantics(str, enum.Enum):
    DELTA = "delta"  # append
    SNAPSHOT = "snapshot"  # replace prior text for the segment


@dataclass
class TransportCapabilities:
    """What the transport/protocol can expose. Declared vs observed.

    ``declared`` comes from the protocol map/handshake; ``observed`` is filled
    during the session (e.g. an input_transcript_final actually arrived).
    """

    input_format: AudioFormat | None = None
    output_format: AudioFormat | None = None
    duplex: bool = False
    incremental_output: bool = False  # streaming audio vs burst
    response_ids: bool = False
    input_transcripts: bool = False
    agent_transcripts: bool = False
    response_end_events: bool = False
    stage_events: bool = False
    explicit_input_commit: bool = False  # end_utterance maps to a real message
    cancellation: bool = False
    playback_ack: bool = False
    outcome_events: bool = False
    transcript_semantics: TranscriptSemantics = TranscriptSemantics.DELTA
    declared: dict[str, Any] = field(default_factory=dict)
    observed: dict[str, Any] = field(default_factory=dict)

    def note_observed(self, key: str, value: Any = True) -> None:
        self.observed[key] = value

    def public_dict(self) -> dict[str, Any]:
        return {
            "input_format": self.input_format.public_dict() if self.input_format else None,
            "output_format": self.output_format.public_dict() if self.output_format else None,
            "duplex": self.duplex,
            "incremental_output": self.incremental_output,
            "response_ids": self.response_ids,
            "input_transcripts": self.input_transcripts,
            "agent_transcripts": self.agent_transcripts,
            "response_end_events": self.response_end_events,
            "stage_events": self.stage_events,
            "explicit_input_commit": self.explicit_input_commit,
            "cancellation": self.cancellation,
            "playback_ack": self.playback_ack,
            "outcome_events": self.outcome_events,
            "transcript_semantics": self.transcript_semantics.value,
            "declared": self.declared,
            "observed": self.observed,
        }


@dataclass
class FrameSendLog:
    """Per-frame send evidence for one caller utterance."""

    sequence: int
    planned_send_ns: int
    send_start_ns: int
    send_complete_ns: int
    bytes_sent: int
    sample_offset: int
    sample_count: int

    @property
    def lateness_ns(self) -> int:
        return self.send_start_ns - self.planned_send_ns


class SessionStatus(str, enum.Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentObservation:
    """What the agent said/decided in one completed response.

    Used for branch predicates: finalized agent text plus explicit intent
    events only. No whole-call concatenation for branching.
    """

    response_id: str | None
    final_text: str
    intents: list[str] = field(default_factory=list)

    def contains_any(self, needles: list[str]) -> bool:
        lowered = self.final_text.lower()
        return any(n.lower() in lowered for n in needles)


@dataclass
class InterruptionObservation:
    """One barge-in attempt with its measurement status."""

    step_id: str
    target_response_id: str | None
    at_ns: int  # B: first successful caller media send start
    agent_stopped_ns: int | None  # L - B when observed
    status: str  # observed | not_stopped | missed | unestablished | unsupported | timeout
    basis: str  # measurement basis note
    quiet_confirm_ns: int | None = None  # when quiet-window inference confirmed
    uncertainty_note: str | None = None
    response_active_at_b: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "target_response_id": self.target_response_id,
            "at_ns": self.at_ns,
            "agent_stopped_ns": self.agent_stopped_ns,
            "status": self.status,
            "basis": self.basis,
            "quiet_confirm_ns": self.quiet_confirm_ns,
            "uncertainty_note": self.uncertainty_note,
            "response_active_at_b": self.response_active_at_b,
        }


@dataclass
class TurnRecord:
    """Per-turn grouping of one caller utterance with its agent response."""

    step_id: str
    utterance_id: str
    selected_text: str
    alternative_index: int
    renders_facts: list[str]
    response_id: str | None
    frames: list[FrameSendLog] = field(default_factory=list)
    caller_audio_end_ns: int | None = None  # C_end, paced client estimate
    scheduled_end_ns: int | None = None
    interrupt: InterruptionObservation | None = None
    skipped: bool = False
    # Timing extraction (filled by timing.py)
    e2e_ns: int | None = None
    e2e_overlap_ns: int | None = None  # signed diagnostic when negative
    client_final_asr_ns: int | None = None
    caller_asr_text: str | None = None
    agent_first_audio_ns: int | None = None
    agent_final_text: str | None = None
    response_done_ns: int | None = None
    llm_ttft_ns: int | None = None
    tts_ttfa_ns: int | None = None
    asr_final_to_agent_text_ns: int | None = None
    first_text_to_audio_ns: int | None = None
    timing_notes: list[str] = field(default_factory=list)

    @property
    def is_interruption_turn(self) -> bool:
        return self.interrupt is not None

    @property
    def sent(self) -> bool:
        return bool(self.frames)


@dataclass
class SessionRecord:
    """Everything one probe session captured, minus secrets."""

    session_id: str
    scenario: Any  # ScenarioScript
    status: SessionStatus
    status_reason: str | None
    environment: str  # "local" | "kaggle" | "mock"
    created_utc: str
    turns: list[TurnRecord] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)
    capabilities: TransportCapabilities | None = None
    synthesis: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = ""
    outcome_rule_used: str | None = None
    outcome_evidence: dict[str, Any] = field(default_factory=dict)
    connection: dict[str, Any] = field(default_factory=dict)
    transcript_segments: list[dict[str, Any]] = field(default_factory=list)
    session_origin_ns: int = 0
    total_duration_ns: int = 0
    counters: dict[str, int] = field(default_factory=dict)
