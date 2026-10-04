"""Response correlation, timing extraction, and barge-in observation.

Every measurement states its basis and clock domain. Local arithmetic uses
session-relative integer nanoseconds from one shared monotonic clock; provider
timestamps are stored verbatim and never mixed in. Missing measurements stay
``None`` with a recorded availability reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from .clock import NS_PER_MS, Clock
from .errors import CapabilityError
from .models import (
    AgentEvent,
    EventKind,
    InterruptionObservation,
    TranscriptSemantics,
    TransportCapabilities,
    TurnRecord,
)

AGGREGATION_LABEL = "mean_of_observed_nonoverlap_turns"


@dataclass
class ResponseState:
    """Live view of one agent response, updated per received event."""

    response_id: str
    synthetic: bool = False
    started_ns: int | None = None
    first_audio_ns: int | None = None
    last_audio_ns: int | None = None
    audio_frames: int = 0
    audio_bytes: int = 0
    first_text_ns: int | None = None
    final_text: str | None = None
    asr_final_ns: int | None = None  # first final caller transcript during response window
    done_ns: int | None = None
    ended_reason: str | None = None  # "event" | "quiet" | "close"
    ambiguous: bool = False  # old/new overlap that serial correlation cannot split
    stage_events: dict[str, int] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.done_ns is not None


class ResponseTracker:
    """Tracks response state from normalized events.

    With provider response IDs every event attributes directly. Without them a
    serial association is used: at most one open response; events while none is
    open start a synthetic one. A second response opening while another is open
    and unfinished marks both ambiguous, which disables precise barge-in timing.
    """

    def __init__(self, capabilities: TransportCapabilities) -> None:
        self.capabilities = capabilities
        self.responses: dict[str, ResponseState] = {}
        self.order: list[str] = []
        self._synthetic_counter = 0
        self.ambiguous_overlap = False

    # -- updates ----------------------------------------------------------

    def apply(self, event: AgentEvent) -> str:
        """Attribute the event to a response; returns the response id used."""
        rid = event.response_id
        if rid is None:
            rid = self._serial_target(event)
            if rid in ("session", "caller"):
                return rid  # non-response streams never open response state
        state = self.responses.get(rid)
        if state is None:
            state = self._open(rid, synthetic=event.response_id is None, at_ns=event.receipt_ns)
        if event.kind == EventKind.RESPONSE_START and state.started_ns is None:
            state.started_ns = event.receipt_ns
        elif event.kind == EventKind.AGENT_AUDIO:
            if state.first_audio_ns is None:
                state.first_audio_ns = event.receipt_ns
            state.last_audio_ns = event.receipt_ns
            state.audio_frames += 1
            state.audio_bytes += len(event.audio.data) if event.audio else 0
        elif event.kind in (EventKind.AGENT_TEXT_DELTA, EventKind.AGENT_TEXT_FINAL):
            if state.first_text_ns is None:
                state.first_text_ns = event.receipt_ns
            if event.kind == EventKind.AGENT_TEXT_FINAL or self.capabilities.transcript_semantics == TranscriptSemantics.SNAPSHOT:
                state.final_text = event.text
            elif event.text:
                state.final_text = (state.final_text or "") + event.text
        elif event.kind == EventKind.CALLER_TRANSCRIPT_FINAL:
            if state.asr_final_ns is None:
                state.asr_final_ns = event.receipt_ns
        elif event.kind == EventKind.RESPONSE_DONE:
            state.done_ns = event.receipt_ns
            state.ended_reason = "event"
        elif event.kind == EventKind.STAGE and event.stage:
            state.stage_events[event.stage] = event.receipt_ns
        return rid

    _RESPONSE_SCOPED: ClassVar[frozenset[EventKind]] = frozenset({
        EventKind.AGENT_AUDIO,
        EventKind.AGENT_TEXT_DELTA,
        EventKind.AGENT_TEXT_FINAL,
        EventKind.RESPONSE_START,
        EventKind.RESPONSE_DONE,
        EventKind.STAGE,
        EventKind.OUTCOME,
        EventKind.INTERRUPTED,
    })

    def _serial_target(self, event: AgentEvent) -> str:
        if event.kind in (EventKind.CALLER_TRANSCRIPT_DELTA, EventKind.CALLER_TRANSCRIPT_FINAL):
            # Caller transcripts belong to the input stream; they precede the
            # response and must never open a synthetic response of their own.
            open_states = [s for s in self.responses.values() if not s.done]
            if open_states:
                state = open_states[-1]
                if state.asr_final_ns is None and event.kind == EventKind.CALLER_TRANSCRIPT_FINAL:
                    state.asr_final_ns = event.receipt_ns
                return state.response_id
            return "caller"
        if event.kind not in self._RESPONSE_SCOPED:
            return "session"  # session-level events never open synthetic responses
        open_states = [s for s in self.responses.values() if not s.done]
        if event.kind == EventKind.RESPONSE_START:
            if open_states:
                self.ambiguous_overlap = True
                for state in open_states:
                    state.ambiguous = True
            self._synthetic_counter += 1
            return f"serial-{self._synthetic_counter}"
        if open_states:
            return open_states[-1].response_id
        self._synthetic_counter += 1
        return f"serial-{self._synthetic_counter}"

    def _open(self, rid: str, *, synthetic: bool, at_ns: int) -> ResponseState:
        state = ResponseState(response_id=rid, synthetic=synthetic, started_ns=at_ns)
        self.responses[rid] = state
        self.order.append(rid)
        return state

    def close_open_responses(self, *, reason: str, at_ns: int) -> None:
        for state in self.responses.values():
            if not state.done:
                state.done_ns = at_ns
                state.ended_reason = reason

    # -- lookups ----------------------------------------------------------

    def by_id(self, rid: str | None) -> ResponseState | None:
        return self.responses.get(rid) if rid else None

    def first_started_after(self, ns: int) -> ResponseState | None:
        """First response whose start is at/after ns (serial correlation)."""
        for rid in self.order:
            state = self.responses[rid]
            if state.started_ns is not None and state.started_ns >= ns:
                return state
        return None


# -- per-turn timing -------------------------------------------------------


def extract_turn_timing(turn: TurnRecord, response: ResponseState | None, session_events: list[AgentEvent]) -> None:
    """Fill timing fields on one turn from correlated response state.

    Missing evidence leaves fields None; availability reasons land in
    ``timing_notes``.
    """
    if turn.caller_audio_end_ns is None:
        turn.timing_notes.append("caller_audio_end unavailable")
        return
    c_end = turn.caller_audio_end_ns
    if response is None:
        turn.timing_notes.append("no correlated agent response")
        return
    if response.first_audio_ns is not None:
        delta = response.first_audio_ns - c_end
        if delta >= 0:
            turn.e2e_ns = delta
        else:
            # Response onset before caller end: keep signed diagnostic, exclude
            # from ordinary nonnegative aggregates.
            turn.e2e_overlap_ns = delta
            turn.timing_notes.append("e2e signed overlap: response audio before caller end")
    else:
        turn.timing_notes.append("e2e unavailable: no agent audio for correlated response")
    turn.agent_first_audio_ns = response.first_audio_ns
    turn.agent_final_text = response.final_text
    turn.response_done_ns = response.done_ns

    asr_receipt = _final_caller_transcript_ns(turn.utterance_id, session_events)
    if asr_receipt is not None:
        delta = asr_receipt - c_end
        if delta >= 0:
            turn.client_final_asr_ns = delta
        else:
            turn.timing_notes.append("client final ASR before caller end (signed diagnostic kept)")
            turn.client_final_asr_ns = None
    else:
        turn.timing_notes.append("client final ASR unavailable")

    # Stage timings only when explicit stage events support the legacy names.
    llm_start = response.stage_events.get("llm_start")
    llm_first = response.stage_events.get("llm_first_token") or response.stage_events.get("llm_token_first")
    tts_start = response.stage_events.get("tts_start")
    if llm_start is not None and llm_first is not None:
        turn.llm_ttft_ns = llm_first - llm_start
    if tts_start is not None and response.first_audio_ns is not None:
        turn.tts_ttfa_ns = response.first_audio_ns - tts_start
    # Proxies (kept out of the legacy fields, reported separately).
    if asr_receipt is not None and response.first_text_ns is not None:
        turn.asr_final_to_agent_text_ns = response.first_text_ns - asr_receipt
    if response.first_text_ns is not None and response.first_audio_ns is not None:
        turn.first_text_to_audio_ns = response.first_audio_ns - response.first_text_ns


def _final_caller_transcript_ns(utterance_id: str, events: list[AgentEvent]) -> int | None:
    for event in events:
        if (
            event.kind == EventKind.CALLER_TRANSCRIPT_FINAL
            and event.utterance_id == utterance_id
        ):
            return event.receipt_ns
    return None


def ordinary_turn_means(turns: list[TurnRecord]) -> tuple[dict[str, float], dict[str, int]]:
    """Arithmetic mean of valid ordinary-turn observations per stage field.

    Ordinary = fully sent, not an interruption turn, no signed overlap.
    """
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    fields = (
        ("stt_ms", "client_final_asr_ns"),
        ("llm_ttft_ms", "llm_ttft_ns"),
        ("tts_ttfa_ms", "tts_ttfa_ns"),
        ("e2e_ms", "e2e_ns"),
    )
    for stage_name, attr in fields:
        values = [
            getattr(t, attr) / 1e6
            for t in turns
            if t.sent and not t.is_interruption_turn and getattr(t, attr) is not None
        ]
        if values:
            means[stage_name] = sum(values) / len(values)
            counts[stage_name] = len(values)
    return means, counts


# -- barge-in observation --------------------------------------------------


class BargeInObserver:
    """Observes one natural barge-in stop without sending any cancellation."""

    def __init__(self, *, target: ResponseState, clock: Clock, agent_quiet_ms: int, observation_timeout_ms: int) -> None:
        self.target = target
        self.clock = clock
        self.quiet_ns = agent_quiet_ms * NS_PER_MS
        self.timeout_ns = observation_timeout_ms * NS_PER_MS

    async def observe(self, b_ns: int) -> InterruptionObservation:
        target = self.target
        deadline = b_ns + self.timeout_ns
        active_at_b = target.started_ns is not None and not target.done
        while True:
            now = self.clock.now_ns()
            last = target.last_audio_ns if target.last_audio_ns is not None else b_ns
            frames_after_b = target.last_audio_ns is not None and target.last_audio_ns > b_ns
            quiet_since = now - max(last, b_ns)
            if frames_after_b and (target.done or quiet_since >= self.quiet_ns):
                obs = InterruptionObservation(
                    step_id="",
                    target_response_id=target.response_id,
                    at_ns=b_ns,
                    agent_stopped_ns=target.last_audio_ns - b_ns,
                    status="observed",
                    basis=(
                        "last received agent-audio frame after barge-in start; "
                        + ("explicit response end" if target.done else "quiet-window inference")
                    ),
                    response_active_at_b=active_at_b,
                )
                if not target.done:
                    obs.quiet_confirm_ns = now
                return self._finish(obs)
            if not frames_after_b and target.done and target.done_ns >= b_ns:
                # Demonstrably active at B and ended without any later frame.
                return self._finish(
                    InterruptionObservation(
                        step_id="",
                        target_response_id=target.response_id,
                        at_ns=b_ns,
                        agent_stopped_ns=0,
                        status="observed",
                        basis="explicit response end with no post-barge-in audio frame",
                        uncertainty_note="observed zero: no frame was received after the barge-in",
                        response_active_at_b=active_at_b,
                    )
                )
            if now >= deadline:
                lower = (target.last_audio_ns - b_ns) if frames_after_b else None
                return self._finish(
                    InterruptionObservation(
                        step_id="",
                        target_response_id=target.response_id,
                        at_ns=b_ns,
                        agent_stopped_ns=None,
                        status="not_stopped",
                        basis=(
                            "right-censored: no confirmed end within the observation window; "
                            f"lower bound {lower / 1e6:.1f}ms" if lower is not None
                            else "right-censored: no confirmed end within the observation window"
                        ),
                        response_active_at_b=active_at_b,
                    )
                )
            await self.clock.sleep_ns(min(self.quiet_ns, 20 * NS_PER_MS))

    def _finish(self, obs: InterruptionObservation) -> InterruptionObservation:
        if self.target.ambiguous:
            obs.status = "unsupported"
            obs.basis = "correlation ambiguous between old and new response; precise timing disabled"
            obs.agent_stopped_ns = None
        return obs


def require_barge_in_support(capabilities: TransportCapabilities) -> None:
    if not capabilities.duplex or not capabilities.incremental_output:
        raise CapabilityError(
            "barge-in measurement requires duplex transport with incremental output",
            remediation="burst-completed audio cannot prove ongoing speech at the interrupt point",
        )
