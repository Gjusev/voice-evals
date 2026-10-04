"""Response correlation, timing extraction, and barge-in observation units."""

import asyncio

from voice_evals.probe.audio import AudioFormat, tone_pcm
from voice_evals.probe.clock import VirtualClock
from voice_evals.probe.models import (
    AgentEvent,
    AgentObservation,
    AudioPayload,
    EventKind,
    FrameSendLog,
    InterruptionObservation,
    TranscriptSemantics,
    TransportCapabilities,
    TurnRecord,
)
from voice_evals.probe.timing import (
    BargeInObserver,
    ResponseTracker,
    extract_turn_timing,
    ordinary_turn_means,
)

FMT = AudioFormat(sample_rate=16000)
CAPS = TransportCapabilities(
    duplex=True,
    incremental_output=True,
    response_ids=True,
    input_transcripts=True,
    agent_transcripts=True,
    response_end_events=True,
)


def audio_event(rid: str, receipt_ns: int) -> AgentEvent:
    pcm = tone_pcm(FMT, 20)
    return AgentEvent(
        kind=EventKind.AGENT_AUDIO,
        receipt_ns=receipt_ns,
        response_id=rid,
        direction="agent",
        audio=AudioPayload(data=pcm, samples=FMT.sample_count(pcm), audio_format=FMT),
    )


def test_correlation_with_provider_ids() -> None:
    tracker = ResponseTracker(CAPS)
    tracker.apply(AgentEvent(kind=EventKind.RESPONSE_START, receipt_ns=100, response_id="r1"))
    tracker.apply(audio_event("r1", 150))
    tracker.apply(audio_event("r1", 200))
    tracker.apply(AgentEvent(kind=EventKind.RESPONSE_DONE, receipt_ns=300, response_id="r1"))
    state = tracker.by_id("r1")
    assert state is not None
    assert state.first_audio_ns == 150 and state.last_audio_ns == 200 and state.done_ns == 300


def test_serial_correlation_without_ids() -> None:
    no_ids = TransportCapabilities(**{**vars(CAPS), "response_ids": False})
    tracker = ResponseTracker(no_ids)
    # Caller transcript must not open a response of its own.
    rid = tracker.apply(AgentEvent(kind=EventKind.CALLER_TRANSCRIPT_FINAL, receipt_ns=10, text="hi", utterance_id="u1"))
    assert rid == "caller"
    assert tracker.responses == {}
    rid = tracker.apply(AgentEvent(kind=EventKind.AGENT_AUDIO, receipt_ns=50))
    assert rid == "serial-1"
    rid = tracker.apply(AgentEvent(kind=EventKind.AGENT_TEXT_FINAL, receipt_ns=60, text="hello"))
    assert rid == "serial-1"
    tracker.apply(AgentEvent(kind=EventKind.RESPONSE_DONE, receipt_ns=70))
    rid = tracker.apply(AgentEvent(kind=EventKind.AGENT_AUDIO, receipt_ns=100))
    assert rid == "serial-2"


def test_overlapping_unlabeled_responses_flag_ambiguity() -> None:
    no_ids = TransportCapabilities(**{**vars(CAPS), "response_ids": False})
    tracker = ResponseTracker(no_ids)
    tracker.apply(AgentEvent(kind=EventKind.RESPONSE_START, receipt_ns=10))
    tracker.apply(AgentEvent(kind=EventKind.RESPONSE_START, receipt_ns=20))
    assert tracker.ambiguous_overlap
    assert tracker.responses["serial-1"].ambiguous  # the overlapped old response
    clock = VirtualClock()
    observer = BargeInObserver(
        target=tracker.responses["serial-1"], clock=clock, agent_quiet_ms=300, observation_timeout_ms=1000
    )
    observation = asyncio.run(observer.observe(15))
    assert observation.status == "unsupported" and observation.agent_stopped_ns is None


def test_session_level_events_never_open_responses() -> None:
    tracker = ResponseTracker(CAPS)
    rid = tracker.apply(AgentEvent(kind=EventKind.SESSION_READY, receipt_ns=0))
    assert rid == "session" and tracker.responses == {}


def test_snapshot_vs_delta_transcripts() -> None:
    snapshot_caps = TransportCapabilities(**{**vars(CAPS), "transcript_semantics": TranscriptSemantics.SNAPSHOT})
    tracker = ResponseTracker(snapshot_caps)
    tracker.apply(AgentEvent(kind=EventKind.AGENT_TEXT_DELTA, receipt_ns=10, response_id="r1", text="Hello"))
    tracker.apply(AgentEvent(kind=EventKind.AGENT_TEXT_DELTA, receipt_ns=20, response_id="r1", text="Hello there"))
    assert tracker.by_id("r1").final_text == "Hello there"  # replaced, never concatenated

    delta_tracker = ResponseTracker(CAPS)
    delta_tracker.apply(AgentEvent(kind=EventKind.AGENT_TEXT_DELTA, receipt_ns=10, response_id="r1", text="Hello "))
    delta_tracker.apply(AgentEvent(kind=EventKind.AGENT_TEXT_DELTA, receipt_ns=20, response_id="r1", text="there"))
    assert delta_tracker.by_id("r1").final_text == "Hello there"


def _turn(step_id: str = "a", e2e_ns=None, asr_ns=None, sent=True, interrupt=None, llm_ns=None, tts_ns=None) -> TurnRecord:
    turn = TurnRecord(
        step_id=step_id,
        utterance_id=f"utt-{step_id}",
        selected_text="hello",
        alternative_index=0,
        renders_facts=[],
        response_id="r1",
        e2e_ns=e2e_ns,
        client_final_asr_ns=asr_ns,
        llm_ttft_ns=llm_ns,
        tts_ttfa_ns=tts_ns,
        interrupt=interrupt,
    )
    if sent:
        turn.frames.append(
            FrameSendLog(0, 0, 0, 1_000_000, 1_000_000, 640, 320)
        )
        turn.caller_audio_end_ns = 1_000_000
    return turn


def test_ordinary_means_exclude_interruption_and_overlap_turns() -> None:
    ordinary = _turn("a", e2e_ns=800_000_000, asr_ns=200_000_000)
    interrupted = _turn("b", e2e_ns=100_000_000, asr_ns=100_000_000, interrupt=InterruptionObservation("b", None, 0, 0, "observed", "test"))
    overlap = _turn("c", e2e_ns=None, asr_ns=100_000_000)
    overlap.e2e_overlap_ns = -50_000_000  # signed: response began before caller end
    means, counts = ordinary_turn_means([ordinary, interrupted, overlap])
    assert counts["e2e_ms"] == 1
    assert means["e2e_ms"] == 800.0
    # Interruption turns are excluded from ordinary means entirely (documented).
    assert counts["stt_ms"] == 2
    assert means["stt_ms"] == (200_000_000 + 100_000_000) / 2 / 1e6


def test_extract_turn_timing_keeps_missing_data_honest() -> None:
    turn = _turn()
    extract_turn_timing(turn, None, [])
    assert turn.e2e_ns is None
    assert any("no correlated agent response" in note for note in turn.timing_notes)

    events = [
        AgentEvent(kind=EventKind.CALLER_TRANSCRIPT_FINAL, receipt_ns=250_000_000, utterance_id="utt-a", text="hi", direction="caller"),
    ]

    class _State:
        first_audio_ns = 900_000_000
        last_audio_ns = 900_000_000
        final_text = "hello there"
        first_text_ns = 400_000_000
        done_ns = 1_000_000_000
        started_ns = 300_000_000
        stage_events: dict = {}  # noqa: RUF012

    turn2 = _turn()
    turn2.caller_audio_end_ns = 200_000_000
    extract_turn_timing(turn2, _State(), events)
    assert turn2.e2e_ns == 700_000_000
    assert turn2.client_final_asr_ns == 50_000_000
    # Proxies reported, legacy stage names stay None without stage events
    assert turn2.llm_ttft_ns is None and turn2.tts_ttfa_ns is None
    assert turn2.asr_final_to_agent_text_ns == 150_000_000
    assert turn2.first_text_to_audio_ns == 500_000_000


def test_llm_tts_only_with_explicit_stage_events() -> None:
    class _State:
        first_audio_ns = 900_000_000
        last_audio_ns = 900_000_000
        final_text = "x"
        first_text_ns = 400_000_000
        done_ns = None
        started_ns = 300_000_000
        stage_events = {"llm_start": 300_000_000, "llm_first_token": 350_000_000, "tts_start": 500_000_000}  # noqa: RUF012

    turn = _turn()
    turn.caller_audio_end_ns = 200_000_000
    extract_turn_timing(turn, _State(), [])
    assert turn.llm_ttft_ns == 50_000_000
    assert turn.tts_ttfa_ns == 400_000_000


def test_barge_in_observer_zero_stop() -> None:
    clock = VirtualClock()

    class _State:
        response_id = "r1"
        started_ns = 0
        first_audio_ns = 50
        last_audio_ns = 100  # nothing after B
        audio_frames = 3
        done_ns = 500
        final_text = None
        first_text_ns = None
        stage_events: dict = {}  # noqa: RUF012
        ended_reason = "event"
        ambiguous = False
        done = property(lambda self: self.done_ns is not None)

    observer = BargeInObserver(target=_State(), clock=clock, agent_quiet_ms=300, observation_timeout_ms=1000)
    observation = asyncio.run(observer.observe(150))
    assert observation.status == "observed"
    assert observation.agent_stopped_ns == 0
    assert observation.uncertainty_note


def test_barge_in_observer_censored_on_ongoing_audio() -> None:
    clock = VirtualClock()

    class _StillTalking:
        """Audio keeps arriving: the quiet window never confirms a stop."""

        response_id = "r1"
        started_ns = 0
        first_audio_ns = 50
        audio_frames = 10_000
        done_ns = None
        final_text = None
        first_text_ns = None
        stage_events: dict = {}  # noqa: RUF012
        ended_reason = None
        ambiguous = False

        @property
        def last_audio_ns(self):
            return clock.now_ns() - 50_000_000  # fresh audio 50ms ago, always

        done = property(lambda self: self.done_ns is not None)

    observer = BargeInObserver(
        target=_StillTalking(), clock=clock, agent_quiet_ms=300, observation_timeout_ms=1000
    )
    observation = asyncio.run(observer.observe(150))
    assert observation.status == "not_stopped"
    assert observation.agent_stopped_ns is None
    assert "right-censored" in observation.basis


def test_agent_observation_predicate() -> None:
    observation = AgentObservation(response_id="r1", final_text="May I have your name, please?")
    assert observation.contains_any(["your name"])
    assert not observation.contains_any(["WHAT TIME"])
