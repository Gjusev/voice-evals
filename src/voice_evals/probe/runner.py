"""Full-duplex session runner: paced sending, live receiving, bounded waits.

One task consumes ``transport.events()``; one serialized sender paces caller
frames on cumulative sample deadlines so drift does not accumulate. The
receiver stays active during waits, caller speech, and interruption
measurement. Python 3.10 compatible: ``create_task``/``wait_for``-style
bounded waits via polling on the injected clock, explicit cancellation, and
``gather(..., return_exceptions=True)``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import AudioFormat, frame_pcm
from .clock import NS_PER_MS, Clock, MonotonicClock
from .config import ProbeConfig
from .conversion import check_eligibility, evaluate_outcome, to_call_record
from .errors import (
    BargeInMissed,
    ProbeError,
    ProtocolError,
    RecordingError,
    ResponseTimeout,
    ScenarioError,
    ScenarioUnmet,
    TransportError,
)
from .models import (
    AgentEvent,
    AgentObservation,
    CallerAudioFrame,
    FrameSendLog,
    SessionRecord,
    SessionStatus,
    TurnRecord,
)
from .recording import SessionRecorder, new_session_id
from .reporting import score_session
from .scenario import ScenarioScript, SelectedUtterance
from .timing import (
    AGGREGATION_LABEL,
    BargeInObserver,
    ResponseTracker,
    extract_turn_timing,
)
from .transports.base import AgentTransport
from .voices.base import CallerVoice

_POLL_NS = 10 * NS_PER_MS


class SessionRunner:
    """Runs one scripted probe session and records everything it observes."""

    def __init__(
        self,
        *,
        caller: CallerVoice,
        transport: AgentTransport,
        config: ProbeConfig,
        clock: Clock | None = None,
    ) -> None:
        self.caller = caller
        self.transport = transport
        self.config = config
        self.clock = clock or MonotonicClock()
        self._last_report: Any = None

    async def run(self, script: ScenarioScript, *, output_dir: Path) -> SessionRecord:
        session_id = self.config.session_id or new_session_id()
        session = SessionRecord(
            session_id=session_id,
            scenario=script,
            status=SessionStatus.FAILED,
            status_reason=None,
            environment=self.config.environment,
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        recorder = SessionRecorder(
            Path(output_dir),
            session_id=session_id,
            config_public=self.config.public_dict(),
            script_dict=script.script_dict(),
            protocol_hash=None,
            limits={
                "max_event_journal_entries": self.config.max_event_journal_entries,
                "max_audio_bytes_per_direction": self.config.max_audio_bytes_per_direction,
            },
        )
        self._origin: int = 0
        self._tracker: ResponseTracker | None = None
        self._receiver: asyncio.Task | None = None
        try:
            await self._run_inner(session, script, recorder)
        except ProbeError as error:
            session.status = SessionStatus.FAILED
            session.status_reason = error.category
            session.errors.append(error.to_dict())
        except asyncio.CancelledError:
            session.status = SessionStatus.CANCELLED
            session.status_reason = "cancelled"
            raise
        except Exception as error:
            session.status = SessionStatus.FAILED
            session.status_reason = "unexpected_error"
            session.errors.append({"category": "unexpected_error", "message": repr(error), "remediation": None})
            raise
        finally:
            await self._finalize(session, recorder)
        return session

    # -- phases -------------------------------------------------------------

    async def _run_inner(self, session: SessionRecord, script: ScenarioScript, recorder: SessionRecorder) -> None:
        config = self.config
        # A scenario that interrupts requests interruption evidence: the gate
        # fails closed on missing/censored attempts even without --max-barge-in-stop-ms.
        if any(s.cue.mode == "interrupt" for s in script.steps):
            config.request_barge_in_gate = True
        # VALIDATING: structural checks that need no transport
        self._check_synthesis_limits(script)
        # PREPARING_AUDIO: synthesize every reachable clip before connecting, so
        # caller TTS latency never contaminates agent measurements.
        _selected, clips = await self._prepare_audio(script, session)
        session.synthesis = [
            {
                "step_id": step_id,
                "text": sel.text,
                "alternative_index": sel.alternative_index,
                "samples": clip.samples,
                "duration_ns": clip.duration_ns,
                "content_hash": clip.content_hash,
                "provenance": clip.provenance,
            }
            for step_id, (sel, clip) in clips.items()
        ]
        # CONNECTING
        connect_start = self.clock.now_ns()
        try:
            capabilities = await self._wait_for(
                lambda: self.transport.open(config.transport),
                config.connect_timeout_ms,
                "transport open",
            )
        except ResponseTimeout as error:
            raise TransportError("transport did not open in time") from error
        socket_open = self.clock.now_ns()
        script.validate_for_probe(capabilities)
        session.capabilities = capabilities
        self._origin = self.clock.now_ns()
        self._tracker = ResponseTracker(capabilities)
        session.connection = {
            "connect_start_offset_ns": self._rel(connect_start),
            "socket_open_offset_ns": self._rel(socket_open),
            "ready_offset_ns": 0,
        }
        session.session_origin_ns = 0
        self._receiver = asyncio.ensure_future(self._receive_loop(session, recorder, capabilities))
        # ACTIVE: script steps
        turn_by_step: dict[str, TurnRecord] = {}
        skipped_steps: set[str] = set()
        call_deadline = self.clock.now_ns() + config.call_timeout_ms * NS_PER_MS
        stop_reason: str | None = None
        for step in script.steps:
            if self.clock.now_ns() > call_deadline:
                stop_reason = "call_timeout"
                break
            if any(ref in skipped_steps for ref in (step.cue.response_to, step.when.response_to if step.when else None)):
                raise ScenarioError(
                    f"step {step.id!r} depends on a skipped step (unreachable dependency)"
                )
            try:
                turn = await self._run_step(
                    session, script, step, clips, turn_by_step, skipped_steps, recorder
                )
            except (ScenarioUnmet, BargeInMissed) as error:
                session.errors.append(error.to_dict())
                stop_reason = error.category
                break
            if turn is not None:
                turn_by_step[step.id] = turn
            else:
                skipped_steps.add(step.id)
        # DRAIN: await the final turn's response and late transcripts.
        last = turn_by_step.get(script.steps[-1].id)
        if last is not None and last.caller_audio_end_ns is not None and not stop_reason:
            await self._drain(last, config)
        session.status = SessionStatus.COMPLETED if not stop_reason else SessionStatus.FAILED
        session.status_reason = stop_reason
        session.total_duration_ns = self.clock.now_ns() - self._origin_abs()

    async def _run_step(
        self,
        session: SessionRecord,
        script: ScenarioScript,
        step: Any,
        clips: dict[str, tuple[SelectedUtterance, Any]],
        turn_by_step: dict[str, TurnRecord],
        skipped_steps: set[str],
        recorder: SessionRecorder,
    ) -> TurnRecord | None:
        config = self.config
        selected = clips[step.id][0]
        clip = clips[step.id][1]
        utterance_id = f"utt-{step.id}"
        turn = TurnRecord(
            step_id=step.id,
            utterance_id=utterance_id,
            selected_text=selected.text,
            alternative_index=selected.alternative_index,
            renders_facts=list(selected.revealed_facts),
            response_id=None,
        )
        cue = step.cue
        target_state = None
        if cue.response_to:
            referenced = turn_by_step.get(cue.response_to)
            if referenced is None:
                raise ScenarioError(
                    f"step {step.id!r} cue references {cue.response_to!r}, which did not run"
                )
        if cue.mode == "wait_agent_end":
            timeout_ms = min(cue.timeout_ms or config.response_timeout_ms, config.response_timeout_ms)
            await self._wait_until(
                lambda: (state := self._state_for_turn(referenced)) is not None and state.done,
                timeout_ms,
                f"agent response end for {cue.response_to!r}",
            )
            target_state = self._state_for_turn(referenced)
        elif cue.mode == "interrupt":
            await self._wait_until(
                lambda: (state := self._state_for_turn(referenced)) is not None
                and state.first_audio_ns is not None,
                cue.timeout_ms or config.first_response_timeout_ms,
                f"first agent audio for {cue.response_to!r}",
            )
            target_state = self._state_for_turn(referenced)
        # Branch predicate over the referenced completed response.
        if step.when is not None:
            when_target = turn_by_step.get(step.when.response_to)
            if when_target is None:
                raise ScenarioError(
                    f"step {step.id!r} 'when' references {step.when.response_to!r}, which did not run"
                )
            observation = AgentObservation(
                response_id=when_target.response_id,
                final_text=self._final_text_for(when_target),
            )
            picked = script.select_utterance(step.id, observation)
            if picked is None:
                if step.on_unmatched == "fail":
                    raise ScenarioUnmet(
                        f"step {step.id!r}: agent response to {step.when.response_to!r} "
                        f"matched none of {list(step.when.any_text)}"
                    )
                turn.skipped = True
                session.turns.append(turn)
                session.counters["skipped_steps"] = session.counters.get("skipped_steps", 0) + 1
                return None
        # Send: paced on cumulative sample duration.
        await self._send_utterance(
            session, turn, clip, step, target_state, recorder
        )
        session.turns.append(turn)
        if turn.skipped:
            return None  # skipped (missed interrupt window with if_missed=skip)
        return turn

    async def _send_utterance(
        self,
        session: SessionRecord,
        turn: TurnRecord,
        clip: Any,
        step: Any,
        target_state: Any,
        recorder: SessionRecorder,
    ) -> None:
        config = self.config
        fmt: AudioFormat = clip.audio_format
        frames = frame_pcm(clip.pcm, fmt, config.transport.frame_ms)
        planned_start = self.clock.now_ns()
        offset_samples = 0
        b_ns: int | None = None
        interrupt_cue = step.cue.mode == "interrupt"
        active_at_arm = None
        if interrupt_cue:
            assert target_state is not None
            first_audio = target_state.first_audio_ns
            assert first_audio is not None
            fire_at = first_audio + (step.cue.after_ms or 0) * NS_PER_MS
            await self._sleep_until_rel(fire_at)
            active_at_arm = self._response_active(target_state)
            if not active_at_arm:
                from .models import InterruptionObservation

                turn.interrupt = InterruptionObservation(
                    step_id=step.id,
                    target_response_id=target_state.response_id,
                    at_ns=self._rel(self.clock.now_ns()),
                    agent_stopped_ns=None,
                    status="missed",
                    basis=(
                        "target response not observably active at after_ms "
                        f"({step.cue.after_ms}ms after its first audio): burst output or already ended"
                    ),
                )
                session.counters["missed_interrupts"] = session.counters.get("missed_interrupts", 0) + 1
                if step.cue.if_missed == "fail":
                    session.turns.append(turn)
                    raise BargeInMissed(turn.interrupt.basis)
                turn.skipped = True
                return  # _run_step appends the turn and marks the step skipped
        for sequence, payload in enumerate(frames):
            planned = planned_start + fmt.duration_ns(offset_samples)
            await self._sleep_until_abs(planned)
            frame = CallerAudioFrame(
                utterance_id=turn.utterance_id,
                sequence=sequence,
                payload=payload,
                sample_offset=offset_samples,
                sample_count=fmt.sample_count(payload),
                audio_format=fmt,
                planned_send_ns=self._rel(planned),
            )
            receipt = await self.transport.send_audio(frame)
            rel_start = self._rel(receipt.send_start_ns)
            rel_complete = self._rel(receipt.send_complete_ns)
            if b_ns is None and interrupt_cue:
                b_ns = rel_start
            turn.frames.append(
                FrameSendLog(
                    sequence=sequence,
                    planned_send_ns=self._rel(planned),
                    send_start_ns=rel_start,
                    send_complete_ns=rel_complete,
                    bytes_sent=receipt.bytes_sent,
                    sample_offset=offset_samples,
                    sample_count=fmt.sample_count(payload),
                )
            )
            recorder.add_caller_audio(turn.utterance_id, payload, fmt)
            offset_samples += fmt.sample_count(payload)
        if turn.frames:
            last = turn.frames[-1]
            turn.caller_audio_end_ns = last.send_complete_ns + fmt.duration_ns(last.sample_count)
            turn.scheduled_end_ns = last.planned_send_ns + fmt.duration_ns(last.sample_count)
            recorder.mark_caller_complete(turn.utterance_id)
        if self._tracker is not None and self._tracker.capabilities.explicit_input_commit:
            await self.transport.end_utterance(turn.utterance_id)
        # Natural barge-in measurement (no cancellation is ever sent).
        if interrupt_cue and b_ns is not None and active_at_arm:
            assert target_state is not None
            observer = BargeInObserver(
                target=target_state,
                clock=_RelClock(self.clock, self._origin_abs),
                agent_quiet_ms=config.agent_quiet_ms,
                observation_timeout_ms=config.barge_observation_timeout_ms,
            )
            observation = await observer.observe(b_ns)
            observation.step_id = step.id
            observation.response_active_at_b = True  # verified at arm time, before sending
            turn.interrupt = observation

    async def _drain(self, last_turn: TurnRecord, config: ProbeConfig) -> None:
        assert self._tracker is not None
        deadline = self.clock.now_ns() + config.drain_timeout_ms * NS_PER_MS

        def settled() -> bool:
            state = self._state_for_turn(last_turn)
            return state is not None and state.done

        try:
            await self._wait_until(lambda: settled(), config.response_timeout_ms, "final response completion")
        except ResponseTimeout:
            pass  # bounded: late evidence stays diagnostic; missing data stays missing
        # Extra bounded window for late final transcripts only.
        while self.clock.now_ns() < deadline:
            await self.clock.sleep_ns(_POLL_NS)

    # -- receiver -----------------------------------------------------------

    async def _receive_loop(
        self, session: SessionRecord, recorder: SessionRecorder, capabilities: Any
    ) -> None:
        assert self._tracker is not None
        try:
            async for event in self.transport.events():
                rel = event.receipt_ns - self._origin_abs()
                normalized = AgentEvent(
                    kind=event.kind,
                    receipt_ns=rel,
                    raw_type=event.raw_type,
                    response_id=event.response_id,
                    utterance_id=event.utterance_id,
                    segment_id=event.segment_id,
                    direction=event.direction,
                    text=event.text,
                    audio=event.audio,
                    stage=event.stage,
                    provider_time=event.provider_time,
                    provider_clock=event.provider_clock,
                    payload=event.payload,
                )
                rid = self._tracker.apply(normalized)
                await recorder.write_event(normalized)
                if normalized.audio is not None:
                    recorder.add_agent_audio(rid, normalized.audio.data, normalized.audio.audio_format)
                capabilities.note_observed(f"saw_{normalized.kind.value}")
                session.events.append(normalized)
                self._update_transcripts(session, normalized)
        except asyncio.CancelledError:
            raise
        except (TransportError, ProtocolError) as error:
            session.errors.append(error.to_dict())
        except RecordingError as error:
            # Recorder overflow/disk failure: stop consuming, fail visibly.
            session.errors.append(error.to_dict())
        except Exception as error:  # noqa: BLE001 - receiver must never kill the run silently
            session.errors.append({"category": "receiver_error", "message": repr(error), "remediation": None})

    def _update_transcripts(self, session: SessionRecord, event: AgentEvent) -> None:
        if event.kind.value not in (
            "caller_transcript_final",
            "agent_text_final",
        ) or event.text is None:
            return
        session.transcript_segments.append(
            {
                "role": "caller" if event.kind.value == "caller_transcript_final" else "agent",
                "text": event.text,
                "utterance_id": event.utterance_id,
                "response_id": event.response_id,
                "final_ns": event.receipt_ns,
            }
        )

    # -- finalize -------------------------------------------------------------

    async def _finalize(self, session: SessionRecord, recorder: SessionRecorder) -> None:
        if self._receiver is not None:
            self._receiver.cancel()
            await asyncio.gather(self._receiver, return_exceptions=True)
        try:
            await self.transport.aclose()
        except Exception as error:  # noqa: BLE001 - best-effort close during finalize
            session.errors.append({"category": "close_error", "message": repr(error), "remediation": None})
        await self.caller.aclose()
        if any(e.get("category") == "recording_error" for e in session.errors):
            session.status = SessionStatus.FAILED
            session.status_reason = "recording_error"
        if self._tracker is not None:
            self._tracker.close_open_responses(
                reason="close", at_ns=self.clock.now_ns() - self._origin_abs()
            )
            self._correlate_and_extract(session)
        session.outcome, session.outcome_rule_used, session.outcome_evidence = evaluate_outcome(session)
        # ASR text per turn (final transcripts matched by utterance id).
        for turn in session.turns:
            event = next(
                (
                    e
                    for e in session.events
                    if e.kind.value == "caller_transcript_final" and e.utterance_id == turn.utterance_id
                ),
                None,
            )
            turn.caller_asr_text = event.text if event else None
        eligible, reasons = check_eligibility(session)
        report = score_session(session, self.config)
        from ..gates import (  # local import: gates live outside probe
            metric_gate_problems,
            probe_gate_problems,
        )

        problems: list[str] = []
        if eligible:
            problems += metric_gate_problems(
                report.result,
                {
                    "max_wer": self.config.max_wer,
                    "min_task_completion": self.config.min_task_completion,
                    "min_fact_coverage": self.config.min_fact_coverage,
                    "max_e2e_p95_ms": self.config.max_e2e_p95_ms,
                    "max_hallucination_rate": self.config.max_hallucination_rate,
                },
            )
        problems += probe_gate_problems(
            report.result,
            request_barge_in_gate=self.config.request_barge_in_gate,
            max_barge_in_stop_ms=self.config.max_barge_in_stop_ms,
            max_e2e_p95_ms=self.config.max_e2e_p95_ms,
        )
        # Operational errors dominate quality failures.
        operational = {
            "transport_error", "caller_auth", "caller_voice_error", "caller_format",
            "recording_error", "config_invalid", "protocol_failure", "protocol_map_invalid",
            "capability_unsupported", "cancelled",
        }
        if any(e.get("category") in operational for e in session.errors):
            problems.insert(0, "operational/execution failure present")
        report.result["gate_passed"] = not problems
        report.probe["gate_reasons"] = problems
        record = to_call_record(session)
        rows = []
        if record is not None:
            row = record_to_row(record)
            rows.append(row)
        try:
            manifest_extra = {
                "scoring": {
                    "eligible": eligible,
                    "exclusion_reasons": reasons,
                    "aggregation": AGGREGATION_LABEL,
                    "replayable": record is not None,
                },
                "stage_support": dict(report.probe.get("metric_support", {})),
            }
            recorder.write_calls(rows, replayable=record is not None)
            recorder.write_result({**report.result})
            # Manifest finalizes last so its file map hashes every artifact.
            await recorder.finalize(session, manifest_extra)
        except RecordingError as error:
            session.errors.append(error.to_dict())
        self._last_report = report

    def _correlate_and_extract(self, session: SessionRecord) -> None:
        assert self._tracker is not None
        tracker = self._tracker
        for turn in session.turns:
            if turn.skipped or not turn.frames:
                continue
            state = None
            if turn.caller_audio_end_ns is not None:
                state = tracker.first_started_after(turn.caller_audio_end_ns)
            if state is None:
                state = tracker.by_id(turn.response_id)
            if state is not None:
                turn.response_id = state.response_id
            extract_turn_timing(turn, state, session.events)

    # -- prepare / helpers ------------------------------------------------------

    def _check_synthesis_limits(self, script: ScenarioScript) -> None:
        config = self.config
        for step in script.steps:
            for text in (step.utterance.text, *step.utterance.alternatives):
                rendered = text  # placeholders only shrink/shuffle; check raw length
                if len(rendered) > config.max_utterance_chars:
                    raise ScenarioError(
                        f"step {step.id!r}: text exceeds max_utterance_chars={config.max_utterance_chars}"
                    )
        total = sum(
            len(s.utterance.text) + sum(len(a) for a in s.utterance.alternatives)
            for s in script.steps
        )
        if total > config.max_total_synthesis_chars:
            raise ScenarioError(
                f"scenario total text {total} exceeds max_total_synthesis_chars={config.max_total_synthesis_chars}"
            )

    async def _prepare_audio(
        self, script: ScenarioScript, session: SessionRecord
    ) -> tuple[dict[str, SelectedUtterance], dict[str, tuple[SelectedUtterance, Any]]]:
        seed = self.config.effective_seed(script.seed)
        selected: dict[str, SelectedUtterance] = {}
        clips: dict[str, tuple[SelectedUtterance, Any]] = {}
        cache: dict[str, Any] = {}
        for step in script.steps:
            pick = script.select_utterance(step.id, None)
            if pick is None:
                raise ScenarioError(f"step {step.id!r} selection unexpectedly failed")
            selected[step.id] = pick
            if pick.text in cache:
                clips[step.id] = (pick, cache[pick.text])
                continue
            clip = await self.caller.synthesize(
                pick.text, audio_format=self.config.input_format, seed=seed
            )
            cache[pick.text] = clip
            clips[step.id] = (pick, clip)
        return selected, clips

    def _response_active(self, state: Any) -> bool:
        """Genuinely incremental output or trustworthy media progress."""
        if state is None or state.done:
            return False
        now = self._rel(self.clock.now_ns())
        last = state.last_audio_ns
        if last is None:
            return False
        if now - last > self.config.agent_quiet_ms * NS_PER_MS:
            return False
        return state.audio_frames > 0

    def _final_text_for(self, turn: TurnRecord) -> str:
        state = self._state_for_turn(turn)
        return (state.final_text or "") if state else ""

    def _state_for_turn(self, turn: TurnRecord) -> Any:
        """Correlated response state; correlates lazily once audio ended."""
        if self._tracker is None:
            return None
        state = self._tracker.by_id(turn.response_id)
        if state is None and turn.caller_audio_end_ns is not None:
            state = self._tracker.first_started_after(turn.caller_audio_end_ns)
            if state is not None:
                turn.response_id = state.response_id
        return state

    async def _wait_for(self, factory: Any, timeout_ms: float, what: str) -> Any:
        """Bounded wait for a coroutine result (transport open etc.)."""
        task = asyncio.ensure_future(factory())
        deadline = self.clock.now_ns() + int(timeout_ms * NS_PER_MS)
        while not task.done():
            remaining = deadline - self.clock.now_ns()
            if remaining <= 0:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise ResponseTimeout(what, timeout_ms)
            await self.clock.sleep_ns(min(_POLL_NS, remaining))
        return task.result()

    async def _wait_until(self, predicate: Any, timeout_ms: float, what: str) -> None:
        deadline = self.clock.now_ns() + int(timeout_ms * NS_PER_MS)
        while not predicate():
            if self.clock.now_ns() >= deadline:
                raise ResponseTimeout(what, timeout_ms)
            await self.clock.sleep_ns(_POLL_NS)

    async def _sleep_until_abs(self, deadline_ns: int) -> None:
        while True:
            remaining = deadline_ns - self.clock.now_ns()
            if remaining <= 0:
                return
            await self.clock.sleep_ns(min(_POLL_NS, remaining))

    async def _sleep_until_rel(self, rel_deadline_ns: int) -> None:
        await self._sleep_until_abs(self._origin_abs() + rel_deadline_ns)

    def _origin_abs(self) -> int:
        return self._origin

    def _rel(self, abs_ns: int) -> int:
        return abs_ns - self._origin

    @property
    def last_report(self) -> Any:
        return getattr(self, "_last_report", None)


def record_to_row(record: Any) -> dict[str, Any]:
    """Serialize a CallRecord into the v0.1 replay row format."""
    row: dict[str, Any] = {
        "id": record.id,
        "scenario": {
            "name": record.scenario.name,
            "expected_outcome": record.scenario.expected_outcome,
            "required_facts": list(record.scenario.required_facts),
            "forbidden_facts": list(record.scenario.forbidden_facts),
        },
        "asr_transcript": record.asr_transcript,
        "ground_truth_transcript": record.ground_truth_transcript,
        "agent_transcript": record.agent_transcript,
        "outcome": record.outcome,
    }
    if record.stage_timings is not None:
        # Full float precision: a live session and its exported replay record
        # must produce identical legacy scores.
        row["stage_timings_ms"] = record.stage_timings.values()
    if record.interruptions:
        row["interruptions"] = [
            {
                "at_ms": i.at_ms,
                "agent_stopped_ms": i.agent_stopped_ms,
            }
            for i in record.interruptions
        ]
    return row


class _RelClock:
    """Clock view that reports session-relative time to observers."""

    def __init__(self, clock: Clock, origin_abs: callable) -> None:
        self._clock = clock
        self._origin = origin_abs

    def now_ns(self) -> int:
        return self._clock.now_ns() - self._origin()

    async def sleep_ns(self, duration_ns: int) -> None:
        await self._clock.sleep_ns(duration_ns)
