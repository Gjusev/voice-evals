"""MockTransport and MockCallerVoice. Standard library only, no sockets.

These are deterministic test doubles, not a real TTS/STT engine:

- MockCallerVoice "synthesizes" PCM whose sample amplitudes water-mark the
  text bytes (band 1000..1255). It is simulated audio with correct sample
  accounting, not speech synthesis.
- MockTransport decodes that watermark as its "ASR" and plays scripted replies
  as streamed tone frames. Latencies (VAD, thinking, streaming) are simulated
  defaults, not provider measurements.

Both accept an injected clock (real or virtual) so tests run without sleeping
and the offline demo exercises the full runner/recorder/evaluator path.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .audio import AudioFormat, tone_pcm
from .clock import NS_PER_MS, Clock, MonotonicClock
from .config import TransportConfig
from .models import (
    AgentEvent,
    AudioPayload,
    CallerAudioFrame,
    ControlReceipt,
    EventKind,
    SendReceipt,
    SynthesizedAudio,
    TransportCapabilities,
)

_WATERMARK_BASE = 1000
_WATERMARK_MAX = 1255
_MS_PER_BYTE = 50  # simulated speech pacing
_BASE_MS = 800


def encode_text_pcm(text: str, audio_format: AudioFormat) -> bytes:
    """Deterministic PCM that embeds the text bytes as amplitude watermark."""
    marked = text.encode("utf-8")
    duration_ms = _BASE_MS + _MS_PER_BYTE * len(marked)
    samples = audio_format.samples_for_ms(duration_ms)
    tones = tone_pcm(audio_format, duration_ms, amplitude=4000.0)  # outside watermark band
    assert len(tones) >= samples * audio_format.bytes_per_sample()
    out = bytearray(tones[: samples * audio_format.bytes_per_sample()])
    for i, value in enumerate(marked):
        struct.pack_into("<h", out, i * 2, _WATERMARK_BASE + value)
    return bytes(out)


def decode_text_pcm(pcm: bytes) -> str:
    """Recover watermarked text; simulated ASR, not recognition."""
    marks = bytearray()
    for offset in range(0, len(pcm) - 1, 2):
        amp = struct.unpack_from("<h", pcm, offset)[0]
        if _WATERMARK_BASE <= amp <= _WATERMARK_MAX:
            marks.append(amp - _WATERMARK_BASE)
        else:
            break
    return marks.decode("utf-8", errors="replace")


@dataclass
class MockReplyRule:
    match: tuple[str, ...]  # case-insensitive substrings of recognized caller text
    reply: str
    long: bool = False  # stream a long, interruptible response


DEFAULT_REPLY_RULES: tuple[MockReplyRule, ...] = (
    MockReplyRule(
        match=("book an appointment", "arrange a"),
        reply="Of course, I'd be happy to help you book an appointment. May I have your name, please?",
    ),
    MockReplyRule(
        match=("my name is", "put it under"),
        reply=(
            "Thank you. I have availability on Tuesday. I can offer you 10:00 in the morning, "
            "which is our first opening. Would 10:00 work for you? I also have later slots on "
            "Wednesday if that would suit you better, but Tuesday morning at 10:00 is the "
            "closest to your request."
        ),
        long=True,
    ),
    MockReplyRule(
        match=("confirm", "yes"),
        reply=(
            "Perfect. Your appointment is confirmed for Tuesday at 11:00 for Alex Morgan. "
            "You will receive a confirmation shortly. Is there anything else I can help you with?"
        ),
    ),
    MockReplyRule(
        match=("interrupt", "not ten", "instead", "make that"),
        reply=(
            "I'm sorry for the confusion. Let me correct that right away: I will update your "
            "appointment to 11:00 instead of 10:00. Tuesday at 11:00 it is."
        ),
    ),
)

FALLBACK_REPLY = "I'm sorry, could you please repeat that?"


@dataclass
class MockAgentPolicy:
    """Scripted, deterministic agent behavior for tests and the offline demo."""

    rules: tuple[MockReplyRule, ...] = DEFAULT_REPLY_RULES
    fallback_reply: str = FALLBACK_REPLY
    vad_silence_ms: int = 200  # simulated server VAD end-of-utterance
    processing_ms: int = 150  # simulated ASR->first-audio processing
    stream_interval_ms: int = 40  # one audio event every interval
    stream_frame_ms: int = 80  # audio duration per event
    short_reply_ms: int = 1500  # total streamed audio for short replies
    long_reply_ms: int = 6000  # total streamed audio for long (interruptible) replies
    interrupt_stop_ms: int = 260  # stop old audio within this of barge-in
    ignore_interrupts: bool = False  # keep talking: exercises censored barge-in
    asr_substitutions: dict[str, str] = field(default_factory=dict)

    def reply_for(self, recognized: str) -> tuple[str, bool]:
        lowered = recognized.lower()
        for rule in self.rules:
            if any(m.lower() in lowered for m in rule.match):
                return rule.reply, rule.long
        return self.fallback_reply, False

    def asr_hears(self, recognized: str) -> str:
        out = recognized
        for source, target in self.asr_substitutions.items():
            out = out.replace(source, target)
        return out


@dataclass
class _UtteranceBuffer:
    utterance_id: str
    pcm: bytearray = field(default_factory=bytearray)
    last_frame_ns: int = 0
    first_frame_ns: int = 0

    def extend(self, payload: bytes, now_ns: int) -> None:
        if not self.pcm:
            self.first_frame_ns = now_ns
        self.pcm.extend(payload)
        self.last_frame_ns = now_ns


@dataclass
class _ResponseRun:
    response_id: str
    reply: str
    total_ms: int
    started_ns: int
    stopped: bool = False
    frames_sent: int = 0
    interrupt_stop_at_ns: int | None = None


class MockTransport:
    """Full-duplex scripted agent transport. No sockets, no network."""

    def __init__(
        self,
        *,
        policy: MockAgentPolicy | None = None,
        clock: Clock | None = None,
        audio_format: AudioFormat | None = None,
    ) -> None:
        self.policy = policy or MockAgentPolicy()
        self.clock = clock or MonotonicClock()
        self.audio_format = audio_format or AudioFormat(sample_rate=16000)
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._buffer: _UtteranceBuffer | None = None
        self._runs: list[_ResponseRun] = []  # concurrent streams (barge-in overlap)
        self._response_counter = 0
        self._closed = False
        self._tasks: list[asyncio.Task] = []
        self._open_ns = 0
        self._ready_emitted = False

    # -- AgentTransport ----------------------------------------------------

    async def open(self, config: TransportConfig) -> TransportCapabilities:
        self._open_ns = self.clock.now_ns()
        caps = TransportCapabilities(
            input_format=self.audio_format,
            output_format=self.audio_format,
            duplex=True,
            incremental_output=True,
            response_ids=True,
            input_transcripts=True,
            agent_transcripts=True,
            response_end_events=True,
            stage_events=False,
            explicit_input_commit=False,  # native simulated VAD: end_utterance is a no-op
            cancellation=True,
            playback_ack=False,
            outcome_events=False,
        )
        caps.note_observed("opened")
        self._start_background()
        await self._emit(EventKind.SESSION_READY, direction="system")
        self._ready_emitted = True
        return caps

    def events(self) -> AsyncIterator[AgentEvent]:
        return self._consume()

    async def _consume(self) -> AsyncIterator[AgentEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def send_audio(self, frame: CallerAudioFrame) -> SendReceipt:
        now = self.clock.now_ns()
        if self._buffer is None or self._buffer.utterance_id != frame.utterance_id:
            if self._buffer is not None and self._buffer.pcm:
                # New utterance while a previous buffer is unsent-complete: treat
                # prior as finished (server VAD would have closed it already).
                self._schedule_processing(self._buffer)
            self._buffer = _UtteranceBuffer(frame.utterance_id)
            if any(not run.stopped for run in self._runs):
                self._on_barge_in(now)
        assert self._buffer is not None
        self._buffer.extend(frame.payload, now)
        self._maybe_vad_complete(now)
        return SendReceipt(
            utterance_id=frame.utterance_id,
            sequence=frame.sequence,
            send_start_ns=now,
            send_complete_ns=self.clock.now_ns(),
            bytes_sent=len(frame.payload),
        )

    async def end_utterance(self, utterance_id: str) -> None:
        # Native simulated VAD: explicit commit is a no-op.
        return None

    async def cancel_response(self, response_id: str) -> ControlReceipt:
        run = next((r for r in self._runs if r.response_id == response_id), None)
        if run is None:
            return ControlReceipt(kind="cancel_response", sent_ns=self.clock.now_ns(), ok=False, detail="no such active response")
        run.stopped = True
        await self._emit(
            EventKind.RESPONSE_DONE, response_id=response_id, direction="system"
        )
        return ControlReceipt(kind="cancel_response", sent_ns=self.clock.now_ns(), ok=True)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # -- internals ---------------------------------------------------------

    def _start_background(self) -> None:
        self._tasks.append(asyncio.ensure_future(self._vad_loop()))
        self._tasks.append(asyncio.ensure_future(self._stream_loop()))

    def _on_barge_in(self, now_ns: int) -> None:
        if self.policy.ignore_interrupts:
            return
        for run in self._runs:
            if not run.stopped and run.interrupt_stop_at_ns is None:
                run.interrupt_stop_at_ns = now_ns + self.policy.interrupt_stop_ms * NS_PER_MS

    def _maybe_vad_complete(self, now_ns: int) -> None:
        buffer = self._buffer
        if buffer is None or not buffer.pcm:
            return
        silence = now_ns - buffer.last_frame_ns
        if silence >= self.policy.vad_silence_ms * NS_PER_MS:
            self._schedule_processing(buffer)
            self._buffer = None

    async def _vad_loop(self) -> None:
        """Simulated server VAD: close the utterance after inbound silence."""
        while not self._closed:
            await self.clock.sleep_ns(20 * NS_PER_MS)
            self._maybe_vad_complete(self.clock.now_ns())

    def _schedule_processing(self, buffer: _UtteranceBuffer) -> None:
        recognized = decode_text_pcm(bytes(buffer.pcm))
        heard = self.policy.asr_hears(recognized)
        utterance_id = buffer.utterance_id
        asyncio.ensure_future(self._process_utterance(utterance_id, recognized, heard))

    async def _process_utterance(self, utterance_id: str, recognized: str, heard: str) -> None:
        await self._emit(
            EventKind.CALLER_TRANSCRIPT_FINAL, utterance_id=utterance_id, direction="caller", text=heard
        )
        await self.clock.sleep_ns(self.policy.processing_ms * NS_PER_MS)
        self._response_counter += 1
        response_id = f"mock-r{self._response_counter}"
        await self._emit(
            EventKind.RESPONSE_START, response_id=response_id, direction="system",
            payload={"recognized": recognized},
        )
        reply, long = self.policy.reply_for(recognized)
        total_ms = self.policy.long_reply_ms if long else self.policy.short_reply_ms
        self._runs.append(
            _ResponseRun(
                response_id=response_id, reply=reply, total_ms=total_ms, started_ns=self.clock.now_ns()
            )
        )
        await self._emit_text(reply, response_id)

    async def _emit_text(self, reply: str, response_id: str) -> None:
        words = reply.split()
        half = max(1, len(words) // 2)
        await self._emit(
            EventKind.AGENT_TEXT_DELTA, response_id=response_id, direction="agent",
            text=" ".join(words[:half]),
        )
        await self._emit(
            EventKind.AGENT_TEXT_FINAL, response_id=response_id, direction="agent", text=reply
        )

    async def _stream_loop(self) -> None:
        """Stream every active response incrementally, honoring barge-in stops.

        Concurrent runs model a duplex agent that keeps talking over a
        barge-in it ignores (each run keeps its own stop conditions).
        """
        while not self._closed:
            await self.clock.sleep_ns(self.policy.stream_interval_ms * NS_PER_MS)
            for run in list(self._runs):
                if run.stopped:
                    continue
                now = self.clock.now_ns()
                if run.interrupt_stop_at_ns is not None and now >= run.interrupt_stop_at_ns:
                    run.stopped = True
                    await self._emit(
                        EventKind.RESPONSE_DONE, response_id=run.response_id, direction="system"
                    )
                    continue
                elapsed_ms = (now - run.started_ns) / NS_PER_MS
                if elapsed_ms >= run.total_ms:
                    run.stopped = True
                    await self._emit(
                        EventKind.RESPONSE_DONE, response_id=run.response_id, direction="system"
                    )
                    continue
                pcm = tone_pcm(
                    self.audio_format, self.policy.stream_frame_ms, amplitude=5000.0
                )
                await self._emit(
                    EventKind.AGENT_AUDIO,
                    response_id=run.response_id,
                    direction="agent",
                    audio=AudioPayload(
                        data=pcm,
                        samples=self.audio_format.sample_count(pcm),
                        audio_format=self.audio_format,
                        sequence=run.frames_sent,
                    ),
                )
                run.frames_sent += 1
            self._runs = [r for r in self._runs if not r.stopped]

    async def _emit(
        self,
        kind: EventKind,
        *,
        response_id: str | None = None,
        utterance_id: str | None = None,
        direction: str | None = None,
        text: str | None = None,
        audio: AudioPayload | None = None,
        payload: dict | None = None,
    ) -> None:
        if self._closed:
            return
        event = AgentEvent(
            kind=kind,
            receipt_ns=self.clock.now_ns(),
            raw_type=f"mock_{kind.value}",
            response_id=response_id,
            utterance_id=utterance_id,
            direction=direction,
            text=text,
            audio=audio,
            payload=payload,
        )
        await self._queue.put(event)


class MockCallerVoice:
    """Deterministic watermark "TTS". Complete clips; no streaming."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock = clock or MonotonicClock()

    async def synthesize(
        self,
        text: str,
        *,
        audio_format: AudioFormat,
        seed: int | None = None,
    ) -> SynthesizedAudio:
        pcm = encode_text_pcm(text, audio_format)
        return SynthesizedAudio(
            text=text,
            pcm=pcm,
            audio_format=audio_format,
            samples=audio_format.sample_count(pcm),
            provenance={
                "engine": "mock-watermark",
                "note": "simulated TTS: text watermarked in sample amplitudes, not speech",
                "seed": seed,
            },
        )

    async def aclose(self) -> None:
        return None
