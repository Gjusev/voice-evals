"""Local reference voice-agent server for validating the probe's WebSocket path.

Speaks exactly the bundled ``voice-evals-default-v1`` protocol
(``src/voice_evals/resources/protocols/default-v1.json``) over a real socket,
so ``WebSocketAgentTransport`` can be exercised end to end without any provider.

DEVELOPMENT EXAMPLE, NOT A PRODUCT AGENT:
- dialogue policy is SCRIPTED: it reuses DEFAULT_REPLY_RULES / FALLBACK_REPLY via
  ``MockAgentPolicy`` from ``voice_evals.probe.testing``;
- TTS is deterministic PCM tones (``tone_pcm``), not speech;
- STT is selected by ``--asr``:
  * ``none``      -- no input_transcript_final events; probe sessions that need
    caller transcripts are honestly NOT SCORED;
  * ``watermark`` -- decodes the mock caller's amplitude watermark
    (``decode_text_pcm``); works only with MockCallerVoice/fixture clips;
  * ``scribe``    -- real STT: the utterance PCM is wrapped as WAV and POSTed
    to the ElevenLabs speech-to-text API. The key is read from the
    ELEVENLABS_API_KEY environment variable at call time; it is never logged
    and never written to any file.

Run from the repository root: ``uv run python examples/reference-agent/server.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import itertools
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from websockets.asyncio.server import serve

from voice_evals.probe.audio import AudioFormat, pcm_to_wav_bytes, tone_pcm
from voice_evals.probe.testing import MockAgentPolicy, decode_text_pcm

FMT = AudioFormat(sample_rate=16000)  # default-v1: pcm_s16le mono 16 kHz both ways
VAD_SILENCE_MS = 250  # finalize an open utterance after this much inbound silence
VAD_TICK_MS = 50
STREAM_FRAME_MS = 80  # audio duration per output_audio event
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
NS_PER_MS = 1_000_000
SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def _now_ns() -> int:
    return time.monotonic_ns()


def _log(tag: str, message: str) -> None:
    # Never log auth headers, endpoints, or query strings; ids and counts only.
    print(f"[ref-agent conn={tag}] {message}", flush=True)


@dataclass
class _Buffer:
    utterance_id: str
    pcm: bytearray = field(default_factory=bytearray)
    last_frame_ns: int = 0


@dataclass
class _Run:
    response_id: str
    total_ms: int
    started_ns: int
    stop_at_ns: int | None = None  # barge-in deadline
    done_sent: bool = False
    frames_sent: int = 0


class ReferenceSession:
    """One WebSocket connection speaking default-v1."""

    def __init__(self, connection: Any, opts: argparse.Namespace, tag: str) -> None:
        self.conn = connection
        self.opts = opts
        self.tag = tag
        self.policy = MockAgentPolicy()  # scripted replies: DEFAULT_REPLY_RULES + FALLBACK
        self.buffer: _Buffer | None = None
        self.runs: list[_Run] = []
        self.response_counter = 0
        self.ready_sent = False
        self.closed = False
        self.tasks: list[asyncio.Task] = []

    # -- lifecycle ----------------------------------------------------------

    async def run(self) -> None:
        _log(self.tag, "connection open")
        self.tasks = [asyncio.ensure_future(self._vad_loop()), asyncio.ensure_future(self._stream_loop())]
        try:
            async for raw in self.conn:
                await self._on_raw(raw)
        except Exception as error:  # noqa: BLE001 - keep the server alive per connection
            _log(self.tag, f"receive loop ended ({type(error).__name__})")
        finally:
            self.closed = True
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            _log(self.tag, "connection closed")

    async def _send(self, message: dict[str, Any]) -> None:
        if self.closed:
            return
        try:
            await self.conn.send(json.dumps(message, separators=(",", ":")))
        except Exception:  # noqa: BLE001 - peer went away; loops check self.closed
            self.closed = True

    # -- inbound ------------------------------------------------------------

    async def _on_raw(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            return  # default-v1 client frames are JSON text
        try:
            message = json.loads(raw)
        except ValueError:
            _log(self.tag, "non-JSON text message ignored")
            return
        if not isinstance(message, dict):
            return
        mtype = message.get("type")
        if mtype == "session_start" and not self.ready_sent:
            self.ready_sent = True
            await self._send({"type": "session_ready"})
            _log(self.tag, "session_start -> session_ready")
        elif mtype == "input_audio":
            await self._on_audio(message)
        elif mtype == "input_commit":
            self._on_commit(str(message.get("utterance_id", "")))
        elif mtype == "cancel_response":
            await self._on_cancel(str(message.get("response_id", "")))
        # unknown event types are ignored, per the protocol contract

    async def _on_audio(self, message: dict[str, Any]) -> None:
        try:
            pcm = base64.b64decode(str(message.get("audio", "")), validate=True)
        except (binascii.Error, ValueError):
            _log(self.tag, "input_audio with invalid base64 ignored")
            return
        utterance_id = str(message.get("utterance_id", ""))
        if not utterance_id or not pcm:
            return
        buffer = self.buffer
        if buffer is None or buffer.utterance_id != utterance_id:
            if buffer is not None and buffer.pcm:
                self._finalize(buffer)  # server VAD would have closed it already
            self.buffer = _Buffer(utterance_id=utterance_id)
            if any(not run.done_sent for run in self.runs):
                self._on_barge_in()
        assert self.buffer is not None
        self.buffer.pcm.extend(pcm)
        self.buffer.last_frame_ns = _now_ns()

    def _on_commit(self, utterance_id: str) -> None:
        buffer = self.buffer
        if buffer is not None and buffer.utterance_id == utterance_id:
            _log(self.tag, f"input_commit {utterance_id} ({len(buffer.pcm)} bytes)")
            self._finalize(buffer)

    async def _on_cancel(self, response_id: str) -> None:
        run = next((r for r in self.runs if r.response_id == response_id and not r.done_sent), None)
        if run is None:
            return
        run.done_sent = True
        await self._send({"type": "response_done", "response_id": response_id})
        _log(self.tag, f"cancel_response {response_id}: response_done sent")

    def _on_barge_in(self) -> None:
        if self.opts.ignore_interrupts:
            _log(self.tag, "new utterance during active response: ignored (--ignore-interrupts)")
            return
        deadline = _now_ns() + self.opts.interrupt_stop_ms * NS_PER_MS
        for run in self.runs:
            if not run.done_sent and run.stop_at_ns is None:
                run.stop_at_ns = deadline
                _log(self.tag, f"barge-in: {run.response_id} stops in {self.opts.interrupt_stop_ms}ms")

    def _finalize(self, buffer: _Buffer) -> None:
        self.buffer = None
        self.tasks.append(asyncio.ensure_future(self._respond(buffer)))
        self.tasks = [t for t in self.tasks if not t.done()]

    # -- STT + scripted reply -------------------------------------------------

    async def _respond(self, buffer: _Buffer) -> None:
        text: str | None = None
        if self.opts.asr == "watermark":
            text = decode_text_pcm(bytes(buffer.pcm))
        elif self.opts.asr == "scribe":
            text = await self._scribe(bytes(buffer.pcm))  # None = no transcript event
        self.response_counter += 1
        response_id = f"ra-{self.response_counter}"
        # Simulated ASR->first-event processing, like MockAgentPolicy.processing_ms:
        # a zero-delay agent reacts to input_commit (which follows the final frame's
        # socket send) before the client's paced caller-audio-end estimate, so its
        # transcript and response events would arrive before C_end and never
        # measure stt_ms / open a correlatable response.
        await asyncio.sleep(self.opts.processing_ms / 1000)
        if text is not None:
            await self._send(
                {"type": "input_transcript_final", "utterance_id": buffer.utterance_id, "text": text}
            )
        await self._send({"type": "response_start", "response_id": response_id})
        reply, long = self.policy.reply_for(text)
        total_ms = self.opts.long_reply_ms if long else self.opts.short_reply_ms
        words = reply.split()
        half = max(1, len(words) // 2)
        await self._send(
            {"type": "output_text_delta", "response_id": response_id, "text": " ".join(words[:half])}
        )
        await self._send({"type": "output_text_final", "response_id": response_id, "text": reply})
        self.runs.append(
            _Run(response_id=response_id, total_ms=total_ms, started_ns=_now_ns())
        )
        _log(
            self.tag,
            f"{buffer.utterance_id} -> {response_id}: reply {len(reply)} chars, "
            f"{total_ms}ms stream, asr text: {text[:60]!r}",
        )

    async def _scribe(self, pcm: bytes) -> str | None:
        """Real STT via ElevenLabs speech-to-text; None = no transcript event."""
        key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not key:
            _log(self.tag, "scribe: ELEVENLABS_API_KEY not set; emitting no transcript")
            return None
        import httpx

        wav = pcm_to_wav_bytes(pcm, FMT)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    SCRIBE_URL,
                    headers={"xi-api-key": key},
                    data={"model_id": "scribe_v1"},
                    files={"file": ("utterance.wav", wav, "audio/wav")},
                )
        except Exception as error:  # noqa: BLE001 - one safe line, never the key
            _log(self.tag, f"scribe: request failed ({type(error).__name__}); emitting no transcript")
            return None
        if response.status_code != 200:
            _log(self.tag, f"scribe: HTTP {response.status_code}; emitting no transcript")
            return None
        try:
            return str(response.json().get("text", ""))
        except ValueError:
            _log(self.tag, "scribe: non-JSON response; emitting no transcript")
            return None

    # -- background loops ------------------------------------------------------

    async def _vad_loop(self) -> None:
        """Simulated server VAD: close the utterance after inbound silence."""
        while not self.closed:
            await asyncio.sleep(VAD_TICK_MS / 1000)
            buffer = self.buffer
            if (
                buffer is not None
                and buffer.pcm
                and _now_ns() - buffer.last_frame_ns >= VAD_SILENCE_MS * NS_PER_MS
            ):
                _log(self.tag, f"vad: {buffer.utterance_id} silent {VAD_SILENCE_MS}ms -> finalize")
                self._finalize(buffer)

    async def _stream_loop(self) -> None:
        """Stream tone frames per active response; honor barge-in stops."""
        frame_b64 = base64.b64encode(tone_pcm(FMT, STREAM_FRAME_MS, amplitude=5000.0)).decode("ascii")
        while not self.closed:
            await asyncio.sleep(self.opts.stream_interval_ms / 1000)
            for run in list(self.runs):
                if run.done_sent:
                    continue
                now = _now_ns()
                if run.stop_at_ns is not None and now >= run.stop_at_ns:
                    run.done_sent = True
                    await self._send({"type": "response_done", "response_id": run.response_id})
                    _log(self.tag, f"{run.response_id}: stopped by barge-in after {run.frames_sent} frames")
                    continue
                if (now - run.started_ns) / NS_PER_MS >= run.total_ms:
                    run.done_sent = True
                    await self._send({"type": "response_done", "response_id": run.response_id})
                    _log(self.tag, f"{run.response_id}: natural end after {run.frames_sent} frames")
                    continue
                await self._send(
                    {
                        "type": "output_audio",
                        "response_id": run.response_id,
                        "seq": run.frames_sent,
                        "audio": frame_b64,
                    }
                )
                run.frames_sent += 1
            self.runs = [r for r in self.runs if not r.done_sent]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reference voice-agent server speaking voice-evals-default-v1 "
            "(development example: scripted policy, tone TTS, STT per --asr mode)"
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default 8765)")
    parser.add_argument(
        "--asr",
        choices=("none", "watermark", "scribe"),
        default="none",
        help=(
            "STT stage: none = no input_transcript_final (sessions honestly NOT SCORED); "
            "watermark = decode the mock caller watermark (mock callers only); "
            "scribe = real STT via ElevenLabs speech-to-text (ELEVENLABS_API_KEY)"
        ),
    )
    parser.add_argument("--interrupt-stop-ms", type=int, default=260, help="barge-in stop latency")
    parser.add_argument(
        "--processing-ms",
        type=int,
        default=150,
        help="simulated ASR->first-event delay (like MockAgentPolicy.processing_ms)",
    )
    parser.add_argument("--stream-interval-ms", type=int, default=40, help="one output_audio per interval")
    parser.add_argument("--short-reply-ms", type=int, default=1500, help="streamed audio for short replies")
    parser.add_argument("--long-reply-ms", type=int, default=6000, help="streamed audio for long replies")
    parser.add_argument(
        "--ignore-interrupts",
        action="store_true",
        help="keep streaming past barge-ins (censored-run testing)",
    )
    return parser.parse_args(argv)


async def _serve(opts: argparse.Namespace) -> None:
    counter = itertools.count(1)

    async def handler(connection: Any) -> None:
        await ReferenceSession(connection, opts, str(next(counter))).run()

    async with serve(
        handler,
        opts.host,
        opts.port,
        max_size=MAX_MESSAGE_BYTES,
        compression=None,
    ):
        await asyncio.get_running_loop().create_future()


def main(argv: list[str] | None = None) -> None:
    opts = parse_args(argv)
    print("reference agent: scripted policy, tone TTS, STT per --asr mode")
    print(
        f"listening on ws://{opts.host}:{opts.port} "
        f"(asr={opts.asr}, interrupt-stop-ms={opts.interrupt_stop_ms}, "
        f"stream-interval-ms={opts.stream_interval_ms}, "
        f"short-reply-ms={opts.short_reply_ms}, long-reply-ms={opts.long_reply_ms}, "
        f"ignore-interrupts={opts.ignore_interrupts})"
    )
    try:
        asyncio.run(_serve(opts))
    except KeyboardInterrupt:
        print("\nreference agent: shut down")


if __name__ == "__main__":
    main()
