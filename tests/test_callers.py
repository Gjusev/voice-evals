"""Caller voices and the WebSocket transport, tested without real sockets.

ElevenLabs/HTTP callers run against httpx.MockTransport handlers; the
WebSocket transport runs against a fake connector. No unit test may open a
socket or reach a provider.
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from voice_evals.probe.audio import AudioFormat, pcm_to_wav_bytes, tone_pcm
from voice_evals.probe.clock import VirtualClock
from voice_evals.probe.config import CallerConfig, TransportConfig
from voice_evals.probe.errors import (
    CallerAuthError,
    CallerFormatError,
    CallerVoiceError,
    CapabilityError,
    ConfigError,
)
from voice_evals.probe.voices.elevenlabs import ElevenLabsCallerVoice
from voice_evals.probe.voices.fixture import FixtureCallerVoice
from voice_evals.probe.voices.http import HttpCallerVoice

FMT = AudioFormat(sample_rate=16000)


# -- ElevenLabs -----------------------------------------------------------


def elevenlabs_client(handler, **env) -> ElevenLabsCallerVoice:
    config = CallerConfig(kind="elevenlabs", voice_id="voice-1", **env)
    return ElevenLabsCallerVoice(
        config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), clock=VirtualClock()
    )


def test_elevenlabs_native_pcm(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "output_format=pcm_16000" in str(request.url)
        assert request.headers["xi-api-key"] == "sk-test"
        return httpx.Response(200, content=tone_pcm(FMT, 500))

    voice = elevenlabs_client(handler)
    clip = asyncio.run(voice.synthesize("hello", audio_format=FMT, seed=7))
    assert clip.samples == FMT.samples_for_ms(500)
    assert clip.provenance["engine"] == "elevenlabs"
    assert "sk-test" not in json.dumps(clip.provenance)
    asyncio.run(voice.aclose())


def test_elevenlabs_rejects_mp3_body(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"ID3\x04\x00rest-of-mp3")

    voice = elevenlabs_client(handler)
    with pytest.raises(CallerFormatError, match="mp3"):
        asyncio.run(voice.synthesize("hello", audio_format=FMT))
    asyncio.run(voice.aclose())


def test_elevenlabs_auth_not_retried(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-bad")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"detail": {"message": "invalid_api_key"}})

    voice = elevenlabs_client(handler)
    with pytest.raises(CallerAuthError):
        asyncio.run(voice.synthesize("hello", audio_format=FMT))
    assert calls["n"] == 1  # 401 is terminal, never retried
    asyncio.run(voice.aclose())


def test_elevenlabs_throttle_retries_then_succeeds(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, content=tone_pcm(FMT, 300))

    voice = elevenlabs_client(handler)
    clip = asyncio.run(voice.synthesize("hello", audio_format=FMT))
    assert clip.samples > 0 and calls["n"] == 2
    assert voice.attempts_record[0]["attempts"] == 2
    asyncio.run(voice.aclose())


def test_elevenlabs_missing_key_is_actionable(monkeypatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return httpx.Response(200)

    voice = elevenlabs_client(handler)
    with pytest.raises(CallerAuthError, match="ELEVENLABS_API_KEY"):
        asyncio.run(voice.synthesize("hello", audio_format=FMT))


def test_elevenlabs_requires_voice_id(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    config = CallerConfig(kind="elevenlabs")
    voice = ElevenLabsCallerVoice(config, http_client=httpx.AsyncClient())
    with pytest.raises(ConfigError, match="voice"):
        asyncio.run(voice.synthesize("hello", audio_format=FMT))


# -- HTTP open-model caller -------------------------------------------------


def test_http_caller_raw_pcm_with_headers(monkeypatch) -> None:
    monkeypatch.setenv("CALLER_TTS_URL", "http://tts.internal/tts")
    config = CallerConfig(kind="http", model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", voice_id="speaker-1")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["sample_rate"] == 16000 and body["format"] == "pcm_s16le"
        return httpx.Response(
            200,
            content=tone_pcm(FMT, 400),
            headers={"x-audio-encoding": "pcm_s16le", "x-audio-rate": "16000"},
        )

    voice = HttpCallerVoice(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    clip = asyncio.run(voice.synthesize("Guten Tag", audio_format=FMT))
    assert clip.provenance["model"].startswith("Qwen/")
    asyncio.run(voice.aclose())


def test_http_caller_accepts_uncompressed_wav(monkeypatch) -> None:
    monkeypatch.setenv("CALLER_TTS_URL", "http://tts.internal/tts")
    config = CallerConfig(kind="http", model_id="kokoro")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=pcm_to_wav_bytes(tone_pcm(FMT, 250), FMT), headers={"content-type": "audio/wav"})

    voice = HttpCallerVoice(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    clip = asyncio.run(voice.synthesize("hola", audio_format=FMT))
    assert clip.samples == FMT.samples_for_ms(250)
    asyncio.run(voice.aclose())


def test_http_caller_rejects_rate_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("CALLER_TTS_URL", "http://tts.internal/tts")
    config = CallerConfig(kind="http")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=tone_pcm(FMT, 100), headers={"x-audio-rate": "8000"})

    voice = HttpCallerVoice(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(CallerFormatError, match="resampling belongs in the model service"):
        asyncio.run(voice.synthesize("hi", audio_format=FMT))
    asyncio.run(voice.aclose())


def test_http_caller_missing_url(monkeypatch) -> None:
    monkeypatch.delenv("CALLER_TTS_URL", raising=False)
    voice = HttpCallerVoice(CallerConfig(kind="http"), http_client=httpx.AsyncClient())
    with pytest.raises(CallerVoiceError, match="CALLER_TTS_URL"):
        asyncio.run(voice.synthesize("hi", audio_format=FMT))


# -- Fixture caller ----------------------------------------------------------


def test_fixture_caller_serves_verified_bytes(tmp_path: Path) -> None:
    import hashlib

    pcm = tone_pcm(FMT, 700)
    wav = pcm_to_wav_bytes(pcm, FMT)
    (tmp_path / "a.wav").write_bytes(wav)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": {"sample_rate": 16000, "channels": 1, "sample_width": 2},
                "clips": [
                    {
                        "text": "fixture line",
                        "file": "a.wav",
                        "sha256": hashlib.sha256(wav).hexdigest(),
                        "provenance": {"source": "synthetic-tone", "note": "not model speech"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    voice = FixtureCallerVoice(manifest)
    clip = asyncio.run(voice.synthesize("fixture line", audio_format=FMT))
    assert clip.pcm == pcm
    assert clip.provenance["sha256"] == hashlib.sha256(wav).hexdigest()
    asyncio.run(voice.aclose())


def test_fixture_caller_checksum_mismatch(tmp_path: Path) -> None:
    pcm = tone_pcm(FMT, 100)
    (tmp_path / "a.pcm").write_bytes(pcm)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"clips": [{"text": "x", "file": "a.pcm", "sha256": "0" * 64}]}),
        encoding="utf-8",
    )
    voice = FixtureCallerVoice(manifest)
    with pytest.raises(CallerFormatError, match="checksum"):
        asyncio.run(voice.synthesize("x", audio_format=FMT))


def test_fixture_caller_rate_mismatch_rejected(tmp_path: Path) -> None:
    fmt24 = AudioFormat(sample_rate=24000)
    (tmp_path / "a.wav").write_bytes(pcm_to_wav_bytes(tone_pcm(fmt24, 100), fmt24))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": [{"text": "x", "file": "a.wav"}]}), encoding="utf-8")
    voice = FixtureCallerVoice(manifest)
    with pytest.raises(CallerFormatError, match="no silent resampling"):
        asyncio.run(voice.synthesize("x", audio_format=FMT))


# -- WebSocket transport with a fake connector --------------------------------


class FakeWebSocket:
    """In-memory server side of a WebSocket speaking default-v1."""

    def __init__(self) -> None:
        self.sent: list[str | bytes] = []
        self.inbox: asyncio.Queue[str | bytes] = asyncio.Queue()
        self.closed = False

    async def server_push(self, message: str | bytes) -> None:
        await self.inbox.put(message)

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True
        await self.inbox.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        return item


def test_websocket_transport_open_send_receive(tmp_path: Path) -> None:
    from voice_evals.probe.models import CallerAudioFrame
    from voice_evals.probe.transports.protocol import load_default_protocol_map
    from voice_evals.probe.transports.websocket import WebSocketAgentTransport

    async def scenario() -> None:
        fake = FakeWebSocket()
        clock = VirtualClock()
        pmap = load_default_protocol_map()

        async def connector(*args, **kwargs):
            return fake

        await fake.server_push(json.dumps({"type": "session_ready"}))
        transport = WebSocketAgentTransport(pmap, clock=clock, connector=connector)
        caps = await transport.open(
            TransportConfig(endpoint="ws://agent.local/ws")
        )
        assert caps.duplex and caps.response_ids
        # Handshake message rendered with a session nonce
        handshake = json.loads(fake.sent[0])
        assert handshake["type"] == "session_start"
        import base64

        pcm = tone_pcm(caps.input_format, 20)
        frame = CallerAudioFrame(
            utterance_id="u1",
            sequence=0,
            payload=pcm,
            sample_offset=0,
            sample_count=caps.input_format.sample_count(pcm),
            audio_format=caps.input_format,
            planned_send_ns=0,
        )
        receipt = await transport.send_audio(frame)
        assert receipt.bytes_sent == len(pcm)
        sent = json.loads(fake.sent[1])
        assert sent["type"] == "input_audio" and sent["seq"] == "0"
        assert base64.b64decode(sent["audio"]) == pcm
        await fake.server_push(
            json.dumps(
                {
                    "type": "output_audio",
                    "response_id": "r1",
                    "seq": 0,
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
        )
        await fake.server_push(json.dumps({"type": "unknown_heartbeat"}))
        await fake.server_push(json.dumps({"type": "response_done", "response_id": "r1"}))
        seen = []
        async for event in transport.events():
            seen.append(event.kind.value)
            if event.kind.value == "response_done":
                break
        assert "agent_audio" in seen and "response_done" in seen
        # end_utterance maps to the map's commit template
        await transport.end_utterance("u1")
        assert json.loads(fake.sent[-1]) == {"type": "input_commit", "utterance_id": "u1"}
        await transport.aclose()
        assert fake.closed

    asyncio.run(scenario())


def test_websocket_transport_cancel_unsupported_map() -> None:
    from voice_evals.probe.transports.protocol import load_default_protocol_map
    from voice_evals.probe.transports.websocket import WebSocketAgentTransport

    async def scenario() -> None:
        fake = FakeWebSocket()
        pmap = load_default_protocol_map()
        pmap.cancel_template = None  # map without cancel support
        await fake.server_push(json.dumps({"type": "session_ready"}))
        transport = WebSocketAgentTransport(
            pmap, clock=VirtualClock(), connector=lambda *a, **k: _async_return(fake)
        )
        await transport.open(TransportConfig(endpoint="ws://x/"))
        with pytest.raises(CapabilityError, match="cancellation"):
            await transport.cancel_response("r1")
        await transport.aclose()

    asyncio.run(scenario())


def _async_return(value):
    async def wrapper():
        return value

    return wrapper()


def test_transport_endpoint_must_be_ws() -> None:
    config = TransportConfig(endpoint="https://agent.example.com")
    with pytest.raises(ConfigError, match="ws:// or wss://"):
        config.resolved_endpoint()


def test_transport_env_resolution(monkeypatch) -> None:
    monkeypatch.setenv("PROBE_TRANSPORT_URL", "wss://agent.example.com/ws?token=hush")
    config = TransportConfig()
    assert config.resolved_endpoint().startswith("wss://")
    # public_dict keeps env name and redacts signed URLs
    public = config.public_dict()
    assert public["endpoint_env"] == "PROBE_TRANSPORT_URL"
    assert "hush" not in json.dumps(public)
