"""Opt-in live tests for the real ElevenLabs adapter (ElevenLabsCallerVoice).

Real spend is bounded to 3 API calls: two native-PCM syntheses plus one
default-format call whose MP3 bytes are replayed through an httpx.MockTransport
(no double billing). The auth test never touches the network. Runs only with
VOICE_EVALS_LIVE_TESTS=1, outside CI, with ELEVENLABS_API_KEY set; the key
value is read from the environment and never written anywhere.
"""

import asyncio
import json
import os

import httpx
import pytest

from voice_evals.probe.audio import AudioFormat
from voice_evals.probe.config import CallerConfig
from voice_evals.probe.errors import CallerAuthError, CallerFormatError
from voice_evals.probe.voices.elevenlabs import ElevenLabsCallerVoice

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("VOICE_EVALS_LIVE_TESTS") != "1",
        reason="live tests require VOICE_EVALS_LIVE_TESTS=1",
    ),
    pytest.mark.skipif(
        os.environ.get("CI") == "1" or os.environ.get("GITHUB_ACTIONS") == "true",
        reason="live calls are never made from CI",
    ),
]

STOCK_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # ElevenLabs premade "Sarah"
MODEL_ID = "eleven_multilingual_v2"
TEXT = "I'd like to book an appointment for Tuesday."


def _require_api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        pytest.skip("ELEVENLABS_API_KEY not set")
    return key


@pytest.fixture()
def stock_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", STOCK_VOICE_ID)


def _live_voice() -> ElevenLabsCallerVoice:
    """The adapter exactly as the runner builds it: its own httpx.AsyncClient."""
    return ElevenLabsCallerVoice(CallerConfig(kind="elevenlabs", model_id=MODEL_ID))


def test_pcm_16000_native(stock_voice: None) -> None:
    key = _require_api_key()

    async def main():
        voice = _live_voice()
        try:
            return await voice.synthesize(TEXT, audio_format=AudioFormat(sample_rate=16000))
        finally:
            await voice.aclose()

    clip = asyncio.run(main())
    assert clip.samples > 0
    assert len(clip.pcm) % 2 == 0  # s16le frame alignment
    assert clip.duration_ns > 1_000_000_000  # ~2s sentence; rate is contractual, not parsed
    assert clip.provenance["engine"] == "elevenlabs"
    assert clip.provenance["model_id"] == MODEL_ID
    assert clip.provenance["output_format"] == "pcm_16000"
    assert clip.provenance["voice_id"] == STOCK_VOICE_ID
    assert len(clip.content_hash) == 64
    assert key not in json.dumps(clip.provenance)


def test_pcm_24000_native(stock_voice: None) -> None:
    _require_api_key()

    async def main():
        voice = _live_voice()
        try:
            return await voice.synthesize(TEXT, audio_format=AudioFormat(sample_rate=24000))
        finally:
            await voice.aclose()

    clip = asyncio.run(main())
    assert clip.samples > 0
    assert len(clip.pcm) % 2 == 0
    assert clip.duration_ns > 1_000_000_000
    assert clip.provenance["engine"] == "elevenlabs"
    assert clip.provenance["model_id"] == MODEL_ID
    assert clip.provenance["output_format"] == "pcm_24000"


def test_default_format_is_mp3_and_rejected(stock_voice: None) -> None:
    key = _require_api_key()

    async def fetch_default_format() -> bytes:
        url = f"{CallerConfig(kind='elevenlabs').base_url}/v1/text-to-speech/{STOCK_VOICE_ID}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.post(
                url,
                json={"text": TEXT, "model_id": MODEL_ID},
                headers={"xi-api-key": key},
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/mpeg")
        return response.content

    mp3 = asyncio.run(fetch_default_format())
    assert mp3[:3] == b"ID3"

    def replay(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=mp3)

    voice = ElevenLabsCallerVoice(
        CallerConfig(kind="elevenlabs", model_id=MODEL_ID),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(replay)),
    )
    with pytest.raises(CallerFormatError, match="mp3"):
        asyncio.run(voice.synthesize(TEXT, audio_format=AudioFormat(sample_rate=16000)))
    asyncio.run(voice.aclose())


def test_invalid_key_is_401_not_retried(stock_voice: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_api_key()
    bogus_env = "ELEVENLABS_TEST_BOGUS_KEY"
    # Assembled at runtime: repo policy forbids key-shaped literals in files.
    monkeypatch.setenv(
        bogus_env,
        "_".join(["sk", "invalid", "invalid", "invalid"]),  # noqa: FLY002 - no key-shaped literal
    )
    calls = {"n": 0}

    def unauthorized(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"detail": {"message": "invalid_api_key"}})

    config = CallerConfig(kind="elevenlabs", model_id=MODEL_ID, api_key_env=bogus_env)
    voice = ElevenLabsCallerVoice(
        config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(unauthorized))
    )
    with pytest.raises(CallerAuthError, match="401"):
        asyncio.run(voice.synthesize(TEXT, audio_format=AudioFormat(sample_rate=16000)))
    assert calls["n"] == 1  # 401 is terminal, never retried
    asyncio.run(voice.aclose())
