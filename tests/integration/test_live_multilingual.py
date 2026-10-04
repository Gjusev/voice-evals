"""Opt-in live test: multilingual clips through real Scribe STT, per-language WER.

Feeds each checksummed real-speech fixture (EN/DE/ES, ElevenLabs TTS) to
ElevenLabs Scribe and measures the word error rate of the TTS -> STT round
trip per language. This is a property of that round trip, not of any voice
agent. Never runs in CI; requires VOICE_EVALS_LIVE_TESTS=1 and the API key.
"""

import asyncio
import io
import json
import os
import wave
from pathlib import Path

import httpx
import pytest

from voice_evals.evaluate import word_error_rate

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
    pytest.mark.skipif(
        not os.environ.get("ELEVENLABS_API_KEY"),
        reason="ELEVENLABS_API_KEY not set",
    ),
]

MULTILINGUAL = (
    Path(__file__).resolve().parents[2] / "evals" / "fixtures" / "audio" / "multilingual"
)
# TTS -> STT round trip in a quiet audio path; loose enough for accented
# multilingual TTS, tight enough to catch a broken clip or wrong language.
MAX_WER_PER_CLIP = 0.35


def _clip_to_wav_bytes(path: Path) -> bytes:
    return path.read_bytes()


async def scribe_transcribe(wav: bytes) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            data={"model_id": "scribe_v1"},  # omit language_code: auto-detect
            files={"file": ("clip.wav", wav, "audio/wav")},
        )
    assert response.status_code == 200, response.text[:200]
    return str(response.json().get("text", ""))


def test_multilingual_round_trip_wer() -> None:
    manifest = json.loads((MULTILINGUAL / "manifest.json").read_text(encoding="utf-8"))
    per_language: dict[str, list[float]] = {}
    for clip in manifest["clips"]:
        wav = _clip_to_wav_bytes(MULTILINGUAL / clip["file"])
        hypothesis = asyncio.run(scribe_transcribe(wav))
        wer = word_error_rate(clip["text"], hypothesis)
        per_language.setdefault(clip["language"], []).append(wer)
        print(f"{clip['file']} [{clip['language']}] wer={wer:.4f} heard={hypothesis!r}")
        assert wer <= MAX_WER_PER_CLIP, f"{clip['file']}: WER {wer:.3f} > {MAX_WER_PER_CLIP}"
    for language, wers in sorted(per_language.items()):
        mean = sum(wers) / len(wers)
        print(f"language {language}: mean round-trip WER {mean:.4f} over {len(wers)} clips")
        assert 0.0 <= mean <= MAX_WER_PER_CLIP


def test_wav_headers_valid() -> None:
    for clip in json.loads((MULTILINGUAL / "manifest.json").read_text(encoding="utf-8"))["clips"]:
        with wave.open(io.BytesIO((MULTILINGUAL / clip["file"]).read_bytes()), "rb") as handle:
            assert handle.getframerate() == 16000 and handle.getnchannels() == 1
