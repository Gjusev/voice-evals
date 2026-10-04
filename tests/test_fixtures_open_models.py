"""Open-model fixture integrity: checksummed Kokoro clips at 24 kHz native."""

import asyncio
import json
import wave
from pathlib import Path

from voice_evals.probe.audio import AudioFormat
from voice_evals.probe.errors import CallerFormatError
from voice_evals.probe.voices.fixture import FixtureCallerVoice

OPEN_MODELS = (
    Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "audio" / "open-models"
)
NATIVE = AudioFormat(sample_rate=24000)
REQUEST_RATE = AudioFormat(sample_rate=16000)


def load_manifest() -> dict:
    return json.loads((OPEN_MODELS / "manifest.json").read_text(encoding="utf-8"))


def test_kokoro_clips_present_and_pinned() -> None:
    manifest = load_manifest()
    assert manifest["note"].startswith("open-model fixtures (Kokoro-82M")
    clips = manifest["clips"]
    assert {clip["language"] for clip in clips} == {"en", "es"}
    for clip in clips:
        provenance = clip["provenance"]
        assert provenance["engine"] == "kokoro"
        assert provenance["model"] == "hexgrad/Kokoro-82M"
        assert provenance["revision"], "revision must be pinned, never null"
        assert provenance["license"] == "Apache-2.0"
        assert provenance["native_sample_rate"] == 24000
        assert provenance["resampled_to_16000"] is False  # no silent resampling
    assert manifest.get("skipped") in ([], None)  # nothing degraded silently


def test_fixture_caller_serves_24k_native_bytes() -> None:
    voice = FixtureCallerVoice(OPEN_MODELS / "manifest.json")
    for clip in load_manifest()["clips"]:
        synthesized = asyncio.run(voice.synthesize(clip["text"], audio_format=NATIVE))
        assert synthesized.provenance["sha256"] == clip["sha256"], clip["file"]
        assert synthesized.audio_format.sample_rate == 24000
        assert synthesized.samples > NATIVE.samples_for_ms(1000)
    asyncio.run(voice.aclose())


def test_rate_mismatch_refuses_without_resampling() -> None:
    voice = FixtureCallerVoice(OPEN_MODELS / "manifest.json")
    text = load_manifest()["clips"][0]["text"]
    try:
        asyncio.run(voice.synthesize(text, audio_format=REQUEST_RATE))
        raise AssertionError("16 kHz request against 24 kHz fixtures must fail")
    except CallerFormatError as error:
        assert "no silent resampling" in str(error)
    finally:
        asyncio.run(voice.aclose())


def test_wavs_are_24k_mono_s16() -> None:
    for clip in load_manifest()["clips"]:
        with wave.open(str(OPEN_MODELS / clip["file"]), "rb") as handle:
            assert handle.getframerate() == 24000
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getcomptype() == "NONE"
