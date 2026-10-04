"""Multilingual real-speech fixtures: checksummed bytes and honest labels."""

import asyncio
import json
import wave
from pathlib import Path

from voice_evals.probe.audio import AudioFormat
from voice_evals.probe.voices.fixture import FixtureCallerVoice

MULTILINGUAL = (
    Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "audio" / "multilingual"
)
FMT = AudioFormat(sample_rate=16000)


def load_manifest() -> dict:
    return json.loads((MULTILINGUAL / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_covers_three_languages() -> None:
    manifest = load_manifest()
    clips = manifest["clips"]
    languages = {clip["language"] for clip in clips}
    assert languages == {"en", "de", "es"}
    assert len(clips) >= 6


def test_fixture_caller_serves_exact_verified_bytes() -> None:
    voice = FixtureCallerVoice(MULTILINGUAL / "manifest.json")
    for clip in load_manifest()["clips"]:
        synthesized = asyncio.run(voice.synthesize(clip["text"], audio_format=FMT))
        # The caller verifies and carries the checksum of the stored file.
        assert synthesized.provenance["sha256"] == clip["sha256"], clip["file"]
        assert synthesized.audio_format.sample_rate == 16000
        assert synthesized.samples > FMT.samples_for_ms(1000)  # >1s of real speech
    asyncio.run(voice.aclose())


def test_wavs_are_readable_16k_mono() -> None:
    for clip in load_manifest()["clips"]:
        with wave.open(str(MULTILINGUAL / clip["file"]), "rb") as handle:
            assert handle.getframerate() == 16000
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getcomptype() == "NONE"


def test_provenance_is_labeled_honestly() -> None:
    """Real-speech clips must never masquerade as open-model output."""
    for clip in load_manifest()["clips"]:
        provenance = clip["provenance"]
        assert provenance["engine"] == "elevenlabs"
        assert provenance["model"] == "eleven_multilingual_v2"
        assert provenance.get("voice")
        assert "NOT open-model" in provenance["note"]
        assert provenance.get("generated_utc")


def test_missing_text_is_actionable() -> None:
    voice = FixtureCallerVoice(MULTILINGUAL / "manifest.json")
    from voice_evals.probe.errors import CallerVoiceError

    try:
        asyncio.run(voice.synthesize("text that was never recorded", audio_format=FMT))
        raise AssertionError("should have failed")
    except CallerVoiceError as error:
        assert "no fixture clip" in str(error)
    finally:
        asyncio.run(voice.aclose())
