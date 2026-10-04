"""Generate the multilingual real-speech caller fixtures (live, ElevenLabs).

NOT open-model output: these clips are real speech synthesized with the
hosted ``eleven_multilingual_v2`` model because serving Qwen3-TTS/Kokoro
locally was out of scope. The manifest records the exact provenance; never
relabel these clips as any open model's output.

Regenerate (needs ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID, spends one TTS
call per clip):

    uv run python evals/fixtures/audio/multilingual/generate.py

Output: one WAV per text plus manifest.json with sha256 checksums and
provenance. The offline suite verifies these bytes without any network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from voice_evals.probe.audio import AudioFormat, pcm_to_wav_bytes
from voice_evals.probe.config import CallerConfig
from voice_evals.probe.voices.elevenlabs import ElevenLabsCallerVoice

HERE = Path(__file__).resolve().parent
FMT = AudioFormat(sample_rate=16000)
VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # premade "Sarah"
MODEL = "eleven_multilingual_v2"

CLIPS: list[dict[str, str]] = [
    {"language": "en", "name": "en-request", "text": "I'd like to book an appointment for Tuesday."},
    {"language": "en", "name": "en-name", "text": "My name is Alex Morgan."},
    {"language": "en", "name": "en-correct", "text": "Sorry, could we make that 11:00 instead?"},
    {"language": "de", "name": "de-request", "text": "Ich möchte einen Termin am Dienstag."},
    {"language": "de", "name": "de-name", "text": "Mein Name ist Alex Morgan."},
    {"language": "es", "name": "es-request", "text": "Quisiera reservar una cita el martes."},
    {"language": "es", "name": "es-name", "text": "Me llamo Alex Morgan."},
]


async def synthesize_with_retry(voice: ElevenLabsCallerVoice, text: str, attempts: int = 3):
    """Prep-time retry: hosted TTS occasionally answers 200 with an MP3 body
    despite output_format=pcm_16000. The adapter rejects that (correctly);
    regenerating fixtures is not timing-sensitive, so retry here."""
    import asyncio

    last = None
    for i in range(attempts):
        try:
            return await voice.synthesize(text, audio_format=FMT)
        except Exception as error:  # noqa: BLE001 - prep tool retries any synthesis failure
            last = error
            print(f"  attempt {i + 1} failed: {error}", flush=True)
            await asyncio.sleep(1.5 * (i + 1))
    raise SystemExit(f"synthesis failed after {attempts} attempts: {last}")


async def main() -> None:
    config = CallerConfig(kind="elevenlabs", voice_id=VOICE, model_id=MODEL)
    voice = ElevenLabsCallerVoice(config)
    entries = []
    try:
        for clip in CLIPS:
            synthesized = await synthesize_with_retry(voice, clip["text"])
            wav_name = f"{clip['name']}.wav"
            wav = pcm_to_wav_bytes(synthesized.pcm, FMT)
            (HERE / wav_name).write_bytes(wav)
            entries.append(
                {
                    "text": clip["text"],
                    "file": wav_name,
                    "sha256": hashlib.sha256(wav).hexdigest(),
                    "language": clip["language"],
                    "provenance": {
                        "engine": "elevenlabs",
                        "model": MODEL,
                        "voice": VOICE,
                        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "sample_rate": 16000,
                        "encoding": "pcm_s16le",
                        "note": (
                            "real speech from hosted ElevenLabs TTS; NOT open-model output. "
                            "Open-model clips (Qwen3-TTS/Kokoro) remain pending model serving."
                        ),
                    },
                }
            )
            print(f"{wav_name}: {synthesized.samples} samples ({synthesized.duration_ns/1e9:.2f}s)")
    finally:
        await voice.aclose()
    manifest = {
        "format": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1, "sample_width": 2},
        "note": "multilingual real-speech caller fixtures; per-clip provenance is authoritative",
        "clips": entries,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} clips + manifest.json")


if __name__ == "__main__":
    asyncio.run(main())
