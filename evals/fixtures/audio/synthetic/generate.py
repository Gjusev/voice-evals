"""Regenerate the synthetic fixture clips (watermarked tones, not speech)."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from voice_evals.probe.audio import AudioFormat, pcm_to_wav_bytes
from voice_evals.probe.testing import encode_text_pcm

HERE = Path(__file__).resolve().parent
TEXTS = [
    "I'd like to book an appointment for Tuesday.",
    "My name is Alex Morgan.",
    "Sorry to interrupt. I need 11:00, not ten.",
    "Ich möchte einen Termin am Dienstag.",
    "Quisiera reservar una cita el martes.",
]


def main() -> None:
    fmt = AudioFormat(sample_rate=16000)
    clips = []
    for index, text in enumerate(TEXTS):
        name = f"clip-{index:02d}.wav"
        pcm = encode_text_pcm(text, fmt)
        wav = pcm_to_wav_bytes(pcm, fmt)
        (HERE / name).write_bytes(wav)
        clips.append(
            {
                "text": text,
                "file": name,
                "sha256": hashlib.sha256(wav).hexdigest(),
                "language": "de" if "Termin" in text else "es" if "cita" in text else "en",
                "provenance": {
                    "source": "synthetic-watermark-tone",
                    "note": "deterministic watermarked tone from voice_evals.probe.testing; not speech, not model output",
                },
            }
        )
    (HERE / "manifest.json").write_text(
        json.dumps(
            {
                "format": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1, "sample_width": 2},
                "note": "synthetic fixtures for offline FixtureCallerVoice exercise; see ../README.md",
                "clips": clips,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(clips)} synthetic clips + manifest")


if __name__ == "__main__":
    main()
