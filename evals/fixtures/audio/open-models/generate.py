"""Generate the open-model caller fixtures locally with Kokoro-82M.

Run inside the scratch venv (see below); writes WAVs + checksummed manifest
into this directory. Kokoro has no German voice; Qwen3-TTS covers German but
requires bf16 hardware (failed on Kaggle T4 with a CUDA device-side assert,
and its package forces CUDA even with device_map=cpu). German open-model
clips stay pending servable bf16 hardware. Recorded: 2026-10-04.

    uv venv .venv-clips --seed
    VIRTUAL_ENV=.venv-clips uv pip install --no-deps kokoro
    VIRTUAL_ENV=.venv-clips uv pip install misaki[en] loguru huggingface_hub \
        numpy soundfile torch transformers
    .venv-clips/Scripts/python evals/fixtures/audio/open-models/generate.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = "hexgrad/Kokoro-82M"
RATE = 24000  # Kokoro native output rate

CLIPS = [
    {"language": "en", "lang_code": "a", "voice": "af_heart", "name": "en-request", "text": "I'd like to book an appointment for Tuesday."},
    {"language": "en", "lang_code": "a", "voice": "af_heart", "name": "en-name", "text": "My name is Alex Morgan."},
    {"language": "en", "lang_code": "a", "voice": "af_heart", "name": "en-correct", "text": "Sorry, could we make that 11:00 instead?"},
    {"language": "es", "lang_code": "e", "voice": "ef_dora", "name": "es-request", "text": "Quisiera reservar una cita el martes."},
    {"language": "es", "lang_code": "e", "voice": "ef_dora", "name": "es-name", "text": "Me llamo Alex Morgan."},
]


def main() -> int:
    import numpy as np
    import soundfile as sf
    from huggingface_hub import HfApi
    from kokoro import KPipeline

    revision = HfApi().model_info(MODEL_ID).sha
    print(f"{MODEL_ID} revision={revision}", flush=True)

    entries = []
    skipped = []
    pipelines: dict[str, object] = {}
    for clip in CLIPS:
        try:
            pipeline = pipelines.get(clip["lang_code"])
            if pipeline is None:
                pipeline = KPipeline(lang_code=clip["lang_code"])
                pipelines[clip["lang_code"]] = pipeline
            audio = None
            for result in pipeline(clip["text"], voice=clip["voice"], speed=1.0):
                audio = result.audio
                break  # short single-sentence clips
            if audio is None:
                raise RuntimeError("no audio generated")
        except Exception as error:  # noqa: BLE001 - degrade per clip
            print(f"SKIP {clip['name']} [{clip['language']}]: {type(error).__name__}: {error}", flush=True)
            skipped.append(clip["name"])
            continue
        audio = np.asarray(audio).squeeze()
        peak = float(np.max(np.abs(audio))) or 1.0
        pcm16 = (audio / peak * 32767.0).astype("<i2")
        name = f"{clip['name']}.wav"
        sf.write(HERE / name, pcm16, RATE, subtype="PCM_16")
        wav_bytes = (HERE / name).read_bytes()
        entries.append(
            {
                "text": clip["text"],
                "file": name,
                "sha256": hashlib.sha256(wav_bytes).hexdigest(),
                "language": clip["language"],
                "provenance": {
                    "engine": "kokoro",
                    "model": MODEL_ID,
                    "revision": revision,
                    "package": "kokoro==0.9.4 (--no-deps) + modern transformers/misaki",
                    "voice": clip["voice"],
                    "native_sample_rate": RATE,
                    "resampled_to_16000": False,
                    "compute_dtype": "float32-cpu",
                    "license": "Apache-2.0",
                    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "reviewed_by": "pending-human-review",
                },
            }
        )
        print(f"{name}: {len(pcm16)} samples ({len(pcm16)/RATE:.2f}s)", flush=True)

    if not entries:
        print("no clips generated", file=sys.stderr)
        return 1
    manifest = {
        "format": {"encoding": "pcm_s16le", "sample_rate": RATE, "channels": 1, "sample_width": 2},
        "note": (
            "open-model fixtures (Kokoro-82M, CPU float32, generated locally). German is "
            "pending: Kokoro has no German voice and Qwen3-TTS (covers de) requires bf16 "
            "hardware (device-side assert on Kaggle T4 fp16; package forces CUDA on cpu)."
        ),
        "skipped": skipped,
        "clips": entries,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} clips + manifest | skipped: {skipped or 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
