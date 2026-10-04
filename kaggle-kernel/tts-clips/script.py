"""Kaggle CPU kernel: generate open-model caller clips with Kokoro-82M.

History (2026-10-04): Qwen3-TTS-12Hz-0.6B requires bfloat16; float16 on T4
died with a CUDA device-side assert during generation, device_map=cpu did not
help because the qwen-tts package moves its code predictor to CUDA
internally, and Kaggle GPUs (T4/P100) predate bf16. Recorded in
evals/models/qwen3-tts-12hz-0.6b-customvoice.json. Kokoro-82M (Apache-2.0,
82M params) runs fine on plain CPU float32.

Kokoro supports the requested English and Spanish; German is not in its
language set, so the German open-model clip stays pending a servable
multilingual model (Qwen3-TTS covers de but needs bf16 hardware).
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKING = Path("/kaggle/working")
OUT = WORKING / "open-models"
MODEL_ID = "hexgrad/Kokoro-82M"
TARGET_RATE = 24000  # Kokoro native rate; no resampling needed

CLIPS = [
    {"language": "en", "voice": "af_heart", "name": "en-request", "text": "I'd like to book an appointment for Tuesday."},
    {"language": "en", "voice": "af_heart", "name": "en-name", "text": "My name is Alex Morgan."},
    {"language": "en", "voice": "af_heart", "name": "en-correct", "text": "Sorry, could we make that 11:00 instead?"},
    {"language": "es", "voice": "ef_dora", "name": "es-request", "text": "Quisiera reservar una cita el martes."},
    {"language": "es", "voice": "ef_dora", "name": "es-name", "text": "Me llamo Alex Morgan."},
]


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(cmd)}")


def main() -> int:
    print("=== voice-evals open-model clips (Kokoro-82M, CPU float32) ===", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "kokoro", "soundfile"])

    revision = None
    try:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(MODEL_ID).sha
    except Exception as error:  # noqa: BLE001 - provenance best-effort, null if unknown
        print(f"revision lookup failed: {error}", flush=True)
    print(f"model: {MODEL_ID} revision={revision}", flush=True)

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for clip in CLIPS:
        try:
            pipeline = KPipeline(lang_code=clip["voice"][0])  # 'a' = american english, 'e' = spanish
            generator = pipeline(clip["text"], voice=clip["voice"], speed=1.0)
            audio = None
            for result in generator:
                audio = result.audio
                break  # short single-sentence clips
            if audio is None:
                raise RuntimeError("no audio generated")
        except Exception as error:  # noqa: BLE001 - degrade per clip, never lose the batch
            print(f"SKIP {clip['name']} [{clip['language']}]: {type(error).__name__}: {error}", flush=True)
            continue
        audio = np.asarray(audio).squeeze()
        peak = float(np.max(np.abs(audio))) or 1.0
        pcm16 = (audio / peak * 32767.0).astype("<i2")
        name = f"{clip['name']}.wav"
        sf.write(OUT / name, pcm16, TARGET_RATE, subtype="PCM_16")
        wav_bytes = (OUT / name).read_bytes()
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
                    "package": "kokoro (PyPI latest at generation time)",
                    "voice": clip["voice"],
                    "native_sample_rate": TARGET_RATE,
                    "resampled_to_16000": False,
                    "compute_dtype": "float32-cpu",
                    "license": "Apache-2.0",
                    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "reviewed_by": "pending-human-review",
                },
            }
        )
        print(f"{name}: {len(pcm16)} samples ({len(pcm16)/TARGET_RATE:.2f}s)", flush=True)

    manifest = {
        "format": {"encoding": "pcm_s16le", "sample_rate": TARGET_RATE, "channels": 1, "sample_width": 2},
        "note": (
            "open-model fixtures (Kokoro-82M, CPU float32). German remains pending: "
            "Kokoro has no German voice and Qwen3-TTS (which covers de) requires bf16 "
            "hardware unavailable on Kaggle T4/P100."
        ),
        "clips": entries,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} clips + manifest (24 kHz native)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
