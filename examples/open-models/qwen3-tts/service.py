"""Qwen3-TTS -> voice-evals HTTP caller contract. Deployment skeleton.

NOT RUN HERE: this wrapper was not executed against a live checkpoint in this
repository; it documents the adapter boundary and the required provenance.
Deploy on your own GPU host, then:

    export QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
    export QWEN_TTS_REVISION=<pinned git sha>   # recorded in provenance
    export QWEN_TTS_VOICE=<speaker id>
    uvicorn service:app --host 0.0.0.0 --port 8080

Then: voice-eval probe scenario.json --caller http \\
          --caller-url-env CALLER_TTS_URL --caller-model $QWEN_TTS_MODEL
with CALLER_TTS_URL=http://<host>:8080/tts

Any resampling to 16/24 kHz happens HERE (in the service) and must be
recorded in the response provenance — the harness never resamples.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Response  # type: ignore[import-not-found]

app = FastAPI()

MODEL = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
REVISION = os.environ.get("QWEN_TTS_REVISION")  # null stays null, never guessed
VOICE = os.environ.get("QWEN_TTS_VOICE")
NATIVE_RATE = 24000  # Qwen3-TTS native output rate; resampled below when needed

# Lazy import: the model stack is only loaded in the deployment environment.
_engine = None


def get_engine():
    global _engine  # noqa: PLW0602 - assigned by the deployment's loader
    if _engine is None:
        # Deployment-specific: load the pinned checkpoint here.
        # Example (pseudo):
        #   from transformers import AutoModelForTextToWaveform  # or vLLM serving
        #   _engine = AutoModelForTextToWaveform.from_pretrained(MODEL, revision=REVISION)
        raise RuntimeError("implement checkpoint loading for your deployment")
    return _engine


@app.post("/tts")
async def tts(body: dict) -> Response:
    text = str(body.get("text", ""))
    if not text:
        return Response(status_code=422, content="text required")
    if body.get("format") != "pcm_s16le":
        return Response(status_code=422, content="only pcm_s16le is served")
    rate = int(body.get("sample_rate", 16000))
    if rate not in (16000, 24000):
        return Response(status_code=422, content="sample_rate must be 16000 or 24000")

    engine = get_engine()
    pcm = synthesize(engine, text, voice=body.get("voice") or VOICE, seed=body.get("seed"), rate=rate)
    return Response(
        content=pcm,
        headers={
            "X-Audio-Encoding": "pcm_s16le",
            "X-Audio-Rate": str(rate),
            "X-Audio-Channels": "1",
            "X-Audio-Sample-Width": "2",
            "X-Provenance-Model": MODEL,
            "X-Provenance-Revision": REVISION or "unknown",
            "X-Provenance-Resampled": "true" if rate != NATIVE_RATE else "false",
        },
    )


def synthesize(engine, text: str, *, voice, seed, rate: int) -> bytes:
    """Deploy-specific synthesis; return mono s16le PCM at ``rate``."""
    raise NotImplementedError("implement for your deployment")
