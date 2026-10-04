# Caller audio fixtures

## `multilingual/` — real speech, three languages (present)

Checksummed WAV clips (16 kHz mono s16le) of the scenario caller lines in
English, German and Spanish. Generated 2026-10-04 by
`multilingual/generate.py` with hosted ElevenLabs
`eleven_multilingual_v2` (premade voice Sarah), because serving
Qwen3-TTS/Kokoro locally was out of scope. **These are not open-model
clips**; each manifest entry records engine/model/voice/timestamp, and the
note says so explicitly. Regeneration spends one TTS call per clip:

    uv run python evals/fixtures/audio/multilingual/generate.py

Measured TTS -> Scribe STT round trip on these clips (2026-10-04, live opt-in
test `tests/integration/test_live_multilingual.py`):

| Language | Mean WER | Dominant error |
| --- | --- | --- |
| en | 0.067 | final period dropped on short sentences |
| de | 0.083 | same punctuation deletion |
| es | 0.208 | same punctuation deletion (shorter sentences weigh more) |

That is a property of the ElevenLabs TTS -> Scribe round trip on these exact
bytes, not of any voice agent, and not an open-model measurement.

## `synthetic/` — watermarked tones (present)

Deterministic synthetic PCM/WAV with an embedded text watermark, produced by
`voice_evals.probe.testing.encode_text_pcm`. They exercise
`FixtureCallerVoice` offline and are **explicitly not speech and not any
model's output**. Regenerate with:

    uv run python evals/fixtures/audio/synthetic/generate.py

## `open-models/` — Kokoro-82M clips (present, generated locally)

Real open-model speech generated 2026-10-04 **locally on CPU float32** with
`hexgrad/Kokoro-82M` (Apache-2.0, revision pinned per clip): English
(`af_heart`) and Spanish (`ef_dora`) at the model's native 24 kHz, no
resampling. Regenerate with `generate.py` (see its docstring for the scratch
venv; kokoro 0.9.4 pins `transformers==4.12.2`, which has no Python 3.13
wheels, so install it `--no-deps` with modern deps).

Measured Kokoro -> Scribe STT round trip (2026-10-04, same live method as the
multilingual set): **en mean WER 0.114, es 0.208**. Notable: Scribe hears
"1100" for "11:00" (numeral normalization), besides the usual dropped final
period.

**German is pending bf16 hardware.** Qwen3-TTS covers German but requires
bfloat16: on a Kaggle T4, float16 died with a CUDA device-side assert during
generation and `device_map=cpu` does not help because the `qwen-tts` package
moves its code predictor to CUDA internally. Kaggle's GPU shapes (T4/P100)
and the local GTX 1080 are all pre-Ampere. Full account in
`evals/models/qwen3-tts-12hz-0.6b-customvoice.json`.

Do not relabel the `multilingual/` (ElevenLabs) or `synthetic/` clips as
open-model output. Manifest shape for any future additions:

```json
{
  "format": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1, "sample_width": 2},
  "clips": [
    {
      "text": "Ich möchte einen Termin am Dienstag.",
      "file": "open-models/de-request.wav",
      "sha256": "<hash>",
      "language": "de",
      "provenance": {
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "revision": "<pinned git sha>",
        "server": "<image/service digest>",
        "voice": "<speaker id>",
        "settings": {"seed": 17},
        "license": "Apache-2.0",
        "reviewed_by": "human"
      }
    }
  ]
}
```
