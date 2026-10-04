# Open-model serving recipes

Separate environments: these recipes run **outside** the voice-evals package.
The harness never loads weights or performs inference — it only speaks the
documented client contracts to services you host.

## 1. Caller TTS service (HTTP contract)

Any server that answers this contract works with `--caller http`:

```
POST $CALLER_TTS_URL
Authorization: Bearer $CALLER_TTS_AUTH          (optional)
Content-Type: application/json

{"text": "...", "model": "...", "voice": "...", "language": "en",
 "seed": 17, "format": "pcm_s16le", "sample_rate": 16000, "channels": 1}
```

Response, either:

- raw headerless PCM (mono s16le, 16 or 24 kHz) with explicit headers
  `X-Audio-Encoding: pcm_s16le`, `X-Audio-Rate`, `X-Audio-Channels`,
  `X-Audio-Sample-Width`; or
- one uncompressed PCM WAV body (`Content-Type: audio/wav`). Compressed WAV
  and MP3/Opus are rejected — resample/transcode in the service, not here.

### Qwen3-TTS wrapper (primary recipe)

`qwen3-tts/service.py` is a minimal FastAPI wrapper skeleton that adapts a
Qwen3-TTS checkpoint to the contract above, including any resampling done in
the service (record it in provenance). It pins model/revision/voice via env.
Deploy it on your own GPU host; point `CALLER_TTS_URL` at it.

Status: contract shape tested offline against mocks (`tests/test_callers.py`);
the wrapper itself has **not** been run against a live checkpoint here.

### Chatterbox Turbo / Kokoro-82M

Same contract. Chatterbox: preserve upstream watermarking. Kokoro: useful for
pregenerating fixture clips during bundle preparation.

## 2. Open voice-agent stack (WebSocket)

The probe speaks any WebSocket endpoint matching a protocol map. For an
open stack (e.g. Voxtral Mini Realtime STT + your LLM + TTS behind a small
gateway):

1. Implement the server side of `voice-evals-default-v1`
   (`src/voice_evals/resources/protocols/default-v1.json`): input_audio
   envelopes in, output_audio/output_text_*/input_transcript_final/
   response_done out, base64 PCM.
2. Or write your own map JSON (see `resources/schemas/protocol-v1.schema.json`)
   and pass `--protocol-map your-map.json`.
3. The external agent owns STT, LLM, TTS, turn-taking, and the real transport.
   voice-evals does not implement a voice-agent framework.

`voxtral-gateway/README.md` sketches the reference gateway contract. Qwen3-Omni
and Moshi are future duplex targets needing native codec adapters — their
upstream protocols are not interchangeable with this map.

## Provenance rules

- Record model repo + revision, server image digest, voice, generation
  settings, quantization (if known), language, and audio hashes per clip.
- Unknown fields stay `null`; never infer them.
- Never relabel fixture or mock audio as a model's measured output, and never
  present mock traces as live provider results.
