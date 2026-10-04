# Pinned model/server recipes (metadata only)

This directory records *provenance metadata* for open-model caller voices and
agent stacks. The harness never downloads weights or runs inference; recipes
point at externally hosted services that satisfy the documented HTTP caller
contract (`src/voice_evals/probe/voices/http.py`) or a WebSocket protocol map.

Status legend: **tested-offline** = exercised through mocks only; **pending**
= planned, requires external hosting/credentials not available here. Nothing
in this directory is a measured benchmark of the named models.

## Caller voices (TTS)

| Recipe | Model | License | Pin | Status |
| --- | --- | --- | --- | --- |
| `qwen3-tts-12hz-0.6b-customvoice.json` | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | Apache-2.0 (weights) | revision recorded in recipe; pin at deploy | tested-offline (HTTP contract via mock); live synthesis pending external service |
| `chatterbox-turbo.json` | resemble-ai/chatterbox (Turbo) | MIT (family; preserve upstream watermarking) | repo + server digest at deploy | pending adapter smoke check |
| `kokoro-82m.json` | hexgrad/Kokoro-82M | Apache-2.0 (weights) | revision at deploy | fixture-generation source only; upstream deps stay outside the harness |

## Agent-side components

| Component | Model | License | Role | Status |
| --- | --- | --- | --- | --- |
| `voxtral-mini-4b-realtime.json` | mistralai/Voxtral-Mini-4B-Realtime-2602 | Apache-2.0 | streaming STT inside an externally hosted voice-agent stack; its transcripts may populate caller ASR only when they are the agent's actual input transcript | pending external stack |
| Qwen3-Omni / Moshi | see `examples/open-models/` | mixed (Moshi weights CC-BY-4.0, code MIT/Apache — do not blanket-label Apache) | future end-to-end duplex targets; native codec/serving needs separate adapters | documented future work |

Unknown provider details must remain `null` in recipe files — never guessed.
