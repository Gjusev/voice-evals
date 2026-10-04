# Reference voice-agent server (default-v1)

A **local development example** that speaks exactly the bundled
`voice-evals-default-v1` protocol
(`src/voice_evals/resources/protocols/default-v1.json`) over a real WebSocket,
so the probe's real network path — `WebSocketAgentTransport`, `websockets`
client, base64 PCM framing, pacing, event ordering, barge-in — can be
validated with zero providers and zero cost.

**This is not a product agent.** It is clearly labeled in its startup banner
(`reference agent: scripted policy, tone TTS, STT per --asr mode`):

- **Dialogue policy is scripted.** It reuses `DEFAULT_REPLY_RULES` and
  `FALLBACK_REPLY` (via `MockAgentPolicy` from `voice_evals.probe.testing`)
  — the same deterministic appointment-dialogue rules as the offline mock.
- **TTS is deterministic PCM tones** (`tone_pcm`), not speech.
- **STT is a stage you select** with `--asr` (below); two of the three modes
  are simulated.

## STT modes (`--asr`)

| mode        | what happens                                                                                                                     | caller pairing                                        |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `none`      | Emits **no** `input_transcript_final` events. The probe scores this honestly: sessions that need caller transcripts are **not scored** (no transcripts, no WER). | any                                                    |
| `watermark` | Decodes the mock caller's amplitude watermark (`decode_text_pcm` from `voice_evals.probe.testing`). Simulated ASR, instant.       | `MockCallerVoice` clips (or fixtures of them) only     |
| `scribe`    | **Real STT.** The utterance PCM is wrapped as WAV and POSTed to `https://api.elevenlabs.io/v1/speech-to-text` (`model_id=scribe_v1`); the response's `text` becomes the transcript (typically ~0.5–3 s per utterance). | real-speech callers (e.g. `--caller elevenlabs`)       |

`scribe` reads the key from the `ELEVENLABS_API_KEY` environment variable at
call time only. It is never logged and never written to any file; on HTTP or
network errors the server logs one safe line (status / exception type, no key)
and emits no transcript event, exactly like `none` for that utterance.

## Run

```bash
# from the repository root
uv run python examples/reference-agent/server.py --asr watermark
# banner: reference agent: scripted policy, tone TTS, STT per --asr mode
# listening on ws://127.0.0.1:8765 (asr=watermark, ...)
```

### Flags

| flag                   | default | meaning                                                       |
| ---------------------- | ------- | ------------------------------------------------------------- |
| `--host` / `--port`    | `127.0.0.1` / `8765` | bind address                                    |
| `--asr`                | `none`  | STT mode: `none` \| `watermark` \| `scribe`                   |
| `--interrupt-stop-ms`  | `260`   | barge-in: stop the current stream this long after new utterance audio |
| `--processing-ms`      | `150`   | simulated ASR->first-event delay: `input_transcript_final` and `response_start` wait this out after commit |
| `--stream-interval-ms` | `40`    | one `output_audio` event per interval (80 ms of tone each)    |
| `--short-reply-ms`     | `1500`  | total streamed audio for short replies                        |
| `--long-reply-ms`      | `6000`  | total streamed audio when the matched rule is long            |
| `--ignore-interrupts`  | off     | keep streaming past barge-ins (censored-run testing)          |

## Protocol behavior (default-v1 server side)

- `session_start` → `session_ready`.
- `input_audio` frames append to a per-utterance buffer (base64 s16le mono
  16 kHz). A frame for a **new** utterance id while a response is streaming is
  a barge-in: the current stream stops after `--interrupt-stop-ms` (skipped,
  not cancelled, frames) and emits `response_done`. With
  `--ignore-interrupts` the stream continues.
- Utterances finalize on `input_commit` **or** by simulated VAD (250 ms with
  no new frames).
- On finalize: `input_transcript_final` (per `--asr`), `response_start`
  (`ra-N`), `output_text_delta` (first half) + `output_text_final` (scripted
  reply), streamed `output_audio` frames, then `response_done` at the natural
  end, on a barge-in stop, or on `cancel_response`.
- Max message size 4 MB, compression off. Logs contain ids/counts/timings
  only — never auth headers, endpoints, or query strings.

## Point the probe at it

The realistic CLI pairing is real caller TTS + `--asr scribe` (real STT both
sides of the socket):

```bash
export ELEVENLABS_API_KEY=...   # caller TTS + server scribe STT (env only)
export ELEVENLABS_VOICE_ID=...  # caller voice
uv run python examples/reference-agent/server.py --asr scribe &
uv run voice-eval probe src/voice_evals/resources/scenarios/appointment-v2.json \
  --transport ws://127.0.0.1:8765 \
  --caller elevenlabs \
  --max-barge-in-stop-ms 800
```

`--asr none` runs the same way but scores nothing (honest NOT SCORED
sessions — useful for transport/latency debugging).

`--asr watermark` pairs only with `MockCallerVoice` audio, and the CLI
deliberately refuses `--mock`/mock callers together with a live transport
(`ProbeConfig.validate`), so watermark mode is driven from a small script
(the smoke test below) or from fixture clips of watermark PCM, not from the
CLI.

## Smoke test (no API key needed)

Start the server with `--asr watermark`, connect
`WebSocketAgentTransport(load_default_protocol_map(),
connector=websockets.asyncio.client.connect)`, assert `capabilities.duplex`,
synthesize `MockCallerVoice().synthesize("I would like to book an
appointment for Tuesday.")`, send 20 ms frames paced in real time, commit via
`transport.end_utterance`, and collect events until the first
`response_done`. Expected and verified (Windows, Python 3.13, websockets
15.0.1):

- `caller_transcript_final` with the exact spoken text;
- `output_text_final` = the name-question reply ("... May I have your name,
  please?");
- 32 `agent_audio` frames (>= 5 required);
- `response_done` ~1.66-1.70 s after commit (150 ms `--processing-ms` +
  1500 ms short reply on a 40 ms tick); commit -> transcript ~155 ms (the
  processing guard — ASR completes before the reply pipeline starts);
  commit -> first audio ~200 ms. With `--processing-ms 0` the same script
  measures ~1.54 s / < 1 ms / ~47 ms instead.

The same script against `--asr scribe` (real STT, real ElevenLabs TTS caller)
returned the exact transcript with commit -> transcript ~0.96 s (~0.81 s STT
round trip plus the 150 ms `--processing-ms` guard), and against
`--asr scribe` with no `ELEVENLABS_API_KEY` it emitted no transcript and fell
back to `FALLBACK_REPLY` — the honest no-key behavior.
