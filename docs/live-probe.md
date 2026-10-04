# Live probe guide

[Back to the README](../README.md)

```bash
pip install "voice-evals[probe]"   # adds httpx + websockets
```

The harness becomes a synthetic caller: it TTS-generates caller lines from a
scenario script (v2 JSON, bundled example:
[`src/voice_evals/resources/scenarios/appointment-v2.json`](../src/voice_evals/resources/scenarios/appointment-v2.json)),
calls your agent over its real transport, records the session, and scores it
with the same evaluator. Try the fully offline mock demo (~25s, no
credentials, no network):

```bash
make demo-probe
# or: voice-eval probe src/voice_evals/resources/scenarios/appointment-v2.json \
#       --mock --output-dir out/probe-demo --max-wer 0.05 --max-barge-in-stop-ms 500
```

A real call against an agent speaking the bundled reference protocol:

```bash
export ELEVENLABS_API_KEY=...      # caller TTS
export ELEVENLABS_VOICE_ID=...     # an API key alone does not identify a voice
export PROBE_TRANSPORT_URL=wss://your-agent.example.com/voice
export PROBE_AGENT_API_KEY=...     # agent auth, never the ElevenLabs key

voice-eval probe src/voice_evals/resources/scenarios/appointment-v2.json \
  --caller elevenlabs --voice-id "$ELEVENLABS_VOICE_ID" \
  --output-dir out/probe-live --json
```

Open-model caller instead of ElevenLabs (the model service runs elsewhere; see
[`examples/open-models/`](../examples/open-models/README.md)):

```bash
export CALLER_TTS_URL=http://your-tts-service:8080/tts
voice-eval probe scenario.json --caller http \
  --caller-model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --transport "$PROBE_TRANSPORT_URL" --output-dir out/probe-open
```

Prepared real speech instead of TTS:
`--caller fixture --fixture-manifest evals/fixtures/audio/synthetic/manifest.json`.

### Local reference agent (real sockets, no third-party agent needed)

[`examples/reference-agent/`](../examples/reference-agent/README.md) is a
localhost WebSocket agent speaking exactly `default-v1` ; scripted policy,
tone TTS, and three STT modes (`none` = honest NOT SCORED, `watermark` = pairs
with `--mock`, `scribe` = **real ElevenLabs Scribe STT**). It validates the
probe's real wire path end to end without any external agent service.

### Recorded live validation (2026-10-04)

The [sanitized metric summary](validation/live-probe-summary.json) preserves selected values and a SHA-256 of the original result. This documents a previous run; the real provider tests were not rerun for the README update.

The repository records validation of three live layers. Reproduction tests are opt-in
(`VOICE_EVALS_LIVE_TESTS=1` + secrets, never in CI):

| Layer | What ran | Result |
| --- | --- | --- |
| Caller TTS | `ElevenLabsCallerVoice` against the real API: native `pcm_16000`/`pcm_24000`, MP3-body rejection, 401-not-retried | 4/4 passed (`tests/integration/test_live_elevenlabs.py`) |
| Real WebSocket E2E | Full `SessionRunner` over a real socket vs the reference agent (watermark mode) | scored, replay-equal, barge-in observed (`tests/integration/test_live_probe_local_agent.py`) |
| Full real probe | CLI `voice-eval probe` with the real ElevenLabs caller (`eleven_multilingual_v2`, stock voice Sarah) → real WSS transport → reference agent with real Scribe STT | **completed + scored**: WER **0.1379** (real TTS→STT round trip: Scribe heard "hour" for "instead", dropped punctuation), task 1.0, E2E turn p50 1086ms/p95 2034ms, barge-in stop 232.6ms; exported `calls.jsonl` replays to identical scores; no secret in any artifact |

**Completed does not mean gate passed:** the saved live run exceeded its configured WER limit of 0.10 (`mean_wer = 0.1379`, `gate_passed = false`). Its turn-level latency percentiles differ from the legacy call-level aggregate: the latter contains one mean-of-turns call record (1380.1 ms), so its p50/p95/p99 are identical.

That WER is a measurement of the ElevenLabs-TTS → Scribe-STT round
trip through the probe's real wire path; it says nothing about any agent's
intelligence (the reference agent's dialogue policy is scripted). The run used the stock voice Sarah (`EXAVITQu4vr4xnSDxMaL`).

### What the probe records

Every run writes a complete, replayable recording directory:

```text
out/probe/manifest.json   sanitized manifest: config, chosen alternatives, timings,
                          interruption observations, provenance, file hashes
out/probe/events.jsonl    append-only normalized event journal
out/probe/calls.jsonl     exported v0.1 replay record (empty+marked when not replayable)
out/probe/result.json     v0.1 result fields + additive "probe" object + gate_passed
out/probe/audio/caller/*.wav   exactly the bytes sent, per utterance
out/probe/audio/agent/*.wav    exactly the bytes received, per response
```

A live session and its exported `calls.jsonl` replay to identical legacy
scores. Exit codes: `0` observed+scored and gates passed, `1` measured or
behavioral failure (including NOT SCORED runs with explicit reasons), `2`
config/transport/provider/recording error.

### Honesty rules baked into the output

- Client-observed events cannot reveal hidden STT/LLM/TTS stages. Legacy
  `llm_ttft_ms`/`tts_ttfa_ms` are populated **only** when explicit stage events
  exist; otherwise proxies are reported under separate names and stage fields
  stay `null` with a recorded reason.
- STT latency is labeled `client_final_asr` (includes endpointing + network).
- A barge-in that never confirms a stop is right-censored (`not_stopped`), with
  a lower bound ; never a fabricated duration.
- Ineligible sessions print **NOT SCORED** with exclusion reasons; placeholder
  zero-fields are conventions, not measurements.
- Secrets are resolved at execution time and never serialized: manifests
  record env variable names, never values or signed URLs.
- Mock latency is simulated and is never presented as a provider benchmark.

### Scenario v2 in one minute

Four caller turns with a conditional branch and a deliberate interruption
(full schema: `src/voice_evals/resources/schemas/scenario-v2.schema.json`):

```json
{"id": "correct_time", "intent": "correct_appointment_time",
 "cue": {"mode": "interrupt", "response_to": "provide_name",
          "after_ms": 600, "timeout_ms": 15000, "if_missed": "fail"},
 "utterance": {"text": "Sorry to interrupt. I need {{corrected_time}}, not ten.",
                "alternatives": ["Sorry, could we make that {{corrected_time}} instead?"],
                "reveals": ["corrected_time"]}}
```

- Alternatives are chosen by a stable hash of `seed`+step id ; deterministic,
  independent of branch execution order.
- `{{placeholders}}` must be declared facts, and every substituted fact must be
  listed in `reveals` (mismatch is a validation error).
- Branches (`when`/`on_unmatched`) are bounded and deterministic: literal
  substring predicates over one completed response. No LLM branching in v0.2.
- `outcome_rule.source: final_agent_text` is an explicit proxy for what the
  agent *reported* ; not evidence a calendar write succeeded.

### Transport: one protocol map, no magic

A WebSocket URL does not specify an audio protocol. The probe speaks any
endpoint covered by a declarative JSON map (bundled reference:
[`default-v1.json`](../src/voice_evals/resources/protocols/default-v1.json)):
PCM formats per direction, handshake, outbound envelopes with a fixed
substitution allowlist, an inbound discriminator with JSON-pointer selectors,
declared capabilities, and env-referenced auth. Write your own map against
[`protocol-v1.schema.json`](../src/voice_evals/resources/schemas/protocol-v1.schema.json)
; no Python, Jinja, or scripts are ever evaluated from a map. Native PCM only
(mono s16le 16/24 kHz): no hidden codecs or resampling; unsupported formats
fail preflight. "Works with any agent" means any agent whose protocol fits the
map; a telephony webhook or arbitrary binary protocol needs a future adapter.
