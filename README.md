# voice-evals

**Evaluation harness for voice agents: replay scoring plus a scripted live probe — WER, latency budgets, barge-in behavior and task outcomes, with CI gates.**

Text-agent evals are everywhere. Voice adds four layers that nobody has open-sourced well: transcription quality under accents and noise, per-stage latency (voice has a hard "feels instant" budget around 800ms), interruption handling, and whether the call actually achieved its goal. `voice-evals` measures them from recorded calls, offline, with zero credentials — and since v0.2 it can also *make* the calls: a deterministic scripted caller speaks to your live agent over its real WebSocket transport, barges in, records everything, and scores it with the same replay evaluator.

Built by someone who runs a production voice agent ([HeizPro KI](https://github.com/Gjusev), real-time STT/LLM/TTS), not from a spec sheet.

## What it measures

| Layer | Metrics |
| --- | --- |
| Transcription | per-call WER (live STT output vs ground truth), mean and max |
| Latency | end-to-end p50/p95/p99 plus per-stage means: STT, LLM time-to-first-token, TTS time-to-first-audio |
| Behavior | interruption count, median time until agent audio stops after a barge-in |
| Outcome | task completion vs expected outcome, required-fact coverage, hallucination rate (forbidden claims) |

## Quick start (replay, no credentials)

```bash
pip install voice-evals
```

```bash
voice-eval run evals/data/demo_calls.jsonl
```

```text
samples=3 failures=0
wer mean=0.0303 max=0.0909
task_completion=0.6667
fact_coverage=0.8333
hallucination_rate=0.3333
e2e_ms p50=950.0 p95=1103.0 p99=1118.6
stage means: e2e_ms=963.3 llm_ttft_ms=330.0 stt_ms=200.0 tts_ttfa_ms=236.7
interruptions=1 median_barge_in_stop_ms=210.0
```

Gate in CI:

```bash
voice-eval run calls.jsonl --max-wer 0.05 --min-task-completion 0.90 --max-e2e-p95-ms 1000
```

Exit codes: `0` passed, `1` gate failed, `2` dataset error.

## v0.2: live probe

```bash
pip install "voice-evals[probe]"   # adds httpx + websockets
```

The harness becomes a synthetic caller: it TTS-generates caller lines from a
scenario script (v2 JSON, bundled example:
[`src/voice_evals/resources/scenarios/appointment-v2.json`](src/voice_evals/resources/scenarios/appointment-v2.json)),
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
[`examples/open-models/`](examples/open-models/README.md)):

```bash
export CALLER_TTS_URL=http://your-tts-service:8080/tts
voice-eval probe scenario.json --caller http \
  --caller-model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --transport "$PROBE_TRANSPORT_URL" --output-dir out/probe-open
```

Prepared real speech instead of TTS:
`--caller fixture --fixture-manifest evals/fixtures/audio/synthetic/manifest.json`.

### Local reference agent (real sockets, no third-party agent needed)

[`examples/reference-agent/`](examples/reference-agent/README.md) is a
localhost WebSocket agent speaking exactly `default-v1` — scripted policy,
tone TTS, and three STT modes (`none` = honest NOT SCORED, `watermark` = pairs
with `--mock`, `scribe` = **real ElevenLabs Scribe STT**). It validates the
probe's real wire path end to end without any external agent service.

### Live validation performed (2026-10-04, this repository)

All three live layers were exercised and are committed as opt-in tests
(`VOICE_EVALS_LIVE_TESTS=1` + secrets, never in CI):

| Layer | What ran | Result |
| --- | --- | --- |
| Caller TTS | `ElevenLabsCallerVoice` against the real API: native `pcm_16000`/`pcm_24000`, MP3-body rejection, 401-not-retried | 4/4 passed (`tests/integration/test_live_elevenlabs.py`) |
| Real WebSocket E2E | Full `SessionRunner` over a real socket vs the reference agent (watermark mode) | scored, replay-equal, barge-in observed (`tests/integration/test_live_probe_local_agent.py`) |
| Full real probe | CLI `voice-eval probe` with the real ElevenLabs caller (`eleven_multilingual_v2`, stock voice Sarah) → real WSS transport → reference agent with real Scribe STT | **completed + scored**: WER **0.1379** (real TTS→STT round trip: Scribe heard "hour" for "instead", dropped punctuation), task 1.0, E2E turn p50 1086ms/p95 2034ms, barge-in stop 232.6ms; exported `calls.jsonl` replays to identical scores; no secret in any artifact |

That WER is a genuine measurement of the ElevenLabs-TTS → Scribe-STT round
trip through the probe's real wire path; it says nothing about any agent's
intelligence (the reference agent's dialogue policy is scripted). The
originally supplied private voice id was not present in the account
(`voice_not_found`); the documented stock premade voice Sarah
(`EXAVITQu4vr4xnSDxMaL`) was used instead.

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
  a lower bound — never a fabricated duration.
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

- Alternatives are chosen by a stable hash of `seed`+step id — deterministic,
  independent of branch execution order.
- `{{placeholders}}` must be declared facts, and every substituted fact must be
  listed in `reveals` (mismatch is a validation error).
- Branches (`when`/`on_unmatched`) are bounded and deterministic: literal
  substring predicates over one completed response. No LLM branching in v0.2.
- `outcome_rule.source: final_agent_text` is an explicit proxy for what the
  agent *reported* — not evidence a calendar write succeeded.

### Transport: one protocol map, no magic

A WebSocket URL does not specify an audio protocol. The probe speaks any
endpoint covered by a declarative JSON map (bundled reference:
[`default-v1.json`](src/voice_evals/resources/protocols/default-v1.json)):
PCM formats per direction, handshake, outbound envelopes with a fixed
substitution allowlist, an inbound discriminator with JSON-pointer selectors,
declared capabilities, and env-referenced auth. Write your own map against
[`protocol-v1.schema.json`](src/voice_evals/resources/schemas/protocol-v1.schema.json)
— no Python, Jinja, or scripts are ever evaluated from a map. Native PCM only
(mono s16le 16/24 kHz): no hidden codecs or resampling; unsupported formats
fail preflight. "Works with any agent" means any agent whose protocol fits the
map; a telephony webhook or arbitrary binary protocol needs a future adapter.

## Dataset format (replay)

JSONL, one recorded call per line (JSON arrays also work):

```json
{
  "id": "call-001",
  "scenario": {
    "name": "book-appointment",
    "expected_outcome": "booked",
    "required_facts": ["Tuesday", "appointment"],
    "forbidden_facts": ["discount"]
  },
  "asr_transcript": "what the agent's STT heard",
  "ground_truth_transcript": "what the caller actually said",
  "agent_transcript": "everything the agent said",
  "outcome": "booked",
  "stage_timings_ms": {"stt_ms": 190, "llm_ttft_ms": 310, "tts_ttfa_ms": 220, "e2e_ms": 820},
  "interruptions": [{"at_ms": 4200, "agent_stopped_ms": 210}]
}
```

Only `id`, `scenario.name`, `scenario.expected_outcome` and `ground_truth_transcript` are required. Everything else scores `n/a` when absent, so you can start with transcripts only and add timings later. v0.2 is fully API- and schema-compatible with v0.1.1; all changes are additive.

A larger frozen regression corpus (123 calls incl. hand-checked WER cases) ships in [`evals/data/`](evals/data/README.md).

## Reproduce in Kaggle

[![Kernel A: offline benchmark](https://img.shields.io/badge/Kaggle-Kernel_A_%E2%80%93_offline_benchmark-20BEFF?logo=kaggle)](https://www.kaggle.com/code/gjusev/voice-evals-offline-benchmark)
[![Kernel B: live probe](https://img.shields.io/badge/Kaggle-Kernel_B_%E2%80%93_live_probe-20BEFF?logo=kaggle)](https://www.kaggle.com/code/gjusev/voice-evals-live-probe)

> Publication pending — the links activate when the kernels and bundle
> dataset are published with the v0.2.x release.

**Kernel A** ([`kaggle-kernel/offline/`](kaggle-kernel/offline/script.py)) proves
harness reproducibility with `enable_internet=false`: it installs the exact
published wheel and pinned dependency wheels from a checksummed Kaggle dataset
(`--no-index --find-links --require-hashes`; never PyPI at runtime), then
asserts the unchanged v0.1 demo metrics, the frozen regression corpus with
hand-checked WERs, a full four-turn mock probe (conditional branch, overlap
barge-in, artifacts, live-to-replay equality), the fixture-caller path, and
the offline test suite. Outputs land in `/kaggle/working` with provenance and
artifact checksums.

**Kernel B** ([`kaggle-kernel/probe/`](kaggle-kernel/probe/script.py)) is a live
diagnostic: with Kaggle Secrets `ELEVENLABS_API_KEY` + `PROBE_TRANSPORT_URL`
it runs one scripted probe over outbound WSS; with secrets missing it runs the
full mock session through the same runner/recorder/evaluator and prints only
the missing secret names — no network attempt. A failed live attempt keeps its
artifacts and additionally writes a separately named mock diagnostic; the mock
score is never substituted for the live result. The open-model variant uses
`CALLER_TTS_URL` with `caller=http` instead of ElevenLabs credentials.

**Latency location warning:** live Kernel B latency is measured from the
Kaggle datacenter and includes network transit/RTT and client scheduling. It
is not directly comparable with local runs; mock latency is simulated and is
not a provider benchmark.

Secrets setup: notebook sidebar → Add-ons → Secrets → attach
`ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` (a stock voice id), and your
agent's `PROBE_TRANSPORT_URL` (+ optional `PROBE_AGENT_API_KEY`). A service on
your localhost is not reachable from Kaggle — expose an authenticated public
WSS endpoint. Your endpoint must match a protocol map; the default map covers
agents implementing the reference protocol.

Publishing (from the repository, host-side):

```bash
KAGGLE_API_TOKEN=... python -m kaggle kernels push -p kaggle-kernel/offline
KAGGLE_API_TOKEN=... python -m kaggle kernels status gjusev/voice-evals-offline-benchmark
KAGGLE_API_TOKEN=... python -m kaggle kernels output gjusev/voice-evals-offline-benchmark -p out/kaggle-offline/

KAGGLE_API_TOKEN=... python -m kaggle kernels push -p kaggle-kernel/probe
KAGGLE_API_TOKEN=... python -m kaggle kernels status gjusev/voice-evals-live-probe
KAGGLE_API_TOKEN=... python -m kaggle kernels output gjusev/voice-evals-live-probe -p out/kaggle-probe/
```

PowerShell: set the token once with `$env:KAGGLE_API_TOKEN="..."`, then run
the same commands without the Bash assignment prefix. Forks must change the
metadata `id` and the commands' owner. `KAGGLE_API_TOKEN` is a publisher
credential, not a kernel runtime secret. Bundle build details:
[`kaggle-kernel/bundle/README.md`](kaggle-kernel/bundle/README.md).

## Python API

```python
from voice_evals import evaluate, load_dataset

result = evaluate(load_dataset("calls.jsonl"))
print(result.summary())
print(result.e2e_p95_ms, result.hallucination_rate)
```

Programmatic probe (same machinery as the CLI):

```python
import asyncio
from pathlib import Path
from voice_evals.probe import (
    MockCallerVoice, MockTransport, ProbeConfig, ScenarioScript, SessionRunner,
)

async def main() -> None:
    script = ScenarioScript.load("scenario.json")
    config = ProbeConfig(environment="mock", output_dir=Path("out/probe"))
    runner = SessionRunner(caller=MockCallerVoice(), transport=MockTransport(), config=config)
    session = await runner.run(script, output_dir=config.output_dir)
    runner.last_report.print_summary()

asyncio.run(main())
```

The base install (jiwer only) imports the whole probe package, runner, mocks
and fixtures; `httpx`/`websockets` are imported only inside their adapters via
the `[probe]` extra.

## Roadmap

- Twilio/Telnyx media-stream adapters behind the same transport interface.
- Dedicated OpenAI Realtime adapter (append/commit/truncate semantics).
- LLM judge for graded outcomes instead of exact outcome matching.
- Regression-gate GitHub Action comparing against a committed baseline.

## Development

```bash
make install          # editable install incl. probe extra + dev tools
make test             # offline pytest (default: -m "not live")
make test-integration # opt-in live tests (VOICE_EVALS_LIVE_TESTS=1 + secrets)
make lint             # ruff
make build            # wheel + sdist
make demo-probe       # offline four-turn mock probe demo
make kaggle-bundle    # stage the internet-disabled bundle (validation mode)
```

Unit tests and CI never call real services. Live integration tests require
both `VOICE_EVALS_LIVE_TESTS=1` and the relevant secrets, and are additionally
guarded against running under CI.

## License

Apache 2.0. See [LICENSE](LICENSE).
