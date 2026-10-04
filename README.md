# voice-evals

**Evaluation harness for voice agents: WER, latency budgets, barge-in behavior and task outcomes, with CI gates.**

Text-agent evals are everywhere. Voice adds four layers that nobody has open-sourced well: transcription quality under accents and noise, per-stage latency (voice has a hard "feels instant" budget around 800ms), interruption handling, and whether the call actually achieved its goal. `voice-evals` measures them from recorded calls, offline, with zero credentials.

Built by someone who runs a production voice agent ([HeizPro KI](https://github.com/Gjusev), real-time STT/LLM/TTS), not from a spec sheet.

## What it measures

| Layer | Metrics |
| --- | --- |
| Transcription | per-call WER (live STT output vs ground truth), mean and max |
| Latency | end-to-end p50/p95/p99 plus per-stage means: STT, LLM time-to-first-token, TTS time-to-first-audio |
| Behavior | interruption count, median time until agent audio stops after a barge-in |
| Outcome | task completion vs expected outcome, required-fact coverage, hallucination rate (forbidden claims) |

## Quick start

```bash
pip install voice-evals
```

Score the bundled demo dataset, no credentials needed:

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

## Dataset format

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

Only `id`, `scenario.name`, `scenario.expected_outcome` and `ground_truth_transcript` are required. Everything else scores `n/a` when absent, so you can start with transcripts only and add timings later.

## Python API

```python
from voice_evals import evaluate, load_dataset

result = evaluate(load_dataset("calls.jsonl"))
print(result.summary())
print(result.e2e_p95_ms, result.hallucination_rate)
```

## Roadmap

- **v0.2 live probe**: the harness becomes a synthetic caller. It TTS-generates caller lines from a scenario script, calls your agent over its real transport (WebSocket first, Twilio/Telnyx next), records the session and scores it live, including deliberate barge-ins.
- LLM judge for graded outcomes instead of exact outcome matching.
- Regression-gate GitHub Action comparing against a committed baseline.

## Development

```bash
make install   # editable install with dev extras
make test      # pytest, no network
make lint      # ruff
make build     # wheel + sdist
```

## License

Apache 2.0. See [LICENSE](LICENSE).
