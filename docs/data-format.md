# Replay dataset and Python API

[Back to the README](../README.md)

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

Only `id`, `scenario.name`, `scenario.expected_outcome` and `ground_truth_transcript` are required. Missing transcripts and outcomes default to empty strings and still participate in scoring. Only unavailable timing values and unmeasured interruption-stop durations are omitted from their aggregates (`n/a` when none exist). Supply the observed transcripts and outcome for meaningful scores. v0.2 is fully API- and schema-compatible with v0.1.1; all changes are additive.

A larger frozen regression corpus (123 calls incl. hand-checked WER cases) ships in [`evals/data/`](../evals/data/README.md).

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

## Metric semantics

- WER is the arithmetic mean of per-call word error rates, not a word-weighted corpus WER. Insertions can make WER exceed 1.0. Empty reference plus non-empty hypothesis is assigned 1.0.
- Outcome matching trims whitespace and ignores case. It compares the recorded outcome to the expected label.
- Fact coverage uses case-insensitive substring matches in the agent transcript or recorded outcome; an empty required-facts list scores 1.0.
- `hallucination_rate` is the share of calls containing at least one configured forbidden phrase in the agent transcript. It is not an open-ended factuality judge.
- Replay latency percentiles aggregate the provided per-call values. Live probe `probe.turn_latency` separately reports observed turn-level percentiles.
- The interruption-stop median includes measured stops only. Inspect probe counts and censoring reasons before interpreting it.
