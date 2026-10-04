# Demo transcript

[Back to the README](../README.md#watch-the-demo) · [Watch the MP4](assets/voice-evals-demo.mp4)

The video contains instrumental music and two soft reveal sounds. There is no spoken narration. All on-screen text is English.

| Time | On screen |
| --- | --- |
| 0:00–0:03.3 | “voice-evals. Can your voice agent handle an interruption?” “Reproducible evaluation for voice agents.” |
| 0:03.3–0:10.9 | “Put the call under test.” A real replay command and selected captured metrics, labeled “Synthetic replay demo · No API keys”. |
| 0:10.9–0:17.45 | “Keep the evidence.” The artifact directory and replay command. “Record the call. Replay the score.” “Eligible probe sessions export to the same offline evaluator.” |
| 0:17.45–0:22 | “voice-evals. Test the conversation. Not just the transcript.” “Replay scoring · Live probes · CI gates.” Repository URL and illustrative mint waveform artwork. |

The command shown is:

```bash
voice-eval run evals/data/demo_calls.jsonl
```

Selected output values:

```text
samples=3                 failures=0
wer mean=0.0303            max=0.0909
task_completion=0.6667
e2e_ms p50=950.0           p95=1103.0
median_barge_in_stop_ms=210.0
```

These are synthetic demonstration calls, not a provider benchmark. `failures=0` does not mean all tasks completed.

The recording directory contains `audio/`, `events.jsonl`, `manifest.json`, `calls.jsonl` and `result.json`. Eligible recordings can be replayed with:

```bash
voice-eval run out/probe-demo/calls.jsonl
```

[Music, sound and font credits](assets/README.md#video-credits).
