# Kaggle reproduction

[Back to the README](../README.md)

[![Kernel A: offline benchmark](https://img.shields.io/badge/Kaggle-Kernel_A_%E2%80%93_offline_benchmark-20BEFF?logo=kaggle)](https://www.kaggle.com/code/gjusev/voice-evals-offline-benchmark)
[![Kernel B: live probe](https://img.shields.io/badge/Kaggle-Kernel_B_%E2%80%93_live_probe-20BEFF?logo=kaggle)](https://www.kaggle.com/code/gjusev/voice-evals-live-probe)

> The links match the kernel IDs in this repository. Public availability and successful execution should be checked on Kaggle; the local sources are linked below.

**Kernel A** ([`kaggle-kernel/offline/`](../kaggle-kernel/offline/script.py)) proves
harness reproducibility with `enable_internet=false`: it installs the exact
published wheel and pinned dependency wheels from a checksummed Kaggle dataset
(`--no-index --find-links --require-hashes`; never PyPI at runtime), then
asserts the unchanged v0.1 demo metrics, the frozen regression corpus with
hand-checked WERs, a full four-turn mock probe (conditional branch, overlap
barge-in, artifacts, live-to-replay equality), the fixture-caller path, and
the offline test suite. Outputs land in `/kaggle/working` with provenance and
artifact checksums.

**Kernel B** ([`kaggle-kernel/probe/`](../kaggle-kernel/probe/script.py)) is a live
diagnostic: with Kaggle Secrets `ELEVENLABS_API_KEY` + `PROBE_TRANSPORT_URL`
it runs one scripted probe over outbound WSS; with secrets missing it runs the
full mock session through the same runner/recorder/evaluator and prints only
the missing secret names ; no network attempt. A failed live attempt keeps its
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
your localhost is not reachable from Kaggle ; expose an authenticated public
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
[`kaggle-kernel/bundle/README.md`](../kaggle-kernel/bundle/README.md).
