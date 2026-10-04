# Voxtral gateway — reference WebSocket contract sketch

An open voice-agent stack that the probe can call looks like:

```
voice-eval probe ──WS(default-v1 map)──> gateway ──> Voxtral Mini Realtime (STT)
                                         │      ──> your LLM
                                         │      ──> your TTS
                                         └── full-duplex audio + JSON events
```

The gateway, not voice-evals, owns STT/LLM/TTS orchestration, turn-taking,
and the real transport. This sketch documents the wire contract only; no
implementation is provided or implied to have been tested.

## Server side of `voice-evals-default-v1`

Inbound (caller -> agent), JSON text frames:

```json
{"type": "session_start", "session_id": "...", "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}}
{"type": "input_audio", "utterance_id": "utt-request", "seq": 0, "audio": "<base64 s16le mono 20ms>"}
{"type": "input_commit", "utterance_id": "utt-request"}     // optional with native VAD
{"type": "cancel_response", "response_id": "mock-r1"}       // only on explicit request; never during natural barge-in tests
```

Outbound (agent -> caller):

```json
{"type": "session_ready"}
{"type": "response_start", "response_id": "r1"}
{"type": "input_transcript_final", "utterance_id": "utt-request", "text": "<Voxtral STT of the caller audio>"}
{"type": "output_text_delta", "response_id": "r1", "text": "..."}
{"type": "output_text_final", "response_id": "r1", "text": "..."}
{"type": "output_audio", "response_id": "r1", "seq": 0, "audio": "<base64 PCM>"}
{"type": "response_done", "response_id": "r1"}
```

Requirements the measurement layer relies on:

- Stream `output_audio` incrementally (burst-completed audio cannot prove
  ongoing speech for barge-in activity checks).
- Correlate everything with `response_id`; carry `utterance_id` on input
  transcripts so caller-ASR attribution is explicit.
- Keep sending old-response audio while the caller barges in; the harness
  measures cessation of received audio without sending cancellations.
- Never route agent audio back into the caller stream (echo).
- Structured `{"type": "outcome", "value": "..."}` events let scenarios use
  `outcome_rule.source = "transport"` instead of text matching.

A service bound to your localhost is not reachable from Kaggle; expose an
authenticated public WSS endpoint for the Kaggle kernel.
