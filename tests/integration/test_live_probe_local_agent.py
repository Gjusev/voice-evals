"""Opt-in live-path end-to-end probe against the bundled reference agent.

Unlike test_live_probe.py this needs no secrets: it spawns the reference agent
(examples/reference-agent/server.py) on a free localhost port with watermark
ASR, which pairs with MockCallerVoice, and runs the REAL SessionRunner over a
REAL WebSocket using the default protocol map and the real monotonic clock.
Still opt-in (VOICE_EVALS_LIVE_TESTS=1) and blocked in CI: it binds a real
socket and the paced four-turn run takes roughly 25-40 seconds.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("VOICE_EVALS_LIVE_TESTS") != "1",
        reason="live tests require VOICE_EVALS_LIVE_TESTS=1",
    ),
    pytest.mark.skipif(
        os.environ.get("CI") == "1" or os.environ.get("GITHUB_ACTIONS") == "true",
        reason="live calls are never made from CI",
    ),
]

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / "examples" / "reference-agent" / "server.py"
SCENARIO = REPO / "src" / "voice_evals" / "resources" / "scenarios" / "appointment-v2.json"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, proc: subprocess.Popen, timeout_s: float = 20.0) -> None:
    """Block until the agent's socket accepts, or fail with its output."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout is not None else ""
            raise RuntimeError(f"reference agent exited rc={proc.returncode}: {output}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"reference agent did not accept on 127.0.0.1:{port}")


def _run_session(port: int, out: Path):
    from voice_evals.probe.config import ProbeConfig, TransportConfig
    from voice_evals.probe.runner import SessionRunner
    from voice_evals.probe.scenario import ScenarioScript
    from voice_evals.probe.testing import MockCallerVoice
    from voice_evals.probe.transports.protocol import load_default_protocol_map
    from voice_evals.probe.transports.websocket import WebSocketAgentTransport

    async def run():
        config = ProbeConfig(
            environment="local",
            output_dir=out,
            transport=TransportConfig(endpoint=f"ws://127.0.0.1:{port}/"),
        )
        runner = SessionRunner(
            caller=MockCallerVoice(),  # watermark PCM pairs with --asr watermark
            transport=WebSocketAgentTransport(load_default_protocol_map()),
            config=config,  # default MonotonicClock: real timed run (~25-40s)
        )
        return await runner.run(ScenarioScript.load(SCENARIO), output_dir=out)

    return asyncio.run(run())


def test_local_reference_agent_end_to_end(tmp_path: Path) -> None:
    """Real runner over a real localhost WebSocket against the reference agent."""
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(SERVER),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--asr",
            "watermark",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_port(port, proc)
        out = tmp_path / "session"
        session = _run_session(port, out)

        # Session completed with no operational/behavioral errors.
        assert session.status.value == "completed", session.status_reason
        assert session.errors == []

        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        probe = result["probe"]
        assert probe["scoring_status"] == "scored"
        assert probe["run_mode"] == "local"

        # All four scripted turns were fully sent (none skipped).
        assert len(session.turns) == 4
        assert all(turn.sent for turn in session.turns)
        assert not any(turn.skipped for turn in session.turns)

        # Barge-in: one attempt, one observed stop within 400ms.
        interrupts = probe["interruptions"]
        assert interrupts["attempted"] == 1
        assert interrupts["observed"] == 1
        assert interrupts["missed"] == 0 and interrupts["not_stopped"] == 0
        stop_ms = next(
            turn.interrupt.agent_stopped_ns for turn in session.turns if turn.interrupt is not None
        ) / 1e6
        assert stop_ms <= 400.0, stop_ms

        # Artifacts: manifest, exported replay rows, per-stream WAVs.
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        assert manifest["environment"] == "local"
        assert (out / "calls.jsonl").is_file()
        caller_wavs = list((out / "audio" / "caller").glob("*.wav"))
        agent_wavs = list((out / "audio" / "agent").glob("*.wav"))
        assert len(caller_wavs) == 4
        assert len(agent_wavs) == 4

        # Canonical replay equality: the exported calls.jsonl must reproduce
        # the live legacy metrics exactly.
        from voice_evals import evaluate, load_dataset

        replay = evaluate(load_dataset(str(out / "calls.jsonl"))).to_dict()
        for key in ("mean_wer", "task_completion", "median_barge_in_stop_ms"):
            assert replay[key] == result[key], key
        # Watermark ASR decodes the sent text verbatim -> zero WER; the final
        # agent text satisfies the outcome rule.
        assert result["mean_wer"] == 0.0
        assert result["task_completion"] == 1.0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()
