"""Probe CLI: mock demo, gates, exit codes, safe errors. No network."""

import json
from pathlib import Path

from voice_evals.cli import main

SCENARIO = (
    Path(__file__).resolve().parents[1] / "src" / "voice_evals" / "resources" / "scenarios" / "appointment-v2.json"
)

MINI_SCRIPT = {
    "schema_version": 2,
    "name": "mini",
    "expected_outcome": "done",
    "seed": 3,
    "facts": {},
    "steps": [
        {
            "id": "ask",
            "intent": "book_appointment",
            "cue": {"mode": "speak_now"},
            "utterance": {"text": "I'd like to book an appointment."},
        },
        {
            "id": "thanks",
            "intent": "closing",
            "cue": {"mode": "wait_agent_end", "response_to": "ask", "timeout_ms": 15000},
            "utterance": {"text": "Thank you."},
        },
    ],
    "outcome_rule": {
        "source": "final_agent_text",
        "after_step": "thanks",
        "all_of": ["appointment"],
        "value": "done",
    },
}


def mini_scenario(tmp_path: Path) -> str:
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(MINI_SCRIPT), encoding="utf-8")
    return str(path)


def run_probe(tmp_path: Path, *extra: str, scenario: str | None = None) -> int:
    out = tmp_path / "out"
    return main(["probe", scenario or str(SCENARIO), "--mock", "--output-dir", str(out), *extra])


def test_mock_probe_passes_gates_and_writes_artifacts(tmp_path: Path, capsys) -> None:
    output = tmp_path / "scored.json"
    code = run_probe(
        tmp_path, "--max-wer", "0.05", "--max-barge-in-stop-ms", "600", "--output", str(output)
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "gate: PASSED" in captured.out
    assert "interruptions attempted=1 observed=1" in captured.out
    assert "mock latency is simulated" in captured.out
    scored = json.loads(output.read_text(encoding="utf-8"))
    assert scored["samples"] == 1 and scored["gate_passed"] is True
    assert (tmp_path / "out" / "manifest.json").is_file()


def test_mock_probe_slow_barge_in_fails_gate(tmp_path: Path, capsys) -> None:
    code = run_probe(tmp_path, "--max-barge-in-stop-ms", "50")  # mock stops ~230ms
    assert code == 1
    captured = capsys.readouterr()
    assert "gate: FAILED" in captured.out
    assert "barge-in" in captured.out


def test_mock_probe_task_failure_when_agent_breaks_script(tmp_path: Path, monkeypatch) -> None:
    # Agent never asks for the name -> scenario_unmet -> exit 1.
    from voice_evals.probe import MockAgentPolicy, MockReplyRule
    from voice_evals.probe.testing import MockTransport

    policy = MockAgentPolicy(rules=(MockReplyRule(match=("appointment",), reply="Booked."),))
    created = {}

    def fake_build_transport(config, clock=None):
        transport = MockTransport(clock=clock, policy=policy)
        created["transport"] = transport
        return transport

    import voice_evals.probe as probe_pkg

    monkeypatch.setattr(probe_pkg, "build_transport", fake_build_transport)
    out = tmp_path / "out"
    code = main(["probe", str(SCENARIO), "--mock", "--output-dir", str(out)])
    assert code == 1
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["probe"]["session_status"] == "failed"
    assert any("scenario_unmet" == e["category"] for e in result["probe"]["execution_failures"])


def test_missing_scenario_file_is_usage_error(tmp_path: Path, capsys) -> None:
    code = main(["probe", str(tmp_path / "nope.json"), "--mock", "--output-dir", str(tmp_path)])
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_legacy_scenario_rejected_for_probe(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "legacy.json"
    scenario.write_text(
        json.dumps({"name": "old", "expected_outcome": "booked", "required_facts": ["x"]}),
        encoding="utf-8",
    )
    code = run_probe(tmp_path, scenario=str(scenario))
    assert code == 2
    assert "no probe script" in capsys.readouterr().err


def test_contradictory_mock_and_transport_rejected(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("PROBE_TRANSPORT_URL", "ws://localhost:9000/x")
    code = run_probe(tmp_path)
    assert code == 2
    assert "contradictory" in capsys.readouterr().err


def test_json_output_mode(tmp_path: Path, capsys) -> None:
    code = run_probe(tmp_path, "--json", "--max-wer", "0.05", scenario=mini_scenario(tmp_path))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["probe"]["run_mode"] == "mock"
    assert payload["gate_passed"] is True


def test_run_subcommand_unchanged(tmp_path: Path) -> None:
    demo = Path(__file__).resolve().parents[1] / "evals" / "data" / "demo_calls.jsonl"
    assert main(["run", str(demo), "--max-wer", "0.05"]) == 0
    assert main(["run", str(demo), "--max-wer", "0.001", "--json"]) == 1
    assert main(["run", "does-not-exist.jsonl"]) == 2
