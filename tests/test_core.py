"""Runnable self-check against the demo dataset. Run: pytest"""

from pathlib import Path

from voice_evals import evaluate, load_dataset
from voice_evals.cli import main

DEMO = Path(__file__).resolve().parents[1] / "evals" / "data" / "demo_calls.jsonl"


def test_demo_metrics():
    result = evaluate(load_dataset(str(DEMO)))
    assert result.samples == 3
    assert abs(result.mean_wer - (1 / 11) / 3) < 0.01  # one substitution in call-002
    assert result.task_completion == 2 / 3  # call-002 reported "clarifying"
    assert result.fact_coverage == 5 / 6
    assert result.hallucination_rate == 1 / 3  # call-003 offered a forbidden free upgrade
    assert result.e2e_p50_ms == 950.0
    assert result.median_barge_in_stop_ms == 210.0


def test_cli_pass_and_gate():
    assert main(["run", str(DEMO), "--max-wer", "0.05"]) == 0
    assert main(["run", str(DEMO), "--max-wer", "0.001", "--json"]) == 1
    assert main(["run", "does-not-exist.jsonl"]) == 2
