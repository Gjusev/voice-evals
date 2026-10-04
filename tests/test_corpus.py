"""Frozen v0.2 regression corpus: metrics must match expected exactly."""

import json
from pathlib import Path

from voice_evals import evaluate, load_dataset

DATA = Path(__file__).resolve().parents[1] / "evals" / "data"


def load_corpus():
    return evaluate(load_dataset(str(DATA / "replay_v02.jsonl")))


def test_corpus_size_and_composition() -> None:
    result = load_corpus()
    assert result.samples == 123
    details = {d["scenario"] for d in result.details}
    assert "book-appointment" in details
    assert "cancel-order" in details
    assert "edge-case" in details


def test_corpus_metrics_frozen() -> None:
    result = load_corpus()
    expected = json.loads((DATA / "replay_v02.expected.json").read_text(encoding="utf-8"))
    metrics = result.to_dict()
    for key, value in expected["metrics"].items():
        got = metrics[key]
        if isinstance(value, float):
            assert abs(got - value) < 1e-9, f"{key}: {got} != {value}"
        else:
            assert got == value, f"{key}: {got} != {value}"


def test_hand_checked_wer_cases() -> None:
    """Hand-computed WERs, independent of the implementation under test."""
    result = load_corpus()
    by_id = {d["id"]: d["wer"] for d in result.details}
    expected = json.loads((DATA / "replay_v02.expected.json").read_text(encoding="utf-8"))
    for case in expected["hand_checked"]["cases"]:
        assert abs(by_id[case["id"]] - case["hand_computed_wer"]) < 1e-9


def test_documented_regression_gate() -> None:
    """The gate shipped with the corpus: loose enough for synthetic noise,
    tight enough to catch aggregation/parser regressions."""
    result = load_corpus()
    assert result.mean_wer <= 0.7
    assert 0.6 <= result.task_completion <= 0.8
    assert 0.7 <= result.fact_coverage <= 0.9
    assert result.median_barge_in_stop_ms is not None
    assert 0 < result.median_barge_in_stop_ms < 1000
    # Latency tail present (edge rows with ~3.6s e2e) and picked up by p95.
    assert result.e2e_p95_ms > 2000


def test_corpus_covers_edge_shapes() -> None:
    rows = [json.loads(line) for line in (DATA / "replay_v02.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r["asr_transcript"] == "" for r in rows)  # empty hypotheses
    wers = [evaluate([__import__("voice_evals").types.CallRecord.from_dict(r)]).max_wer for r in rows[:1]]
    assert any(
        len(r.get("stage_timings_ms", {})) == 0 for r in rows
    )  # absent timings
    assert any(
        r.get("stage_timings_ms") == {"e2e_ms": r["stage_timings_ms"]["e2e_ms"]}
        for r in rows
        if r.get("stage_timings_ms")
    )  # only-e2e rows (optional stages absent)
    assert any(
        i.get("agent_stopped_ms") is None for r in rows for i in r.get("interruptions", [])
    )  # unmeasured stops
    assert any(
        len(r.get("interruptions", [])) >= 2 for r in rows
    )  # multiple interruptions
    del wers
