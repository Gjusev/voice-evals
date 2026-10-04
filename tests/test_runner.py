"""Runner behavior on mock sessions: overlap, branches, failures, replay."""

import asyncio
import json
from pathlib import Path

from voice_evals import evaluate, load_dataset
from voice_evals.probe import (
    MockAgentPolicy,
    MockCallerVoice,
    MockReplyRule,
    MockTransport,
    ProbeConfig,
    ScenarioScript,
    SessionRunner,
)
from voice_evals.probe.clock import VirtualClock

RESOURCES = Path(__file__).resolve().parents[1] / "src" / "voice_evals" / "resources"
APPOINTMENT = RESOURCES / "scenarios" / "appointment-v2.json"


def make_runner(tmp_path: Path, policy: MockAgentPolicy | None = None, **config_kwargs):
    clock = VirtualClock(slice_ns=1_000_000)
    config = ProbeConfig(environment="mock", output_dir=tmp_path / "out", **config_kwargs)
    transport = MockTransport(clock=clock, policy=policy)
    caller = MockCallerVoice(clock=clock)
    runner = SessionRunner(caller=caller, transport=transport, config=config, clock=clock)
    return runner, config


async def run_default(tmp_path: Path, policy: MockAgentPolicy | None = None, **config_kwargs):
    runner, config = make_runner(tmp_path, policy, **config_kwargs)
    script = ScenarioScript.load(APPOINTMENT)
    session = await runner.run(script, output_dir=config.output_dir)
    return runner, session


def test_full_mock_session_success(tmp_path: Path) -> None:
    runner, session = asyncio.run(run_default(tmp_path))
    assert session.status.value == "completed"
    assert session.errors == []
    report = runner.last_report
    assert report.probe["scoring_status"] == "scored"
    assert report.result["samples"] == 1
    assert report.result["mean_wer"] == 0.0
    assert report.result["task_completion"] == 1.0
    assert report.result["fact_coverage"] == 1.0
    # E2E support on ordinary turns, never on the interruption turn
    assert report.probe["metric_support"]["e2e_ms"] == 3
    # All four turns sent in order with deterministic selection
    texts = [t.selected_text for t in session.turns]
    assert len(texts) == 4
    assert all("{{" not in t for t in texts)


def test_barge_in_overlap_and_stop(tmp_path: Path) -> None:
    runner, session = asyncio.run(run_default(tmp_path))
    interrupt_turn = next(t for t in session.turns if t.interrupt is not None)
    observation = interrupt_turn.interrupt
    assert observation.status == "observed"
    assert observation.agent_stopped_ns is not None
    # Mock stops within interrupt_stop_ms(260ms) + one stream interval(40ms).
    assert 0 <= observation.agent_stopped_ns / 1e6 <= 340
    # Ordinary-turn aggregates exclude the interruption turn (3 of 4 turns).
    assert runner.last_report.probe["metric_support"]["e2e_ms"] == 3
    counts = runner.last_report.probe["interruptions"]
    assert counts["attempted"] == 1 and counts["observed"] == 1


def test_interrupt_turn_overlaps_agent_response(tmp_path: Path) -> None:
    _, session = asyncio.run(run_default(tmp_path))
    turn = next(t for t in session.turns if t.step_id == "correct_time")
    frames = turn.frames
    assert frames, "interrupt utterance must send frames"
    # During the interrupt the target response was active: audio frames of the
    # old response were received after the first caller frame send.
    assert turn.interrupt is not None and turn.interrupt.response_active_at_b is True


def test_censored_barge_in_when_agent_ignores_interrupts(tmp_path: Path) -> None:
    policy = MockAgentPolicy(ignore_interrupts=True)
    runner, session = asyncio.run(run_default(tmp_path, policy=policy))
    observation = next(t.interrupt for t in session.turns if t.interrupt)
    assert observation.status == "not_stopped"
    assert observation.agent_stopped_ns is None  # never fabricated
    counts = runner.last_report.probe["interruptions"]
    assert counts["not_stopped"] == 1
    # A requested gate must fail on the censored attempt.
    assert runner.last_report.gate_passed is False


def test_barge_in_missed_fails_closed(tmp_path: Path) -> None:
    # Short replies finish long before the interrupt point -> missed window.
    policy = MockAgentPolicy(long_reply_ms=300)
    runner, session = asyncio.run(run_default(tmp_path, policy=policy))
    assert session.status.value == "failed"
    assert session.status_reason == "barge_in_missed"
    report = runner.last_report
    counts = report.probe["interruptions"]
    assert counts["missed"] == 1
    assert report.gate_passed is False


def test_barge_in_missed_skip_then_unreachable_dependency(tmp_path: Path) -> None:
    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["steps"][2]["cue"]["if_missed"] = "skip"
    script = ScenarioScript.from_dict(raw)
    policy = MockAgentPolicy(long_reply_ms=300)
    clock = VirtualClock(slice_ns=1_000_000)
    config = ProbeConfig(environment="mock", output_dir=tmp_path / "out")
    runner = SessionRunner(
        caller=MockCallerVoice(clock=clock),
        transport=MockTransport(clock=clock, policy=policy),
        config=config,
        clock=clock,
    )
    session = asyncio.run(runner.run(script, output_dir=config.output_dir))
    skipped = next(t for t in session.turns if t.step_id == "correct_time")
    assert skipped.skipped is True and skipped.interrupt is not None
    assert skipped.interrupt.status == "missed"
    # confirm references the skipped step: explicit unreachable dependency,
    # never an indefinite wait.
    assert session.status.value == "failed"
    assert any("skipped" in str(e.get("message", "")) for e in session.errors)
    confirm = [t for t in session.turns if t.step_id == "confirm"]
    assert confirm == [] or not confirm[0].sent


def test_unreachable_dependency_after_skip(tmp_path: Path) -> None:
    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["steps"][2]["cue"]["if_missed"] = "skip"
    script = ScenarioScript.from_dict(raw)
    policy = MockAgentPolicy(long_reply_ms=300)
    clock = VirtualClock(slice_ns=1_000_000)
    config = ProbeConfig(environment="mock", output_dir=tmp_path / "out")
    runner = SessionRunner(
        caller=MockCallerVoice(clock=clock),
        transport=MockTransport(clock=clock, policy=policy),
        config=config,
        clock=clock,
    )
    session = asyncio.run(runner.run(script, output_dir=config.output_dir))
    # confirm references the skipped correct_time: explicit failure, no hang.
    assert session.status.value == "failed"
    assert "did not run" in (session.status_reason or "") or session.errors


def test_branch_predicate_failure_is_scenario_unmet(tmp_path: Path) -> None:
    # Agent never asks for the name -> provide_name `when` fails.
    policy = MockAgentPolicy(rules=(MockReplyRule(match=("appointment",), reply="Booked. Done."),))
    runner, session = asyncio.run(run_default(tmp_path, policy=policy))
    assert session.status.value == "failed"
    assert session.status_reason == "scenario_unmet"
    assert any(e["category"] == "scenario_unmet" for e in session.errors)
    report = runner.last_report
    assert "behavioral_failures" in report.probe
    assert report.gate_passed is False


def test_response_timeout_is_bounded(tmp_path: Path) -> None:
    # Agent that never answers: policy with no rules still answers fallback...
    # so simulate silence by huge processing delay beyond cue timeout.
    policy = MockAgentPolicy(processing_ms=60_000)
    runner, session = asyncio.run(run_default(tmp_path, policy=policy))
    assert session.status.value == "failed"
    assert session.status_reason == "response_timeout"
    # Not scored: evidence incomplete; diagnostic-only with explicit reason.
    report = runner.last_report
    assert report.probe["scoring_status"] == "not_scored"
    assert report.probe["exclusion_reasons"]


def test_asr_error_injection_scores_nonzero_wer(tmp_path: Path) -> None:
    policy = MockAgentPolicy(asr_substitutions={"Tuesday": "Thursday", "Morgan": "Morgon"})
    runner, _session = asyncio.run(run_default(tmp_path, policy=policy))
    report = runner.last_report
    assert report.probe["scoring_status"] == "scored"
    assert report.result["mean_wer"] > 0.0


def test_live_and_replay_identical_legacy_scores(tmp_path: Path) -> None:
    runner, _session = asyncio.run(run_default(tmp_path))
    live = runner.last_report.result
    calls_path = Path(config_dir(runner)) / "calls.jsonl"
    replay = evaluate(load_dataset(str(calls_path))).to_dict()
    for key in (
        "samples", "mean_wer", "max_wer", "task_completion", "fact_coverage",
        "hallucination_rate", "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
        "interruption_count", "median_barge_in_stop_ms", "stage_means_ms",
    ):
        assert replay[key] == live[key], f"{key} differs between live and replay"


def config_dir(runner: SessionRunner) -> str:
    return str(runner.config.output_dir)


def test_pacing_drift_does_not_accumulate(tmp_path: Path) -> None:
    _, session = asyncio.run(run_default(tmp_path))
    turn = session.turns[0]
    lateness = [f.send_start_ns - f.planned_send_ns for f in turn.frames]
    # Cumulative-sample-anchored pacing: lateness stays bounded, not growing.
    assert max(lateness) < 50_000_000  # < 50ms even after a multi-second clip
    # Caller audio end = last send completion + final frame sample duration.
    last = turn.frames[-1]
    expected_end = last.send_complete_ns + last.sample_count / 16_000 * 1e9
    assert abs(turn.caller_audio_end_ns - expected_end) < 1_000_000


def test_call_timeout_bounds_session(tmp_path: Path) -> None:
    policy = MockAgentPolicy(long_reply_ms=120_000)
    _runner, session = asyncio.run(
        run_default(tmp_path, policy=policy, call_timeout_ms=4_000)
    )
    assert session.status.value == "failed"
    assert session.status_reason == "call_timeout"
