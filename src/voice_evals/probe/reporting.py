"""Legacy result plus additive probe diagnostics.

Existing v0.1 result keys and types are preserved verbatim. The probe adds one
``probe`` object and ``gate_passed``; rich turn metrics live there or in the
manifest. Not-scored runs keep every v0.1 field with the documented zero-sample
placeholder convention (numeric rates 0.0, latencies null, ``stage_means_ms={}``,
``details=[]``) and print "NOT SCORED" instead of those placeholder rates.
"""

from __future__ import annotations

import sys
from typing import Any

from ..evaluate import evaluate
from ..types import percentile
from .config import ProbeConfig
from .conversion import check_eligibility, to_call_record
from .models import SessionRecord
from .timing import AGGREGATION_LABEL, ordinary_turn_means


class ProbeReport:
    """Scored-or-not result bundle for one probe session."""

    def __init__(self, result: dict[str, Any], probe: dict[str, Any]) -> None:
        self.result = result
        self.probe = probe

    @property
    def gate_passed(self) -> bool:
        return bool(self.result.get("gate_passed"))

    def print_summary(self, stream: Any = None) -> None:
        stream = stream or sys.stdout
        probe = self.probe
        if probe.get("scoring_status") != "scored":
            stream.write(
                f"NOT SCORED ({probe.get('session_status', '?')}): "
                + "; ".join(probe.get("exclusion_reasons", []) or ["ineligible"])
                + "\n"
            )
            return
        result = self.result
        stream.write(
            f"samples={result['samples']} failures={result['failures']}\n"
            f"wer mean={result['mean_wer']:.4f} max={result['max_wer']:.4f}\n"
            f"task_completion={result['task_completion']:.4f}\n"
            f"fact_coverage={result['fact_coverage']:.4f}\n"
            f"hallucination_rate={result['hallucination_rate']:.4f}\n"
        )
        for key in ("e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms"):
            value = result.get(key)
            stream.write(f"{key}={'n/a' if value is None else f'{value:.1f}'}\n")
        if result.get("stage_means_ms"):
            stage = " ".join(f"{k}={v:.1f}" for k, v in sorted(result["stage_means_ms"].items()))
            stream.write(f"stage means: {stage}\n")
        turn = probe.get("turn_latency", {})
        if turn.get("e2e_p50_ms") is not None:
            stream.write(
                f"turn e2e p50={turn['e2e_p50_ms']:.1f} p95={turn['e2e_p95_ms']:.1f} "
                f"p99={turn['e2e_p99_ms']:.1f} n={turn.get('e2e_count', 0)}\n"
            )
        interruptions = probe.get("interruptions", {})
        if interruptions:
            stream.write(
                "interruptions attempted={attempted} observed={observed} missed={missed} "
                "unsupported={unsupported} not_stopped={not_stopped}\n".format(**{**_DEFAULT_INTERRUPTIONS, **interruptions})
            )
        basis = probe.get("measurement_basis")
        if basis:
            stream.write(f"basis: {basis}\n")

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps({**self.result, "probe": self.probe}, indent=indent)


_DEFAULT_INTERRUPTIONS = {
    "attempted": 0, "observed": 0, "missed": 0, "unsupported": 0, "not_stopped": 0
}


def score_session(session: SessionRecord, config: ProbeConfig, gate_problems: list[str] | None = None) -> ProbeReport:
    """Build the additive report for one finished session."""
    eligible, reasons = check_eligibility(session)
    interruption_counts = _interruption_counts(session)
    _means, support = ordinary_turn_means(session.turns)
    turn_e2e = sorted(
        t.e2e_ns / 1e6 for t in session.turns if t.e2e_ns is not None
    )
    stops = [
        t.interrupt.agent_stopped_ns / 1e6
        for t in session.turns
        if t.interrupt is not None and t.interrupt.agent_stopped_ns is not None
    ]
    behavioral_failures = _behavioral_failures(session)
    probe: dict[str, Any] = {
        "session_id": session.session_id,
        "session_status": session.status.value,
        "run_mode": session.environment,
        "scoring_status": "scored" if eligible else "not_scored",
        "exclusion_reasons": [] if eligible else reasons,
        "attempts": 1,
        "scored_calls": 1 if eligible else 0,
        "excluded_calls": 0 if eligible else 1,
        "execution_failures": session.errors,
        "behavioral_failures": behavioral_failures,
        "metric_support": support,
        "stage_aggregation": AGGREGATION_LABEL,
        "measurement_basis": (
            "local monotonic clock; client send/receive observations only; "
            "mock latency is simulated and not a provider benchmark"
            if session.environment == "mock"
            else "local monotonic clock; client send/receive observations only; "
            "includes network transit and client scheduling"
        ),
        "turn_latency": {
            "e2e_p50_ms": percentile(turn_e2e, 50) if turn_e2e else None,
            "e2e_p95_ms": percentile(turn_e2e, 95) if turn_e2e else None,
            "e2e_p99_ms": percentile(turn_e2e, 99) if turn_e2e else None,
            "e2e_count": len(turn_e2e),
        },
        "interruptions": interruption_counts,
        "outcome_rule": session.outcome_rule_used,
        "clock_domain": "time.perf_counter_ns (session-relative)",
    }
    if gate_problems is None:
        gate_problems = []
    probe["gate_reasons"] = gate_problems

    if eligible:
        record = to_call_record(session)
        assert record is not None  # eligibility checked above
        legacy = evaluate([record]).to_dict()
        result = {**legacy, "probe": probe, "gate_passed": not gate_problems}
        return ProbeReport(result=result, probe=probe)

    # Not scored: v0.1 field types preserved with the zero-sample convention.
    placeholder: dict[str, Any] = {
        "samples": 0,
        "failures": 0,
        "mean_wer": 0.0,
        "max_wer": 0.0,
        "task_completion": 0.0,
        "fact_coverage": 0.0,
        "hallucination_rate": 0.0,
        "e2e_p50_ms": None,
        "e2e_p95_ms": None,
        "e2e_p99_ms": None,
        "stage_means_ms": {},
        "interruption_count": sum(
            1 for t in session.turns if t.interrupt is not None
        ),
        "median_barge_in_stop_ms": percentile(stops, 50) if stops else None,
        "details": [],
    }
    result = {**placeholder, "probe": probe, "gate_passed": False}
    return ProbeReport(result=result, probe=probe)


def _interruption_counts(session: SessionRecord) -> dict[str, Any]:
    attempts = [t.interrupt for t in session.turns if t.interrupt is not None]
    stops = [a.agent_stopped_ns / 1e6 for a in attempts if a.agent_stopped_ns is not None]
    counts: dict[str, Any] = {
        "attempted": len(attempts),
        "observed": sum(1 for a in attempts if a.status == "observed"),
        "missed": sum(1 for a in attempts if a.status == "missed"),
        "unsupported": sum(1 for a in attempts if a.status == "unsupported"),
        "unestablished": sum(1 for a in attempts if a.status == "unestablished"),
        "not_stopped": sum(1 for a in attempts if a.status == "not_stopped"),
    }
    if stops:
        counts["median_stop_ms"] = percentile(stops, 50)
        counts["max_stop_ms"] = max(stops)
    return counts


def _behavioral_failures(session: SessionRecord) -> list[str]:
    failures: list[str] = []
    for error in session.errors:
        if error.get("category") in ("scenario_unmet", "barge_in_missed"):
            failures.append(str(error.get("message")))
    if session.status.value == "completed" and session.scenario.outcome_rule is not None:
        expected = session.scenario.expected_outcome.strip().lower()
        if session.outcome.strip().lower() != expected:
            failures.append(
                f"outcome {session.outcome!r} != expected {session.scenario.expected_outcome!r}"
            )
    return failures
