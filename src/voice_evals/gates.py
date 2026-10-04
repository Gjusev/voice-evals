"""Threshold gates shared by replay ``run`` and probe, plus probe completeness.

Legacy replay gate behavior is unchanged in this release. Probe gates fail
closed: a requested metric without evidence cannot pass its threshold.
"""

from __future__ import annotations

from typing import Any


def metric_gate_problems(result: dict[str, Any], thresholds: dict[str, float | None]) -> list[str]:
    """The five v0.1 metric gates over a legacy result dict."""
    problems: list[str] = []
    max_wer = thresholds.get("max_wer")
    if max_wer is not None and result["mean_wer"] > max_wer:
        problems.append(f"mean_wer {result['mean_wer']:.4f} > allowed {max_wer:.4f}")
    min_tc = thresholds.get("min_task_completion")
    if min_tc is not None and result["task_completion"] < min_tc:
        problems.append(f"task_completion {result['task_completion']:.4f} < required {min_tc:.4f}")
    min_fc = thresholds.get("min_fact_coverage")
    if min_fc is not None and result["fact_coverage"] < min_fc:
        problems.append(f"fact_coverage {result['fact_coverage']:.4f} < required {min_fc:.4f}")
    max_e2e = thresholds.get("max_e2e_p95_ms")
    if max_e2e is not None and result.get("e2e_p95_ms") is not None and result["e2e_p95_ms"] > max_e2e:
        problems.append(f"e2e_p95_ms {result['e2e_p95_ms']:.1f} > allowed {max_e2e:.1f}")
    max_hall = thresholds.get("max_hallucination_rate")
    if max_hall is not None and result["hallucination_rate"] > max_hall:
        problems.append(f"hallucination_rate {result['hallucination_rate']:.4f} > allowed {max_hall:.4f}")
    return problems


def probe_gate_problems(
    report: dict[str, Any],
    *,
    request_barge_in_gate: bool,
    max_barge_in_stop_ms: int | None,
    max_e2e_p95_ms: float | None = None,
) -> list[str]:
    """Completeness gates over the additive ``probe`` report object.

    A requested metric with no supporting observations fails closed; a
    not-scored run can never pass.
    """
    problems: list[str] = []
    probe = report.get("probe") or {}
    if probe.get("scoring_status") != "scored":
        problems.append("probe not scored; gates require a nonzero scored sample")
        return problems
    if report.get("samples", 0) == 0:
        problems.append("scored sample is zero")
    support = probe.get("metric_support", {})
    if max_e2e_p95_ms is not None and support.get("e2e_ms", 0) == 0:
        problems.append("e2e observations unavailable; the requested e2e gate lacks evidence")
    interruptions = probe.get("interruptions", {})
    if request_barge_in_gate:
        attempted = interruptions.get("attempted", 0)
        if attempted == 0:
            problems.append("barge-in gate requested but no interruption was attempted")
        else:
            for key in ("missed", "unsupported", "unestablished", "not_stopped"):
                count = interruptions.get(key, 0)
                if count:
                    problems.append(f"barge-in gate: {count} required attempt(s) {key.replace('_', ' ')}")
            observed = interruptions.get("observed", 0)
            if observed == 0:
                problems.append("barge-in gate: no measured stop durations")
            elif max_barge_in_stop_ms is not None:
                worst = interruptions.get("max_stop_ms")
                if worst is not None and worst > max_barge_in_stop_ms:
                    problems.append(
                        f"max barge-in stop {worst:.1f}ms > allowed {max_barge_in_stop_ms:.1f}ms"
                    )
    return problems
