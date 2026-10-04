"""Score a set of replayed calls."""

from __future__ import annotations

from jiwer import wer as _jiwer_wer

from .types import CallRecord, VoiceEvalResult, percentile


def word_error_rate(truth: str, hypothesis: str) -> float:
    """WER of one utterance. Empty truth with non-empty hypothesis counts as 1.0."""
    if not truth.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return float(_jiwer_wer(truth, hypothesis))


def _contains(text: str, fact: str) -> bool:
    return fact.lower() in text.lower()


def score_record(record: CallRecord) -> dict:
    """Score one call: WER, outcome, facts, hallucinations, timings."""
    scenario = record.scenario
    call_wer = word_error_rate(record.ground_truth_transcript, record.asr_transcript)
    required = scenario.required_facts
    coverage = (
        sum(1 for fact in required if _contains(record.agent_transcript, fact) or _contains(record.outcome, fact))
        / len(required)
        if required
        else 1.0
    )
    hallucinations = [fact for fact in scenario.forbidden_facts if _contains(record.agent_transcript, fact)]
    task_complete = record.outcome.strip().lower() == scenario.expected_outcome.strip().lower()
    return {
        "id": record.id,
        "scenario": scenario.name,
        "wer": call_wer,
        "task_complete": task_complete,
        "fact_coverage": coverage,
        "hallucinations": hallucinations,
        "timings_ms": record.stage_timings.values() if record.stage_timings else {},
        "interruptions": len(record.interruptions),
    }


def evaluate(records: list[CallRecord]) -> VoiceEvalResult:
    """Aggregate record scores into a VoiceEvalResult."""
    if not records:
        raise ValueError("cannot evaluate an empty dataset")

    details = [score_record(r) for r in records]
    wers = [d["wer"] for d in details]
    completions = [d["task_complete"] for d in details]
    coverages = [d["fact_coverage"] for d in details]
    hallucinated = [bool(d["hallucinations"]) for d in details]

    e2e = [d["timings_ms"]["e2e_ms"] for d in details if "e2e_ms" in d["timings_ms"]]
    stage_names = ("stt_ms", "llm_ttft_ms", "tts_ttfa_ms", "e2e_ms")
    stage_means: dict[str, float] = {}
    for stage in stage_names:
        vals = [d["timings_ms"][stage] for d in details if stage in d["timings_ms"]]
        if vals:
            stage_means[stage] = sum(vals) / len(vals)

    stop_times = [
        i.agent_stopped_ms for r in records for i in r.interruptions if i.agent_stopped_ms is not None
    ]

    n = len(details)
    return VoiceEvalResult(
        samples=n,
        failures=0,
        mean_wer=sum(wers) / n,
        max_wer=max(wers),
        task_completion=sum(completions) / n,
        fact_coverage=sum(coverages) / n,
        hallucination_rate=sum(hallucinated) / n,
        e2e_p50_ms=percentile(e2e, 50) if e2e else None,
        e2e_p95_ms=percentile(e2e, 95) if e2e else None,
        e2e_p99_ms=percentile(e2e, 99) if e2e else None,
        stage_means_ms=stage_means,
        interruption_count=sum(d["interruptions"] for d in details),
        median_barge_in_stop_ms=percentile(stop_times, 50) if stop_times else None,
        details=details,
    )
