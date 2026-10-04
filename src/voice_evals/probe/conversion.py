"""Session eligibility and conversion to the canonical replay CallRecord.

Canonical scoring is strict because v0.1 requires numeric WER and fact scores
without an "unavailable" state: ground truth must cover exactly the sent
audio, caller ASR coverage must be complete, agent transcript coverage must be
complete (or confirmed silence), and the outcome observation window must be
finished. Ineligible sessions keep all artifacts but feed nothing into
``evaluate()``.
"""

from __future__ import annotations

from typing import Any

from ..types import CallRecord, Interruption, Scenario, StageTimings
from .models import EventKind, SessionRecord
from .timing import AGGREGATION_LABEL, ordinary_turn_means


def check_eligibility(session: SessionRecord) -> tuple[bool, list[str]]:
    """Strict canonical-scoring eligibility with explicit exclusion reasons."""
    reasons: list[str] = []
    sent = [t for t in session.turns if t.sent and not t.skipped]
    if not sent:
        reasons.append("no fully sent caller utterances")
    for turn in sent:
        if turn.caller_asr_text is None:
            reasons.append(f"turn {turn.step_id}: caller ASR incomplete")
        if turn.agent_final_text is None:
            reasons.append(f"turn {turn.step_id}: agent transcript incomplete (no confirmed silence either)")
    if session.outcome_rule_used is None:
        reasons.append("outcome observation incomplete")
    return (not reasons), reasons


def _dedupe_ordered(texts: list[str]) -> str:
    out: list[str] = []
    for text in texts:
        if not out or out[-1] != text:
            out.append(text)
    return " ".join(out)


def to_call_record(session: SessionRecord) -> CallRecord | None:
    """Convert an eligible session into the canonical replay record."""
    eligible, _ = check_eligibility(session)
    if not eligible:
        return None
    sent = [t for t in session.turns if t.sent and not t.skipped]
    scenario: Scenario = session.scenario.to_scenario()
    means, _counts = ordinary_turn_means(session.turns)
    stage = StageTimings(
        stt_ms=means.get("stt_ms"),
        llm_ttft_ms=means.get("llm_ttft_ms"),
        tts_ttfa_ms=means.get("tts_ttfa_ms"),
        e2e_ms=means.get("e2e_ms"),
    )
    interruptions = [
        Interruption(
            at_ms=t.interrupt.at_ns / 1e6,
            agent_stopped_ms=(t.interrupt.agent_stopped_ns / 1e6) if t.interrupt.agent_stopped_ns is not None else None,
        )
        for t in session.turns
        if t.interrupt is not None
    ]
    return CallRecord(
        id=session.session_id,
        scenario=scenario,
        asr_transcript=_dedupe_ordered([t.caller_asr_text or "" for t in sent]),
        ground_truth_transcript=" ".join(t.selected_text for t in sent),
        agent_transcript=_dedupe_ordered([t.agent_final_text or "" for t in sent]),
        outcome=session.outcome,
        stage_timings=stage if means else None,
        interruptions=interruptions,
    )


def evaluate_outcome(session: SessionRecord) -> tuple[str, str | None, dict[str, Any]]:
    """Resolve the session outcome per its rule. Never copies expected_outcome."""
    rule = session.scenario.outcome_rule
    if rule is None:
        return "", None, {"reason": "no outcome rule"}
    if rule.source == "transport":
        events = [e for e in session.events if e.kind == EventKind.OUTCOME]
        if not events:
            return "", rule.source, {"reason": "no structured outcome event observed"}
        value = str((events[-1].payload or {}).get("value") or "")
        return value, rule.source, {"source_event": "outcome", "observed": value or None}
    # final_agent_text
    target = next((t for t in session.turns if t.step_id == rule.after_step), None)
    if target is None or target.agent_final_text is None:
        return "", rule.source, {"reason": f"final agent text after step {rule.after_step!r} unavailable"}
    text = target.agent_final_text
    lowered = text.lower()
    found = {needle: needle.lower() in lowered for needle in rule.all_of}
    matched = all(found.values())
    evidence = {
        "after_step": rule.after_step,
        "all_of": found,
        "final_agent_text": text,
        "limitation": (
            "text confirmation is a proxy for what the agent reported; it is not "
            "evidence that a calendar write succeeded"
        ),
    }
    return (rule.value if matched else ""), rule.source, evidence


def stage_aggregation_note() -> dict[str, str]:
    return {"aggregation": AGGREGATION_LABEL}
