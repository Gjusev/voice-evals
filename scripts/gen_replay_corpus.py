"""Generate the frozen v0.2 synthetic replay corpus.

Deterministic given SEED. This is harness regression data — substitutions,
deletions, insertions, empty hypotheses, WER>1, failed outcomes, fact
omissions, forbidden phrases, latency tails, absent optional stage timings,
and measured/unmeasured interruption stops. It is NOT evidence of any live
model's quality.

Run (from the repo root, after changes to templates only):
    uv run python scripts/gen_replay_corpus.py

Outputs:
    evals/data/replay_v02.jsonl          the frozen corpus
    evals/data/replay_v02.expected.json  expected metrics + hand-checked cases

The three HAND_CHECKED calls carry manually computed WERs so expected outputs
are not merely produced by the implementation under test.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_evals.evaluate import evaluate
from voice_evals.types import load_dataset

SEED = 20260211
OUT = Path(__file__).resolve().parents[1] / "evals" / "data" / "replay_v02.jsonl"
EXPECTED = OUT.with_suffix(".expected.json")

APPOINTMENT_FACTS = ["Tuesday", "appointment", "11:00", "Alex Morgan"]
SUPPORT_TOPICS = [
    ("reschedule-delivery", "rescheduled", ["Thursday", "delivery"], ["refund"]),
    ("cancel-order", "cancelled", ["cancelled", "confirmation email"], ["free upgrade"]),
    ("support-ticket", "resolved", ["ticket", "escalated"], ["compensation"]),
    ("change-address", "updated", ["address", "updated"], ["voucher"]),
]

# Hand-checked WER cases (computed by hand, asserted by tests and Kernel A):
#   "book a table for four" vs "book a table for or": 1 substitution / 5 words = 0.2
#   "cancel my order" vs "cancel my order please now": 2 insertions / 3 words ~ 0.6667
#   "yes" vs "no yes maybe": delete "no" + insert "maybe" = 2 edits / 1 word = 2.0 (> 1)
HAND_CHECKED = [
    {
        "id": "call-hand-001",
        "scenario": {"name": "book-appointment", "expected_outcome": "booked", "required_facts": ["Tuesday", "appointment"], "forbidden_facts": ["discount"]},
        "asr_transcript": "book a table for or",
        "ground_truth_transcript": "book a table for four",
        "agent_transcript": "You are booked for Tuesday. The appointment is confirmed.",
        "outcome": "booked",
        "stage_timings_ms": {"stt_ms": 200, "llm_ttft_ms": 310, "tts_ttfa_ms": 220, "e2e_ms": 800},
        "hand_wer": 0.2,
    },
    {
        "id": "call-hand-002",
        "scenario": {"name": "cancel-order", "expected_outcome": "cancelled", "required_facts": ["cancelled", "confirmation email"], "forbidden_facts": ["free upgrade"]},
        "asr_transcript": "cancel my order please now",
        "ground_truth_transcript": "cancel my order",
        "agent_transcript": "Your order has been cancelled and a confirmation email was sent.",
        "outcome": "cancelled",
        "stage_timings_ms": {"stt_ms": 180, "e2e_ms": 700},
        "hand_wer": 2 / 3,
    },
    {
        "id": "call-hand-003",
        "scenario": {"name": "confirm-yes", "expected_outcome": "confirmed", "required_facts": ["confirmed"], "forbidden_facts": []},
        "asr_transcript": "no yes maybe",
        "ground_truth_transcript": "yes",
        "agent_transcript": "Confirmed.",
        "outcome": "confirmed",
        "interruptions": [{"at_ms": 3000, "agent_stopped_ms": None}],
        "hand_wer": 2.0,
    },
]


def _edits(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    mode = rng.random()
    if mode < 0.30:  # substitution
        i = rng.randrange(len(words))
        words[i] = {"for": "or", "four": "for", "Tuesday": "Thursday", "eleven": "ten", "order": "older", "email": "emails", "cancelled": "canceled"}.get(words[i], words[i] + "x")
    elif mode < 0.50:  # deletion
        words.pop(rng.randrange(len(words)))
    elif mode < 0.70:  # insertion(s)
        for _ in range(rng.randint(1, 2)):
            words.insert(rng.randrange(len(words) + 1), rng.choice(["um", "please", "yes", "now"]))
    elif mode < 0.78:  # heavy insertions -> WER > 1 territory on short truths
        words = words[:1] + ["maybe", "so", "like"] * 2 + words[:0]
    # else: clean copy
    return " ".join(words)


def _timings(rng: random.Random, tail: bool) -> dict:
    base = {"stt_ms": rng.randint(140, 320), "llm_ttft_ms": rng.randint(260, 460), "tts_ttfa_ms": rng.randint(180, 320), "e2e_ms": rng.randint(680, 1150)}
    roll = rng.random()
    if roll < 0.15:
        return {}  # no timings at all
    if roll < 0.35:
        return {"e2e_ms": base["e2e_ms"]}  # only e2e: optional stages absent
    if roll < 0.50:
        return {"stt_ms": base["stt_ms"], "e2e_ms": base["e2e_ms"]}
    if tail:
        base["e2e_ms"] = rng.randint(2200, 4200)  # latency tail
    return base


def _interruptions(rng: random.Random) -> list[dict]:
    roll = rng.random()
    if roll < 0.30:
        return []
    if roll < 0.65:
        return [{"at_ms": float(rng.randint(2000, 9000)), "agent_stopped_ms": float(rng.randint(120, 480))}]
    if roll < 0.85:
        return [{"at_ms": float(rng.randint(2000, 9000)), "agent_stopped_ms": None}]  # unmeasured stop
    return [
        {"at_ms": float(rng.randint(2000, 5000)), "agent_stopped_ms": float(rng.randint(120, 400))},
        {"at_ms": float(rng.randint(6000, 9000)), "agent_stopped_ms": None},
    ]


def appointment_call(rng: random.Random, i: int) -> dict:
    truth = rng.choice(
        [
            "I'd like to book an appointment for Tuesday.",
            "Can I book an appointment for Tuesday at eleven?",
            "Please book an appointment for Alex Morgan on Tuesday.",
            "I want to book a table for four on Tuesday evening.",
        ]
    )
    fail = rng.random() < 0.25
    forbidden = rng.random() < 0.20
    agent = (
        "The appointment is booked for Tuesday at 11:00 for Alex Morgan."
        if not fail
        else "I could not complete the booking; the slot is unavailable."
    )
    if forbidden:
        agent += " As an apology, here is a free upgrade."
    omit = rng.random() < 0.25
    if omit:
        agent = agent.replace(" for Alex Morgan", "").replace("Alex Morgan ", "")
    empty_asr = rng.random() < 0.08
    return {
        "id": f"call-appt-{i:03d}",
        "scenario": {
            "name": "book-appointment",
            "expected_outcome": "booked",
            "required_facts": APPOINTMENT_FACTS,
            "forbidden_facts": ["free upgrade", "discount"],
        },
        "asr_transcript": "" if empty_asr else _edits(truth, rng),
        "ground_truth_transcript": truth,
        "agent_transcript": agent,
        "outcome": "" if fail else "booked",
        "stage_timings_ms": _timings(rng, tail=rng.random() < 0.15),
        "interruptions": _interruptions(rng),
    }


def support_call(rng: random.Random, i: int) -> dict:
    name, outcome, required, forbidden = rng.choice(SUPPORT_TOPICS)
    truth = {
        "reschedule-delivery": "Please reschedule my delivery to Thursday.",
        "cancel-order": "Yes, please cancel my order.",
        "support-ticket": "My ticket needs to be escalated today.",
        "change-address": "I need my address updated on the account.",
    }[name]
    fail = rng.random() < 0.3
    agent = {
        "reschedule-delivery": "Your delivery has been rescheduled to Thursday.",
        "cancel-order": "Your order has been cancelled and you will receive a confirmation email.",
        "support-ticket": "Your ticket has been escalated to a specialist.",
        "change-address": "Your address has been updated on the account.",
    }[name]
    if fail:
        agent = "I'm afraid I cannot process that request right now."
    if rng.random() < 0.18:
        agent += f" Here is a {forbidden[0]} for the trouble."
    return {
        "id": f"call-supp-{i:03d}",
        "scenario": {"name": name, "expected_outcome": outcome, "required_facts": required, "forbidden_facts": forbidden},
        "asr_transcript": _edits(truth, rng),
        "ground_truth_transcript": truth,
        "agent_transcript": agent,
        "outcome": "" if fail else outcome,
        "stage_timings_ms": _timings(rng, tail=rng.random() < 0.12),
        "interruptions": _interruptions(rng),
    }


def fault_call(rng: random.Random, i: int) -> dict:
    kind = rng.choice(["empty-asr", "wer-above-one", "no-timings", "timeout-ish", "all-missing-optional"])
    truth = rng.choice(["Yes.", "No.", "Confirm.", "Cancel it."])
    asr = {
        "empty-asr": "",
        "wer-above-one": "well so like " + rng.choice(["yes", "no"]) + " maybe",
        "no-timings": _edits(truth, rng),
        "timeout-ish": _edits(truth, rng),
        "all-missing-optional": _edits(truth, rng),
    }[kind]
    row = {
        "id": f"call-edge-{i:03d}",
        "scenario": {
            "name": "edge-case",
            "expected_outcome": "handled",
            "required_facts": ["handled"],
            "forbidden_facts": ["guarantee"],
        },
        "asr_transcript": asr,
        "ground_truth_transcript": truth,
        "agent_transcript": "The request was handled." if rng.random() < 0.7 else "",
        "outcome": "handled" if rng.random() < 0.7 else "",
    }
    if kind == "no-timings":
        row["stage_timings_ms"] = {}
    elif kind == "timeout-ish":
        row["stage_timings_ms"] = {"stt_ms": 900, "llm_ttft_ms": 1400, "tts_ttfa_ms": 700, "e2e_ms": 3600}
    elif kind == "all-missing-optional":
        row["stage_timings_ms"] = {"e2e_ms": 1500}
    row["interruptions"] = _interruptions(rng)
    return row


def main() -> None:
    rng = random.Random(SEED)
    rows = list(HAND_CHECKED)
    for i in range(1, 41):
        rows.append(appointment_call(rng, i))
    for i in range(1, 41):
        rows.append(support_call(rng, i))
    for i in range(1, 41):
        rows.append(fault_call(rng, i))
    assert len(rows) == 3 + 120
    OUT.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")

    result = evaluate(load_dataset(str(OUT))).to_dict()
    hand = [
        {"id": row["id"], "hand_computed_wer": row["hand_wer"]}
        for row in HAND_CHECKED
    ]
    expected = {
        "dataset": OUT.name,
        "seed": SEED,
        "samples": result["samples"],
        "provenance": (
            "synthetic harness regression corpus, deterministically generated by "
            "scripts/gen_replay_corpus.py; not evidence of live model quality"
        ),
        "hand_checked": {
            "note": "WERs computed by hand from word-level edit distances; independent of the implementation under test",
            "cases": hand,
        },
        "metrics": {k: v for k, v in result.items() if k != "details"},
    }
    EXPECTED.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {OUT}")
    print(json.dumps(expected["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
