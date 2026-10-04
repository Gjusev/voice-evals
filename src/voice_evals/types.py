"""Typed schemas for voice-evals datasets and results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any


class DatasetError(ValueError):
    """Raised when a dataset row cannot be parsed."""


@dataclass
class Scenario:
    """What a call was supposed to achieve, and what it must never claim."""

    name: str
    expected_outcome: str
    required_facts: list[str] = field(default_factory=list)
    forbidden_facts: list[str] = field(default_factory=list)


@dataclass
class StageTimings:
    """Latency budget of one turn, in milliseconds.

    stt_ms      transcription time for the caller utterance
    llm_ttft_ms time to first LLM token
    tts_ttfa_ms time to first synthesised audio byte
    e2e_ms      caller stops speaking -> agent audio starts
    """

    stt_ms: float | None = None
    llm_ttft_ms: float | None = None
    tts_ttfa_ms: float | None = None
    e2e_ms: float | None = None

    def values(self) -> dict[str, float]:
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class Interruption:
    """One barge-in event: caller interrupted at at_ms."""

    at_ms: float
    agent_stopped_ms: float | None = None  # how long until agent audio stopped


@dataclass
class CallRecord:
    """One recorded call, replayed offline."""

    id: str
    scenario: Scenario
    asr_transcript: str  # what the agent's STT heard (caller side)
    ground_truth_transcript: str  # what the caller actually said
    agent_transcript: str  # what the agent said, full call
    outcome: str  # outcome the agent reported
    stage_timings: StageTimings | None = None
    interruptions: list[Interruption] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CallRecord":
        try:
            scenario = Scenario(
                name=str(row["scenario"]["name"]),
                expected_outcome=str(row["scenario"]["expected_outcome"]),
                required_facts=[str(f) for f in row["scenario"].get("required_facts", [])],
                forbidden_facts=[str(f) for f in row["scenario"].get("forbidden_facts", [])],
            )
            timings_raw = row.get("stage_timings_ms") or {}
            timings = (
                StageTimings(
                    stt_ms=_num(timings_raw.get("stt_ms")),
                    llm_ttft_ms=_num(timings_raw.get("llm_ttft_ms")),
                    tts_ttfa_ms=_num(timings_raw.get("tts_ttfa_ms")),
                    e2e_ms=_num(timings_raw.get("e2e_ms")),
                )
                if timings_raw
                else None
            )
            interruptions = [
                Interruption(
                    at_ms=float(i["at_ms"]),
                    agent_stopped_ms=_num(i.get("agent_stopped_ms")),
                )
                for i in row.get("interruptions", [])
            ]
            return cls(
                id=str(row["id"]),
                scenario=scenario,
                asr_transcript=str(row.get("asr_transcript", "")),
                ground_truth_transcript=str(row["ground_truth_transcript"]),
                agent_transcript=str(row.get("agent_transcript", "")),
                outcome=str(row.get("outcome", "")),
                stage_timings=timings,
                interruptions=interruptions,
            )
        except KeyError as error:
            raise DatasetError(f"missing field {error} in record {row.get('id', '?')!r}") from error
        except (TypeError, ValueError) as error:
            raise DatasetError(f"invalid record {row.get('id', '?')!r}: {error}") from error


def _num(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_dataset(path: str) -> list[CallRecord]:
    """Load a .jsonl (one record per line) or .json (array) dataset."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if path.endswith(".jsonl"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise DatasetError("JSON dataset must be an array of records")
    return [CallRecord.from_dict(row) for row in rows]


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile of a non-empty list."""
    if not values:
        raise ValueError("percentile of empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass
class VoiceEvalResult:
    """Aggregate scores over a replayed dataset."""

    samples: int
    failures: int
    mean_wer: float
    max_wer: float
    task_completion: float
    fact_coverage: float
    hallucination_rate: float
    e2e_p50_ms: float | None
    e2e_p95_ms: float | None
    e2e_p99_ms: float | None
    stage_means_ms: dict[str, float]
    interruption_count: int
    median_barge_in_stop_ms: float | None
    details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "failures": self.failures,
            "mean_wer": self.mean_wer,
            "max_wer": self.max_wer,
            "task_completion": self.task_completion,
            "fact_coverage": self.fact_coverage,
            "hallucination_rate": self.hallucination_rate,
            "e2e_p50_ms": self.e2e_p50_ms,
            "e2e_p95_ms": self.e2e_p95_ms,
            "e2e_p99_ms": self.e2e_p99_ms,
            "stage_means_ms": self.stage_means_ms,
            "interruption_count": self.interruption_count,
            "median_barge_in_stop_ms": self.median_barge_in_stop_ms,
            "details": self.details,
        }

    def summary(self) -> str:
        lines = [
            f"samples={self.samples} failures={self.failures}",
            f"wer mean={self.mean_wer:.4f} max={self.max_wer:.4f}",
            f"task_completion={self.task_completion:.4f}",
            f"fact_coverage={self.fact_coverage:.4f}",
            f"hallucination_rate={self.hallucination_rate:.4f}",
            _fmt_line("e2e_ms", [self.e2e_p50_ms, self.e2e_p95_ms, self.e2e_p99_ms], ("p50", "p95", "p99")),
        ]
        if self.stage_means_ms:
            stage = " ".join(f"{k}={v:.1f}" for k, v in sorted(self.stage_means_ms.items()))
            lines.append(f"stage means: {stage}")
        lines.append(
            f"interruptions={self.interruption_count} "
            f"median_barge_in_stop_ms="
            f"{'n/a' if self.median_barge_in_stop_ms is None else f'{self.median_barge_in_stop_ms:.1f}'}"
        )
        return "\n".join(lines)


def _fmt_line(label: str, values: list[float | None], names: tuple[str, ...]) -> str:
    parts = [f"{n}={'n/a' if v is None else f'{v:.1f}'}" for n, v in zip(names, values)]
    return f"{label} " + " ".join(parts)
