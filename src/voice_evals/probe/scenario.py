"""Scenario v2: scripted caller behavior with deterministic selection.

A ``ScenarioScript`` extends the replay ``Scenario`` with a bounded step list,
fact placeholders, cue rules (speak now / wait / interrupt), conditional
branches, and an outcome rule. Unknown fields are rejected so misspelled cues
fail loudly. A legacy scenario loads with no steps for replay use; probing it
raises an actionable validation error.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import Scenario
from .errors import ScenarioError
from .models import AgentObservation, TransportCapabilities

SCHEMA_VERSION = 2
MAX_STEPS = 100
_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


@dataclass(frozen=True)
class Cue:
    """When the caller speaks. One linear state cannot express barge-in, so
    ``interrupt`` deliberately overlaps caller sending with agent responding."""

    mode: str  # "speak_now" | "wait_agent_end" | "interrupt"
    response_to: str | None = None
    timeout_ms: int | None = None
    after_ms: int | None = None  # interrupt only: offset from response first audio
    if_missed: str | None = None  # interrupt only: "fail" | "skip"


@dataclass(frozen=True)
class Condition:
    """Deterministic branch predicate over one completed agent response."""

    response_to: str
    any_text: tuple[str, ...]


@dataclass(frozen=True)
class Utterance:
    text: str
    alternatives: tuple[str, ...] = ()
    reveals: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScriptStep:
    id: str
    intent: str  # caller metadata only; never sent on the audio channel
    cue: Cue
    utterance: Utterance
    when: Condition | None = None
    on_unmatched: str | None = None  # "fail" | "skip", required iff when is set


@dataclass(frozen=True)
class OutcomeRule:
    source: str  # "transport" | "final_agent_text"
    after_step: str | None = None
    all_of: tuple[str, ...] = ()
    value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.source}
        if self.after_step is not None:
            out["after_step"] = self.after_step
        if self.all_of:
            out["all_of"] = list(self.all_of)
        if self.value is not None:
            out["value"] = self.value
        return out


@dataclass(frozen=True)
class SelectedUtterance:
    step_id: str
    text: str
    alternative_index: int
    revealed_facts: tuple[str, ...]


def _render(text: str, facts: Mapping[str, str], step_id: str) -> tuple[str, tuple[str, ...]]:
    used: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in facts:
            raise ScenarioError(f"step {step_id!r}: placeholder {name!r} is not a declared fact")
        used.append(name)
        return facts[name]

    return _PLACEHOLDER.sub(_sub, text), tuple(dict.fromkeys(used))


class ScenarioScript(Scenario):
    """Replay Scenario plus a probeable caller script."""

    schema_version: int
    seed: int
    facts: dict[str, str]
    steps: tuple[ScriptStep, ...]
    outcome_rule: OutcomeRule | None

    def __init__(
        self,
        name: str,
        expected_outcome: str,
        required_facts: list[str] | None = None,
        forbidden_facts: list[str] | None = None,
        *,
        schema_version: int = SCHEMA_VERSION,
        seed: int = 0,
        facts: dict[str, str] | None = None,
        steps: tuple[ScriptStep, ...] = (),
        outcome_rule: OutcomeRule | None = None,
    ) -> None:
        super().__init__(
            name=name,
            expected_outcome=expected_outcome,
            required_facts=required_facts or [],
            forbidden_facts=forbidden_facts or [],
        )
        self.schema_version = schema_version
        self.seed = seed
        self.facts = dict(facts or {})
        self.steps = tuple(steps)
        self.outcome_rule = outcome_rule

    # -- loading ---------------------------------------------------------

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScenarioScript:
        try:
            name = str(value["name"])
            expected = str(value["expected_outcome"])
        except KeyError as error:
            raise ScenarioError(f"missing required field {error}") from error
        known = {
            "name",
            "expected_outcome",
            "required_facts",
            "forbidden_facts",
            "schema_version",
            "seed",
            "facts",
            "steps",
            "outcome_rule",
        }
        unknown = set(value) - known
        if unknown:
            raise ScenarioError(f"unknown scenario field(s): {sorted(unknown)}")
        if "steps" in value or "outcome_rule" in value or "schema_version" in value:
            version = value.get("schema_version")
            if version != SCHEMA_VERSION:
                raise ScenarioError(
                    f"schema_version must be {SCHEMA_VERSION}, got {version!r}"
                )
            steps = tuple(cls._parse_step(i, s) for i, s in enumerate(value.get("steps") or []))
            rule = cls._parse_outcome(value.get("outcome_rule"))
            script = cls(
                name,
                expected,
                [str(f) for f in value.get("required_facts", [])],
                [str(f) for f in value.get("forbidden_facts", [])],
                schema_version=SCHEMA_VERSION,
                seed=int(value.get("seed", 0)),
                facts={str(k): str(v) for k, v in (value.get("facts") or {}).items()},
                steps=steps,
                outcome_rule=rule,
            )
            script._validate_semantics()
            return script
        # Legacy replay scenario: no script; probing must fail with guidance.
        return cls(
            name,
            expected,
            [str(f) for f in value.get("required_facts", [])],
            [str(f) for f in value.get("forbidden_facts", [])],
            schema_version=0,
            seed=0,
            facts={},
            steps=(),
            outcome_rule=None,
        )

    @classmethod
    def load(cls, path: str | Path) -> ScenarioScript:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def _parse_step(index: int, raw: Any) -> ScriptStep:
        if not isinstance(raw, dict):
            raise ScenarioError(f"step {index} must be an object")
        known = {"id", "intent", "cue", "when", "on_unmatched", "utterance"}
        unknown = set(raw) - known
        if unknown:
            raise ScenarioError(f"step {index}: unknown field(s) {sorted(unknown)}")
        try:
            step_id = str(raw["id"])
            intent = str(raw["intent"])
            cue_raw = raw["cue"]
            utt_raw = raw["utterance"]
        except KeyError as error:
            raise ScenarioError(f"step {index}: missing field {error}") from error
        cue = ScenarioScript._parse_cue(index, step_id, cue_raw)
        when_raw = raw.get("when")
        on_unmatched = raw.get("on_unmatched")
        when: Condition | None = None
        if when_raw is not None:
            if not isinstance(when_raw, dict):
                raise ScenarioError(f"step {step_id!r}: 'when' must be an object")
            unknown = set(when_raw) - {"response_to", "any_text"}
            if unknown:
                raise ScenarioError(f"step {step_id!r}: unknown 'when' field(s) {sorted(unknown)}")
            try:
                any_text = when_raw["any_text"]
                when = Condition(
                    response_to=str(when_raw["response_to"]),
                    any_text=tuple(str(t) for t in any_text),
                )
            except KeyError as error:
                raise ScenarioError(f"step {step_id!r}: 'when' missing {error}") from error
            if not when.any_text:
                raise ScenarioError(f"step {step_id!r}: 'when.any_text' must be non-empty")
        else:
            if on_unmatched is not None:
                raise ScenarioError(
                    f"step {step_id!r}: 'on_unmatched' requires 'when' (dependentRequired)"
                )
        if when is not None and on_unmatched not in ("fail", "skip"):
            raise ScenarioError(f"step {step_id!r}: 'when' requires on_unmatched fail|skip")
        if not isinstance(utt_raw, dict):
            raise ScenarioError(f"step {step_id!r}: 'utterance' must be an object")
        unknown = set(utt_raw) - {"text", "alternatives", "reveals"}
        if unknown:
            raise ScenarioError(f"step {step_id!r}: unknown utterance field(s) {sorted(unknown)}")
        try:
            text = str(utt_raw["text"])
        except KeyError as error:
            raise ScenarioError(f"step {step_id!r}: utterance missing text") from error
        if not text:
            raise ScenarioError(f"step {step_id!r}: utterance text must be non-empty")
        utterance = Utterance(
            text=text,
            alternatives=tuple(str(a) for a in utt_raw.get("alternatives", [])),
            reveals=tuple(str(r) for r in utt_raw.get("reveals", [])),
        )
        return ScriptStep(id=step_id, intent=intent, cue=cue, utterance=utterance, when=when, on_unmatched=on_unmatched)

    @staticmethod
    def _parse_cue(index: int, step_id: str, raw: Any) -> Cue:
        if not isinstance(raw, dict) or "mode" not in raw:
            raise ScenarioError(f"step {step_id!r}: cue must be an object with a mode")
        mode = raw["mode"]
        allowed = {
            "speak_now": set(),
            "wait_agent_end": {"response_to", "timeout_ms"},
            "interrupt": {"response_to", "after_ms", "timeout_ms", "if_missed"},
        }
        if mode not in allowed:
            raise ScenarioError(f"step {step_id!r}: unknown cue mode {mode!r}")
        fields = set(raw) - {"mode"}
        expected = allowed[mode]
        if fields != expected:
            missing = expected - fields
            extra = fields - expected
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"unexpected {sorted(extra)}")
            raise ScenarioError(f"step {step_id!r}: cue {mode} fields: " + "; ".join(parts))
        if mode == "speak_now":
            return Cue(mode="speak_now")
        timeout_ms = int(raw["timeout_ms"])
        if timeout_ms < 1:
            raise ScenarioError(f"step {step_id!r}: cue timeout_ms must be >= 1")
        if mode == "wait_agent_end":
            return Cue(mode=mode, response_to=str(raw["response_to"]), timeout_ms=timeout_ms)
        after_ms = int(raw["after_ms"])
        if after_ms < 0:
            raise ScenarioError(f"step {step_id!r}: interrupt after_ms must be >= 0")
        if raw["if_missed"] not in ("fail", "skip"):
            raise ScenarioError(f"step {step_id!r}: interrupt if_missed must be fail|skip")
        return Cue(
            mode="interrupt",
            response_to=str(raw["response_to"]),
            after_ms=after_ms,
            timeout_ms=timeout_ms,
            if_missed=str(raw["if_missed"]),
        )

    @staticmethod
    def _parse_outcome(raw: Any) -> OutcomeRule:
        if not isinstance(raw, dict):
            raise ScenarioError("outcome_rule must be an object")
        source = raw.get("source")
        if source == "transport":
            if set(raw) != {"source"}:
                raise ScenarioError("outcome_rule source=transport takes no other fields")
            return OutcomeRule(source="transport")
        if source == "final_agent_text":
            required = {"source", "after_step", "all_of", "value"}
            if set(raw) != required:
                raise ScenarioError(
                    f"outcome_rule final_agent_text requires exactly {sorted(required)}"
                )
            return OutcomeRule(
                source="final_agent_text",
                after_step=str(raw["after_step"]),
                all_of=tuple(str(t) for t in raw["all_of"]),
                value=str(raw["value"]),
            )
        raise ScenarioError(f"outcome_rule source must be transport|final_agent_text, got {source!r}")

    # -- validation ------------------------------------------------------

    def _validate_semantics(self) -> None:
        if not self.steps:
            raise ScenarioError("v2 scenario requires at least one step")
        if len(self.steps) > MAX_STEPS:
            raise ScenarioError(f"scenario exceeds {MAX_STEPS} steps")
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ScenarioError(f"duplicate step id(s): {duplicates}")
        if self.steps[0].cue.mode != "speak_now":
            raise ScenarioError("the first step must cue speak_now")
        seen: set[str] = set()
        for step in self.steps:
            for ref in self._step_references(step):
                if ref not in seen:
                    raise ScenarioError(
                        f"step {step.id!r} references {ref!r}, which is not a prior step"
                    )
            if (
                step.when is not None
                and step.cue.response_to
                and step.when.response_to != step.cue.response_to
            ):
                raise ScenarioError(
                    f"step {step.id!r}: 'when.response_to' must match the cue's response_to"
                )
            substituted: set[str] = set()
            for text in (step.utterance.text, *step.utterance.alternatives):
                _, used = _render(text, self.facts, step.id)
                substituted.update(used)
            reveals = set(step.utterance.reveals)
            missing = substituted - reveals
            unused = reveals - substituted
            if missing:
                raise ScenarioError(
                    f"step {step.id!r}: substituted fact(s) {sorted(missing)} not declared in reveals"
                )
            if unused:
                raise ScenarioError(
                    f"step {step.id!r}: reveal declaration(s) {sorted(unused)} never substituted"
                )
            seen.add(step.id)
        if self.outcome_rule is None:
            raise ScenarioError("v2 scenario requires an outcome_rule")
        if self.outcome_rule.after_step is not None and self.outcome_rule.after_step not in seen:
            raise ScenarioError(
                f"outcome_rule.after_step {self.outcome_rule.after_step!r} does not exist"
            )

    @staticmethod
    def _step_references(step: ScriptStep) -> list[str]:
        refs = []
        if step.cue.response_to:
            refs.append(step.cue.response_to)
        if step.when is not None:
            refs.append(step.when.response_to)
        return refs

    def validate_for_probe(self, capabilities: TransportCapabilities) -> None:
        """Raise actionable errors when this script cannot run on a transport."""
        if not self.steps or self.outcome_rule is None:
            raise ScenarioError(
                f"scenario {self.name!r} has no probe script (legacy replay scenario); "
                "provide a schema_version=2 script with steps and outcome_rule"
            )
        for step in self.steps:
            if step.cue.mode == "interrupt":
                if not capabilities.duplex:
                    raise ScenarioError(
                        f"step {step.id!r} needs barge-in but the transport is not duplex"
                    )
                if not capabilities.incremental_output:
                    raise ScenarioError(
                        f"step {step.id!r} needs interruptible streaming output; "
                        "burst-completed audio cannot prove ongoing speech"
                    )
        if self.outcome_rule.source == "transport" and not capabilities.outcome_events:
            raise ScenarioError(
                "outcome_rule source=transport requires structured outcome events; "
                "the transport/protocol map does not declare them"
            )

    # -- selection -------------------------------------------------------

    def alternative_index(self, step_id: str) -> int:
        """Deterministic alternative index from a stable hash of seed+step.

        Independent of branch execution order: computing it for one step never
        depends on which other steps ran.
        """
        step = self.step_by_id(step_id)
        candidates = 1 + len(step.utterance.alternatives)
        digest = hashlib.sha256(f"{self.seed}:{step_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % candidates

    def select_utterance(self, step_id: str, observation: AgentObservation) -> SelectedUtterance | None:
        """Select and render one alternative, or None when a branch predicate fails."""
        step = self.step_by_id(step_id)
        if (
            step.when is not None
            and observation is not None
            and not observation.contains_any(list(step.when.any_text))
        ):
            return None
        index = self.alternative_index(step_id)
        candidates = (step.utterance.text, *step.utterance.alternatives)
        text, used = _render(candidates[index], self.facts, step_id)
        return SelectedUtterance(step_id=step_id, text=text, alternative_index=index, revealed_facts=used)

    def step_by_id(self, step_id: str) -> ScriptStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise ScenarioError(f"unknown step id {step_id!r}")

    # -- projection ------------------------------------------------------

    def to_scenario(self) -> Scenario:
        """Project the replay fields; extensions stay out of legacy rows."""
        return Scenario(
            name=self.name,
            expected_outcome=self.expected_outcome,
            required_facts=list(self.required_facts),
            forbidden_facts=list(self.forbidden_facts),
        )

    def script_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "expected_outcome": self.expected_outcome,
            "seed": self.seed,
            "steps": [
                {
                    "id": s.id,
                    "intent": s.intent,
                    "cue": {
                        key: value
                        for key, value in {
                            "mode": s.cue.mode,
                            "response_to": s.cue.response_to,
                            "timeout_ms": s.cue.timeout_ms,
                            "after_ms": s.cue.after_ms,
                            "if_missed": s.cue.if_missed,
                        }.items()
                        if value is not None
                    },
                    "utterance": {
                        "text": s.utterance.text,
                        "alternatives": list(s.utterance.alternatives),
                        "reveals": list(s.utterance.reveals),
                    },
                }
                for s in self.steps
            ],
            "outcome_rule": self.outcome_rule.to_dict() if self.outcome_rule else None,
        }
        if self.required_facts:
            out["required_facts"] = list(self.required_facts)
        if self.forbidden_facts:
            out["forbidden_facts"] = list(self.forbidden_facts)
        if self.facts:
            out["facts"] = dict(self.facts)
        return out
