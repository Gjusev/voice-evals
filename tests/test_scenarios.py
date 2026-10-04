"""Scenario v2 parsing, validation, selection, and branch semantics."""

from pathlib import Path

import pytest

from voice_evals.probe import (
    AgentObservation,
    ScenarioError,
    ScenarioScript,
    TransportCapabilities,
)
from voice_evals.probe.scenario import SCHEMA_VERSION

RESOURCES = Path(__file__).resolve().parents[1] / "src" / "voice_evals" / "resources"
APPOINTMENT = RESOURCES / "scenarios" / "appointment-v2.json"


def load_appointment() -> ScenarioScript:
    return ScenarioScript.load(APPOINTMENT)


def patch(raw: dict, **changes) -> dict:
    out = dict(raw)
    out.update(changes)
    return out


def test_bundled_example_loads() -> None:
    script = load_appointment()
    assert script.schema_version == SCHEMA_VERSION
    assert script.seed == 17
    assert len(script.steps) == 4
    assert script.steps[0].cue.mode == "speak_now"
    assert script.steps[2].cue.mode == "interrupt"
    assert script.outcome_rule is not None and script.outcome_rule.source == "final_agent_text"


def test_unknown_fields_rejected() -> None:
    raw = __import__("json").loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["stpes"] = raw.pop("steps")
    with pytest.raises(ScenarioError, match="unknown scenario field"):
        ScenarioScript.from_dict(raw)


def test_missing_schema_version_rejected() -> None:
    import json

    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    del raw["schema_version"]
    with pytest.raises(ScenarioError, match="schema_version"):
        ScenarioScript.from_dict(raw)


def test_reference_must_be_prior_step() -> None:
    import json

    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["steps"][1]["cue"]["response_to"] = "confirm"  # forward reference
    with pytest.raises(ScenarioError, match="not a prior step"):
        ScenarioScript.from_dict(raw)


def test_first_step_must_speak_now() -> None:
    import json

    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["steps"][0]["cue"] = {"mode": "wait_agent_end", "response_to": "request", "timeout_ms": 100}
    with pytest.raises(ScenarioError, match="first step"):
        ScenarioScript.from_dict(raw)


def test_unknown_placeholder_rejected() -> None:
    import json

    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["steps"][0]["utterance"]["text"] = "Book for {{nope}}."
    with pytest.raises(ScenarioError, match="not a declared fact"):
        ScenarioScript.from_dict(raw)


def test_reveals_must_cover_substituted_facts() -> None:
    import json

    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["steps"][0]["utterance"]["reveals"] = []
    with pytest.raises(ScenarioError, match="not declared in reveals"):
        ScenarioScript.from_dict(raw)


def test_unused_reveal_rejected() -> None:
    import json

    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["steps"][0]["utterance"]["reveals"] = ["preferred_day", "caller_name"]
    with pytest.raises(ScenarioError, match="never substituted"):
        ScenarioScript.from_dict(raw)


def test_when_requires_on_unmatched() -> None:
    import json

    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    del raw["steps"][1]["on_unmatched"]
    with pytest.raises(ScenarioError, match="on_unmatched"):
        ScenarioScript.from_dict(raw)


def test_duplicate_step_ids_rejected() -> None:
    import json

    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["steps"][1]["id"] = "request"
    with pytest.raises(ScenarioError, match="duplicate step id"):
        ScenarioScript.from_dict(raw)


def test_cue_field_mismatch_rejected() -> None:
    import json

    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["steps"][2]["cue"].pop("if_missed")
    with pytest.raises(ScenarioError, match="cue interrupt fields"):
        ScenarioScript.from_dict(raw)


def test_legacy_scenario_loads_without_steps() -> None:
    legacy = ScenarioScript.from_dict(
        {"name": "old", "expected_outcome": "booked", "required_facts": ["x"]}
    )
    assert legacy.steps == ()
    assert legacy.outcome_rule is None
    caps = TransportCapabilities(duplex=True, incremental_output=True)
    with pytest.raises(ScenarioError, match="no probe script"):
        legacy.validate_for_probe(caps)


def test_selection_deterministic_and_order_independent() -> None:
    script = load_appointment()
    first = script.alternative_index("request")
    again = load_appointment().alternative_index("request")
    assert first == again
    assert 0 <= first <= 1  # text + 1 alternative


def test_select_utterance_branch_predicate() -> None:
    script = load_appointment()
    step = script.step_by_id("provide_name")
    assert step.when is not None
    matched = script.select_utterance(
        "provide_name", AgentObservation(response_id="r1", final_text="May I have your name, please?")
    )
    assert matched is not None and "{{" not in matched.text
    unmatched = script.select_utterance(
        "provide_name", AgentObservation(response_id="r1", final_text="What time works?")
    )
    assert unmatched is None


def test_validate_for_probe_interrupt_requirements() -> None:
    script = load_appointment()
    no_duplex = TransportCapabilities(duplex=False, incremental_output=True)
    with pytest.raises(ScenarioError, match="not duplex"):
        script.validate_for_probe(no_duplex)
    burst = TransportCapabilities(duplex=True, incremental_output=False)
    with pytest.raises(ScenarioError, match="interruptible streaming"):
        script.validate_for_probe(burst)
    ok = TransportCapabilities(duplex=True, incremental_output=True)
    script.validate_for_probe(ok)


def test_validate_for_probe_transport_outcome_needs_events() -> None:
    import json

    raw = json.loads(APPOINTMENT.read_text(encoding="utf-8"))
    raw["outcome_rule"] = {"source": "transport"}
    script = ScenarioScript.from_dict(raw)
    caps = TransportCapabilities(duplex=True, incremental_output=True, outcome_events=False)
    with pytest.raises(ScenarioError, match="outcome_rule source=transport"):
        script.validate_for_probe(caps)


def test_to_scenario_projects_replay_fields_only() -> None:
    script = load_appointment()
    scenario = script.to_scenario()
    assert scenario.name == "appointment-with-correction"
    assert scenario.expected_outcome == "booked"
    assert scenario.required_facts == ["appointment", "Tuesday", "11:00", "Alex Morgan"]
    assert not hasattr(scenario, "steps")


def test_round_trip_script_dict() -> None:
    script = load_appointment()
    clone = ScenarioScript.from_dict(script.script_dict())
    assert clone.seed == script.seed
    assert [s.id for s in clone.steps] == [s.id for s in script.steps]


def test_human_facing_scenario_copy_matches_resource() -> None:
    """evals/scenarios/ copies are checked against the packaged resources."""
    copy_path = (
        Path(__file__).resolve().parents[1] / "evals" / "scenarios" / "appointment-v2.json"
    )
    assert copy_path.read_text(encoding="utf-8") == APPOINTMENT.read_text(encoding="utf-8")
