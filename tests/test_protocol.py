"""Protocol map validation, normalization, encoding, and schema parity."""

import base64
import json
from pathlib import Path

import pytest

from voice_evals.probe.audio import AudioFormat
from voice_evals.probe.errors import ProtocolError, ProtocolMapError
from voice_evals.probe.models import EventKind
from voice_evals.probe.scenario import ScenarioScript
from voice_evals.probe.transports.protocol import ProtocolMap, load_default_protocol_map

RESOURCES = Path(__file__).resolve().parents[1] / "src" / "voice_evals" / "resources"
FORMAT = AudioFormat(sample_rate=16000)


def load_default() -> ProtocolMap:
    return load_default_protocol_map()


def test_default_map_loads_and_declares_capabilities() -> None:
    pmap = load_default()
    assert pmap.input_format.sample_rate == 16000
    assert pmap.output_format.sample_rate == 16000
    caps = pmap.capabilities
    assert caps.duplex and caps.incremental_output and caps.response_ids
    assert caps.input_transcripts and caps.agent_transcripts and caps.response_end_events
    assert not caps.stage_events  # proxies only


def test_unknown_top_level_field_rejected() -> None:
    raw = json.loads((RESOURCES / "protocols" / "default-v1.json").read_text(encoding="utf-8"))
    raw["evil"] = True
    with pytest.raises(ProtocolMapError, match="unknown protocol map field"):
        ProtocolMap(raw)


def test_non_allowlisted_token_rejected() -> None:
    pmap = load_default()
    bad = {"type": "input_audio", "oops": "{{caller_facts}}"}
    with pytest.raises(ProtocolMapError, match="non-allowlisted token"):
        ProtocolMap._validate_template(pmap, bad)


def test_inline_token_substitution_in_auth_template() -> None:
    pmap = load_default()
    pairs = pmap.auth_header_values({"PROBE_AGENT_API_KEY": "s3cret"})
    assert pairs == [("Authorization", "Bearer s3cret")]


def test_unknown_substitution_value_is_error() -> None:
    pmap = load_default()
    with pytest.raises(ProtocolMapError, match="missing substitution"):
        pmap.render(pmap.outbound_envelope, {"utterance_id": "u1"})  # seq/audio missing


def test_normalize_full_event_flow() -> None:
    pmap = load_default()
    audio_b64 = base64.b64encode(b"\x01\x00" * 320).decode("ascii")
    messages = [
        {"type": "session_ready"},
        {"type": "response_start", "response_id": "r1"},
        {"type": "input_transcript_final", "utterance_id": "u1", "text": "hello"},
        {"type": "output_text_final", "response_id": "r1", "text": "hi there"},
        {"type": "output_audio", "response_id": "r1", "seq": 0, "audio": audio_b64},
        {"type": "response_done", "response_id": "r1"},
    ]
    kinds = []
    for i, message in enumerate(messages):
        event = pmap.normalize(json.dumps(message), receipt_ns=i * 1000, output_format=FORMAT)
        assert event is not None
        kinds.append(event.kind)
        if event.audio is not None:
            assert event.audio.samples == 320
            assert event.direction == "agent"
            assert event.audio.sequence == 0
    assert kinds == [
        EventKind.SESSION_READY,
        EventKind.RESPONSE_START,
        EventKind.CALLER_TRANSCRIPT_FINAL,
        EventKind.AGENT_TEXT_FINAL,
        EventKind.AGENT_AUDIO,
        EventKind.RESPONSE_DONE,
    ]


def test_unknown_event_ignored() -> None:
    pmap = load_default()
    event = pmap.normalize('{"type": "keepalive"}', receipt_ns=0, output_format=FORMAT)
    assert event is None


def test_invalid_base64_is_typed_failure() -> None:
    pmap = load_default()
    with pytest.raises(ProtocolError, match="base64"):
        pmap.normalize(
            '{"type": "output_audio", "audio": "!!!not-base64!!!"}',
            receipt_ns=0,
            output_format=FORMAT,
        )


def test_unsampled_audio_rejected() -> None:
    pmap = load_default()
    odd = base64.b64encode(b"\x01\x00\x02").decode("ascii")
    with pytest.raises(ProtocolError, match="sample-aligned"):
        pmap.normalize(
            json.dumps({"type": "output_audio", "audio": odd}),
            receipt_ns=0,
            output_format=FORMAT,
        )


def test_binary_frame_with_binary_map() -> None:
    raw = json.loads((RESOURCES / "protocols" / "default-v1.json").read_text(encoding="utf-8"))
    raw["outbound"]["audio"]["media"] = "binary"
    raw["inbound"]["events"]["output_audio"]["media"] = "binary"
    pmap = ProtocolMap(raw)
    event = pmap.normalize(b"\x00\x01" * 160, receipt_ns=5, output_format=FORMAT)
    assert event is not None and event.kind == EventKind.AGENT_AUDIO
    assert event.audio.samples == 160


def test_binary_frame_without_binary_map_rejected() -> None:
    pmap = load_default()
    with pytest.raises(ProtocolError, match="no binary"):
        pmap.normalize(b"\x00\x01" * 160, receipt_ns=5, output_format=FORMAT)


def test_encode_audio_frame_json() -> None:
    pmap = load_default()
    message = json.loads(pmap.encode_audio_frame("u1", 3, b"\xff\xfe"))
    assert message["type"] == "input_audio"
    assert message["utterance_id"] == "u1"
    assert message["seq"] == "3"
    assert base64.b64decode(message["audio"]) == b"\xff\xfe"


def test_control_templates() -> None:
    pmap = load_default()
    commit = json.loads(pmap.encode_control(pmap.end_utterance_template, {"utterance_id": "u9"}))
    assert commit == {"type": "input_commit", "utterance_id": "u9"}
    cancel = json.loads(pmap.encode_control(pmap.cancel_template, {"response_id": "r2"}))
    assert cancel == {"type": "cancel_response", "response_id": "r2"}


def test_auth_headers_resolve_env_values() -> None:
    pmap = load_default()
    pairs = pmap.auth_header_values({"PROBE_AGENT_API_KEY": "tok-123"})
    assert pairs == [("Authorization", "Bearer tok-123")]


def test_ambiguous_media_mapping_rejected() -> None:
    raw = json.loads((RESOURCES / "protocols" / "default-v1.json").read_text(encoding="utf-8"))
    raw["inbound"]["events"]["output_audio2"] = dict(raw["inbound"]["events"]["output_audio"])
    with pytest.raises(ProtocolMapError, match="ambiguous media"):
        ProtocolMap(raw)


def test_json_pointer_traversal() -> None:
    pmap = load_default()
    message = {"type": "outcome", "value": {"nested": "booked"}}
    raw = json.loads((RESOURCES / "protocols" / "default-v1.json").read_text(encoding="utf-8"))
    raw["inbound"]["events"]["outcome"]["selectors"]["value"] = "$.value.nested"
    pmap = ProtocolMap(raw)
    event = pmap.normalize(json.dumps(message), receipt_ns=0, output_format=FORMAT)
    assert event is not None and event.payload == {"value": "booked"}


def test_bundled_maps_agree_with_published_schemas() -> None:
    """jsonschema (dev dependency) checks the published contract itself."""
    jsonschema = pytest.importorskip("jsonschema")
    protocol_raw = json.loads((RESOURCES / "protocols" / "default-v1.json").read_text(encoding="utf-8"))
    protocol_schema = json.loads((RESOURCES / "schemas" / "protocol-v1.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(protocol_raw, protocol_schema)
    scenario_raw = json.loads((RESOURCES / "scenarios" / "appointment-v2.json").read_text(encoding="utf-8"))
    scenario_schema = json.loads((RESOURCES / "schemas" / "scenario-v2.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(scenario_raw, scenario_schema)


def test_schema_rejects_invalid_scenario() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    schema = json.loads((RESOURCES / "schemas" / "scenario-v2.schema.json").read_text(encoding="utf-8"))
    bad = {"schema_version": 1, "name": "x", "expected_outcome": "y", "steps": [], "outcome_rule": {}}
    with pytest.raises(ValidationError):
        jsonschema.validate(bad, schema)


def test_parser_and_schema_agree_on_examples() -> None:
    """Every schema-valid example must also parse with the runtime parser."""
    import copy

    jsonschema = pytest.importorskip("jsonschema")

    schema = json.loads((RESOURCES / "schemas" / "scenario-v2.schema.json").read_text(encoding="utf-8"))
    base = json.loads((RESOURCES / "scenarios" / "appointment-v2.json").read_text(encoding="utf-8"))
    variants = [base]
    skip_variant = copy.deepcopy(base)
    skip_variant["steps"][1]["on_unmatched"] = "skip"
    variants.append(skip_variant)
    minimal = {
        "schema_version": 2,
        "name": "one-shot",
        "expected_outcome": "done",
        "steps": [{"id": "a", "intent": "ask", "cue": {"mode": "speak_now"}, "utterance": {"text": "hi"}}],
        "outcome_rule": {"source": "transport"},
    }
    variants.append(minimal)
    for variant in variants:
        jsonschema.validate(variant, schema)
        ScenarioScript.from_dict(variant)  # must not raise


def test_map_declared_frame_ms_is_exposed() -> None:
    """audio.input.frame_ms is not dead config: the CLI uses it as the pacing
    default unless --frame-ms overrides it."""
    pmap = load_default()
    assert pmap.input_frame_ms == 20
    raw = json.loads((RESOURCES / "protocols" / "default-v1.json").read_text(encoding="utf-8"))
    raw["audio"]["input"]["frame_ms"] = 60
    assert ProtocolMap(raw).input_frame_ms == 60
