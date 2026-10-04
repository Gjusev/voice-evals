"""Declarative protocol map: validation, normalization, frame encoding.

A WebSocket URL does not specify an audio protocol. A map declares the wire
contract: audio formats, handshake, outbound envelopes with a fixed
substitution allowlist, an inbound event discriminator with JSON-pointer
selectors, declared capabilities, and env-referenced authentication. No
Python, Jinja, JSONPath, or scripts are ever evaluated from a map.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Any

from ..audio import AudioFormat
from ..errors import ProtocolError, ProtocolMapError
from ..models import (
    AgentEvent,
    AudioPayload,
    EventKind,
    TranscriptSemantics,
    TransportCapabilities,
)

PROTOCOL_SCHEMA_VERSION = 1

ALLOWED_TOKENS = {"session_id", "utterance_id", "seq", "audio_b64", "response_id", "env_value"}
AUDIO_KINDS = {"agent_audio"}
MEDIA_JSON_BASE64 = "json_base64"
MEDIA_BINARY = "binary"

_EVENT_KIND_ALIASES = {kind.value: kind for kind in EventKind}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolMapError(message)


def _pointer_get(message: dict[str, Any], pointer: str) -> Any:
    """JSON Pointer traversal ('$.a.b' or '/a/b'); missing targets are None."""
    if pointer.startswith("$."):
        parts = [p for p in pointer[2:].split(".") if p]
    else:
        parts = [p for p in pointer.lstrip("/").split("/") if p]
    node: Any = message
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


class EventMapping:
    """One inbound event mapping: kind plus field selectors."""

    def __init__(self, event_type: str, spec: dict[str, Any]) -> None:
        unknown = set(spec) - {"kind", "direction", "media", "selectors"}
        _require(not unknown, f"inbound event {event_type!r}: unknown field(s) {sorted(unknown)}")
        kind_name = str(spec.get("kind", ""))
        _require(
            kind_name in _EVENT_KIND_ALIASES,
            f"inbound event {event_type!r}: unknown kind {kind_name!r}",
        )
        self.kind = _EVENT_KIND_ALIASES[kind_name]
        self.direction = spec.get("direction") or self._default_direction(self.kind)
        self.media = spec.get("media") or MEDIA_JSON_BASE64
        _require(
            self.media in (MEDIA_JSON_BASE64, MEDIA_BINARY),
            f"inbound event {event_type!r}: media must be json_base64 or binary",
        )
        selectors = spec.get("selectors") or {}
        _require(isinstance(selectors, dict), f"inbound event {event_type!r}: selectors must be an object")
        allowed = {"data", "text", "response_id", "utterance_id", "sequence", "value", "stage", "code", "message"}
        unknown = set(selectors) - allowed
        _require(not unknown, f"inbound event {event_type!r}: unknown selector(s) {sorted(unknown)}")
        for key, ptr in selectors.items():
            _require(isinstance(ptr, str) and ptr[:1] in ("$", "/"), f"selector {key!r} must be a JSON pointer")
        self.selectors: dict[str, str] = selectors

    @staticmethod
    def _default_direction(kind: EventKind) -> str:
        if kind in (EventKind.CALLER_TRANSCRIPT_DELTA, EventKind.CALLER_TRANSCRIPT_FINAL):
            return "caller"
        if kind in (EventKind.SESSION_READY, EventKind.ERROR, EventKind.STAGE, EventKind.RESPONSE_START, EventKind.RESPONSE_DONE):
            return "system"
        return "agent"

    def extract(self, message: dict[str, Any]) -> dict[str, Any]:
        return {key: _pointer_get(message, ptr) for key, ptr in self.selectors.items()}


class ProtocolMap:
    """Parsed, validated protocol map."""

    def __init__(self, raw: dict[str, Any]) -> None:
        _require(isinstance(raw, dict), "protocol map must be a JSON object")
        _require(
            raw.get("protocol_version") == PROTOCOL_SCHEMA_VERSION,
            f"protocol_version must be {PROTOCOL_SCHEMA_VERSION}",
        )
        self.name = str(raw.get("name", "unnamed"))
        known = {"protocol_version", "name", "description", "audio", "handshake", "outbound", "inbound", "auth", "capabilities"}
        unknown = set(raw) - known
        _require(not unknown, f"unknown protocol map field(s): {sorted(unknown)}")

        audio = raw.get("audio") or {}
        self.input_format, self.input_delivery, self.input_frame_ms = self._parse_audio(
            audio.get("input"), "audio.input"
        )
        self.output_format, self.output_delivery, _ = self._parse_audio(
            audio.get("output"), "audio.output"
        )

        handshake = raw.get("handshake")
        self.handshake_send: dict[str, Any] | None = None
        self.handshake_ready_type: str | None = None
        if handshake is not None:
            _require(isinstance(handshake, dict), "handshake must be an object")
            unknown = set(handshake) - {"send", "ready_event"}
            _require(not unknown, f"handshake unknown field(s): {sorted(unknown)}")
            send = handshake.get("send")
            _require(send is None or isinstance(send, dict), "handshake.send must be an object")
            self.handshake_send = send
            ready = handshake.get("ready_event")
            if ready is not None:
                _require(isinstance(ready, dict), "handshake.ready_event must be an object")
                self.handshake_ready_type = str(ready.get("type", ""))
                _require(bool(self.handshake_ready_type), "handshake.ready_event needs a type")

        outbound = raw.get("outbound") or {}
        unknown = set(outbound) - {"audio", "end_utterance", "cancel"}
        _require(not unknown, f"outbound unknown field(s): {sorted(unknown)}")
        audio_out = outbound.get("audio") or {}
        unknown = set(audio_out) - {"media", "envelope"}
        _require(not unknown, f"outbound.audio unknown field(s): {sorted(unknown)}")
        self.outbound_media = str(audio_out.get("media", MEDIA_JSON_BASE64))
        _require(
            self.outbound_media in (MEDIA_JSON_BASE64, MEDIA_BINARY),
            "outbound.audio.media must be json_base64 or binary",
        )
        self.outbound_envelope = audio_out.get("envelope")
        if self.outbound_media == MEDIA_JSON_BASE64:
            _require(isinstance(self.outbound_envelope, dict), "outbound.audio.envelope must be an object")
            self._validate_template(self.outbound_envelope)
        self.end_utterance_template = self._optional_template(outbound, "end_utterance")
        self.cancel_template = self._optional_template(outbound, "cancel")

        inbound = raw.get("inbound") or {}
        unknown = set(inbound) - {"discriminator", "events"}
        _require(not unknown, f"inbound unknown field(s): {sorted(unknown)}")
        self.discriminator = str(inbound.get("discriminator", "type"))
        events = inbound.get("events") or {}
        _require(isinstance(events, dict) and events, "inbound.events must be a non-empty object")
        self.events: dict[str, EventMapping] = {t: EventMapping(t, s) for t, s in events.items()}
        audio_events = [m for m in self.events.values() if m.kind == EventKind.AGENT_AUDIO]
        _require(len(audio_events) <= 1, "ambiguous media mapping: at most one agent_audio event")

        auth = raw.get("auth")
        self.auth_headers: list[dict[str, str]] = []
        if auth is not None:
            _require(isinstance(auth, dict), "auth must be an object")
            unknown = set(auth) - {"headers"}
            _require(not unknown, f"auth unknown field(s): {sorted(unknown)}")
            for header in auth.get("headers", []):
                _require(isinstance(header, dict), "auth.headers entries must be objects")
                _require(
                    set(header) == {"name", "env", "template"},
                    "auth header needs exactly name, env, template",
                )
                self.auth_headers.append(
                    {"name": str(header["name"]), "env": str(header["env"]), "template": str(header["template"])}
                )

        self.capabilities = self._parse_capabilities(raw.get("capabilities") or {})

    # -- parsing helpers ---------------------------------------------------

    @staticmethod
    def _parse_audio(spec: Any, where: str) -> tuple[AudioFormat, str, int | None]:
        _require(isinstance(spec, dict), f"{where} must be an object")
        unknown = set(spec) - {"encoding", "sample_rate", "channels", "sample_width", "frame_ms", "delivery"}
        _require(not unknown, f"{where}: unknown field(s) {sorted(unknown)}")
        fmt = AudioFormat(
            sample_rate=int(spec.get("sample_rate", 16000)),
            channels=int(spec.get("channels", 1)),
            sample_width=int(spec.get("sample_width", 2)),
            encoding=str(spec.get("encoding", "pcm_s16le")),
        )
        delivery = str(spec.get("delivery", "streaming"))
        _require(delivery in ("streaming", "burst"), f"{where}.delivery must be streaming|burst")
        frame_ms_raw = spec.get("frame_ms")
        frame_ms = int(frame_ms_raw) if frame_ms_raw is not None else None
        return fmt, delivery, frame_ms

    def _optional_template(self, outbound: dict[str, Any], key: str) -> dict[str, Any] | None:
        template = outbound.get(key)
        if template is None:
            return None
        _require(isinstance(template, dict), f"outbound.{key} must be an object")
        self._validate_template(template)
        return template

    def _validate_template(self, node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                self._validate_template(value)
        elif isinstance(node, list):
            for value in node:
                self._validate_template(value)
        elif isinstance(node, str):
            tokens = _tokens_in(node)
            if len(tokens) > 1:
                raise ProtocolMapError(f"template value {node!r} combines multiple substitution tokens")
            if tokens and tokens[0] not in ALLOWED_TOKENS:
                raise ProtocolMapError(
                    f"template value {node!r} uses non-allowlisted token {tokens[0]!r}; "
                    f"allowed: {sorted(ALLOWED_TOKENS)}"
                )

    def _parse_capabilities(self, spec: dict[str, Any]) -> TransportCapabilities:
        unknown = set(spec) - {
            "duplex", "incremental_output", "response_ids", "input_transcripts",
            "agent_transcripts", "response_end_events", "stage_events",
            "explicit_input_commit", "cancellation", "playback_ack",
            "outcome_events", "transcript_semantics",
        }
        _require(not unknown, f"capabilities unknown field(s): {sorted(unknown)}")
        semantics = TranscriptSemantics(str(spec.get("transcript_semantics", "delta")))
        return TransportCapabilities(
            input_format=self.input_format,
            output_format=self.output_format,
            duplex=bool(spec.get("duplex", False)),
            incremental_output=self.output_delivery == "streaming" and bool(spec.get("incremental_output", True)),
            response_ids=bool(spec.get("response_ids", False)),
            input_transcripts=bool(spec.get("input_transcripts", False)),
            agent_transcripts=bool(spec.get("agent_transcripts", False)),
            response_end_events=bool(spec.get("response_end_events", False)),
            stage_events=bool(spec.get("stage_events", False)),
            explicit_input_commit=bool(spec.get("explicit_input_commit", False)),
            cancellation=bool(spec.get("cancellation", False)),
            playback_ack=bool(spec.get("playback_ack", False)),
            outcome_events=bool(spec.get("outcome_events", False)),
            transcript_semantics=semantics,
            declared=dict(spec),
        )

    # -- rendering ---------------------------------------------------------

    def render(self, template: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
        """Deep-render a template using only allowlisted token substitutions."""

        def walk(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: walk(v) for k, v in node.items()}
            if isinstance(node, list):
                return [walk(v) for v in node]
            if isinstance(node, str):
                tokens = _tokens_in(node)
                if not tokens:
                    return node
                out = node
                for token in dict.fromkeys(tokens):
                    if token not in values:
                        raise ProtocolMapError(f"missing substitution value for {token!r}")
                    out = out.replace("{{" + token + "}}", values[token])
                return out
            return node

        return walk(template)

    def auth_header_values(self, resolved_env: dict[str, str]) -> list[tuple[str, str]]:
        pairs = []
        for header in self.auth_headers:
            value = resolved_env.get(header["env"], "")
            rendered = self.render({"v": header["template"]}, {"env_value": value})["v"]
            pairs.append((header["name"], rendered))
        return pairs

    # -- inbound normalization ----------------------------------------------

    def normalize(
        self,
        raw: str | bytes,
        *,
        receipt_ns: int,
        output_format: AudioFormat,
    ) -> AgentEvent | None:
        """Normalize one inbound application message; None = ignored unknown.

        Raises ProtocolError on malformed mapped events (invalid base64,
        sample misalignment, oversized frames, ambiguous mapping).
        """
        if isinstance(raw, bytes):
            return self._normalize_binary(raw, receipt_ns=receipt_ns, output_format=output_format)
        try:
            message = json.loads(raw)
        except ValueError as error:
            raise ProtocolError(f"inbound text message is not JSON: {error}") from error
        if not isinstance(message, dict):
            raise ProtocolError("inbound JSON message must be an object")
        event_type = message.get(self.discriminator)
        mapping = self.events.get(str(event_type)) if event_type is not None else None
        if mapping is None:
            return None  # unknown nonessential events are counted and ignored
        fields = mapping.extract(message)
        event = AgentEvent(
            kind=mapping.kind,
            receipt_ns=receipt_ns,
            raw_type=str(event_type),
            response_id=_opt_str(fields.get("response_id")),
            utterance_id=_opt_str(fields.get("utterance_id")),
            direction=mapping.direction,
        )
        if mapping.kind == EventKind.AGENT_AUDIO:
            data_b64 = fields.get("data")
            _require_str(data_b64, f"{event_type}: audio data selector missing")
            try:
                payload = base64.b64decode(str(data_b64), validate=True)
            except (binascii.Error, ValueError) as error:
                raise ProtocolError(f"{event_type}: invalid base64 audio: {error}") from error
            event.audio = self._audio_payload(payload, fields, output_format)
        elif mapping.kind in (EventKind.AGENT_TEXT_DELTA, EventKind.AGENT_TEXT_FINAL,
                              EventKind.CALLER_TRANSCRIPT_DELTA, EventKind.CALLER_TRANSCRIPT_FINAL):
            _require_str(fields.get("text"), f"{event_type}: text selector missing")
            event.text = str(fields.get("text"))
        elif mapping.kind == EventKind.OUTCOME:
            event.payload = {"value": fields.get("value")}
        elif mapping.kind == EventKind.STAGE:
            _require_str(fields.get("stage"), f"{event_type}: stage selector missing")
            event.stage = str(fields.get("stage"))
            event.response_id = _opt_str(fields.get("response_id"))
        elif mapping.kind == EventKind.ERROR:
            event.payload = {"code": fields.get("code"), "message": fields.get("message")}
        return event

    def _normalize_binary(
        self, raw: bytes, *, receipt_ns: int, output_format: AudioFormat
    ) -> AgentEvent | None:
        audio_maps = [m for m in self.events.values() if m.kind == EventKind.AGENT_AUDIO and m.media == MEDIA_BINARY]
        if not audio_maps:
            raise ProtocolError("binary frame received but the map declares no binary agent audio")
        payload = self._audio_payload(raw, {}, output_format)
        return AgentEvent(
            kind=EventKind.AGENT_AUDIO,
            receipt_ns=receipt_ns,
            raw_type="binary",
            direction="agent",
            audio=payload,
        )

    def _audio_payload(
        self, payload: bytes, fields: dict[str, Any], output_format: AudioFormat
    ) -> AudioPayload:
        if not payload or len(payload) % output_format.bytes_per_sample() != 0:
            raise ProtocolError("agent audio payload is empty or not sample-aligned")
        seq = fields.get("sequence")
        return AudioPayload(
            data=payload,
            samples=output_format.sample_count(payload),
            audio_format=output_format,
            sequence=int(seq) if isinstance(seq, (int, float)) else None,
        )

    # -- outbound encoding ---------------------------------------------------

    def encode_audio_frame(self, utterance_id: str, seq: int, pcm: bytes) -> str | bytes:
        """Encode one outbound audio frame per the map (JSON+base64 or binary)."""
        if self.outbound_media == MEDIA_BINARY:
            return pcm
        rendered = self.render(
            self.outbound_envelope,
            {
                "utterance_id": utterance_id,
                "seq": str(seq),
                "audio_b64": base64.b64encode(pcm).decode("ascii"),
            },
        )
        return json.dumps(rendered, separators=(",", ":"))

    def encode_control(self, template: dict[str, Any], values: dict[str, str]) -> str:
        return json.dumps(self.render(template, values), separators=(",", ":"))


def _tokens_in(value: str) -> list[str]:
    return re.findall(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", value)


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _require_str(value: Any, message: str) -> None:
    if not isinstance(value, str) or not value:
        raise ProtocolError(message)


def load_default_protocol_map() -> ProtocolMap:
    """Load the bundled reference protocol (voice-evals-default-v1)."""
    from importlib import resources

    raw = json.loads(
        (resources.files("voice_evals") / "resources" / "protocols" / "default-v1.json").read_text(encoding="utf-8")
    )
    return ProtocolMap(raw)
