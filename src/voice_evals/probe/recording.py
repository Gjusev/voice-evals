"""Session recording: journal, audio files, atomic manifest finalization.

Layout (one output directory per session):

    manifest.json      sanitized, atomically finalized
    events.jsonl       append-only normalized event journal
    calls.jsonl        exported replayable CallRecord (empty when not replayable)
    result.json        legacy v0.1 result fields plus additive ``probe`` object
    audio/caller/<utterance-id>.wav
    audio/agent/<response-id>.wav

Hard termination can leave an explicitly incomplete manifest and a truncated
final journal line; recovery reads complete lines only. Secrets never reach
artifacts: only env variable names and sanitized values are recorded.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import AudioFormat, pcm_to_wav_bytes
from .errors import RecordingError
from .models import AgentEvent, SessionRecord

MANIFEST_SCHEMA_VERSION = 1
_FLUSH_EVERY = 200


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_session_id() -> str:
    return f"sess-{uuid.uuid4().hex[:12]}"


@dataclass
class _AudioStream:
    audio_format: AudioFormat
    chunks: list[bytes] = field(default_factory=list)
    bytes_written: int = 0
    complete: bool = False

    def extend(self, payload: bytes) -> None:
        self.chunks.append(payload)
        self.bytes_written += len(payload)

    def wav_bytes(self) -> bytes:
        return pcm_to_wav_bytes(b"".join(self.chunks), self.audio_format)


class SessionRecorder:
    """Bounded journal writer plus audio/manifest/result persistence."""

    def __init__(
        self,
        output_dir: Path,
        *,
        session_id: str,
        config_public: dict[str, Any],
        script_dict: dict[str, Any],
        protocol_hash: str | None,
        limits: dict[str, int],
    ) -> None:
        self.dir = Path(output_dir)
        self.audio_dir = self.dir / "audio"
        (self.audio_dir / "caller").mkdir(parents=True, exist_ok=True)
        (self.audio_dir / "agent").mkdir(parents=True, exist_ok=True)
        self.journal_path = self.dir / "events.jsonl"
        self.manifest_path = self.dir / "manifest.json"
        self._journal = self.journal_path.open("w", encoding="utf-8")
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=10_000)
        self._writer_task = asyncio.ensure_future(self._writer_loop())
        self._entries = 0
        self._flushed_upto = 0
        self.caller_streams: dict[str, _AudioStream] = {}
        self.agent_streams: dict[str, _AudioStream] = {}
        self._max_entries = limits.get("max_event_journal_entries", 100_000)
        self._max_audio_bytes = limits.get("max_audio_bytes_per_direction", 200 * 1024 * 1024)
        self._caller_bytes = 0
        self._agent_bytes = 0
        self._files: dict[str, str] = {}
        self.finalized = False
        self._script_dict = script_dict
        self._protocol_hash = protocol_hash
        self._write_initial_manifest(session_id, config_public, script_dict, protocol_hash)

    # -- manifest ---------------------------------------------------------

    def _write_initial_manifest(
        self,
        session_id: str,
        config_public: dict[str, Any],
        script_dict: dict[str, Any],
        protocol_hash: str | None,
    ) -> None:
        manifest = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "session_id": session_id,
            "status": "running",
            "status_reason": None,
            "complete": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {
                "voice_evals": _package_version(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "config": config_public,
            "script": script_dict,
            "script_sha256": _sha256(json.dumps(script_dict, sort_keys=True).encode("utf-8")),
            "protocol_map_sha256": protocol_hash,
        }
        self._atomic_write(self.manifest_path, json.dumps(manifest, indent=2))

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    # -- journal ----------------------------------------------------------

    async def write_event(self, event: AgentEvent) -> None:
        """Queue one journal entry; timestamping happened before insertion."""
        if self._entries >= self._max_entries:
            raise RecordingError(
                f"event journal exceeded {self._max_entries} entries",
                remediation="raise limits.max_event_journal_entries or shorten the session",
            )
        try:
            self._queue.put_nowait(event.to_journal_dict())
        except asyncio.QueueFull as error:
            raise RecordingError("journal writer backpressure exceeded bound") from error
        self._entries += 1

    async def _writer_loop(self) -> None:
        buffer: list[dict[str, Any]] = []
        while True:
            item = await self._queue.get()
            if item is None:
                if buffer:
                    self._flush(buffer)
                self._journal.flush()
                return
            buffer.append(item)
            if len(buffer) >= _FLUSH_EVERY:
                self._flush(buffer)
                buffer = []

    def _flush(self, buffer: list[dict[str, Any]]) -> None:
        try:
            for entry in buffer:
                self._journal.write(json.dumps(entry, separators=(",", ":")) + "\n")
            self._journal.flush()
        except OSError as error:
            raise RecordingError(f"journal write failed: {error}") from error
        self._flushed_upto += len(buffer)

    # -- audio ------------------------------------------------------------

    def add_caller_audio(self, utterance_id: str, payload: bytes, audio_format: AudioFormat) -> None:
        stream = self.caller_streams.get(utterance_id)
        if stream is None:
            stream = _AudioStream(audio_format)
            self.caller_streams[utterance_id] = stream
        self._caller_bytes += len(payload)
        if self._caller_bytes > self._max_audio_bytes:
            raise RecordingError("caller audio exceeded max_audio_bytes_per_direction")
        stream.extend(payload)

    def add_agent_audio(self, response_id: str, payload: bytes, audio_format: AudioFormat) -> None:
        stream = self.agent_streams.get(response_id)
        if stream is None:
            stream = _AudioStream(audio_format)
            self.agent_streams[response_id] = stream
        self._agent_bytes += len(payload)
        if self._agent_bytes > self._max_audio_bytes:
            raise RecordingError("agent audio exceeded max_audio_bytes_per_direction")
        stream.extend(payload)

    def mark_caller_complete(self, utterance_id: str) -> None:
        stream = self.caller_streams.get(utterance_id)
        if stream is not None:
            stream.complete = True

    # -- finalization -------------------------------------------------------

    async def finalize(self, session: SessionRecord, manifest_extra: dict[str, Any]) -> dict[str, Any]:
        """Flush artifacts and atomically finalize the manifest."""
        if self.finalized:
            return {}
        self.finalized = True
        await self._queue.put(None)
        await asyncio.gather(self._writer_task, return_exceptions=False)
        self._journal.close()
        for utterance_id, stream in self.caller_streams.items():
            path = self.audio_dir / "caller" / f"{utterance_id}.wav"
            path.write_bytes(stream.wav_bytes())
            self._files[path.relative_to(self.dir).as_posix()] = _file_sha256(path)
        for response_id, stream in self.agent_streams.items():
            path = self.audio_dir / "agent" / f"{response_id}.wav"
            path.write_bytes(stream.wav_bytes())
            self._files[path.relative_to(self.dir).as_posix()] = _file_sha256(path)
        manifest = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "session_id": session.session_id,
            "status": session.status.value,
            "status_reason": session.status_reason,
            "complete": session.status.value == "completed",
            "created_utc": session.created_utc,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {
                "voice_evals": _package_version(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "environment": session.environment,
            "script": self._script_dict,
            "script_sha256": _sha256(json.dumps(self._script_dict, sort_keys=True).encode("utf-8")),
            "protocol_map_sha256": self._protocol_hash,
            "session_origin_ns": 0,
            "total_duration_ns": session.total_duration_ns,
            "connection": session.connection,
            "capabilities": session.capabilities.public_dict() if session.capabilities else None,
            "turns": [_turn_dict(t) for t in session.turns],
            "interruptions": [t.interrupt.to_dict() for t in session.turns if t.interrupt],
            "transcript_segments": session.transcript_segments,
            "synthesis": session.synthesis,
            "errors": session.errors,
            "outcome": {
                "value": session.outcome,
                "rule": session.outcome_rule_used,
                "evidence": session.outcome_evidence,
            },
            "counters": session.counters,
            "journal_entries": self._entries,
            "files": self._files,
            "cleanup": {"journal_closed": True, "audio_flushed": True},
            **manifest_extra,
        }
        self._atomic_write(self.manifest_path, json.dumps(manifest, indent=2))
        return manifest

    def write_calls(self, rows: list[dict[str, Any]], replayable: bool) -> None:
        path = self.dir / "calls.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        self._files["calls.jsonl"] = _file_sha256(path)
        self._files["calls_replayable"] = "true" if replayable else "false"

    def write_result(self, result: dict[str, Any]) -> None:
        path = self.dir / "result.json"
        self._atomic_write(path, json.dumps(result, indent=2))
        self._files["result.json"] = _file_sha256(path)


def _turn_dict(turn: Any) -> dict[str, Any]:
    out = {
        "step_id": turn.step_id,
        "utterance_id": turn.utterance_id,
        "selected_text": turn.selected_text,
        "alternative_index": turn.alternative_index,
        "reveals": list(turn.renders_facts),
        "response_id": turn.response_id,
        "skipped": turn.skipped,
        "sent": turn.sent,
        "frames": [vars(f) for f in turn.frames],
        "caller_audio_end_ns": turn.caller_audio_end_ns,
        "scheduled_end_ns": turn.scheduled_end_ns,
        "timing": {
            "e2e_ns": turn.e2e_ns,
            "e2e_overlap_ns": turn.e2e_overlap_ns,
            "client_final_asr_ns": turn.client_final_asr_ns,
            "llm_ttft_ns": turn.llm_ttft_ns,
            "tts_ttfa_ns": turn.tts_ttfa_ns,
            "asr_final_to_agent_text_ns": turn.asr_final_to_agent_text_ns,
            "first_text_to_audio_ns": turn.first_text_to_audio_ns,
            "agent_first_audio_ns": turn.agent_first_audio_ns,
            "response_done_ns": turn.response_done_ns,
            "notes": turn.timing_notes,
        },
        "caller_asr_text": turn.caller_asr_text,
        "agent_final_text": turn.agent_final_text,
    }
    if turn.interrupt is not None:
        out["interrupt"] = turn.interrupt.to_dict()
    return out


def _package_version() -> str:
    try:
        from voice_evals import __version__

        return __version__
    except Exception:  # noqa: BLE001 - pragma: no cover - defensive
        return "unknown"


def read_journal(path: str | Path) -> list[dict[str, Any]]:
    """Recovery helper: read complete journal lines only."""
    entries = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError:
                break  # truncated final line after hard termination
    return entries
