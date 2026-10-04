"""Checksum-addressed prerecorded caller voice.

A fixture manifest maps exact texts to PCM/WAV files with sha256 checksums and
provenance (model, revision, license, language, generator). This is the
strongest-provenance caller: reviewed real audio, pinned bytes, no inference
at probe time. Fixture audio with model provenance must never be relabeled as
another model's output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..audio import AudioFormat, wav_to_pcm
from ..errors import CallerFormatError, CallerVoiceError
from ..models import SynthesizedAudio


class FixtureCallerVoice:
    """Serves prepared clips addressed by exact text; bytes are verified."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CallerVoiceError(f"fixture manifest unreadable: {error}") from error
        self._root = self.manifest_path.parent
        self._format = AudioFormat(
            sample_rate=int(raw.get("format", {}).get("sample_rate", 16000)),
            channels=int(raw.get("format", {}).get("channels", 1)),
            sample_width=int(raw.get("format", {}).get("sample_width", 2)),
            encoding=str(raw.get("format", {}).get("encoding", "pcm_s16le")),
        )
        self._clips: dict[str, dict[str, Any]] = {
            str(clip["text"]): clip for clip in raw.get("clips", [])
        }
        self.provenance_note = str(
            raw.get("note", "prepared fixture audio; see manifest provenance per clip")
        )

    async def synthesize(
        self,
        text: str,
        *,
        audio_format: AudioFormat,
        seed: int | None = None,
    ) -> SynthesizedAudio:
        clip = self._clips.get(text)
        if clip is None:
            raise CallerVoiceError(
                f"no fixture clip for text {text[:60]!r}...",
                remediation="extend the fixture manifest or use a synthesizing caller",
            )
        path = self._root / str(clip["file"])
        try:
            data = path.read_bytes()
        except OSError as error:
            raise CallerVoiceError(f"fixture file unreadable: {path.name}") from error
        digest = hashlib.sha256(data).hexdigest()
        expected = clip.get("sha256")
        if expected and digest != expected:
            raise CallerFormatError(
                f"fixture checksum mismatch for {path.name}",
                remediation="regenerate the manifest hash or restore the original file",
            )
        if clip.get("file", "").endswith(".wav") or data[:4] == b"RIFF":
            pcm, served_format = wav_to_pcm(data)
        else:
            pcm, served_format = data, self._format
        if served_format.sample_rate != audio_format.sample_rate:
            raise CallerFormatError(
                f"fixture is {served_format.sample_rate} Hz but the transport needs "
                f"{audio_format.sample_rate} Hz; no silent resampling"
            )
        return SynthesizedAudio(
            text=text,
            pcm=pcm,
            audio_format=served_format,
            samples=served_format.sample_count(pcm),
            provenance={
                "engine": "fixture",
                "file": str(clip.get("file")),
                "sha256": digest,
                **{k: v for k, v in (clip.get("provenance") or {}).items()},
            },
        )

    async def aclose(self) -> None:
        return None
