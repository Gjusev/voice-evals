"""PCM framing, pacing helpers and WAV I/O. Standard library only.

v0.2 wire support is PCM signed 16-bit little-endian, mono, 16 kHz or 24 kHz.
No resampling, no codecs: unsupported formats fail preflight.
"""

from __future__ import annotations

import hashlib
import io
import math
import struct
import wave
from dataclasses import dataclass

from .errors import CallerFormatError

SUPPORTED_RATES = (16000, 24000)
SUPPORTED_SAMPLE_WIDTH = 2
SUPPORTED_CHANNELS = 1


@dataclass(frozen=True)
class AudioFormat:
    """Describes one PCM stream. Independent per direction."""

    sample_rate: int = 16000
    channels: int = SUPPORTED_CHANNELS
    sample_width: int = SUPPORTED_SAMPLE_WIDTH  # bytes per sample
    encoding: str = "pcm_s16le"

    def validate(self) -> None:
        if self.encoding != "pcm_s16le":
            raise CallerFormatError(
                f"unsupported encoding {self.encoding!r}; v0.2 supports pcm_s16le only"
            )
        if self.sample_rate not in SUPPORTED_RATES:
            raise CallerFormatError(
                f"unsupported sample rate {self.sample_rate}; supported: {list(SUPPORTED_RATES)}"
            )
        if self.channels != SUPPORTED_CHANNELS or self.sample_width != SUPPORTED_SAMPLE_WIDTH:
            raise CallerFormatError("v0.2 supports mono 16-bit PCM only")

    def bytes_per_sample(self) -> int:
        return self.channels * self.sample_width

    def frame_bytes(self, frame_ms: int) -> int:
        """Byte size of one frame, on sample boundaries."""
        return self.samples_for_ms(frame_ms) * self.bytes_per_sample()

    def samples_for_ms(self, ms: float) -> int:
        return int(self.sample_rate * ms / 1000.0)

    def duration_ns(self, samples: int) -> int:
        if self.sample_rate == 0:
            return 0
        return samples * 1_000_000_000 // self.sample_rate

    def sample_count(self, payload: bytes) -> int:
        return len(payload) // self.bytes_per_sample()

    def public_dict(self) -> dict[str, object]:
        return {
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
        }


@dataclass(frozen=True)
class AudioChunk:
    """PCM bytes plus their exact format and derived sample accounting."""

    data: bytes
    audio_format: AudioFormat
    samples: int

    @classmethod
    def from_pcm(cls, data: bytes, audio_format: AudioFormat) -> AudioChunk:
        width = audio_format.bytes_per_sample()
        if len(data) % width != 0:
            raise CallerFormatError(
                f"PCM payload of {len(data)} bytes is not sample-aligned for width {width}"
            )
        return cls(data=data, audio_format=audio_format, samples=len(data) // width)

    @property
    def duration_ns(self) -> int:
        return self.audio_format.duration_ns(self.samples)

    def content_hash(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def frame_pcm(
    pcm: bytes,
    audio_format: AudioFormat,
    frame_ms: int,
) -> list[bytes]:
    """Split PCM on sample boundaries into frames of ~frame_ms.

    The final short frame is valid and unpadded; callers record its true
    sample duration. Frame sizes never cross a sample boundary.
    """
    width = audio_format.bytes_per_sample()
    frame_size = audio_format.frame_bytes(frame_ms)
    if frame_size <= 0 or frame_size % width != 0:
        raise ValueError(f"frame_ms={frame_ms} yields non-sample-aligned size {frame_size}")
    return [pcm[i : i + frame_size] for i in range(0, len(pcm), frame_size)]


def pcm_to_wav_bytes(pcm: bytes, audio_format: AudioFormat) -> bytes:
    """Wrap raw PCM in a WAV container using stdlib ``wave``."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(audio_format.channels)
        handle.setsampwidth(audio_format.sample_width)
        handle.setframerate(audio_format.sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def wav_to_pcm(data: bytes) -> tuple[bytes, AudioFormat]:
    """Validate an uncompressed PCM WAV with stdlib ``wave`` and strip it.

    Compressed WAV fails: v0.2 accepts uncompressed PCM only.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            comptype = handle.getcomptype()
            pcm = handle.readframes(handle.getnframes())
    except wave.Error as error:
        raise CallerFormatError(f"not a readable WAV container: {error}") from error
    if comptype != "NONE":
        raise CallerFormatError("compressed WAV is not accepted; serve uncompressed PCM")
    fmt = AudioFormat(sample_rate=rate, channels=channels, sample_width=width)
    if width != SUPPORTED_SAMPLE_WIDTH or channels != SUPPORTED_CHANNELS or rate not in SUPPORTED_RATES:
        raise CallerFormatError(
            f"WAV {rate}Hz/{channels}ch/{width * 8}bit is outside v0.2 PCM support "
            f"(mono 16-bit 16/24 kHz): resample in the model service"
        )
    return pcm, fmt


def looks_like_pcm_container(data: bytes) -> str | None:
    """Detect recognizable non-PCM bodies when raw PCM was requested.

    Headerless PCM cannot prove its own rate; this only catches obvious
    MP3/ID3/RIFF/Ogg/FLAC bodies so failures are actionable.
    """
    if not data:
        return None
    head = data[:8]
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    if head.startswith(b"RIFF"):
        return "wav"
    if head.startswith(b"OggS"):
        return "ogg"
    if head.startswith(b"fLaC"):
        return "flac"
    if data.lstrip()[:1] in (b"{", b"["):
        return "json"
    return None


def silence_bytes(audio_format: AudioFormat, duration_ms: float) -> bytes:
    """Zero PCM of the given duration (used for optional timeline WAVs)."""
    samples = audio_format.samples_for_ms(duration_ms)
    return b"\x00\x00" * samples


def tone_pcm(
    audio_format: AudioFormat,
    duration_ms: float,
    *,
    freq_hz: float = 220.0,
    amplitude: float = 6000.0,
) -> bytes:
    """Deterministic tone PCM for tests and synthetic fixtures (not speech)."""
    samples = audio_format.samples_for_ms(duration_ms)
    return b"".join(
        struct.pack(
            "<h",
            int(amplitude * math.sin(2 * math.pi * freq_hz * n / audio_format.sample_rate)),
        )
        for n in range(samples)
    )
