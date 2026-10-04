"""PCM framing, sample accounting, WAV I/O, and container detection."""


import pytest

from voice_evals.probe.audio import (
    AudioChunk,
    AudioFormat,
    frame_pcm,
    looks_like_pcm_container,
    pcm_to_wav_bytes,
    tone_pcm,
    wav_to_pcm,
)
from voice_evals.probe.errors import CallerFormatError
from voice_evals.probe.testing import decode_text_pcm, encode_text_pcm


def test_frame_sizes_on_sample_boundaries() -> None:
    fmt = AudioFormat(sample_rate=16000)
    pcm = b"\x01\x00" * 1601  # 1601 samples: not frame-aligned
    frames = frame_pcm(pcm, fmt, 20)
    sizes = [len(f) for f in frames]
    assert sizes[0] == 640  # 20ms at 16kHz mono s16
    assert all(s == 640 for s in sizes[:-1])
    assert sizes[-1] == 2  # final short frame: one sample, unpadded
    assert b"".join(frames) == pcm
    assert sum(fmt.sample_count(f) for f in frames) == fmt.sample_count(pcm)


def test_frame_bytes_24k() -> None:
    fmt = AudioFormat(sample_rate=24000)
    assert fmt.frame_bytes(20) == 960


def test_duration_from_samples_not_chunks() -> None:
    fmt = AudioFormat(sample_rate=16000)
    chunk = AudioChunk.from_pcm(b"\x00\x00" * 16000, fmt)
    assert chunk.samples == 16000
    assert chunk.duration_ns == 1_000_000_000


def test_unaligned_payload_rejected() -> None:
    with pytest.raises(CallerFormatError, match="sample-aligned"):
        AudioChunk.from_pcm(b"\x01\x02\x03", AudioFormat())


def test_unsupported_format_fails_preflight() -> None:
    with pytest.raises(CallerFormatError, match="pcm_s16le"):
        AudioFormat(encoding="pcm_mulaw").validate()
    with pytest.raises(CallerFormatError, match="sample rate"):
        AudioFormat(sample_rate=8000).validate()
    with pytest.raises(CallerFormatError, match="mono"):
        AudioFormat(channels=2).validate()


def test_wav_round_trip() -> None:
    fmt = AudioFormat(sample_rate=24000)
    pcm = tone_pcm(fmt, 100)
    wav = pcm_to_wav_bytes(pcm, fmt)
    assert wav[:4] == b"RIFF"
    out, out_fmt = wav_to_pcm(wav)
    assert out == pcm
    assert out_fmt.sample_rate == 24000 and out_fmt.channels == 1 and out_fmt.sample_width == 2


def test_compressed_wav_rejected() -> None:
    # Hand-built WAV with a non-PCM fmt tag (6 = a-law): stdlib wave only
    # writes NONE, so forge the fmt chunk bytes directly.
    fmt_chunk = b"fmt " + (16).to_bytes(4, "little") + (6).to_bytes(2, "little")
    fmt_chunk += (1).to_bytes(2, "little") + (16000).to_bytes(4, "little")
    fmt_chunk += (32000).to_bytes(4, "little") + (2).to_bytes(2, "little")
    fmt_chunk += (16).to_bytes(2, "little")
    data = b"\x00\x00" * 100
    data_chunk = b"data" + len(data).to_bytes(4, "little") + data
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    wav = b"RIFF" + riff_size.to_bytes(4, "little") + b"WAVE" + fmt_chunk + data_chunk
    with pytest.raises(CallerFormatError, match="not a readable WAV|compressed"):
        wav_to_pcm(wav)


def test_container_detection() -> None:
    assert looks_like_pcm_container(b"ID3\x04\x00abc") == "mp3"
    assert looks_like_pcm_container(b"\xff\xfb\x90\x00") == "mp3"
    assert looks_like_pcm_container(b"RIFFxxxxWAVE") == "wav"
    assert looks_like_pcm_container(b'OggS\x00\x02') == "ogg"
    assert looks_like_pcm_container(b'{"detail":"quota"}') == "json"
    assert looks_like_pcm_container(b"\x12\x34\x56") is None  # plain PCM


def test_watermark_codec_round_trip_multilingual() -> None:
    fmt = AudioFormat(sample_rate=16000)
    for text in (
        "I'd like to book an appointment for Tuesday.",
        "Ich möchte einen Termin am Dienstag.",
        "Quisiera reservar una cita el martes.",
    ):
        pcm = encode_text_pcm(text, fmt)
        chunk = AudioChunk.from_pcm(pcm, fmt)
        assert chunk.samples >= len(text.encode("utf-8"))
        assert decode_text_pcm(pcm) == text


def test_watermark_duration_scales_with_text() -> None:
    fmt = AudioFormat(sample_rate=16000)
    short = AudioChunk.from_pcm(encode_text_pcm("hi", fmt), fmt)
    long = AudioChunk.from_pcm(encode_text_pcm("a much longer sentence here", fmt), fmt)
    assert long.samples > short.samples
