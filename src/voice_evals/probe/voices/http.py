"""Generic HTTP caller voice for externally hosted open-model TTS servers.

Documented contract (v0.2):

    POST {CALLER_TTS_URL}
    Authorization: Bearer <optional env-referenced token>
    {"text": str, "model": str, "voice": str, "language": str,
     "seed": int|null, "format": "pcm_s16le",
     "sample_rate": 16000|24000, "channels": 1}

    200 -> raw headerless PCM bytes with explicit format headers
           X-Audio-Encoding: pcm_s16le
           X-Audio-Rate: <sample rate>
           X-Audio-Channels: 1
           X-Audio-Sample-Width: 2
       or a single uncompressed PCM WAV body (Content-Type audio/wav);
       compressed WAV is rejected.

The model service may wrap an upstream server (vLLM, chatterbox-server, ...)
to meet this contract, including any resampling — that happens in the service
and must be recorded in its provenance, not here. No inference enters the
package. Requires ``httpx`` from the ``probe`` extra.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..audio import AudioChunk, AudioFormat, looks_like_pcm_container, wav_to_pcm
from ..clock import NS_PER_MS, Clock, MonotonicClock
from ..config import CallerConfig
from ..errors import CallerAuthError, CallerFormatError, CallerVoiceError
from ..models import SynthesizedAudio


class HttpCallerVoice:
    """Open-model caller voice behind a small documented HTTP contract."""

    def __init__(
        self,
        config: CallerConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self._client = http_client
        self._owns_client = http_client is None
        self.clock = clock or MonotonicClock()

    def _url(self) -> str:
        url = self.config.url or os.environ.get(self.config.url_env, "").strip()
        if not url:
            raise CallerVoiceError(
                f"caller TTS URL missing (env {self.config.url_env})",
                remediation=f"pass --caller-url or set {self.config.url_env}",
            )
        if not url.startswith(("http://", "https://")):
            raise CallerFormatError("caller TTS URL must be http(s)")
        return url

    async def synthesize(
        self,
        text: str,
        *,
        audio_format: AudioFormat,
        seed: int | None = None,
    ) -> SynthesizedAudio:
        url = self._url()
        headers: dict[str, str] = {}
        if self.config.auth_env:
            token = os.environ.get(self.config.auth_env, "").strip()
            if not token:
                raise CallerAuthError(
                    f"missing caller auth (env {self.config.auth_env})",
                    remediation=f"export {self.config.auth_env}=...",
                )
            headers["authorization"] = f"Bearer {token}"
        body: dict[str, Any] = {
            "text": text,
            "model": self.config.model_id,
            "voice": self.config.voice_id or "default",
            "language": self.config.language,
            "seed": seed,
            "format": audio_format.encoding,
            "sample_rate": audio_format.sample_rate,
            "channels": audio_format.channels,
        }
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        attempts = 0
        while True:
            attempts += 1
            try:
                response = await self._client.post(url, json=body, headers=headers)
            except httpx.HTTPError as error:
                if attempts > self.config.retries:
                    raise CallerVoiceError(f"caller TTS unreachable: {type(error).__name__}") from error
                await self.clock.sleep_ns(500 * NS_PER_MS)
                continue
            if response.status_code in (401, 402, 403):
                raise CallerAuthError(f"caller TTS rejected auth (HTTP {response.status_code})")
            if response.status_code == 429 or response.status_code >= 500:
                if attempts > self.config.retries:
                    raise CallerVoiceError(f"caller TTS HTTP {response.status_code} after retries")
                retry_after = response.headers.get("retry-after")
                try:
                    delay_ms = min(float(retry_after or 0.5) * 1000.0, 10_000.0)
                except ValueError:
                    delay_ms = 500.0
                await self.clock.sleep_ns(int(delay_ms * NS_PER_MS))
                continue
            if response.status_code != 200:
                raise CallerVoiceError(f"caller TTS HTTP {response.status_code}")
            pcm, served_format, served = self._parse(response, audio_format)
            return SynthesizedAudio(
                text=text,
                pcm=pcm,
                audio_format=served_format,
                samples=served_format.sample_count(pcm),
                provenance={
                    "engine": "http-caller",
                    "model": self.config.model_id,
                    "voice": self.config.voice_id,
                    "language": self.config.language,
                    "seed": seed,
                    "served_format": served,
                    "url_env": self.config.url_env,
                    "http_attempts": attempts,
                },
            )

    def _parse(self, response: httpx.Response, requested: AudioFormat) -> tuple[bytes, AudioFormat, str]:
        content = response.content
        if not content:
            raise CallerFormatError("caller TTS returned an empty body")
        content_type = (response.headers.get("content-type") or "").lower()
        if "wav" in content_type or content[:4] == b"RIFF":
            pcm, fmt = wav_to_pcm(content)
            return pcm, fmt, "wav"
        container = looks_like_pcm_container(content)
        if container is not None:
            raise CallerFormatError(
                f"caller TTS returned recognizable {container} content; the contract "
                "serves headerless PCM (with X-Audio-* headers) or uncompressed WAV"
            )
        rate_header = response.headers.get("x-audio-rate")
        if rate_header:
            rate = int(rate_header)
            if rate != requested.sample_rate:
                raise CallerFormatError(
                    f"caller TTS served {rate} Hz while {requested.sample_rate} Hz was "
                    "requested; resampling belongs in the model service, not the harness"
                )
        width_header = response.headers.get("x-audio-sample-width")
        if width_header and int(width_header) != requested.sample_width:
            raise CallerFormatError("caller TTS sample width does not match the request")
        chunk = AudioChunk.from_pcm(content, requested)
        return chunk.data, requested, "raw-pcm"

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
