"""ElevenLabs caller voice requesting native PCM.

``POST /v1/text-to-speech/{voice_id}?output_format=pcm_16000`` (or pcm_24000)
returns headerless PCM. Headerless PCM cannot prove its own sample rate: the
request, the provider contract, and conformance fixtures are the source of
truth; recognizable MP3/WAV/error bodies are rejected. Requires ``httpx`` from
the ``probe`` extra.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..audio import AudioChunk, AudioFormat, looks_like_pcm_container
from ..clock import NS_PER_MS, Clock, MonotonicClock
from ..config import CallerConfig
from ..errors import CallerAuthError, CallerFormatError, CallerVoiceError
from ..models import SynthesizedAudio

_MAX_CLIP_BYTES = 20 * 1024 * 1024
_MAX_RETRY_AFTER_MS = 10_000


class ElevenLabsCallerVoice:
    """Complete-clip synthesis; audio is prepared before the call connects."""

    def __init__(
        self,
        config: CallerConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        max_bytes: int = _MAX_CLIP_BYTES,
    ) -> None:
        self.config = config
        self._client = http_client
        self._owns_client = http_client is None
        self.clock = clock or MonotonicClock()
        self.max_bytes = max_bytes
        self.attempts_record: list[dict[str, Any]] = []

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._client

    async def synthesize(
        self,
        text: str,
        *,
        audio_format: AudioFormat,
        seed: int | None = None,
    ) -> SynthesizedAudio:
        api_key = self._api_key()
        voice_id = self.config.resolved_voice_id()
        output_format = f"pcm_{audio_format.sample_rate}"
        url = (
            f"{self.config.base_url.rstrip('/')}/v1/text-to-speech/{voice_id}"
            f"?output_format={output_format}"
        )
        body = {"text": text, "model_id": self.config.model_id}
        if seed is not None:
            body["seed"] = seed
        headers = {"xi-api-key": api_key, "accept": "audio/*"}

        attempts = 0
        status: int | None = None
        last_error: str | None = None
        while attempts <= self.config.retries:
            attempts += 1
            client = await self._ensure_client()
            try:
                response = await client.post(url, json=body, headers=headers)
            except httpx.HTTPError as error:
                last_error = f"transport: {type(error).__name__}"
                await self._retry_backoff(response=None)
                continue
            status = response.status_code
            if status in (401, 402, 422):
                # Authentication and exhausted quota are not retried.
                raise CallerAuthError(
                    f"ElevenLabs rejected the request (HTTP {status})",
                    remediation=f"check the key referenced by {self.config.api_key_env} and quota",
                )
            if status == 200:
                self.attempts_record.append(
                    {"status": status, "attempts": attempts, "bytes": len(response.content)}
                )
                return self._parse_pcm(response.content, text, audio_format, attempts)
            if status == 429 or status >= 500:
                last_error = f"HTTP {status}"
                await self._retry_backoff(response)
                continue
            raise CallerVoiceError(f"ElevenLabs HTTP {status}")
        raise CallerVoiceError(
            f"ElevenLabs synthesis failed after {attempts - 1} attempt(s): {last_error}",
            remediation="throttling/5xx persisted; not retried mid-call to avoid timing contamination",
        )

    def _parse_pcm(
        self, content: bytes, text: str, audio_format: AudioFormat, attempts: int
    ) -> SynthesizedAudio:
        if not content:
            raise CallerFormatError("ElevenLabs returned an empty body for a PCM request")
        container = looks_like_pcm_container(content)
        if container is not None:
            raise CallerFormatError(
                f"ElevenLabs returned recognizable {container} content when raw PCM "
                f"was requested (output_format=pcm_{audio_format.sample_rate})"
            )
        if len(content) > self.max_bytes:
            raise CallerFormatError(f"clip exceeds {self.max_bytes} bytes")
        chunk = AudioChunk.from_pcm(content, audio_format)
        return SynthesizedAudio(
            text=text,
            pcm=chunk.data,
            audio_format=audio_format,
            samples=chunk.samples,
            provenance={
                "engine": "elevenlabs",
                "model_id": self.config.model_id,
                "voice_id": self.config.resolved_voice_id(),
                "output_format": f"pcm_{audio_format.sample_rate}",
                "http_attempts": attempts,
                "note": "duration derived from sample count; headerless PCM rate is contractual",
            },
        )

    def _api_key(self) -> str:
        import os

        value = os.environ.get(self.config.api_key_env, "").strip()
        if not value:
            raise CallerAuthError(
                f"missing API key (env {self.config.api_key_env})",
                remediation=f"export {self.config.api_key_env}=...",
            )
        return value

    async def _retry_backoff(self, response: httpx.Response | None) -> None:
        delay_ms = 500.0
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    delay_ms = min(float(retry_after) * 1000.0, _MAX_RETRY_AFTER_MS)
                except ValueError:
                    pass
        await self.clock.sleep_ns(int(delay_ms * NS_PER_MS))

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
