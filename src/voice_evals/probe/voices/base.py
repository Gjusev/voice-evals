"""CallerVoice protocol. Adapters never receive expected outcomes or facts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..audio import AudioFormat
from ..models import SynthesizedAudio


@runtime_checkable
class CallerVoice(Protocol):
    async def synthesize(
        self,
        text: str,
        *,
        audio_format: AudioFormat,
        seed: int | None = None,
    ) -> SynthesizedAudio:
        """Synthesize one complete clip in the exact requested native format."""
        ...

    async def aclose(self) -> None:
        ...
