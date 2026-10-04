"""AgentTransport protocol. Transports never see scenario expectations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from ..config import TransportConfig
from ..models import (
    AgentEvent,
    CallerAudioFrame,
    ControlReceipt,
    SendReceipt,
    TransportCapabilities,
)


@runtime_checkable
class AgentTransport(Protocol):
    async def open(self, config: TransportConfig) -> TransportCapabilities:
        """Acquire a usable session and report negotiated capabilities."""
        ...

    def events(self) -> AsyncIterator[AgentEvent]:
        """Exactly one consumer iterates this stream for the whole session."""
        ...

    async def send_audio(self, frame: CallerAudioFrame) -> SendReceipt:
        """Send one caller frame; stamps send start/complete at the socket boundary."""
        ...

    async def end_utterance(self, utterance_id: str) -> None:
        """Commit input when the protocol requires it; no-op for native VAD."""
        ...

    async def cancel_response(self, response_id: str) -> ControlReceipt:
        """Explicit cancellation; raises CapabilityError when unsupported."""
        ...

    async def aclose(self) -> None:
        ...
