"""Real full-duplex WebSocket transport (WS for local dev, WSS verified).

Uses ``websockets`` (>=15, <16) from the ``probe`` extra. The transport stamps
receipt before normalization or disk I/O and send timestamps at the socket
boundary. Maximum message size is bounded and compression is disabled for PCM.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..audio import AudioFormat
from ..clock import Clock, MonotonicClock
from ..config import TransportConfig
from ..errors import CapabilityError, ProtocolError, TransportError
from ..models import (
    AgentEvent,
    CallerAudioFrame,
    ControlReceipt,
    EventKind,
    SendReceipt,
    TransportCapabilities,
)
from .protocol import ProtocolMap

Connector = Callable[..., Any]


class WebSocketAgentTransport:
    """Raw WS/WSS client driven by a declarative protocol map."""

    def __init__(
        self,
        protocol_map: ProtocolMap,
        *,
        clock: Clock | None = None,
        connector: Connector | None = None,
    ) -> None:
        self.map = protocol_map
        self.clock = clock or MonotonicClock()
        self._connector = connector
        self._ws: Any = None
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._receive_task: asyncio.Task | None = None
        self._capabilities: TransportCapabilities | None = None
        self._unknown_events = 0
        self._closed = False
        self._ready_seen = False

    async def open(self, config: TransportConfig) -> TransportCapabilities:
        if self._connector is None:
            from websockets.asyncio.client import connect as _connect

            self._connector = _connect
        endpoint = config.resolved_endpoint()
        headers: list[tuple[str, str]] = []
        if self.map.auth_headers:
            import os

            resolved = {name: os.environ.get(name, "") for name in {h["env"] for h in self.map.auth_headers}}
            headers = self.map.auth_header_values(resolved)
        try:
            self._ws = await asyncio.wait_for(
                self._connector(
                    endpoint,
                    additional_headers=dict(headers) if headers else None,
                    max_size=config.max_message_bytes,
                    max_queue=16,
                    compression=None,  # PCM does not compress; disable by default
                    open_timeout=max(1, config.handshake_timeout_ms // 1000),
                ),
                timeout=max(1, config.handshake_timeout_ms / 1000),
            )
        except asyncio.TimeoutError as error:
            raise TransportError("websocket open/handshake timed out") from error
        except Exception as error:  # websockets raises many typed errors
            raise TransportError(f"websocket connection failed: {type(error).__name__}") from error
        caps = self.map.capabilities
        if caps.input_format is not None:
            caps.input_format.validate()
        if caps.output_format is not None:
            caps.output_format.validate()
        self._capabilities = caps
        caps.note_observed("socket_open")
        self._receive_task = asyncio.ensure_future(self._receive_loop(caps))
        if self.map.handshake_send is not None:
            message = self.map.render(self.map.handshake_send, {"session_id": _session_nonce()})
            await self._ws.send(json.dumps(message, separators=(",", ":")))
        if self.map.handshake_ready_type is not None:
            await self._await_ready(config)
        return caps

    async def _await_ready(self, config: TransportConfig) -> None:
        deadline = self.clock.now_ns() + config.handshake_timeout_ms * 1_000_000
        while not self._ready_seen:
            if self.clock.now_ns() >= deadline:
                raise TransportError(
                    "protocol ready event not observed within the handshake timeout",
                    remediation="check the protocol map's handshake.ready_event against the server",
                )
            await self.clock.sleep_ns(10_000_000)

    async def _receive_loop(self, caps: TransportCapabilities) -> None:
        output_format: AudioFormat = caps.output_format or AudioFormat()
        try:
            async for raw in self._ws:
                receipt_ns = self.clock.now_ns()  # stamped before normalization/IO
                try:
                    event = self.map.normalize(raw, receipt_ns=receipt_ns, output_format=output_format)
                except ProtocolError as error:
                    event = AgentEvent(
                        kind=EventKind.ERROR,
                        receipt_ns=receipt_ns,
                        direction="system",
                        payload={"code": "protocol_failure", "message": error.message},
                    )
                if event is None:
                    self._unknown_events += 1
                    continue
                if event.kind == EventKind.SESSION_READY:
                    self._ready_seen = True
                await self._queue.put(event)
        except asyncio.CancelledError:
            return
        except Exception as error:  # noqa: BLE001 - socket close and protocol failures
            if not self._closed:
                await self._queue.put(
                    AgentEvent(
                        kind=EventKind.ERROR,
                        receipt_ns=self.clock.now_ns(),
                        direction="system",
                        payload={
                            "code": "transport_error",
                            "message": f"{type(error).__name__}",
                        },
                    )
                )
        finally:
            await self._queue.put(None)

    def events(self) -> AsyncIterator[AgentEvent]:
        return self._consume()

    async def _consume(self) -> AsyncIterator[AgentEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def send_audio(self, frame: CallerAudioFrame) -> SendReceipt:
        if self._ws is None:
            raise TransportError("transport not open")
        encoded = self.map.encode_audio_frame(frame.utterance_id, frame.sequence, frame.payload)
        start = self.clock.now_ns()
        try:
            await self._ws.send(encoded)
        except Exception as error:
            raise TransportError(f"websocket send failed: {type(error).__name__}") from error
        return SendReceipt(
            utterance_id=frame.utterance_id,
            sequence=frame.sequence,
            send_start_ns=start,
            send_complete_ns=self.clock.now_ns(),
            bytes_sent=len(frame.payload),
        )

    async def end_utterance(self, utterance_id: str) -> None:
        caps = self._capabilities
        if caps is None or not caps.explicit_input_commit or self.map.end_utterance_template is None:
            return  # native server VAD: no commit message required
        message = self.map.render(self.map.end_utterance_template, {"utterance_id": utterance_id})
        await self._send_control(json.dumps(message, separators=(",", ":")))

    async def cancel_response(self, response_id: str) -> ControlReceipt:
        caps = self._capabilities
        if caps is None or not caps.cancellation or self.map.cancel_template is None:
            raise CapabilityError(
                "response cancellation is not supported by this protocol map",
                remediation="natural barge-in testing never needs cancellation",
            )
        message = self.map.render(self.map.cancel_template, {"response_id": response_id})
        sent_ns = self.clock.now_ns()
        await self._send_control(json.dumps(message, separators=(",", ":")))
        return ControlReceipt(kind="cancel_response", sent_ns=sent_ns, ok=True)

    async def _send_control(self, text: str) -> None:
        if self._ws is None:
            raise TransportError("transport not open")
        try:
            await self._ws.send(text)
        except Exception as error:
            raise TransportError(f"websocket control send failed: {type(error).__name__}") from error

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._receive_task is not None:
            self._receive_task.cancel()
            await asyncio.gather(self._receive_task, return_exceptions=True)
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001, S110 - best-effort close
                pass


def _session_nonce() -> str:
    import secrets

    return secrets.token_hex(8)
