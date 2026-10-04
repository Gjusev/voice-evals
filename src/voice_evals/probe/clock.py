"""Monotonic clock interface plus a deterministic virtual clock for tests.

All durations use integer nanoseconds from ``time.perf_counter_ns()``. The
session origin is subtracted once; stored timestamps are session-relative.
UTC is used for run identity only, never for duration arithmetic.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    def now_ns(self) -> int:
        """Current monotonic nanoseconds."""
        ...

    async def sleep_ns(self, duration_ns: int) -> None:
        """Sleep for ``duration_ns`` (>= 0), yielding control to the loop."""
        ...


class MonotonicClock:
    """Wall-advancing clock backed by ``perf_counter_ns``/``asyncio.sleep``."""

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    async def sleep_ns(self, duration_ns: int) -> None:
        if duration_ns > 0:
            await asyncio.sleep(duration_ns / 1e9)


NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


class VirtualClock:
    """Deterministic clock: sleeping advances virtual time without waiting.

    ``sleep_ns`` advances the shared virtual now in slices (default 1 ms) and
    yields to the event loop after each slice, so concurrently sleeping tasks
    (paced sender, streaming mock agent) interleave in order. Timestamps are
    exact up to one slice. ``advance_ns`` jumps time without yielding.
    """

    def __init__(self, start_ns: int = 0, slice_ns: int = NS_PER_MS) -> None:
        self._now = start_ns
        self.slice_ns = max(1, slice_ns)

    def now_ns(self) -> int:
        return self._now

    def advance_ns(self, duration_ns: int) -> None:
        self._now += max(0, duration_ns)

    async def sleep_ns(self, duration_ns: int) -> None:
        remaining = max(0, duration_ns)
        while remaining > 0:
            step = min(remaining, self.slice_ns)
            self.advance_ns(step)
            remaining -= step
            await asyncio.sleep(0)
