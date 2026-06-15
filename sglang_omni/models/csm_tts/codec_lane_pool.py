# SPDX-License-Identifier: Apache-2.0
"""Codec-lane pool — the codec/vocoder stage's concurrency as a first-class,
budgeted admission resource.

On commodity GPUs the realtime ceiling of a streaming-TTS server is frequently
set by the **codec (Mimi decode) stage**, not by the language model: each
active stream needs a codec "lane" (a unit of decode capacity) every frame
period, and once those lanes are all in use, admitting another stream only
adds queueing — i.e. underrun (see :mod:`frame_metrics` for the frame-clock
breaking-point model this pairs with). The robust answer is the same one
SGLang/vLLM use for KV blocks: treat the scarce resource as an explicitly
sized pool with acquire/release discipline and saturation telemetry, and shed
(fast 429) rather than queue when it is exhausted.

This module gives the vocoder stage that pool. It is a bounded counting
semaphore with:

- non-blocking ``try_acquire`` (admission control: fail fast, do not block the
  decode thread waiting for a lane), and a blocking ``acquire`` for callers
  that genuinely want to wait;
- idempotent-safe ``release``;
- a ``lane`` context manager for exception-safe hold/return;
- saturation counters (``acquired`` / ``rejected`` / peak ``in_use``) so the
  ceiling shows up in telemetry instead of as mysterious stutter.

Sizing: the lane count is the codec stage's max in-flight decode width — it
tracks the lane capacity, never a hardcoded 1 (R0 §7 lesson (c)). The default
mirrors the pipeline's ``DEFAULT_MAX_CONCURRENCY`` so the codec pool and the
engine's running-request cap are sized together; deployments raise it via the
vocoder factory's ``codec_lanes`` arg once the codec stage's measured
per-frame cost shows headroom.

R0 §9 (codec-lane-pool sizing) + §7 (admission / 429 backpressure).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

__all__ = ["CodecLanePool", "CodecLanePoolStats", "CodecLaneExhausted"]


class CodecLaneExhausted(RuntimeError):
    """Raised by :meth:`CodecLanePool.acquire` when no lane frees within the
    timeout. The serving layer maps this to a fast 429 (shed, don't queue)."""


@dataclass
class CodecLanePoolStats:
    """Point-in-time snapshot of codec-lane occupancy + saturation history."""

    capacity: int
    in_use: int
    available: int
    peak_in_use: int
    total_acquired: int
    total_rejected: int

    @property
    def saturated(self) -> bool:
        """True when every lane is currently held — the codec stage is the
        binding constraint right now."""
        return self.available == 0

    @property
    def utilization(self) -> float:
        """Fraction of lanes currently in use, in ``[0, 1]``."""
        return self.in_use / self.capacity if self.capacity else 0.0

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "codec_lane_capacity": self.capacity,
            "codec_lanes_in_use": self.in_use,
            "codec_lanes_available": self.available,
            "codec_lane_peak_in_use": self.peak_in_use,
            "codec_lane_total_acquired": self.total_acquired,
            "codec_lane_total_rejected": self.total_rejected,
            "codec_lane_saturated": self.saturated,
            "codec_lane_utilization": round(self.utilization, 4),
        }


class CodecLanePool:
    """Bounded acquire/release pool of codec decode lanes.

    Thread-safe: the vocoder stage acquires/releases from its worker thread(s)
    while admission/telemetry may read :meth:`stats` concurrently.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"codec lane capacity must be >= 1, got {capacity}")
        self._capacity = int(capacity)
        self._cond = threading.Condition(threading.Lock())
        self._in_use = 0
        self._peak_in_use = 0
        self._total_acquired = 0
        self._total_rejected = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def try_acquire(self) -> bool:
        """Non-blocking admission check: take a lane if one is free.

        Returns True on success (caller MUST :meth:`release` exactly once),
        False if the pool is saturated (caller sheds — fast 429 — rather than
        queueing into multi-second latency).
        """
        with self._cond:
            if self._in_use >= self._capacity:
                self._total_rejected += 1
                return False
            self._in_use += 1
            self._total_acquired += 1
            if self._in_use > self._peak_in_use:
                self._peak_in_use = self._in_use
            return True

    def acquire(self, timeout: float | None = None) -> None:
        """Blocking acquire (waits up to ``timeout`` seconds for a free lane).

        Raises :class:`CodecLaneExhausted` if no lane frees in time. Most
        admission paths should prefer :meth:`try_acquire`; this exists for
        callers that intentionally back-pressure with a short, bounded wait.
        """
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._in_use < self._capacity, timeout=timeout
            ):
                self._total_rejected += 1
                raise CodecLaneExhausted(
                    f"no codec lane available within {timeout}s "
                    f"(capacity={self._capacity}, in_use={self._in_use})"
                )
            self._in_use += 1
            self._total_acquired += 1
            if self._in_use > self._peak_in_use:
                self._peak_in_use = self._in_use

    def release(self) -> None:
        """Return one lane to the pool. Never drops below zero (idempotent
        against a double-release on an abort/teardown race)."""
        with self._cond:
            if self._in_use > 0:
                self._in_use -= 1
                self._cond.notify()

    @contextmanager
    def lane(self, *, blocking: bool = False, timeout: float | None = None) -> Iterator[bool]:
        """Context manager that holds a lane for the block's duration.

        With ``blocking=False`` (default) it yields the result of
        :meth:`try_acquire` — the body must check the yielded bool and shed if
        it is False. With ``blocking=True`` it waits (and raises
        :class:`CodecLaneExhausted` on timeout), always yielding True. The lane
        is released on exit, including on exception, so an aborted/raising
        decode never leaks a lane.
        """
        acquired = False
        try:
            if blocking:
                self.acquire(timeout=timeout)
                acquired = True
            else:
                acquired = self.try_acquire()
            yield acquired
        finally:
            if acquired:
                self.release()

    def stats(self) -> CodecLanePoolStats:
        with self._cond:
            return CodecLanePoolStats(
                capacity=self._capacity,
                in_use=self._in_use,
                available=self._capacity - self._in_use,
                peak_in_use=self._peak_in_use,
                total_acquired=self._total_acquired,
                total_rejected=self._total_rejected,
            )
