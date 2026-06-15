# SPDX-License-Identifier: Apache-2.0
"""Inter-frame-latency (IFL) SLO instrumentation for the CSM streaming path.

A streaming-TTS server is realtime-bound by its per-frame wall clock, NOT by
VRAM: one Mimi frame is 80 ms = 1920 samples at 12.5 Hz, so the audio consumer
drains one frame every ``FRAME_PERIOD_MS``. A stream *underruns* (audible
stutter / buffer starvation) whenever the serial work to produce the next
frame's audio takes longer than that budget. Aggregate throughput hides this:
the meaningful realtime SLO signal is the distribution of **inter-frame
latency** — the wall-clock gap between successive emitted audio frames — and
the **underrun fraction**, the share of frames whose gap exceeds the frame
clock.

This module is the framework-native (sgl-omni ``ServerArgs``-shaped) home for
that signal. It is intentionally model-scoped and dependency-light: a small
ring buffer of per-frame deltas plus tail-percentile reduction, suitable for
exposing through the stage's existing operational-telemetry surface.

Reporting convention (so cross-host / cross-engine comparisons stay
apples-to-apples): callers that compare two transports should report IFL
**excess over each engine's own single-stream IFL floor**, not raw IFL — the
floor folds in the engine's fixed per-frame compute, and only the excess
reflects scheduling/transport pressure.

R0 §8 (IFL underrun metric) + §9 (frame-clock breaking-point model).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

__all__ = [
    "FRAME_PERIOD_MS",
    "FRAME_RATE_HZ",
    "FrameLatencyTracker",
    "InterFrameLatencyStats",
    "frame_clock_stream_ceiling",
]

# Mimi runs at 12.5 Hz (one 80 ms frame). Inlined here (the same dependency-
# avoidance pattern audio_codec.py / config.py use) so this metrics module
# stays import-light — it does not pull the model/codec stack just to compute
# latency percentiles.
FRAME_RATE_HZ: float = 12.5

# One audio frame is 1/12.5 s = 80 ms. This is the realtime deadline a stream
# must beat per emitted frame; it is the denominator of the underrun test and
# the numerator of the breaking-point capacity model below.
FRAME_PERIOD_MS: float = 1000.0 / FRAME_RATE_HZ


def _percentile(sorted_ms: list[float], q: float) -> float:
    """Linear-interpolated percentile over an already-sorted list (q in
    ``[0, 1]``). Empty → 0.0; used for IFL p50/p95/p99."""
    n = len(sorted_ms)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_ms[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_ms[lo] * (1.0 - frac) + sorted_ms[hi] * frac


@dataclass
class InterFrameLatencyStats:
    """A reduced IFL snapshot over a window of emitted frames.

    All latencies are milliseconds. ``underrun_frac`` is the fraction of
    sampled inter-frame gaps that exceeded ``frame_period_ms`` — the realtime
    SLO breach rate. ``p99 > frame_period_ms`` is the headline "audible
    stutter" condition.
    """

    samples: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    underrun_frac: float = 0.0
    frame_period_ms: float = FRAME_PERIOD_MS

    @property
    def realtime_ok(self) -> bool:
        """True when the p99 inter-frame gap stays within the frame clock —
        the streaming-audio realtime SLO is met across the window."""
        return self.samples > 0 and self.p99_ms <= self.frame_period_ms

    def to_dict(self) -> dict[str, float | int | bool]:
        """Telemetry-friendly dict (mirrors the per-stream metrics shape so it
        slots into the existing ``/v1/streams/{id}/metrics`` surface)."""
        return {
            "ifl_samples": self.samples,
            "ifl_p50_ms": round(self.p50_ms, 3),
            "ifl_p95_ms": round(self.p95_ms, 3),
            "ifl_p99_ms": round(self.p99_ms, 3),
            "ifl_max_ms": round(self.max_ms, 3),
            "ifl_mean_ms": round(self.mean_ms, 3),
            "underrun_frac": round(self.underrun_frac, 4),
            "frame_period_ms": round(self.frame_period_ms, 3),
            "realtime_ok": self.realtime_ok,
        }


@dataclass
class FrameLatencyTracker:
    """Records per-frame emission timestamps and reduces them to IFL stats.

    Thread-safe (the vocoder stage may emit from a worker thread while a
    telemetry endpoint snapshots concurrently). Bounded ring buffer so a long
    stream cannot grow memory without limit — the realtime signal lives in the
    tail of a rolling window, not the full history (R0 §7 lesson (b): keep the
    underrun window short so a recovering engine clears it in seconds).

    Usage::

        tracker = FrameLatencyTracker()
        # ... each time a frame's audio is handed to the consumer:
        tracker.record_frame()   # call once per emitted 80 ms frame
        stats = tracker.snapshot()
    """

    window: int = 512
    frame_period_ms: float = FRAME_PERIOD_MS
    _deltas_ms: list[float] = field(default_factory=list)
    _last_ts: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_frame(self, *, frames: int = 1, now: float | None = None) -> None:
        """Mark ``frames`` audio frames as emitted at wall time ``now``.

        When a single ``on_stream_chunk`` emits more than one frame (the codec
        decodes a stride of frames at once), the wall gap is amortized evenly
        across the frames in the chunk so each frame's effective IFL is
        comparable to the single-frame case — this is what makes the underrun
        test honest for strided emission.
        """
        if frames <= 0:
            return
        ts = time.monotonic() if now is None else now
        with self._lock:
            if self._last_ts is not None:
                gap_ms = (ts - self._last_ts) * 1000.0
                per_frame = gap_ms / frames
                for _ in range(frames):
                    self._append_locked(per_frame)
            self._last_ts = ts

    def _append_locked(self, delta_ms: float) -> None:
        self._deltas_ms.append(delta_ms)
        if len(self._deltas_ms) > self.window:
            # Drop oldest; the realtime signal is the recent tail.
            del self._deltas_ms[: len(self._deltas_ms) - self.window]

    def reset(self) -> None:
        """Clear the window and the last-timestamp anchor (call on a new
        stream so the first frame of a stream is never counted as a gap)."""
        with self._lock:
            self._deltas_ms.clear()
            self._last_ts = None

    def snapshot(self) -> InterFrameLatencyStats:
        """Reduce the current window to an :class:`InterFrameLatencyStats`."""
        with self._lock:
            deltas = list(self._deltas_ms)
        if not deltas:
            return InterFrameLatencyStats(frame_period_ms=self.frame_period_ms)
        ordered = sorted(deltas)
        n = len(ordered)
        underruns = sum(1 for d in ordered if d > self.frame_period_ms)
        return InterFrameLatencyStats(
            samples=n,
            p50_ms=_percentile(ordered, 0.50),
            p95_ms=_percentile(ordered, 0.95),
            p99_ms=_percentile(ordered, 0.99),
            max_ms=ordered[-1],
            mean_ms=sum(ordered) / n,
            underrun_frac=underruns / n,
            frame_period_ms=self.frame_period_ms,
        )


def frame_clock_stream_ceiling(
    per_stream_per_frame_ms: float,
    *,
    frame_period_ms: float = FRAME_PERIOD_MS,
) -> int:
    """Frame-clock breaking-point capacity (R0 §9).

    A streaming-TTS server's realtime ceiling — the number of concurrent
    streams it can keep above water — is bounded by how much serial per-frame
    work fits inside one frame budget::

        concurrent streams  ~=  frame_period_ms / per_stream_per_frame_ms

    This is a *first-order* model (it ignores batching speedups that lower the
    effective per-stream term and the codec-lane-pool hard cap layered on top
    by :mod:`sglang_omni.models.csm_tts.codec_lane_pool`); it is the right
    yardstick for reading a per-frame latency delta as a change in serveable
    concurrency. Returns at least 1 for any positive per-frame cost.
    """
    if per_stream_per_frame_ms <= 0.0:
        raise ValueError("per_stream_per_frame_ms must be > 0")
    return max(1, int(frame_period_ms // per_stream_per_frame_ms))
