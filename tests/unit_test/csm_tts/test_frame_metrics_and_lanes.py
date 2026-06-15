# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the CSM realtime-SLO + codec-admission machinery (O2).

CPU-only, dependency-light (these modules do not import torch / sglang):

- ``frame_metrics`` — inter-frame-latency (IFL) p50/p95/p99 + underrun_frac
  against the 80 ms frame clock, and the frame-clock breaking-point capacity
  model.
- ``codec_lane_pool`` — the codec stage's concurrency as a first-class
  acquire/release admission resource with saturation telemetry.
"""

from __future__ import annotations

import pytest

from sglang_omni.models.csm_tts.codec_lane_pool import (
    CodecLaneExhausted,
    CodecLanePool,
)
from sglang_omni.models.csm_tts.frame_metrics import (
    FRAME_PERIOD_MS,
    FrameLatencyTracker,
    frame_clock_stream_ceiling,
)


# --- IFL / frame metrics ----------------------------------------------------


def test_frame_period_is_80ms() -> None:
    assert FRAME_PERIOD_MS == pytest.approx(80.0)


def _feed(tracker: FrameLatencyTracker, gaps_ms: list[float]) -> None:
    """Feed absolute timestamps so each recorded delta is exactly gaps_ms[i]."""
    ts = 0.0
    tracker.record_frame(now=ts / 1000.0)  # anchor, no delta
    for gap in gaps_ms:
        ts += gap
        tracker.record_frame(now=ts / 1000.0)


def test_ifl_percentiles_and_underrun() -> None:
    tracker = FrameLatencyTracker(window=64, frame_period_ms=80.0)
    # Nine on-budget frames (40 ms) and one underrun (200 ms > 80 ms clock).
    _feed(tracker, [40.0] * 9 + [200.0])
    stats = tracker.snapshot()
    assert stats.samples == 10
    assert stats.p50_ms == pytest.approx(40.0)
    assert stats.max_ms == pytest.approx(200.0)
    assert stats.underrun_frac == pytest.approx(0.1)
    assert stats.p99_ms > FRAME_PERIOD_MS  # tail breaches the frame clock
    assert stats.realtime_ok is False
    d = stats.to_dict()
    for key in ("ifl_p50_ms", "ifl_p99_ms", "underrun_frac", "frame_period_ms"):
        assert key in d


def test_ifl_realtime_ok_when_under_clock() -> None:
    tracker = FrameLatencyTracker(frame_period_ms=80.0)
    _feed(tracker, [50.0] * 20)  # every frame inside the 80 ms budget
    stats = tracker.snapshot()
    assert stats.samples == 20
    assert stats.underrun_frac == 0.0
    assert stats.realtime_ok is True


def test_ifl_strided_emission_is_amortized() -> None:
    # One chunk decoding 4 frames over 320 ms ⇒ 80 ms/frame each (comparable
    # to single-frame emission for the underrun test).
    tracker = FrameLatencyTracker(frame_period_ms=80.0)
    tracker.record_frame(now=0.0)
    tracker.record_frame(frames=4, now=0.320)
    stats = tracker.snapshot()
    assert stats.samples == 4
    assert stats.p50_ms == pytest.approx(80.0)


def test_ifl_window_is_bounded() -> None:
    tracker = FrameLatencyTracker(window=8, frame_period_ms=80.0)
    _feed(tracker, [40.0] * 100)
    assert tracker.snapshot().samples == 8  # only the recent tail is kept


def test_ifl_reset_clears_anchor() -> None:
    tracker = FrameLatencyTracker(frame_period_ms=80.0)
    _feed(tracker, [40.0] * 5)
    assert tracker.snapshot().samples == 5
    tracker.reset()
    assert tracker.snapshot().samples == 0


def test_empty_tracker_snapshot() -> None:
    stats = FrameLatencyTracker().snapshot()
    assert stats.samples == 0
    assert stats.realtime_ok is False  # no evidence of realtime health yet


def test_frame_clock_stream_ceiling() -> None:
    assert frame_clock_stream_ceiling(8.0, frame_period_ms=80.0) == 10
    assert frame_clock_stream_ceiling(80.0, frame_period_ms=80.0) == 1
    assert frame_clock_stream_ceiling(200.0, frame_period_ms=80.0) == 1  # >= 1
    with pytest.raises(ValueError):
        frame_clock_stream_ceiling(0.0)


# --- codec lane pool --------------------------------------------------------


def test_lane_pool_rejects_when_capacity_invalid() -> None:
    with pytest.raises(ValueError):
        CodecLanePool(0)


def test_lane_pool_try_acquire_saturation_and_release() -> None:
    pool = CodecLanePool(2)
    assert pool.try_acquire() is True
    assert pool.try_acquire() is True
    assert pool.try_acquire() is False  # saturated
    stats = pool.stats()
    assert stats.saturated is True
    assert stats.in_use == 2
    assert stats.available == 0
    assert stats.peak_in_use == 2
    assert stats.total_acquired == 2
    assert stats.total_rejected == 1
    assert stats.utilization == pytest.approx(1.0)
    pool.release()
    assert pool.try_acquire() is True  # a lane freed up


def test_lane_pool_release_never_underflows() -> None:
    pool = CodecLanePool(1)
    pool.release()  # release with nothing held is a no-op
    pool.release()
    assert pool.stats().in_use == 0


def test_lane_context_manager_non_blocking() -> None:
    pool = CodecLanePool(1)
    pool.try_acquire()  # exhaust
    with pool.lane() as acquired:
        assert acquired is False  # caller must shed
    # The failed try_acquire must not have leaked or double-released a lane.
    assert pool.stats().in_use == 1


def test_lane_context_manager_releases_on_exception() -> None:
    pool = CodecLanePool(1)
    with pytest.raises(RuntimeError):
        with pool.lane(blocking=True, timeout=0.5) as acquired:
            assert acquired is True
            raise RuntimeError("boom")
    assert pool.stats().in_use == 0  # released despite the exception


def test_lane_pool_blocking_acquire_times_out() -> None:
    pool = CodecLanePool(1)
    pool.try_acquire()
    with pytest.raises(CodecLaneExhausted):
        pool.acquire(timeout=0.01)
    # A timed-out acquire counts as a rejection (admission shed).
    assert pool.stats().total_rejected == 1
