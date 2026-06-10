# SPDX-License-Identifier: Apache-2.0
"""Unit tests for csm_tts.sampler (CPU, no GPU).

PLAN §6.1.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="csm_tts skeleton — sampler not implemented yet (PLAN §1.5/§6.1)"
)


def test_step_reference_vs_frame_finalize_direct_parity():
    """Per-row ``step_reference`` vs batched ``frame_finalize_direct`` across
    phases: running / eos_now / done-freeze / mixed batch."""
    raise NotImplementedError("PLAN §6.1 test_sampler: reference-vs-batched parity")


def test_eos_uses_cb0_to_cb30_only():
    """EOS check covers codebooks 0..30 only — a frame with cb0..30 == 0 and
    cb31 != 0 STILL stops (HF-exact stop(31) semantics)."""
    raise NotImplementedError("PLAN §6.1 test_sampler: cb31-excluded EOS")


def test_stop_sentinel_freeze():
    """Done rows emit STOP_CODE=-1 and stay frozen on subsequent frames."""
    raise NotImplementedError("PLAN §6.1 test_sampler: STOP freeze")


def test_greedy_short_circuit_determinism():
    """temp=0 / top_k=1 rows short-circuit branchlessly to argmax over RAW
    logits — bitwise deterministic across calls."""
    raise NotImplementedError("PLAN §6.1 test_sampler: greedy determinism")


def test_fixed_shape_topk_filter_vs_naive():
    """Fixed-shape ``topk(K_MAX)`` + per-row k-th-value gather matches a naive
    per-row top-k filter for mixed per-row k."""
    raise NotImplementedError("PLAN §6.1 test_sampler: fixed-shape top-k")


def test_forced_frame0_eos_zero_decode_lifecycle():
    """Forced frame-0 EOS (synthetic logits driving cb0..30 → 0 on the
    prefill path): request finishes AT PREFILL, nothing is emitted to the
    vocoder, finish reason set (F7 — the zero-decode lifecycle)."""
    raise NotImplementedError("PLAN §6.1 test_sampler: frame-0 EOS (F7)")
