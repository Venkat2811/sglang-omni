# SPDX-License-Identifier: Apache-2.0
"""Async-decode runner tests for csm_tts — LANDS WITH M4 (async stays off
through M3; ``async_decode_min_batch_size=2`` per the measured bs=1
regression).

PLAN §6.1, §2.6.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="csm_tts skeleton — async decode lands with M4 (PLAN §6.1/§7)"
)


def test_launch_resolve_vs_sync_parity_mixed_batches():
    """launch/resolve halves produce identical collect results to the sync
    path on mixed (running + finishing) batches."""
    raise NotImplementedError("PLAN §6.1 test_async: launch/resolve parity")


def test_next_token_ids_published_at_launch():
    """``result.next_token_ids = _cg_codes_BN[:n, 0].clamp_min(0)`` is set at
    launch with no host sync (STOP_CODE=-1 clamped into embed range)."""
    raise NotImplementedError("PLAN §6.1 test_async: next_token_ids at launch")


def test_overrun_guard_skips():
    """The three guards (padding-row reroute at launch; pre_finished snapshot
    + row drop in resolve; post-drain filter_batch) each skip exactly the
    overrun rows."""
    raise NotImplementedError("PLAN §6.1 test_async: overrun guards")


def test_bs1_eos_edge():
    """bs=1 request hitting EOS under lookahead: no double-collect, no
    emission after done."""
    raise NotImplementedError("PLAN §6.1 test_async: bs=1 EOS edge")


def test_real_pinned_memory_roundtrip():
    """One real-pinned-memory test: ping-pong host buffers round-trip the
    staging [P, 34] snapshot intact."""
    raise NotImplementedError("PLAN §6.1 test_async: pinned-memory roundtrip")
