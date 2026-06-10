# SPDX-License-Identifier: Apache-2.0
"""Unit tests for csm_tts.modeling (CPU, no GPU, no sglang engine).

PLAN §6.1. The depth-decoder greedy parity test is a HARD M1-exit blocker —
it pins the position/offset/head off-by-ones (the #1 bug class here) and the
§1.4 codebooks-head orientation (review F4) before any GPU work.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="csm_tts skeleton — modeling not implemented yet (PLAN §1.4/§6.1)"
)


def test_frame_embedding_matches_hand_rolled_reference():
    """``CsmFrameEmbedding.forward(codes_B32)`` equals the hand-rolled
    ``Σ_k embed[c_k + k*2051]`` over the one [65632, 2048] table (random
    weights, fp32, exact match)."""
    raise NotImplementedError("PLAN §6.1 test_modeling: frame-embed compose")


def test_embed_codebook_offset_correctness():
    """``embed_codebook(codes_B, k)`` reads rows at exactly ``codes + k*2051``
    for every k in 0..31."""
    raise NotImplementedError("PLAN §6.1 test_modeling: embed_codebook offsets")


def test_codebooks_head_index_mapping():
    """``CsmCodebooksHead`` head *k* predicts codebook *k+1*: forward(h, k)
    equals ``h @ weight[k]`` with weight kept in checkpoint layout
    ``[31, 1024, 2051]`` (orientation trap, review F4)."""
    raise NotImplementedError("PLAN §6.1 test_modeling: codebooks-head mapping")


def test_depth_decoder_greedy_parity_vs_hf():
    """Single-frame greedy parity vs HF ``CsmDepthDecoderForCausalLM.generate``
    (tiny-config random weights, temp=0, both fp32, EXACT 31-code match).
    Hard M1-exit blocker (PLAN §7 M1)."""
    raise NotImplementedError("PLAN §6.1 test_modeling: depth greedy parity (M1 blocker)")
