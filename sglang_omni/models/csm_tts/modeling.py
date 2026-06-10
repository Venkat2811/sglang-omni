# SPDX-License-Identifier: Apache-2.0
"""Framework-free torch modules for CSM-1B: frame embedding, codebooks head,
dense depth decoder (4L, static 33-position KV, 31-step inner AR).

These compose with an SGLang ``LlamaForCausalLM`` backbone in
:class:`sglang_omni.models.csm_tts.model.CsmTTSModel`. Everything here is
plain eager torch with static shapes — CUDA-graph-capturable in M4 (the
31-iteration host loop unrolls at capture; no data-dependent branches).

PLAN §1.4.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CsmFrameEmbedding(nn.Module):
    """Fused 32-codebook audio embedding: ONE ``weight [65632, 2048]`` table
    (32 * 2051 rows; tied to the depth decoder's embed table).

    Mirrors HF ``modeling_csm.py:655-666``: register
    ``audio_tokens_offsets = arange(32) * 2051`` as a buffer.
    """

    def __init__(
        self,
        num_codebooks: int = 32,
        codebook_vocab: int = 2051,
        hidden_size: int = 2048,
    ) -> None:
        """Owns ``weight: nn.Parameter [num_codebooks * codebook_vocab, hidden_size]``
        (= ``[65632, 2048]``, bf16 after model cast) and the int64
        ``audio_tokens_offsets [32]`` buffer."""
        super().__init__()
        raise NotImplementedError("skeleton — PLAN §1.4 CsmFrameEmbedding.__init__")

    def forward(self, codes_B32: torch.Tensor) -> torch.Tensor:
        """Frame embed: sum of per-codebook lookups.

        ``F.embedding(codes_B32 + audio_tokens_offsets).sum(-2)``.

        Args:
            codes_B32: int64 ``[B, 32]``, values in ``[0, 2050]``.

        Returns:
            ``[B, 2048]`` (weight dtype, bf16 in the ship config).
        """
        raise NotImplementedError("skeleton — PLAN §1.4 CsmFrameEmbedding.forward")

    def embed_codebook(self, codes_B: torch.Tensor, k: int) -> torch.Tensor:
        """Single-codebook lookup at table offset ``k * 2051`` (depth-step
        feedback: depth position ``p`` embeds the previous code with offset
        ``(p - 1) * 2051``).

        Args:
            codes_B: int64 ``[B]``, values in ``[0, 2050]``.
            k: codebook index in ``[0, 31]``.

        Returns:
            ``[B, 2048]`` (weight dtype).
        """
        raise NotImplementedError("skeleton — PLAN §1.4 CsmFrameEmbedding.embed_codebook")


class CsmCodebooksHead(nn.Module):
    """Per-depth-step output heads, ``weight [31, 1024, 2051]`` stored EXACTLY
    in checkpoint layout. Head *k* predicts codebook *k+1*.

    ⚠ Orientation trap (PLAN §1.4 / review F4): the forward is
    ``h @ weight[step]`` (≡ ``F.linear(h, weight[step].T)``, HF
    ``modeling_csm.py:528-536``). Do NOT "fix" shapes by re-storing the weight
    as ``[31, 2051, 1024]`` — combined with loading the checkpoint tensor
    ``(31, 1024, 2051)`` untransposed, that RUNS and produces garbage audio.
    The §6.1 tiny-config depth parity test pins this orientation and is a hard
    M1-exit blocker.
    """

    def __init__(
        self,
        num_heads: int = 31,
        hidden_size: int = 1024,
        codebook_vocab: int = 2051,
    ) -> None:
        """Owns ``weight: nn.Parameter [31, 1024, 2051]`` (checkpoint layout)."""
        super().__init__()
        raise NotImplementedError("skeleton — PLAN §1.4 CsmCodebooksHead.__init__")

    def forward(self, h_BD: torch.Tensor, step: int) -> torch.Tensor:
        """Logits for depth step ``step`` (predicting codebook ``step + 1``).

        Args:
            h_BD: ``[B, 1024]`` depth hidden (bf16).
            step: head index in ``[0, 30]``.

        Returns:
            ``[B, 2051]`` logits in the WEIGHT dtype — caller casts to fp32
            before sampling (PLAN §1.4 / R9).
        """
        raise NotImplementedError("skeleton — PLAN §1.4 CsmCodebooksHead.forward")


class CsmDepthDecoderLayer(nn.Module):
    """One depth-decoder layer: RMSNorm → SDPA attention (8 heads, head_dim
    128, 2 KV heads GQA, llama3-scaled rope) → RMSNorm → SwiGLU
    (1024 → 8192 → 1024).

    Attention is computed over the FIXED 33-length static KV with an additive
    causal/length mask — shape-static (CG-ready); at 33 positions the
    masked-full compute is cheaper than any varlen cleverness.
    """

    def __init__(self, depth_cfg: Any, layer_idx: int) -> None:
        """Dense q/k/v/o + gate/up/down projections + 2 RMSNorms; no stacking
        (own modules so the §1.6 manual weight copies stay 1:1)."""
        super().__init__()
        raise NotImplementedError("skeleton — PLAN §1.4 CsmDepthDecoderLayer.__init__")

    def forward(
        self,
        hidden_BTD: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_pos: int,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One layer forward over the static 33-position KV.

        Args:
            hidden_BTD: ``[B, T, 1024]`` bf16; T == 2 on the len-2 "depth
                prefill" (step 0+1), T == 1 thereafter.
            rope_cos / rope_sin: precomputed ``[33, 128]`` fp32 cos/sin slices
                for the positions being written.
            k_cache / v_cache: this layer's static-KV slot views
                ``[B, 2, 33, 128]`` bf16 (kv-heads-major), written in place at
                ``write_pos``.
            write_pos: first absolute depth position (0..32) being written.
            attn_mask: additive ``[T, 33]`` (or broadcastable) fp mask encoding
                causal + valid-length.

        Returns:
            ``[B, T, 1024]`` bf16.
        """
        raise NotImplementedError("skeleton — PLAN §1.4 CsmDepthDecoderLayer.forward")


class CsmDepthDecoder(nn.Module):
    """Dense depth decoder: 31-step inner AR per 80 ms frame.

    Owns ``inputs_embeds_projector [1024, 2048]`` (Linear 2048→1024, no bias),
    4 :class:`CsmDepthDecoderLayer`, final RMSNorm, a
    :class:`CsmCodebooksHead`, precomputed 33-position rope cos/sin, and the
    SLOT-INDEXED static KV ``k, v: [4, max_slots, 2, 33, 128]`` bf16.

    Slot- (not pool-) indexed by design (PLAN §2.2): the depth cache is
    frame-local — reset at the start of every frame, dead by the end — so it
    needs no per-request persistence and no rid→row gather/scatter; it lives
    in CG-batch-slot order ``[:bs]``. "Reset per frame" = reset a length
    scalar / mask, no memset.
    """

    def __init__(
        self,
        depth_cfg: Any,
        frame_embedding: CsmFrameEmbedding,
        max_slots: int,
    ) -> None:
        """``frame_embedding`` is SHARED with the outer model (tied table);
        ``max_slots`` sizes the static KV (= CG max batch size)."""
        super().__init__()
        raise NotImplementedError("skeleton — PLAN §1.4 CsmDepthDecoder.__init__")

    def generate_frame(
        self,
        h_B2048: torch.Tensor,
        cb0_B: torch.Tensor,
        temps_B: torch.Tensor,
        top_ks_B: torch.Tensor,
        *,
        bs: int,
    ) -> torch.Tensor:
        """Run the 31-step depth AR for one frame, batched across ``[:bs]``.

        Step 0+1: forward ``[proj(h), proj(embed_codebook(cb0, 0))]`` (len-2
        "depth prefill") → head ``W[0]`` → sample cb1. Then 30 more len-1
        steps: position ``p`` embeds the previous code with table offset
        ``(p - 1) * 2051``, head ``W[p - 1]`` samples ``cb_p`` — total 31
        sampled codes; cb31 is never forwarded (HF parity). Fixed
        31-iteration host loop, no data-dependent branches (CG-capturable as
        one graph in M4). Logits cast fp32 before sampling; per-row branchless
        greedy/top-k via :func:`sampler.sample_codes_batched`.

        Args:
            h_B2048: ``[B, 2048]`` bf16 post-norm backbone hidden.
            cb0_B: int64 ``[B]`` sampled codebook-0 codes.
            temps_B: fp32 ``[B]`` depth temperatures (CSM-private param set).
            top_ks_B: int64 ``[B]`` depth top-k buffer (K_MAX for neutral rows).
            bs: live CG batch size (buffers are sliced ``[:bs]``).

        Returns:
            int64 ``[B, 32]`` — full frame ``[cb0 | cb1..cb31]``.
        """
        raise NotImplementedError("skeleton — PLAN §1.4 CsmDepthDecoder.generate_frame")


__all__ = [
    "CsmCodebooksHead",
    "CsmDepthDecoder",
    "CsmDepthDecoderLayer",
    "CsmFrameEmbedding",
]
