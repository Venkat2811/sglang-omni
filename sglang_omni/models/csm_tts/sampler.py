# SPDX-License-Identifier: Apache-2.0
"""Branchless batched sampling + EOS state machine for CSM.

Simpler than higgs by design: CSM has no delay pattern and no EOC wind-down,
so the three-phase higgs machine collapses to one transition
(``eos_now = all(codes[:, :31] == 0)``; done rows freeze and emit
``STOP_CODE``). Dual sampling param sets: cb0 (rides SGLang's sampling_info)
and depth (CSM-private, from ``req._omni_data``).

PLAN §1.5, §2.5.
"""

from __future__ import annotations

import torch

from sglang_omni.models.csm_tts.utils import (
    CODEBOOK_EOS,
    K_MAX,
    NUM_CODEBOOKS,
    STOP_CODE,
)

__all__ = [
    "CsmBatchedSamplerState",
    "K_MAX",
    "frame_finalize_direct",
    "sample_codes_batched",
    "step_reference",
]


class CsmBatchedSamplerState:
    """Pool-row sampler state surviving across decode steps (higgs row
    discipline: rows are acquired/released by rid while batch composition
    changes under a fixed-shape CUDA graph).

    Per row (pool size ``P``; last row = padding row):

    - ``last_codes``: int64 ``[P, 32]`` — previous frame, feeds
      ``CsmFrameEmbedding`` on the next decode step.
    - ``generation_done``: bool ``[P]`` — EOS latched; row frozen.
    - ``frames_emitted``: int32 ``[P]`` — frames appended so far.

    No ``delay_count`` / ``eoc_countdown`` — CSM has neither (PLAN §2.4).
    """

    def __init__(self, pool_size: int, device: torch.device | str = "cuda") -> None:
        """Allocate ``last_codes [P, 32]`` int64, ``generation_done [P]`` bool,
        ``frames_emitted [P]`` int32 on ``device``."""
        raise NotImplementedError("skeleton — PLAN §1.5 CsmBatchedSamplerState.__init__")

    def reset_row(self, row: int) -> None:
        """Reset one pool row to the fresh-request state
        (``last_codes=0``, ``generation_done=False``, ``frames_emitted=0``)."""
        raise NotImplementedError("skeleton — PLAN §1.5 CsmBatchedSamplerState.reset_row")


def sample_codes_batched(
    logits_BV_fp32: torch.Tensor,
    temperature_B: torch.Tensor,
    top_k_buf_B: torch.Tensor,
    top_p_B: torch.Tensor | None = None,
) -> torch.Tensor:
    """Branchless per-row greedy/top-k sampling (higgs
    ``_sample_independent_batched`` with ``K_MAX=2051``).

    Greedy rows (``temperature <= 1e-5`` or ``top_k == 1``) short-circuit
    branchlessly to argmax over RAW logits; sampled rows use a fixed-shape
    ``topk(K_MAX)`` + per-row k-th-value gather (CG-deterministic at temp=0).
    Used for BOTH cb0 and every one of the 31 depth steps.

    Args:
        logits_BV_fp32: fp32 ``[B, 2051]`` (callers cast before this).
        temperature_B: fp32 ``[B]``.
        top_k_buf_B: int64 ``[B]`` (neutral rows carry ``K_MAX``).
        top_p_B: optional fp32 ``[B]``.

    Returns:
        int64 ``[B]`` sampled codes in ``[0, 2050]``.
    """
    raise NotImplementedError("skeleton — PLAN §1.5 sample_codes_batched")


def frame_finalize_direct(
    codes_B32: torch.Tensor,
    generation_done_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The branchless EOS machine (single transition; PLAN §2.5).

    ``eos_now = (codes[:, :31] == CODEBOOK_EOS).all(-1)`` — cb31 EXCLUDED,
    HF-exact (stop(31)/trim(32) inconsistency is mirrored downstream);
    ``new_done = done | eos_now``; ``out_codes = where(was_done, STOP_CODE,
    codes)``; done rows freeze. Runs on decode steps AND on the
    prefill-sampled frame 0 (frame-0 EOS is legal + empirically real,
    PLAN §1.7 / F7).

    Args:
        codes_B32: int64 ``[B, 32]`` freshly sampled frame.
        generation_done_B: bool ``[B]`` done flags BEFORE this frame.

    Returns:
        ``(out_codes_B32 int64 [B, 32], new_done_B bool [B],
        was_done_B bool [B])``.
    """
    raise NotImplementedError("skeleton — PLAN §1.5 frame_finalize_direct")


def step_reference(
    codes_32: torch.Tensor,
    generation_done: bool,
) -> tuple[torch.Tensor, bool, bool]:
    """Per-row eager mirror of :func:`frame_finalize_direct` for parity tests
    (higgs pattern: tests assert per-row vs batched equality across
    running / eos_now / done-freeze / mixed-batch phases).

    Args:
        codes_32: int64 ``[32]``.
        generation_done: row done flag before this frame.

    Returns:
        ``(out_codes_32 int64 [32], new_done bool, was_done bool)``.
    """
    raise NotImplementedError("skeleton — PLAN §1.5 step_reference")
