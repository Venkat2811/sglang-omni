# SPDX-License-Identifier: Apache-2.0
"""SGLang-integrated CSM-1B model: paged Llama backbone + cb0 head + dense
depth decoder, with the 31-step depth loop INSIDE the scheduler-visible
decode step (1 frame = 1 backbone position = 1 SGLang "token").

Registered as ``CsmForConditionalGeneration`` in
:meth:`sglang_omni.model_runner.sglang_model_runner.SGLModelRunner._register_omni_model`.

The backbone runs under SGLang paged/varlen attention (composed
``LlamaForCausalLM``) — no dense left-padded batch, which sidesteps the known
bf16 B>=2 left-pad KV-corruption trap by construction (PLAN header, §8.R9).

PLAN §1.7, §2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.llama import LlamaForCausalLM
from torch import nn

from sglang_omni.models.csm_tts.hf_config import (
    CsmTtsHfConfig,
    build_backbone_llama_config,
    build_depth_decoder_config,
)
from sglang_omni.models.csm_tts.modeling import (
    CsmCodebooksHead,
    CsmDepthDecoder,
    CsmFrameEmbedding,
)
from sglang_omni.models.csm_tts.sampler import (
    CsmBatchedSamplerState,
    frame_finalize_direct,
    sample_codes_batched,
)
from sglang_omni.models.csm_tts.weight_loader import CsmWeightMapper


@dataclass
class CsmGenParams:
    """Per-request generation params — TWO sampling sets (PLAN §2.3, §2.4):
    cb0 rides SGLang's ``sampling_info``; the depth set is CSM-private
    (defaults (0.9, 50), HF parity) and travels via ``req._omni_data``."""

    temperature: float = 0.9
    top_k: int = 50
    top_p: float | None = None
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    seed: int | None = None


def _flat_sampling_attr(sampling_info: Any, attr: str) -> list | None:
    """Read a flat per-row attribute off SGLang's ``sampling_info`` (one D2H
    per attribute; higgs pattern). Returns ``None`` when absent."""
    raise NotImplementedError("skeleton — PLAN §1.7/§2.1 _flat_sampling_attr")


class CsmTTSModel(nn.Module):
    """CSM-1B for SGLang: ``backbone`` (paged LlamaForCausalLM, 16L/2048/
    32h/8kv/hd64, ctx 2048, tie_word_embeddings=True so the never-used text
    lm_head is free), ``frame_embedding`` ([65632, 2048] fused audio table),
    ``codebook0_head`` (Linear 2048→2051, no bias — loaded from the ckpt's
    ``lm_head.weight``), and ``depth_decoder`` (4L dense, slot-indexed static
    KV ``[4, max_slots, 2, 33, 128]`` bf16).

    Sampler pool: :class:`CsmBatchedSamplerState` with
    ``pool_size = max_batch_size + 1`` (last row = padding row);
    ``_rid_to_row`` / ``_free_rows`` / ``acquire_row`` / ``release_row`` /
    ``reset_request`` follow higgs verbatim.

    CG shadow buffers (pool-sized, sliced ``[:bs]`` in-graph):
    ``_cg_row_indices`` (long), ``_cg_temperature`` / ``_cg_top_p`` (fp32),
    ``_cg_top_k_buf`` (long, = K_MAX neutral), ``_cg_depth_temperature``
    (fp32) + ``_cg_depth_top_k_buf`` (long) — the CSM-delta second param set,
    ``_cg_codes_BN`` (long [P, 32]), ``_cg_collect_staging`` (long [P, 34] =
    ``c0..c31 | was_done | generation_done``), ``_cg_was_done`` (bool),
    ``_cg_active_generation_done`` (bool), ``_cg_active_last_codes``
    (long [P, 32]).

    PLAN §1.7.
    """

    def __init__(
        self,
        config: CsmTtsHfConfig,
        quant_config: Any | None = None,
        prefix: str = "",
        max_batch_size: int = 64,
    ) -> None:
        """Build backbone via ``LlamaForCausalLM(build_backbone_llama_config(
        config), quant_config, prefix=add_prefix("backbone", prefix))`` (ctor
        verified upstream: ``(config, quant_config=None, prefix="")``); build
        frame_embedding / codebook0_head / depth_decoder and cast them to
        backbone bf16 (~1 ULP/step note); allocate the sampler pool + CG
        shadow buffers; ``_output_codes: dict[rid, list[Tensor[32]]]``."""
        super().__init__()
        raise NotImplementedError("skeleton — PLAN §1.7 CsmTTSModel.__init__")

    # --- sampler-pool row lifecycle (higgs verbatim) -------------------------

    def acquire_row(self, req_id: str) -> int:
        """Bind ``req_id`` to a free pool row (reset on acquire). Returns row."""
        raise NotImplementedError("skeleton — PLAN §1.7 CsmTTSModel.acquire_row")

    def release_row(self, req_id: str) -> None:
        """Return the row bound to ``req_id`` to the free list (idempotent)."""
        raise NotImplementedError("skeleton — PLAN §1.7 CsmTTSModel.release_row")

    def reset_request(self, req_id: str) -> None:
        """Abort/result-adapter hook: release the sampler row + drop
        ``_output_codes[req_id]`` (wired as ``OmniScheduler.abort_callback``)."""
        raise NotImplementedError("skeleton — PLAN §1.7 CsmTTSModel.reset_request")

    def get_output_codes(self, req_id: str) -> torch.Tensor:
        """All frames emitted so far for ``req_id``.

        Returns:
            int64 CPU ``[F, 32]`` (includes the EOS frame — HF ``sequences``
            parity; the vocoder never receives it, PLAN §2.5).
        """
        raise NotImplementedError("skeleton — PLAN §1.7 CsmTTSModel.get_output_codes")

    # --- forward -------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        """One scheduler step (PLAN §1.7, §2.1).

        Decode: ``input_embeds = _decode_step_embeds_cg(bs)``; prefill: the
        runner supplies overlay embeds. Then ``hidden = self.backbone.model(
        input_ids, positions, forward_batch, input_embeds)`` — SGLang's
        LlamaModel applies the final RMSNorm (upstream llama.py:400), giving
        the POST-norm hidden CSM's depth conditioning requires. Last-token
        hidden via ``cumsum(extend_seq_lens) - 1`` on prefill. Then
        ``decode_codebooks_batch_cg(hidden)`` (decode) or
        ``decode_codebooks_batch(hidden, req_ids, gen_params)`` (prefill —
        frame 0 IS sampled at prefill, like higgs).

        Args:
            input_ids: int64 ``[T_total]`` (decode: ``[bs]``).
            positions: int64 ``[T_total]``.
            forward_batch: SGLang ``ForwardBatch``.
            input_embeds: optional ``[T_total, 2048]`` bf16 overlay.

        Returns:
            Dummy ``LogitsProcessorOutput(next_token_logits=zeros(bs, 128256,
            fp32))`` — cb0 is published via the runner, logit processors run
            over zeros harmlessly (R1c).
        """
        raise NotImplementedError("skeleton — PLAN §1.7 CsmTTSModel.forward")

    def _decode_step_embeds_cg(self, bs: int) -> torch.Tensor:
        """UNCONDITIONAL ``frame_embedding(_cg_active_last_codes[:bs])`` —
        post-prefill a last frame always exists (no higgs ``delay_count>0``
        text fallback; padding rows embed zeros-garbage that the collect
        discards).

        Returns:
            bf16 ``[bs, 2048]``.
        """
        raise NotImplementedError("skeleton — PLAN §1.7 CsmTTSModel._decode_step_embeds_cg")

    def decode_codebooks_batch_cg(self, hidden_BD: torch.Tensor) -> torch.Tensor:
        """CG decode tail: ``logits0 = codebook0_head(hidden).float()`` →
        ``cb0 = sample_codes_batched(logits0, _cg_temperature, _cg_top_k_buf)``
        → ``codes = depth_decoder.generate_frame(hidden, cb0,
        _cg_depth_temperature, _cg_depth_top_k_buf, bs=bs)`` →
        :func:`frame_finalize_direct` → write ``_cg_codes_BN`` /
        ``_cg_was_done`` / ``_cg_active_*``. No host control flow, no D2H.

        Args:
            hidden_BD: bf16 ``[bs, 2048]`` post-norm backbone hidden.

        Returns:
            int64 ``[bs, 32]`` finalized frame codes (STOP_CODE on frozen rows).
        """
        raise NotImplementedError("skeleton — PLAN §1.7 CsmTTSModel.decode_codebooks_batch_cg")

    def decode_codebooks_batch(
        self,
        hidden_BD: torch.Tensor,
        req_ids: list[str],
        gen_params: list[CsmGenParams],
    ) -> torch.Tensor:
        """Eager pool-indexed variant for PREFILL (rows via ``acquire_row``,
        params from request data). Samples frame 0 (cb0 + full depth loop on
        the last prompt position's hidden) and runs the SAME
        :func:`frame_finalize_direct` EOS machine — frame-0 EOS is legal HF
        behavior and empirically real for this checkpoint (PLAN §1.7 / F7):
        on EOS at prefill → ``_mark_sampler_finished`` immediately, emit
        NOTHING to the vocoder, zero-decode lifecycle. Appends to
        ``_output_codes[rid]``.

        Args:
            hidden_BD: bf16 ``[n_req, 2048]`` last-prompt-position hiddens.
            req_ids: rids aligned with rows of ``hidden_BD``.
            gen_params: per-request dual sampling param sets.

        Returns:
            int64 ``[n_req, 32]`` frame-0 codes.
        """
        raise NotImplementedError("skeleton — PLAN §1.7 CsmTTSModel.decode_codebooks_batch")

    # --- weights -------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Split the checkpoint stream per :class:`CsmWeightMapper` (§1.6):
        backbone names go through ``LlamaForCausalLM.load_weights`` (qkv /
        gate_up stacking + tied-lm_head skip), own params are copied manually
        with shape checks, everything cast to backbone bf16. Asserts the
        M1 census: 538 tensors mapped-or-deliberately-skipped, zero
        unexpected (the depth tied-embed duplicate counts as skipped).

        Returns:
            Set of destination param names that received a tensor.
        """
        raise NotImplementedError("skeleton — PLAN §1.7/§1.6 CsmTTSModel.load_weights")


__all__ = ["CsmGenParams", "CsmTTSModel", "_flat_sampling_attr"]
