# SPDX-License-Identifier: Apache-2.0
"""Frame-step ModelRunner for CSM TTS.

The scheduler sees a normal one-token decode step (1 frame = 1 backbone
position = 1 SGLang "token"); the 31-step depth loop runs inside
``model.forward``'s decode branch (PLAN §2.1). This runner owns everything
OUTSIDE the (future-M4) captured region: CG-buffer population, prefill embed
overlay, the packed one-D2H-per-step collect, stream emission, and the
async launch/resolve halves.

Step anatomy (decode, sync path — PLAN §2.1):

1. ``before_decode → _populate_cg_buffers``: reset padding row; acquire rows;
   lookahead-only GPU-side done-row reroute to the padding row (higgs guard
   #1); fill cb0 sampling buffers from SGLang ``sampling_info`` via
   ``_flat_sampling_attr``; fill DEPTH sampling buffers host-side from
   ``req._omni_data.data`` (depth params never ride sampling_info); gather
   ``pool.{generation_done, last_codes}[rows] → _cg_active_*[:bs]``.
2. ``model.forward`` (see model.py).
3. ``post_decode → _collect_step_outputs_cg``: ``_decode_pack_gpu`` (scatter
   ``_cg_active_* → pool[rows]``; pack staging ``[P, 34]``) → ONE blocking
   D2H ``staging[:n].cpu()`` → ``_decode_collect_host``.

All three async overrun guards stay verbatim from higgs (padding-row reroute
at launch; ``pre_finished`` snapshot + row drop in resolve; post-drain
``filter_batch()``) — they protect the lookahead overlap, not the wind-down
(PLAN §2.5). Async stays off until M4; ``async_decode_min_batch_size=2``
(the measured bs=1 regression).

PLAN §1.11, §2.
"""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.csm_tts.model import _flat_sampling_attr
from sglang_omni.models.csm_tts.sampler import K_MAX
from sglang_omni.models.csm_tts.utils import (
    AUDIO_EOS_TOKEN_ID,
    AUDIO_TOKEN_ID,
    CODEBOOK_EOS,
)
from sglang_omni.scheduling.messages import OutgoingMessage

__all__ = ["CsmTTSModelRunner"]


class CsmTTSModelRunner(ModelRunner):
    """ModelRunner for :class:`~sglang_omni.models.csm_tts.model.CsmTTSModel`."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        """Mirror higgs: keep ``tp_worker`` / ``output_processor``, a lazy
        ``_outbox`` and ``_vocoder_target = "vocoder"``."""
        super().__init__(tp_worker, output_processor)
        raise NotImplementedError("skeleton — PLAN §1.11 CsmTTSModelRunner.__init__")

    def set_stream_outbox(self, outbox: Any) -> None:
        """Wire the OmniScheduler outbox used by :meth:`_emit_code_chunk`."""
        raise NotImplementedError("skeleton — PLAN §1.11 set_stream_outbox")

    # --- prefill --------------------------------------------------------------

    def before_prefill(self, forward_batch: Any, schedule_batch: Any, requests: list) -> None:
        """Stamp ``forward_batch.req_ids`` and install the embed overlay via
        :meth:`_build_prefill_input_embeds`."""
        raise NotImplementedError("skeleton — PLAN §1.11 before_prefill")

    def post_prefill(self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list) -> None:
        """Collect the prefill-sampled frame 0 via :meth:`_collect_step_outputs`."""
        raise NotImplementedError("skeleton — PLAN §1.11 post_prefill")

    def _build_prefill_input_embeds(
        self, forward_batch: Any, requests: list
    ) -> torch.Tensor | None:
        """Prefill embed overlay (PLAN §2.4 row).

        ``embed_tokens`` of the token ids, then paste context-frame embeds at
        the REAL-id ``AUDIO_TOKEN_ID`` (128002) positions (no -100 sentinel),
        and give each ``AUDIO_EOS_TOKEN_ID`` (128003) position the
        all-zeros-frame embed (HF ``_merge_input_ids_with_input_values``
        parity). ``num_ctx_codes_consumed`` keeps the paste chunked-prefill-
        correct across ``chunked_prefill_size`` boundaries.

        Returns:
            bf16 ``[T_total, 2048]`` or ``None`` when no request in the batch
            has context audio.
        """
        raise NotImplementedError("skeleton — PLAN §1.11/§2.4 _build_prefill_input_embeds")

    # --- decode (sync) ----------------------------------------------------------

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        """Stamp ``forward_batch.req_ids``; delegate to
        :meth:`_populate_cg_buffers`."""
        raise NotImplementedError("skeleton — PLAN §1.11 before_decode")

    def post_decode(self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list) -> None:
        """Sync collect: :meth:`_collect_step_outputs_cg`."""
        raise NotImplementedError("skeleton — PLAN §1.11 post_decode")

    def _populate_cg_buffers(
        self, forward_batch: Any, requests: list, *, is_lookahead: bool = False
    ) -> None:
        """Fill the model's CG buffers for one decode step (OUTSIDE any
        captured region) — PLAN §2.1 step 1. cb0 params from SGLang
        ``sampling_info`` (one D2H per attribute via ``_flat_sampling_attr``);
        depth params from ``req._omni_data.data`` host-side, defaulting to
        (0.9, 50) or mirroring cb0 when the request says so; padding row gets
        neutral params (T=1, k=K_MAX)."""
        raise NotImplementedError("skeleton — PLAN §1.11/§2.1 _populate_cg_buffers")

    @staticmethod
    def _extract_decode_sampling_params(forward_batch: Any, n_real: int) -> Any:
        """cb0 (temperature, top_k, top_p) rows from ``sampling_info``,
        length-checked against ``n_real``.

        Returns:
            Tuple of host lists/tensors consumed by ``_populate_cg_buffers``.
        """
        raise NotImplementedError("skeleton — PLAN §1.11 _extract_decode_sampling_params")

    def _collect_step_outputs_cg(
        self, result: Any, forward_batch: Any, requests: list
    ) -> None:
        """Sync decode tail: ``_decode_pack_gpu(n)`` → ONE blocking D2H
        ``staging[:n].cpu()`` → :meth:`_decode_collect_host` (PLAN §2.1
        step 3, §2.6)."""
        raise NotImplementedError("skeleton — PLAN §1.11/§2.6 _collect_step_outputs_cg")

    def _decode_pack_gpu(self, n_real: int) -> torch.Tensor:
        """Scatter ``_cg_active_* → pool[rows]`` and pack the staging row
        ``[codes_0..31 | was_done | generation_done]``.

        Returns:
            int64 GPU view ``_cg_collect_staging[:n_real]`` of shape
            ``[n_real, 34]``.
        """
        raise NotImplementedError("skeleton — PLAN §1.11/§2.6 _decode_pack_gpu")

    def _decode_collect_host(
        self,
        staging_host: torch.Tensor,
        result: Any,
        requests: list,
        *,
        next_token_device: torch.device | None = None,
    ) -> None:
        """Per-row host collect off the staged snapshot (PLAN §2.6):
        skip if ``req.is_chunked > 0`` / ``req.finished()`` / ``was_done``;
        else append ``codes_32`` to ``data.output_frames``; if
        ``generation_done``: :meth:`_mark_sampler_finished`, NO stream emit
        (EOS frames never reach Mimi — fence #1 of PLAN §4.2); else
        :meth:`_emit_code_chunk`; collect cb0 into ``result.next_token_ids``.

        Args:
            staging_host: int64 CPU ``[n_real, 34]``.
        """
        raise NotImplementedError("skeleton — PLAN §1.11/§2.6 _decode_collect_host")

    def _collect_step_outputs(self, result: Any, requests: list) -> None:
        """Prefill/eager collect (frame 0): same emission + finish rules as
        the decode collect, including the frame-0-EOS zero-decode lifecycle
        (PLAN §1.7 / F7); overwrites ``next_token_ids`` with cb0."""
        raise NotImplementedError("skeleton — PLAN §1.11 _collect_step_outputs")

    # --- decode (async halves; OFF until M4) -----------------------------------

    def post_decode_launch(self, result: Any, forward_batch: Any, requests: list) -> Any:
        """Async GPU half: scatter + pack, ``copy_(non_blocking=True)`` into a
        ping-pong pinned host buffer, and publish
        ``result.next_token_ids = _cg_codes_BN[:n, 0].clamp_min(0)`` from GPU
        state with NO host sync (clamp keeps STOP_CODE=-1 in embed range;
        decode embeds read ``last_codes``, so this is bookkeeping only —
        PLAN §2.6).

        Returns:
            The pinned host buffer (the base runner records the CUDA event).
        """
        raise NotImplementedError("skeleton — PLAN §1.11/§2.6 post_decode_launch")

    def post_decode_resolve(
        self, host_buf: Any, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        """Async host half: run :meth:`_decode_collect_host` off the pinned
        snapshot (with the ``pre_finished`` overrun guard)."""
        raise NotImplementedError("skeleton — PLAN §1.11/§2.6 post_decode_resolve")

    # --- emission / finish ------------------------------------------------------

    def _emit_code_chunk(self, sched_req: Any, codes_32: torch.Tensor) -> None:
        """Stream one frame to the vocoder:
        ``OutgoingMessage(type="stream", target="vocoder", data=codes_32 CPU
        int64 [32], metadata=stream_metadata)`` via the outbox. ``new_done``
        rows are never emitted (PLAN §2.5)."""
        raise NotImplementedError("skeleton — PLAN §1.11/§2.6 _emit_code_chunk")

    @staticmethod
    def _mark_sampler_finished(req: Any, generation_done: bool) -> None:
        """Set ``FINISH_MATCHED_TOKEN(matched=CODEBOOK_EOS)`` (cb0 of the EOS
        frame = 0) — upstream only needs *a* finish reason to fire KV release
        (PLAN §2.4)."""
        raise NotImplementedError("skeleton — PLAN §1.11/§2.4 _mark_sampler_finished")
