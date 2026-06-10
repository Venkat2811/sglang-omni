# SPDX-License-Identifier: Apache-2.0
"""SGLang request construction + scheduler adapters for CSM TTS.

PLAN §1.10.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.models.csm_tts.utils import BACKBONE_CTX, DEFAULT_MAX_FRAMES
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


@dataclass
class CsmSGLangRequestData(SGLangARRequestData):
    """Per-request state for the CSM TTS scheduler (engine-side mirror of
    :class:`CsmTtsState` plus runtime bookkeeping)."""

    context_codes: list[Any] | None = None  # CPU int32 [F, 32] per segment
    num_ctx_codes_consumed: int = 0  # chunked-prefill overlay cursor
    output_frames: list[torch.Tensor] = field(default_factory=list)
    generation_done: bool = False
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    engine_start_s: float = 0.0
    stream_metadata: dict[str, Any] | None = None


class _ResettableCsmModel(Protocol):
    def reset_request(self, req_id: str) -> None: ...


_CsmRequestBuilder = Callable[[StagePayload], CsmSGLangRequestData]
_CsmResultAdapter = Callable[[CsmSGLangRequestData], StagePayload]


def _context_fingerprint(state: CsmTtsState) -> str | None:
    """blake2b-16 hex over ALL context-code matrices (2 bytes/code, range
    0..2050) + speaker ids; ``None`` for context-free requests so they share
    the radix subtree.

    Required because different context audios share identical ``128002×F``
    token prefixes — without ``Req.extra_key`` namespacing the radix tree
    would share KV across different voices (higgs lesson, verbatim).

    Returns:
        Short hex digest or ``None``.
    """
    raise NotImplementedError("skeleton — PLAN §1.10 _context_fingerprint")


def build_sglang_csm_request(
    state: CsmTtsState, *, request_id: str = ""
) -> CsmSGLangRequestData:
    """Build the SGLang ``Req`` + request data for one CSM TTS request.

    - ``SamplingParams(max_new_tokens=<frames>, temperature, top_k, top_p,
      sampling_seed=int(seed))`` — the kwarg is ``sampling_seed``, NOT
      ``seed`` (review B6); then ``sampling_params.normalize(tokenizer=None)``
      (mandatory — upstream ``check_finished`` trips on ``len(None)``
      stop_strs otherwise).
    - ``Req(rid, origin_input_text="", origin_input_ids=prompt_ids,
      sampling_params, vocab_size=128256,
      extra_key=_context_fingerprint(state))`` — ``origin_input_text=""``
      required (positional signature, review B5).
    - Stamp ``req._codec_suppress_tokens = None`` and
      ``req._input_embeds_are_projected = False`` (V1 prefill-manager probes).

    Returns:
        :class:`CsmSGLangRequestData`.
    """
    raise NotImplementedError("skeleton — PLAN §1.10 build_sglang_csm_request")


def build_csm_stream_metadata(
    payload: StagePayload, data: CsmSGLangRequestData
) -> dict[str, Any] | None:
    """Stream metadata latched by the vocoder scheduler:
    ``{modality: "audio_codes", stream: True, num_codebooks: 32,
    codebook_size: 2051, [initial_codec_chunk_frames]}``.

    Returns:
        dict for streaming requests, ``None`` for non-streaming.
    """
    raise NotImplementedError("skeleton — PLAN §1.10 build_csm_stream_metadata")


def apply_csm_result(state: CsmTtsState, data: CsmSGLangRequestData) -> None:
    """Write ``output_frames`` + usage (prompt_tokens, completion_frames,
    engine_time_s) back into ``state``."""
    raise NotImplementedError("skeleton — PLAN §1.10 apply_csm_result")


def make_csm_scheduler_adapters(
    model: _ResettableCsmModel,
    max_new_tokens_cap: int | None = DEFAULT_MAX_FRAMES,
) -> tuple[_CsmRequestBuilder, _CsmResultAdapter]:
    """Build the (request_builder, result_adapter) pair for ``OmniScheduler``.

    request_builder: stamps ``engine_start_s`` / ``stage_payload`` /
    ``stream_metadata`` and clamps ``max_new_tokens`` to
    ``min(cap, BACKBONE_CTX - 1 - len(prompt_ids))`` — the scheduler's budget
    is ``max_req_len = min(context_length - 1, max_total_num_tokens - 1)``
    = 2047, NOT 2048 (omni_scheduler.py:150-153; clamping against 2048 admits
    requests the admission check then rejects — review A4/B18).

    result_adapter: writes ``output_frames`` + usage into the state, calls
    ``model.reset_request(rid)`` (frees the sampler row), returns a fresh
    ``StagePayload``.

    Returns:
        ``(request_builder, result_adapter)``.
    """
    raise NotImplementedError("skeleton — PLAN §1.10 make_csm_scheduler_adapters")


__all__ = [
    "CsmSGLangRequestData",
    "apply_csm_result",
    "build_csm_stream_metadata",
    "build_sglang_csm_request",
    "make_csm_scheduler_adapters",
]
