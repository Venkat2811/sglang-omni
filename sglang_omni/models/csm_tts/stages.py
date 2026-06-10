# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the CSM TTS pipeline:
``preprocessing → audio_encoder (Mimi encode, no-op fast path) → tts_engine →
vocoder (Mimi decode)``.

Referenced by dotted path from
:class:`sglang_omni.models.csm_tts.config.CsmTtsPipelineConfig` — this module
imports sglang and is therefore NEVER imported by ``config.py`` (the
registry-silent-skip gotcha, PLAN §1.15).

PLAN §1.14.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.models.csm_tts.model_runner import CsmTTSModelRunner
from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.models.csm_tts.request_builders import make_csm_scheduler_adapters
from sglang_omni.models.csm_tts.text_tokenizer import CsmPromptBuilder
from sglang_omni.models.csm_tts.utils import (
    BACKBONE_CTX,
    DEFAULT_MAX_FRAMES,
    get_or_load_codec,
    load_audio_to_24k,
    resolve_checkpoint,
    to_codes_F32,
)
from sglang_omni.models.csm_tts.vocoder_scheduler import CsmStreamingVocoderScheduler
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

logger = logging.getLogger(__name__)

# The binding constraint on the 3060 is COMPUTE (frame step must beat 80 ms
# real time), not VRAM (PLAN §3.3) — mrr/concurrency default to 8.
DEFAULT_MAX_CONCURRENCY = 8

# Context-audio budget guard: 60 s = 750 backbone positions at 12.5 Hz
# (+1 <|audio_eos|> per segment); text capped ~1500 tokens after reserving
# frames (PLAN §1.14).
MAX_CONTEXT_AUDIO_SECONDS = 60


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> ThreadedSimpleScheduler:
    """CPU preprocessing: parse inputs, tokenize, route the audio path.

    Parses ``payload.request.inputs`` (str | dict with ``input|text``,
    ``speaker``, ``context: [{text, speaker, audio|codes}]``,
    higgs-compatible ``references`` aliasing). Three branches (PLAN §1.14):

    (a) pre-encoded codes → full prompt NOW (F = ``codes.shape[0]``);
    (b) no context → prompt now;
    (c) raw waveform → load/resample 24 kHz mono and DEFER prompt assembly to
        the audio_encoder — the placeholder count must come from the actual
        Mimi encode (kills the off-by-one class; F14, do not optimize away).

    Limits: context audio <= 60 s; text <= ~1500 tokens. Three-tier caching
    copied from higgs (path-hash memo / waveform LRU / encoded-codes LRU as
    CPU int32).

    Returns:
        ``ThreadedSimpleScheduler`` wrapping the preprocess fn.
    """
    raise NotImplementedError("skeleton — PLAN §1.14 create_preprocessing_executor")


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_batch_size: int = DEFAULT_MAX_CONCURRENCY,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    """Mimi ENCODE stage (optional fast-path no-op).

    No-op on the pre-encoded fast path; else ``codec.encode_reference`` →
    ``[F, 32]`` → build the prompt with exactly F ``128002`` placeholders +
    ``128003`` per segment → null the waveform. Startup warmup encode of 1 s
    of zeros. Shares the process-wide Mimi via ``get_or_load_codec``.

    Returns:
        ``SimpleScheduler`` with batched encode.
    """
    raise NotImplementedError("skeleton — PLAN §1.14 create_audio_encoder_executor")


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int | None = DEFAULT_MAX_FRAMES,
    server_args_overrides: dict[str, Any] | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
) -> OmniScheduler:
    """SGLang-backed frame-AR engine (the §3-contract recipe).

    Recipe (PLAN §1.14): ``resolve_checkpoint`` →
    ``build_sglang_server_args(checkpoint_dir, context_length=2048,
    disable_cuda_graph=True (M1-M3; False in M4), cuda_graph_max_bs=8,
    mem_fraction_static=0.5, max_running_requests=8,
    chunked_prefill_size=2048, dtype="bfloat16", **overrides)`` →
    ``server_args.disable_overlap_schedule = True`` (contract-required) →
    ``create_sglang_infrastructure`` → ``CsmTTSModelRunner(model_worker,
    SGLangOutputProcessor(capture_hidden=False, ...))`` →
    ``make_csm_scheduler_adapters`` → ``OmniScheduler(...,
    abort_callback=model.reset_request)`` →
    ``model_runner.set_stream_outbox(scheduler.outbox)``.

    Deliberate deltas vs higgs: NO ``truncate_rope_to_bf16`` (CSM ckpt is
    fp32-trained — leave SGLang's fp32 cos/sin cache alone); the
    ``on_retract`` callable passed to bootstrap must ABORT with an actionable
    error instead of silently re-prefilling (sampler pool has no rollback —
    PLAN §8.R1a).

    Returns:
        Configured ``OmniScheduler``.
    """
    raise NotImplementedError("skeleton — PLAN §1.14 create_sglang_tts_engine_executor")


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "float32",
    max_batch_size: int = DEFAULT_MAX_CONCURRENCY,
    max_batch_wait_ms: int = 2,
    stream_stride: int = 13,
    stream_followup_stride: int = 6,
    stream_overlap_frames: int = 4,
    stream_holdback_frames: int = 2,
) -> CsmStreamingVocoderScheduler:
    """Mimi DECODE stage (fp32 default — conv-transpose stability).

    ``resolve_checkpoint`` → ``get_or_load_codec`` (shared with the encoder)
    → :class:`CsmStreamingVocoderScheduler` with the 12.5 Hz framing knobs
    (PLAN §4.1).

    Returns:
        ``CsmStreamingVocoderScheduler``.
    """
    raise NotImplementedError("skeleton — PLAN §1.14 create_vocoder_executor")


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "MAX_CONTEXT_AUDIO_SECONDS",
    "create_audio_encoder_executor",
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
