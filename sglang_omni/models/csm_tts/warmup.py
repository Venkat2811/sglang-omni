# SPDX-License-Identifier: Apache-2.0
"""Boot pre-warm for the CSM TTS pipeline — pay the cold-start tax at startup,
before the stage advertises readiness, so the first real request is a warmed
(graph-replay, when CUDA graphs are enabled) path rather than a cold prefill.

sgl-omni's structural weakness on the CSM path is *first-audio latency* (TTFA):
the first request after boot eats lazy CUDA-context init, kernel autotune /
Triton JIT, and (when ``cuda_graph`` is enabled) the per-batch-size graph
capture. All of that is deterministic work that has nothing to do with the
request's content, so it should happen during startup while the server is
still reporting "not ready", not on a user's first request. This is the
standard vLLM/SGLang pre-warm discipline applied to the CSM stages.

Two warmups live here, each a no-op-safe helper a stage factory calls right
after it builds its component and before its scheduler loop starts serving:

1. :func:`prewarm_codec_decode` — runs a synthetic Mimi decode at each codec
   batch-size bucket {1 .. max_batch_size}, so the conv-transpose decode
   kernels (and the bucketed ``decode_batch`` path) are autotuned/captured
   before the first stream's audio is needed. This is the warmup that most
   directly attacks first-audio latency, because the codec decode sits on the
   TTFA critical path for the very first emitted chunk.

2. :func:`prewarm_engine_frame_path` — best-effort warm of the frame-AR
   engine's decode-step kernels via the model's own warmup entry point when
   the backend exposes one (e.g. an SGLang model runner's cuda-graph capture /
   dummy-run). Strictly opportunistic: any backend that has no such hook is
   skipped cleanly, and graph capture only happens when the engine was
   configured with graphs enabled.

The pipeline contract (R0 §6) is: ``readyz`` flips true only after these have
run. Because the sgl-omni runner waits for every stage's scheduler to come up
before it serves, performing the warmup inside the factory — synchronously,
before returning the scheduler — places it firmly before readiness with no
change to the generic serving layer.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from sglang_omni.models.csm_tts.audio_codec import CsmMimiCodec
from sglang_omni.models.csm_tts.utils import NUM_CODEBOOKS

logger = logging.getLogger(__name__)

__all__ = ["prewarm_codec_decode", "prewarm_engine_frame_path", "prewarm_disabled"]

# A short synthetic frame count for the warmup decode — long enough to exercise
# the conv-transpose ladder past its first frame (so per-frame kernels, not
# just the prologue, are touched), short enough to keep boot fast.
_WARMUP_FRAMES = 4

# Framework-native operational env toggle (OSS-native, mirrors SGLang's
# SGLANG_* env convention) to DISABLE boot pre-warm. Default: pre-warm ON.
# This exists so an operator can A/B "does boot pre-warm crush first-request
# TTFA?" on the same binary by flipping one env var — exactly the single-knob
# discipline the bench harness uses. Set to "1"/"true"/"on"/"yes" to skip.
_PREWARM_DISABLE_ENV = "SGLANG_OMNI_CSM_DISABLE_PREWARM"


def prewarm_disabled() -> bool:
    """True when boot pre-warm is disabled via ``SGLANG_OMNI_CSM_DISABLE_PREWARM``."""
    return os.environ.get(_PREWARM_DISABLE_ENV, "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def prewarm_codec_decode(
    codec: CsmMimiCodec,
    *,
    max_batch_size: int,
    frames: int = _WARMUP_FRAMES,
) -> None:
    """Run a synthetic Mimi decode at each batch-size bucket {1..max_batch_size}.

    Touches both the single-item ``decode`` path and the bucketed
    ``decode_batch`` path the vocoder uses, so the codec decode kernels are
    autotuned (and graph-captured, where applicable) before the first stream's
    audio chunk is requested. All-zero codes are inside the valid Mimi range,
    so the clamp fence (#2) never fires during warmup — keeping the warmup off
    the OOB-telemetry counters.

    Best-effort: a warmup failure is logged and swallowed (it must never block
    the server from coming up), but is surfaced loudly because a codec that
    can't decode zeros at boot will fail on the first real request too.
    """
    if max_batch_size < 1:
        return
    if prewarm_disabled():
        logger.info(
            "CSM codec decode pre-warm DISABLED via %s; first-audio latency "
            "will pay the cold-start tax (intentional A/B arm)",
            _PREWARM_DISABLE_ENV,
        )
        return
    t0 = time.monotonic()
    try:
        # Single-item path (the streaming on_stream_chunk decode shape).
        single = torch.zeros(frames, NUM_CODEBOOKS, dtype=torch.long)
        codec.decode(single)
        # Bucketed batch path at each bucket size up to the pool width.
        for bs in range(2, max_batch_size + 1):
            batch = [
                torch.zeros(frames, NUM_CODEBOOKS, dtype=torch.long)
                for _ in range(bs)
            ]
            codec.decode_batch(batch)
        # A second pass at bs=1 so any first-call lazy init is excluded from
        # the steady-state path the first request will actually hit.
        codec.decode(single)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover - defensive boot path
        logger.exception(
            "CSM codec decode pre-warm failed (max_batch_size=%d, frames=%d); "
            "first-audio latency will pay the cold-start tax",
            max_batch_size,
            frames,
        )
        return
    logger.info(
        "CSM codec decode pre-warmed for batch sizes 1..%d (%.0f ms)",
        max_batch_size,
        (time.monotonic() - t0) * 1000.0,
    )


def prewarm_engine_frame_path(
    model_worker: Any,
    *,
    cuda_graph_enabled: bool,
) -> None:
    """Best-effort warm of the frame-AR engine's decode-step path at boot.

    When the underlying model runner exposes a warmup / cuda-graph-capture
    entry point (the SGLang convention), call it so the per-batch-size decode
    graphs are captured here, at startup, rather than lazily on the first
    request. Backends without such a hook are skipped cleanly — the codec
    pre-warm above already removes the largest single chunk of first-audio
    cold-start cost, and graph capture only matters when graphs are enabled.

    Never raises: the engine has already been constructed and validated by the
    time this runs; a missing/failed optional warmup hook degrades to "warm on
    first request", it does not block readiness.
    """
    runner = getattr(model_worker, "model_runner", None)
    if runner is None:
        return
    if prewarm_disabled():
        logger.info(
            "CSM engine frame-path pre-warm DISABLED via %s (intentional A/B arm)",
            _PREWARM_DISABLE_ENV,
        )
        return
    t0 = time.monotonic()
    # Try the conventional SGLang capture/warmup entry points in order; the
    # first one present wins. None of these are required to exist.
    candidates = (
        "capture_cuda_graphs" if cuda_graph_enabled else None,
        "init_cuda_graphs" if cuda_graph_enabled else None,
        "warmup",
    )
    for name in candidates:
        if not name:
            continue
        hook = getattr(runner, name, None)
        if not callable(hook):
            continue
        try:
            hook()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # pragma: no cover - optional, backend-dependent
            logger.warning(
                "CSM engine pre-warm hook %r raised; first request will warm "
                "the decode path instead",
                name,
                exc_info=True,
            )
            return
        logger.info(
            "CSM engine frame path pre-warmed via %r (cuda_graph=%s, %.0f ms)",
            name,
            cuda_graph_enabled,
            (time.monotonic() - t0) * 1000.0,
        )
        return
    logger.debug(
        "CSM engine exposes no pre-warm hook (cuda_graph=%s); relying on codec "
        "pre-warm + first-request warmup for the engine decode path",
        cuda_graph_enabled,
    )
