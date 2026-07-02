# SPDX-License-Identifier: Apache-2.0
# Adopts the MOSS-TTS-Local vocoder CUDA-graph pattern
# (sglang_omni/models/moss_tts_local/vocoder_cuda_graph.py, upstream #798/#886).
"""CUDA-graph runner for the CSM Mimi vocoder's STATELESS decode: one graph
per window length T at B=1, captured once at boot (warmup → capture → seal),
never lazily in the serving loop.

Structural simplification vs MOSS: CSM's shipped vocoder path is the
stateless overlap-trim ``MimiModel.decode`` — no streaming KV cache, no conv
padding cache (the bundled codec_config ships ``use_cache=False``), so the
WHOLE decode forward is the capture region and no in-place cache patch
(MOSS's ``patch_codec_attention_cache_for_cuda_graph``) is needed for
bit-identity.

Capture boundary:

- CAPTURED: ``MimiModel.decode(codes[1, 32, T])`` for each warmup T —
  quantizer embed → upsample conv-transpose → decoder transformer
  (sliding-window causal mask) → SEANet conv stack. A pure function of the
  codes at fixed shape: transformers' modeling_mimi has no ``.item()`` host
  syncs on this path and builds masks/positions device-side (verified
  against transformers 5.6.0).
- EAGER (``None`` fallback, selected by :meth:`decode`): any T not captured
  (non-streaming full-utterance decodes run T up to the 125-frame cap), any
  ``B != 1`` (``decode_batch`` multi-item buckets), and non-CUDA inputs.
- NEVER captured: the OPTIONAL stateful
  ``streaming_decode(..., decoder_past_key_values=...)`` path — HF
  ``DynamicCache`` grows across calls (data-dependent shapes), which a CUDA
  graph cannot replay without a static-cache rewrite. That path is not the
  shipped default and stays eager by design.

fp32 discipline: the runner captures whatever dtype the codec was loaded in
(fp32 by default — conv-transpose decode stability); it neither casts inputs
nor outputs.

Bit-identity is the bar: replayed output must ``torch.equal`` the eager
output for the same codes. The real gate is CUDA-only
(tests/unit_test/csm_tts/test_vocoder_cuda_graph.py) and runs on the
RTX 3060.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)


class _CapturedVocoderGraph(NamedTuple):
    """One captured per-T graph and its static replay buffers (named to
    avoid positional unpack)."""

    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor  # int64 [1, 32, T]
    static_audio: torch.Tensor  # [1, 1, T * 1920], codec dtype


class CsmMimiVocoderCudaGraphRunner:
    """Warmup-captured, sealed replay of exact-T CUDA graphs for the
    stateless Mimi decode (B fixed at 1 — the streaming vocoder decodes one
    request's window per call)."""

    def __init__(
        self,
        model,
        *,
        num_codebooks: int = 32,
        max_frames: int,
        max_graphs: int = 32,
        warmup_iters: int = 3,
        min_free_gb: float = 1.0,
    ) -> None:
        """``model`` is the realized ``transformers.MimiModel``
        (``CsmMimiCodec.model``); the caller owns capture policy (which T
        values to warm up — see
        ``CsmStreamingVocoderScheduler._cuda_graph_capture_frames``)."""
        self._model = model
        self._num_codebooks = int(num_codebooks)
        self._device = next(model.parameters()).device
        self._max_frames = int(max_frames)
        self._max_graphs = int(max_graphs)
        self._warmup_iters = int(warmup_iters)
        # Min free VRAM to attempt a capture; below it we skip -> eager, so a
        # VRAM-tight box degrades gracefully instead of OOM-ing. Default is
        # far below MOSS's 3 GB: these are B=1, T<=~13 fp32 graphs (MBs, not
        # the multi-GB B=16 MOSS captures).
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._graphs: dict[int, _CapturedVocoderGraph] = {}
        self._pool = None
        self._sealed = False

    def _is_supported_frame_count(self, frame_count: int) -> bool:
        return 1 <= frame_count <= self._max_frames

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    @torch.no_grad()
    def _capture_frame_count(self, frame_count: int) -> None:
        device = self._device
        static_codes = torch.zeros(
            1, self._num_codebooks, frame_count, dtype=torch.long, device=device
        )
        # Side-stream warmup forces lazy allocs (conv algo / workspaces) out
        # of the capture (MOSS discipline). No state reset is needed between
        # warmup and capture: the stateless decode holds no streaming state.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._model.decode(static_codes)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        # Shared mempool across the T graphs to bound memory; capture order
        # (largest T first) in warmup.
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self._pool, capture_error_mode="thread_local"
        ):
            static_audio = self._model.decode(static_codes).audio_values
        self._graphs[frame_count] = _CapturedVocoderGraph(
            graph=graph,
            static_codes=static_codes,
            static_audio=static_audio,
        )
        logger.info(
            "Captured CSM Mimi vocoder CUDA graph T=%d (B=1) -> audio %s (%d cached)",
            frame_count,
            tuple(static_audio.shape),
            len(self._graphs),
        )

    @torch.no_grad()
    def warmup(self, frames: Iterable[int]) -> None:
        """Capture one graph per T, once, then seal (boot time, GPU
        quiescent). Non-CUDA codecs seal immediately with zero graphs (the
        caller then serves eager)."""
        if self._sealed:
            logger.warning(
                "CsmMimiVocoderCudaGraphRunner.warmup called after seal; ignoring"
            )
            return
        if self._device.type != "cuda":
            self._sealed = True
            logger.info(
                "CSM Mimi vocoder CUDA graphs: codec on %s, nothing to capture "
                "(eager)",
                self._device,
            )
            return
        # Bind capture to the codec's device (stream/pool/graph use the
        # current device; factory-time capture can precede a device switch).
        with torch.cuda.device(self._device):
            # Capture LARGEST T first — the graphs share one mempool;
            # capturing a larger graph after a smaller one grows the pool and
            # invalidates earlier graphs' addresses (replay segfaults).
            for t in sorted(dict.fromkeys(int(x) for x in frames), reverse=True):
                if t in self._graphs:
                    continue
                if not self._is_supported_frame_count(t):
                    logger.warning(
                        "skip CSM Mimi vocoder CG T=%d: outside [1, %d]",
                        t,
                        self._max_frames,
                    )
                    continue
                if len(self._graphs) >= self._max_graphs:
                    logger.warning(
                        "CSM Mimi vocoder CG cap %d reached; skipping rest",
                        self._max_graphs,
                    )
                    break
                # VRAM headroom guard — skip capture (-> eager) rather than
                # risk OOM. Checked per-T because each capture allocates.
                enough, free = self._enough_free_vram()
                if not enough:
                    logger.warning(
                        "CSM Mimi vocoder CG: free VRAM %.1fGB < %.1fGB headroom; "
                        "skipping T=%d+ (eager)",
                        free / 1024**3,
                        self._min_free_bytes / 1024**3,
                        t,
                    )
                    break
                # best-effort: an uncaptured T falls back to eager
                try:
                    self._capture_frame_count(t)
                except Exception as exc:
                    self._graphs.pop(t, None)
                    logger.warning(
                        "CSM Mimi vocoder CG capture failed for T=%d: %s; "
                        "will use eager",
                        t,
                        exc,
                    )
        self._sealed = True
        logger.info(
            "CSM Mimi vocoder CUDA graphs sealed: %d T captured %s",
            len(self._graphs),
            sorted(self._graphs.keys()),
        )

    def captured_frames(self) -> list[int]:
        return sorted(self._graphs.keys())

    @torch.no_grad()
    def decode(self, codes_B32F: torch.Tensor) -> torch.Tensor | None:
        """Replay the captured graph for int64 ``[1, 32, T]`` codes (copy
        codes into the static input, replay), else ``None`` → the caller
        decodes eager. Returns the STATIC audio buffer ``[1, 1, T * 1920]``
        directly; the caller must consume (copy/D2H) it before the next
        replay."""
        if not codes_B32F.is_cuda:
            return None
        if codes_B32F.ndim != 3:
            return None
        b, k, t = codes_B32F.shape
        if b != 1 or k != self._num_codebooks:
            return None
        entry = self._graphs.get(int(t))
        if entry is None:
            return None
        # Replicate the eager input exactly so replay is bit-for-bit
        # identical (stateless decode: codes are the ONLY input).
        entry.static_codes.copy_(codes_B32F)
        entry.graph.replay()
        return entry.static_audio


__all__ = ["CsmMimiVocoderCudaGraphRunner"]
