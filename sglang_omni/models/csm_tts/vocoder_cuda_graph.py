# SPDX-License-Identifier: Apache-2.0
# Adopts the MOSS-TTS-Local vocoder CUDA-graph pattern
# (sglang_omni/models/moss_tts_local/vocoder_cuda_graph.py, upstream #798/#886).
"""CUDA-graph runner for the CSM Mimi vocoder's STATELESS decode: one graph
per window length T at B=1, captured once at boot (warmup → capture → seal),
never lazily in the serving loop.

Structural simplification vs MOSS: CSM's shipped vocoder path is the
stateless overlap-trim ``MimiModel.decode`` — no streaming KV cache, no conv
padding cache (the bundled codec_config ships ``use_cache=False``), so the
WHOLE decode forward is the capture region and no STATE patch (MOSS's
``patch_codec_attention_cache_for_cuda_graph``) is needed for bit-identity.

Two capture-LEGALITY patches are needed instead
(:func:`patch_mimi_codec_for_cuda_graph`, applied by the scheduler's
``warmup_now`` before capture) — both measured on the RTX 3060 against
transformers 5.6.0, both value-identical so patched eager stays bitwise
equal to upstream HF:

1. ``MimiResidualVectorQuantizer.decode`` initializes its accumulator with
   ``torch.tensor(0.0, device=codes.device)`` — a CPU-scalar construction +
   unpinned H2D copy, which raises ``Cannot copy between CPU and CUDA
   tensors during CUDA graph capture`` (and capture_end then warns "The
   CUDA Graph is empty"). Patched to a device-side ``torch.zeros(())``.
2. ``MimiConv1d`` keeps ``kernel_size``/``stride``/``padding_total`` as
   int64 CUDA buffers; ``F.pad`` coerces the resulting 0-d CUDA scalars via
   ``__index__`` → a D2H sync → ``operation not permitted when stream is
   capturing``. Patched to pre-read host ints with the same ceil
   arithmetic (integers here are ≪ 2**24, exact in the upstream fp32
   tensor math, so the results are identical).

Capture boundary:

- CAPTURED: ``MimiModel.decode(codes[1, 32, T])`` for each warmup T —
  quantizer embed → upsample conv-transpose → decoder transformer
  (sliding-window causal mask) → SEANet conv stack — after the two patches
  above. A pure function of the codes at fixed shape.
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
output for the same codes — enforced TWICE: per-T at boot by
:meth:`CsmMimiVocoderCudaGraphRunner._replay_self_check` (a poisoned-output
replay compared against an eager reference; an empty/no-op capture or a
non-bit-identical replay drops that T to eager), and by the CUDA gate in
tests/unit_test/csm_tts/test_vocoder_cuda_graph.py on the RTX 3060.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from types import MethodType
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)


class _CapturedVocoderGraph(NamedTuple):
    """One captured per-T graph and its static replay buffers (named to
    avoid positional unpack)."""

    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor  # int64 [1, 32, T]
    static_audio: torch.Tensor  # [1, 1, T * 1920], codec dtype


_ORIG_RVQ_DECODE_ATTR = "_sglang_omni_original_rvq_decode"
_ORIG_CONV_FORWARD_ATTR = "_sglang_omni_original_conv_forward"


def _cuda_graph_rvq_decode(self, codes: torch.Tensor) -> torch.Tensor:
    """Value-identical rewrite of transformers'
    ``MimiResidualVectorQuantizer.decode`` (5.6.0): the ONLY change is the
    accumulator init — ``torch.zeros((), device=...)`` allocates the 0-d
    fp32 zero device-side, where upstream's ``torch.tensor(0.0, device=...)``
    constructs on CPU and does an unpinned H2D copy (illegal during graph
    capture). Same value, dtype, shape ⇒ every subsequent op is unchanged."""
    quantized_out = torch.zeros((), device=codes.device)
    codes = codes.transpose(0, 1)
    for i, indices in enumerate(codes):
        layer = self.layers[i]
        quantized = layer.decode(indices)
        quantized_out = quantized_out + quantized

    if self.output_proj is not None:
        quantized_out = self.output_proj(quantized_out)
    return quantized_out


def _cuda_graph_conv1d_forward(self, hidden_states, padding_cache=None):
    """Value-identical rewrite of transformers' ``MimiConv1d.forward``
    (5.6.0): the padding arithmetic runs on host ints pre-read at patch time
    (upstream registers ``kernel_size``/``stride``/``padding_total`` as int64
    CUDA buffers; ``F.pad`` coerces those 0-d CUDA scalars via ``__index__``
    → a D2H sync, illegal during graph capture). Exactness: the magnitudes
    here are ≪ 2**24, where upstream's fp32 tensor division/ceil is exact,
    so the host math yields identical padding."""
    length = int(hidden_states.shape[-1])
    kernel_size = self._sglang_omni_kernel_size
    stride = self._sglang_omni_stride
    padding_total = self._sglang_omni_padding_total
    n_frames = math.ceil((length - kernel_size + padding_total) / stride + 1) - 1
    extra_padding = n_frames * stride + kernel_size - padding_total - length

    if not self.causal and padding_cache is not None:
        raise ValueError("`padding_cache` is not supported for non-causal convolutions.")

    if self.causal and padding_cache is not None:
        layer_padding_cache = padding_cache.update(hidden_states, self.layer_idx)
        hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=2)
    elif self.causal:
        hidden_states = self._pad1d(
            hidden_states, (padding_total, extra_padding), mode=self.pad_mode
        )
    else:
        hidden_states = self._pad1d(
            hidden_states,
            (
                self._sglang_omni_padding_left,
                self._sglang_omni_padding_right + extra_padding,
            ),
            mode=self.pad_mode,
        )
    return self.conv(hidden_states)


def patch_mimi_codec_for_cuda_graph(model) -> None:
    """Rebind transformers Mimi's two capture-hostile idioms to
    value-identical, capture-legal forms (module docstring, items 1-2) —
    the MOSS ``patch_codec_attention_cache_for_cuda_graph`` analogue.
    Idempotent (originals stashed once); modules with an unexpected layout
    are skipped with a warning (their captures then fail per-T → eager).
    The patched EAGER path stays bitwise equal to upstream HF, so applying
    the patch never changes served audio."""
    for module in model.modules():
        name = type(module).__name__
        if name == "MimiResidualVectorQuantizer":
            if hasattr(module, _ORIG_RVQ_DECODE_ATTR):
                continue
            decode = getattr(module, "decode", None)
            if not callable(decode) or not hasattr(module, "layers"):
                logger.warning(
                    "MimiResidualVectorQuantizer layout unexpected; skipping "
                    "CUDA-graph zero-init patch (capture may fall back to eager)"
                )
                continue
            setattr(module, _ORIG_RVQ_DECODE_ATTR, decode)
            module.decode = MethodType(_cuda_graph_rvq_decode, module)
        elif name == "MimiConv1d":
            if hasattr(module, _ORIG_CONV_FORWARD_ATTR):
                continue
            required = (
                "kernel_size",
                "stride",
                "padding_total",
                "padding_left",
                "padding_right",
                "conv",
                "_pad1d",
                "pad_mode",
                "causal",
            )
            if not all(hasattr(module, attr) for attr in required):
                logger.warning(
                    "MimiConv1d layout unexpected; skipping CUDA-graph pad "
                    "patch (capture may fall back to eager)"
                )
                continue
            # Reading the int64 buffers syncs — legal HERE (patch time),
            # illegal during capture; that is the whole point of the patch.
            module._sglang_omni_kernel_size = int(module.kernel_size)
            module._sglang_omni_stride = int(module.stride)
            module._sglang_omni_padding_total = int(module.padding_total)
            module._sglang_omni_padding_left = int(module.padding_left)
            module._sglang_omni_padding_right = int(module.padding_right)
            setattr(module, _ORIG_CONV_FORWARD_ATTR, module.forward)
            module.forward = MethodType(_cuda_graph_conv1d_forward, module)


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
        # Eager reference on the SAME static input for the post-capture
        # replay self-check (current stream, before capture).
        eager_ref = self._model.decode(static_codes).audio_values.detach().clone()
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
        # Boot-time bit-identity gate: a silently-empty capture (capture_end
        # warns "The CUDA Graph is empty") replays as a no-op — catch it (and
        # any non-bit-identical replay) HERE and raise so warmup drops this T
        # to eager instead of shipping corrupt audio.
        self._replay_self_check(graph.replay, static_audio, eager_ref)
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

    @staticmethod
    @torch.no_grad()
    def _replay_self_check(
        replay_fn, static_audio: torch.Tensor, eager_ref: torch.Tensor
    ) -> None:
        """Poison the static output with NaN, replay, and require
        ``torch.equal`` with the eager reference for the SAME static input.

        Catches (a) silently-EMPTY captures — wrong stream/device or a
        swallowed in-capture error leaves zero recorded nodes, replay is a
        no-op and the poison survives (the exact RTX 3060 failure mode this
        guards) — and (b) any non-bit-identical replay. Callers drop the T
        (→ eager) on failure. Device-agnostic on purpose so the CPU tier can
        regression-test the no-op-replay mode with a stub ``replay_fn``."""
        static_audio.fill_(float("nan"))
        replay_fn()
        if static_audio.is_cuda:
            torch.cuda.synchronize()
        if bool(torch.isnan(static_audio).any()):
            raise RuntimeError(
                "CUDA-graph replay left the static output poisoned — the "
                "capture recorded no work (empty graph / wrong stream)"
            )
        if not torch.equal(static_audio, eager_ref):
            raise RuntimeError(
                "CUDA-graph replay is not bit-identical to the eager decode "
                "of the same input"
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


__all__ = ["CsmMimiVocoderCudaGraphRunner", "patch_mimi_codec_for_cuda_graph"]
