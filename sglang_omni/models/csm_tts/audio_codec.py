# SPDX-License-Identifier: Apache-2.0
"""Mimi codec wrapper for CSM — loaded from the SAME checkpoint's
``codec_model.*`` subtree (one-artifact pattern; the engine's weight loader
skips that subtree, PLAN §1.6).

fp32 BY DEFAULT (conv-transpose decode stability; bf16 opt-in). Mimi runs at
24 kHz / 12.5 Hz ⇒ one frame = 80 ms = 1920 samples; 32 quantizers with
2048-row codebooks — codes 2048/2049/2050 are OUT OF BOUNDS and must be
clamped before any Mimi call (hard SIGSEGV-class hazard on some kernels,
garbage on others; PLAN §4.2 / §8.R6).

PLAN §1.12.
"""

from __future__ import annotations

from typing import Any

import torch


def sanitize_for_mimi(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Fence #2 of PLAN §4.2: count then ``codes.clamp_(0, 2047)``.

    The clamp count is exported as a counter: under greedy/temp=0 runs
    ``count == 0`` is a hard parity-gate assertion (PLAN §6.2); under
    sampling it is telemetry ONLY — the 2051-way heads can legally emit
    2048-2050 with nonzero probability (F11).

    Args:
        codes: int64 ``[F, 32]`` (any device).

    Returns:
        ``(clamped codes [F, 32], num_clamped int)``.
    """
    raise NotImplementedError("skeleton — PLAN §1.12/§4.2 sanitize_for_mimi")


class CsmMimiCodec:
    """Wraps ``transformers.MimiModel`` for encode (audio_encoder stage) and
    decode (vocoder stage). One process-wide instance per (path, device,
    dtype) via :func:`sglang_omni.models.csm_tts.utils.get_or_load_codec`."""

    SAMPLE_RATE = 24000
    SAMPLES_PER_FRAME = 1920  # asserted from config sampling_rate / frame_rate

    def __init__(self, model: Any, device: str, dtype: torch.dtype) -> None:
        """Hold the realized ``MimiModel`` + device/dtype; assert
        ``sampling_rate / frame_rate == 1920``."""
        raise NotImplementedError("skeleton — PLAN §1.12 CsmMimiCodec.__init__")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> "CsmMimiCodec":
        """Build a ``MimiModel`` from the CSM checkpoint's ``codec_model.*``
        tensors (NOT a separate kyutai repo download).

        Returns:
            :class:`CsmMimiCodec` on ``device`` in ``dtype`` (fp32 default).
        """
        raise NotImplementedError("skeleton — PLAN §1.12 CsmMimiCodec.from_pretrained")

    def encode_reference(self, wav_11S: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Encode one context waveform to Mimi codes.

        Pads to >= 1 frame; transposes HF's ``[1, 32, F]`` output.

        Args:
            wav_11S: float32 ``[1, 1, S]`` mono.
            sample_rate: must be 24000 (caller resamples).

        Returns:
            int64 ``[F, 32]``, values in ``[0, 2047]``.
        """
        raise NotImplementedError("skeleton — PLAN §1.12 CsmMimiCodec.encode_reference")

    def decode(self, codes_F32: torch.Tensor) -> torch.Tensor:
        """Decode a code matrix to a waveform (stateless path).

        Transposes to ``[1, 32, F]``, applies the :func:`sanitize_for_mimi`
        clamp guard, decodes.

        Args:
            codes_F32: int64 ``[F, 32]``.

        Returns:
            float32 ``[1, S]`` with ``S == F * 1920``.
        """
        raise NotImplementedError("skeleton — PLAN §1.12 CsmMimiCodec.decode")

    def decode_batch(self, items: list[torch.Tensor]) -> list[torch.Tensor]:
        """Bucketed batch decode (exact-frame-count buckets, higgs
        ``_bucketed_batch`` copy) for the non-streaming path.

        Args:
            items: list of int64 ``[F_i, 32]``.

        Returns:
            list of float32 ``[1, S_i]`` aligned with the input order.
        """
        raise NotImplementedError("skeleton — PLAN §1.12 CsmMimiCodec.decode_batch")

    def streaming_decode(
        self, codes_F32: torch.Tensor, past_key_values: Any | None
    ) -> tuple[torch.Tensor, Any]:
        """Stateful incremental decode wrapping
        ``MimiModel.decode(..., decoder_past_key_values=...)`` — M4+ path; the
        per-request state object is OWNED by the vocoder scheduler and dropped
        in ``clear_stream_state`` (cross-request state-leak guard; the rime
        MimiCodecAdapter postmortem is the cautionary tale — PLAN §4.3).

        Args:
            codes_F32: int64 ``[F_new, 32]`` delta frames.
            past_key_values: Mimi decoder KV state or ``None`` on first call.

        Returns:
            ``(wave float32 [1, F_new * 1920], new past_key_values)``.
        """
        raise NotImplementedError("skeleton — PLAN §1.12/§4.3 CsmMimiCodec.streaming_decode")


__all__ = ["CsmMimiCodec", "sanitize_for_mimi"]
