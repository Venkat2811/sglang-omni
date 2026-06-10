# SPDX-License-Identifier: Apache-2.0
"""Constants + stage helpers shared across the CSM TTS pipeline.

Key facts (PLAN header + §1.3):

- Mimi: 24 kHz, 12.5 Hz frame rate ⇒ one frame = 80 ms = 1920 samples,
  32 quantizers, codebook size 2048.
- Audio vocab per codebook is 2051: valid Mimi codes 0..2047; specials
  2048/2049/2050 must NEVER reach Mimi (``CODEBOOK_PAD=2050``).
- EOS frame: codebooks 0..30 all == 0 (cb31 excluded); audio trim at the
  first all-32-zero frame; default generation cap 125 frames.
- **No delay-pattern helpers** — CSM has none (unlike higgs).

PLAN §1.3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download

from sglang_omni.models.csm_tts.audio_codec import CsmMimiCodec

# --- CSM constants (real values, pinned by PLAN §1.3) -----------------------
NUM_CODEBOOKS = 32
CODEBOOK_VOCAB = 2051  # per-codebook logits width (2048 Mimi + 3 specials)
MIMI_CODEBOOK_SIZE = 2048  # valid Mimi code range is [0, 2047]
CODEBOOK_EOS = 0
CODEBOOK_PAD = 2050
STOP_CODE = -1  # frozen-row sentinel emitted after generation_done
AUDIO_TOKEN_ID = 128002  # <|AUDIO|> — real in-vocab id (no -100 sentinel)
AUDIO_EOS_TOKEN_ID = 128003  # <|audio_eos|>
BOS = 128000
EOT = 128001
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920  # 24 kHz / 12.5 Hz
FRAME_RATE = 12.5
DEFAULT_MAX_FRAMES = 125
BACKBONE_CTX = 2048
K_MAX = 2051  # fixed-shape top-k width for the branchless sampler

# Shared between audio_encoder + vocoder; one Mimi load saves ~0.4 GB fp32.
_CODEC_CACHE: dict[tuple[str, str, str], CsmMimiCodec] = {}


def resolve_checkpoint(checkpoint: str) -> str:
    """Local dir or HF repo id → local snapshot path (verbatim higgs copy)."""
    if Path(checkpoint).is_dir():
        return checkpoint
    return snapshot_download(checkpoint)


def get_or_load_codec(checkpoint_dir: str, device: str, dtype: str) -> CsmMimiCodec:
    """Process-wide cached :class:`CsmMimiCodec` per ``(path, device, dtype)``.

    One Mimi load serves both audio_encoder (encode) and vocoder (decode).

    Returns:
        ``CsmMimiCodec`` (fp32 by default — conv-transpose decode stability).
    """
    raise NotImplementedError("skeleton — PLAN §1.3 get_or_load_codec")


def encoded_audio_length(num_samples: int) -> int:
    """Closed-form Mimi frame count for ``num_samples`` raw 24 kHz samples.

    Port of HF ``processing_csm.py:93-129`` (causal-conv ladder ceil;
    ≈ ``ceil(num_samples / 1920)``). Used ONLY for pre-encode budget /
    admission estimates on raw waveforms — the pre-encoded fast path needs no
    formula (F is just ``codes.shape[0]``), and the raw-waveform path defers
    the actual placeholder count to the real Mimi encode (PLAN §1.14) so
    prompts can never mismatch. Do NOT promote this estimate to prompt
    assembly (PLAN §1.3 / F14).

    Returns:
        int — number of 80 ms frames F.
    """
    raise NotImplementedError("skeleton — PLAN §1.3 encoded_audio_length")


def to_codes_F32(obj: Any) -> torch.Tensor | None:
    """Coerce client-supplied pre-encoded context codes to ``[F, 32]`` int64.

    Accepts ``torch.Tensor`` / nested lists / numpy; ``None`` / empty → None.
    Raises ``ValueError`` unless the result is 2-D with 32 columns.

    Returns:
        ``torch.LongTensor [F, 32]`` or ``None``.
    """
    raise NotImplementedError("skeleton — PLAN §1.3 to_codes_F32")


def load_audio_to_24k(reference_audio: Any) -> tuple[np.ndarray, int]:
    """Load context audio as 24 kHz mono float32 (higgs loader copy).

    Accepts local path, HTTP/HTTPS URL, or ``{audio_path|path|bytes|base64|
    data}`` dict.

    Returns:
        ``(audio_float32 [S], sample_rate)`` — caller reshapes to ``[1, 1, S]``.
    """
    raise NotImplementedError("skeleton — PLAN §1.3 load_audio_to_24k")


__all__ = [
    "AUDIO_EOS_TOKEN_ID",
    "AUDIO_TOKEN_ID",
    "BACKBONE_CTX",
    "BOS",
    "CODEBOOK_EOS",
    "CODEBOOK_PAD",
    "CODEBOOK_VOCAB",
    "DEFAULT_MAX_FRAMES",
    "EOT",
    "FRAME_RATE",
    "K_MAX",
    "MIMI_CODEBOOK_SIZE",
    "NUM_CODEBOOKS",
    "SAMPLES_PER_FRAME",
    "SAMPLE_RATE",
    "STOP_CODE",
    "encoded_audio_length",
    "get_or_load_codec",
    "load_audio_to_24k",
    "resolve_checkpoint",
    "to_codes_F32",
]
