# SPDX-License-Identifier: Apache-2.0
"""Per-request pipeline state for CSM TTS — THE inter-stage wire schema.

Carried between stages via :class:`sglang_omni.proto.StagePayload.data`.
Fields populate lazily so a deserialised state is valid at any stage boundary.

PLAN §1.8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CsmTtsState:
    """State threaded through preprocessing → audio_encoder → tts_engine →
    vocoder.

    ``context`` entries are ``{speaker_id, text, codes_F32 | waveform}``;
    ``context_codes`` holds one CPU int32 ``[F, 32]`` matrix per context
    segment; ``num_ctx_codes_consumed`` is the chunked-prefill overlay cursor
    (PLAN §2.4); ``max_new_tokens`` counts FRAMES (1 frame = 1 backbone
    position, exactly like HF); ``output_frames`` is a list of ``[32]``
    int rows and INCLUDES the EOS frame (HF ``sequences`` parity — the
    vocoder never receives it).
    """

    # preprocessing
    text: str | None = None
    speaker_id: int = 0
    context: list[dict[str, Any]] = field(default_factory=list)

    # preprocessing / audio_encoder
    prompt_ids: list[int] = field(default_factory=list)
    context_codes: list[Any] | None = None  # per segment: CPU int32 [F, 32]
    num_ctx_codes_consumed: int = 0

    # generation params (dual sampling sets; HF defaults T=0.9 / top_k=50)
    max_new_tokens: int = 125  # frames (DEFAULT_MAX_FRAMES)
    temperature: float = 0.9
    top_k: int = 50
    top_p: float | None = None
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    seed: int | None = None
    stream: bool = False

    # tts_engine outputs
    output_frames: list[Any] | None = None  # list of [32] int rows
    prompt_tokens: int = 0
    completion_frames: int = 0
    engine_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for ``StagePayload.data`` (sparse: omit None/empty;
        higgs pattern)."""
        raise NotImplementedError("skeleton — PLAN §1.8 CsmTtsState.to_dict")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CsmTtsState":
        """Rebuild from ``StagePayload.data`` produced by :meth:`to_dict`."""
        raise NotImplementedError("skeleton — PLAN §1.8 CsmTtsState.from_dict")


__all__ = ["CsmTtsState"]
