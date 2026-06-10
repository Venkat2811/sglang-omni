# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder stage for CSM — Mimi decode with overlap-trim seam
handling at 12.5 Hz.

Follows the higgs ``StreamingSimpleScheduler`` lifecycle exactly (latch
contract, ``_pending_done`` buffering, abort → ``clear_stream_state``, slim
terminal result). Key delta vs higgs (PLAN §4.1): CSM has NO delay pattern,
so ``raw_frames_available = len(rows)`` — frame 0 is decodable the moment it
arrives (no ``rows - N + 1`` reversal warmup). One row = one ``[32]`` tensor
= 80 ms = 1920 samples.

Frame-math knobs at 12.5 Hz (PLAN §4.1; starting points — M3 gate includes a
seam-audibility A/B over overlap ∈ {2, 4, 6}):

- ``stream_stride=13`` (~1.04 s first decode; overridden to 1 by the
  first-chunk knob on streaming paths)
- ``stream_followup_stride=6`` (480 ms steady chunks)
- ``stream_overlap_frames=4`` (320 ms re-decoded left context — Mimi's decode
  path has no conv-state cache, edge frames differ per the HF docstring)
- ``stream_holdback_frames=2`` (160 ms; never emit codec-edge frames
  mid-stream; final flush emits all)

Emission per delta: re-decode ``[emitted - overlap : available - holdback]``,
trim ``overlap * 1920`` samples, emit exactly ``new_frames * 1920`` —
seam-free without crossfade. M2/M3 use this STATELESS overlap-trim decode;
the M4+ stateful ``decoder_past_key_values`` path is mutually exclusive with
overlap re-decode (PLAN §4.3) and stays gated on an A/B.

The three OOB fences (PLAN §4.2): (1) engine never streams EOS/STOP frames;
(2) ``sanitize_for_mimi`` clamp + counter on every matrix entering Mimi;
(3) final flush / non-streaming trim at the first all-32-zero frame (HF
cutoff semantics, generation_csm.py:471-477 — mirrors HF's stop(31)/trim(32)
inconsistency exactly).

PLAN §1.13, §4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.csm_tts.audio_codec import CsmMimiCodec, sanitize_for_mimi
from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler

__all__ = ["CsmStreamState", "CsmStreamingVocoderScheduler"]


@dataclass
class CsmStreamState:
    """Per-request streaming state (dropped in ``clear_stream_state``)."""

    rows: list[torch.Tensor] = field(default_factory=list)  # each int64 [32]
    frames_emitted: int = 0
    samples_emitted: int = 0
    num_codebooks: int | None = None  # latched: 32
    codebook_size: int | None = None  # latched: 2051
    initial_codec_chunk_frames: int | None = None
    clamp_count: int = 0  # sanitize_for_mimi telemetry (gate only at temp=0)
    past_key_values: Any | None = None  # M4+ stateful path ONLY
    done: bool = False


class CsmStreamingVocoderScheduler(StreamingSimpleScheduler):
    """Decode CSM code rows incrementally; batched final decode for the
    non-streaming path (``_vocode_payloads`` + ``decode_batch``)."""

    def __init__(
        self,
        codec: CsmMimiCodec,
        *,
        stream_stride: int = 13,
        stream_followup_stride: int = 6,
        stream_overlap_frames: int = 4,
        stream_holdback_frames: int = 2,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
    ) -> None:
        """Validate knobs, hold the shared :class:`CsmMimiCodec`, init the
        per-request ``_stream_states`` dict, then ``super().__init__(
        self._vocode_payload, batch_compute_fn=self._vocode_payloads,
        max_batch_size=..., max_batch_wait_ms=...)`` (higgs ctor shape)."""
        raise NotImplementedError("skeleton — PLAN §1.13 CsmStreamingVocoderScheduler.__init__")

    # --- StreamingSimpleScheduler hooks ---------------------------------------

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        """``bool(payload.request.params.get("stream", False))``."""
        raise NotImplementedError("skeleton — PLAN §1.13 is_streaming_payload")

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        """Latch the stream contract (``num_codebooks=32`` /
        ``codebook_size=2051``) + ``initial_codec_chunk_frames`` from the
        payload/params (higgs latch discipline; the stage runs with
        ``can_accept_stream_before_payload=True``)."""
        raise NotImplementedError("skeleton — PLAN §1.13 on_streaming_new_request")

    def on_stream_chunk(self, request_id: str, item: Any) -> list[OutgoingMessage]:
        """Append the ``[32]`` int64 row from ``item.data`` to the request's
        ``rows`` and run :meth:`_decode_delta`.

        Returns:
            Zero or more audio-chunk ``OutgoingMessage``s (raw PCM payloads of
            exactly ``new_frames * 1920`` samples each).
        """
        raise NotImplementedError("skeleton — PLAN §1.13 on_stream_chunk")

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        """Final flush (emit all held-back frames, trimmed at the first
        all-32-zero frame — fence #3) + slim terminal result with usage
        (prompt_tokens, completion_frames, engine_time_s). Zero-chunk streams
        (frame-0 EOS, PLAN §1.7/F7) produce a well-formed empty/near-empty
        audio response matching HF ``cutoff_idx=0``."""
        raise NotImplementedError("skeleton — PLAN §1.13 on_stream_done")

    def clear_stream_state(self, request_id: str) -> None:
        """Drop the per-request :class:`CsmStreamState` INCLUDING any Mimi
        ``past_key_values`` — the cross-request state-leak guard (abort path
        calls this too)."""
        raise NotImplementedError("skeleton — PLAN §1.13 clear_stream_state")

    # --- decode internals -------------------------------------------------------

    def _decode_delta(
        self, request_id: str, state: CsmStreamState, *, final: bool = False
    ) -> list[OutgoingMessage]:
        """Stateless overlap-trim incremental decode (PLAN §4.1):
        when enough new frames are available per stride/first-chunk knobs,
        re-decode ``rows[emitted - overlap : available - holdback]`` through
        :func:`sanitize_for_mimi` + ``codec.decode``, trim ``overlap * 1920``
        leading samples, emit exactly ``new_frames * 1920``.

        Returns:
            Audio-chunk messages (possibly empty when below stride).
        """
        raise NotImplementedError("skeleton — PLAN §1.13/§4.1 _decode_delta")

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        """Non-streaming single decode (delegates to :meth:`_vocode_payloads`)."""
        raise NotImplementedError("skeleton — PLAN §1.13 _vocode_payload")

    def _vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        """Non-streaming batch decode over ``state.output_frames`` via
        ``codec.decode_batch`` (exact-frame-count buckets), trimming at the
        first all-32-zero frame (exact HF trim parity — the EOS frame IS in
        ``output_frames``)."""
        raise NotImplementedError("skeleton — PLAN §1.13 _vocode_payloads")

    @staticmethod
    def _build_usage(state: CsmTtsState) -> dict[str, Any] | None:
        """Usage dict for the terminal result (prompt_tokens,
        completion_frames, engine_time_s)."""
        raise NotImplementedError("skeleton — PLAN §1.13 _build_usage")
