# SPDX-License-Identifier: Apache-2.0
# Follows the sglang-omni Higgs `StreamingSimpleScheduler` streaming-vocoder pattern.
"""Streaming vocoder stage for CSM — Mimi decode with overlap-trim seam
handling at 12.5 Hz.

Follows the higgs ``StreamingSimpleScheduler`` lifecycle exactly (latch
contract, ``_pending_done`` buffering, abort → ``clear_stream_state``, slim
terminal result). Key delta vs higgs: CSM has NO delay pattern,
so ``raw_frames_available = len(rows)`` — frame 0 is decodable the moment it
arrives (no ``rows - N + 1`` reversal warmup). One row = one ``[32]`` tensor
= 80 ms = 1920 samples.

Frame-math knobs at 12.5 Hz (starting points; a
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
seam-free without crossfade. This STATELESS overlap-trim decode is the default;
the optional stateful ``decoder_past_key_values`` path is mutually exclusive with
overlap re-decode and stays gated on an A/B.

CUDA graphs (default-ON; MOSS-TTS vocoder pattern, upstream #798/#886):
at boot :meth:`CsmStreamingVocoderScheduler.warmup_now` captures one graph
per streaming window length T at B=1 over the STATELESS
``MimiModel.decode`` and attaches the sealed runner to the codec; every
``codec.decode`` with a captured T replays bit-identically, everything else
(uncaptured T, batched decodes, the optional stateful
``decoder_past_key_values`` path, CPU boxes) serves eager. Escape hatch:
``cuda_graph=False`` (ctor) or env ``CSM_VOCODER_CUDA_GRAPH=0``. See
``vocoder_cuda_graph.py`` for the exact capture boundary and why the
stateless path needs no MOSS-style cache patch.

The three OOB fences: (1) engine never streams EOS/STOP frames;
(2) ``sanitize_for_mimi`` clamp + counter on every matrix entering Mimi;
(3) final flush / non-streaming trim at the first all-32-zero frame (HF
cutoff semantics, generation_csm.py:471-477 — mirrors HF's stop(31)/trim(32)
inconsistency exactly).


"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.csm_tts.audio_codec import (
    CsmMimiCodec,
    sanitize_for_mimi,
    trim_at_first_all_zero_frame,
)
from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.models.tts_streaming import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage, load_state
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler

logger = logging.getLogger(__name__)

__all__ = ["CsmStreamState", "CsmStreamingVocoderScheduler"]

# Known CSM stream contract; used as the latch default when neither the
# payload nor the chunk metadata carries the fields (CsmTtsState has no
# num_codebooks/codebook_size fields — they're model constants).
_NUM_CODEBOOKS = 32
_CODEBOOK_VOCAB = 2051

# Operator escape hatch for the vocoder CUDA graphs (the ctor toggle is not
# reachable from YAML yet — create_vocoder_executor's signature is frozen this
# pass). Default ON; set CSM_VOCODER_CUDA_GRAPH=0 to serve eager.
_CUDA_GRAPH_ENV = "CSM_VOCODER_CUDA_GRAPH"


def _env_cuda_graph_enabled() -> bool:
    return os.environ.get(_CUDA_GRAPH_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


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
    past_key_values: Any | None = None  # stateful path ONLY
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
        cuda_graph: bool = True,
        cuda_graph_frames: list[int] | None = None,
        cuda_graph_min_free_gb: float = 1.0,
    ) -> None:
        """Validate knobs, hold the shared :class:`CsmMimiCodec`, init the
        per-request ``_stream_states`` dict, then ``super().__init__(
        self._vocode_payload, batch_compute_fn=self._vocode_payloads,
        max_batch_size=..., max_batch_wait_ms=...)`` (higgs ctor shape),
        then :meth:`warmup_now` (boot-time CUDA-graph capture; no-op off
        CUDA / when toggled off)."""
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream_stride and stream_followup_stride must be > 0")
        if stream_overlap_frames < 0:
            raise ValueError("stream_overlap_frames must be >= 0")
        if stream_holdback_frames < 0:
            raise ValueError("stream_holdback_frames must be >= 0")
        if cuda_graph_min_free_gb < 0:
            raise ValueError(
                "cuda_graph_min_free_gb must be >= 0 (0 disables the VRAM "
                f"headroom guard); got {cuda_graph_min_free_gb}"
            )
        if cuda_graph_frames is not None:
            invalid = [t for t in cuda_graph_frames if int(t) < 1]
            if not cuda_graph_frames or invalid:
                raise ValueError(
                    "cuda_graph_frames must be a non-empty list of positive "
                    f"ints (>= 1); got {cuda_graph_frames}"
                )

        self._codec = codec
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_frames = int(stream_overlap_frames)
        self._stream_holdback_frames = int(stream_holdback_frames)
        self._sample_rate = CsmMimiCodec.SAMPLE_RATE
        self._samples_per_frame = CsmMimiCodec.SAMPLES_PER_FRAME
        self._stream_states: dict[str, CsmStreamState] = {}
        self._clamp_count_total = 0  # process-wide fence-#2 counter
        # Effective toggle = ctor AND env (env is the operator escape hatch).
        self._cuda_graph = bool(cuda_graph) and _env_cuda_graph_enabled()
        self._cuda_graph_frames = (
            sorted({int(t) for t in cuda_graph_frames}) if cuda_graph_frames else None
        )
        self._cuda_graph_min_free_gb = float(cuda_graph_min_free_gb)
        self._cuda_graph_warmup_attempted = False

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )
        self.warmup_now()

    # --- StreamingSimpleScheduler hooks ---------------------------------------

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        """``bool(payload.request.params.get("stream", False))``."""
        params = payload.request.params
        if not isinstance(params, dict):
            raise TypeError(
                f"CSM request params must be a dict, got {type(params).__name__}"
            )
        return bool(params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        """Latch the stream contract (``num_codebooks=32`` /
        ``codebook_size=2051``) + ``initial_codec_chunk_frames`` from the
        payload/params (higgs latch discipline; the stage runs with
        ``can_accept_stream_before_payload=True``)."""
        state = self._stream_states.setdefault(request_id, CsmStreamState())
        data = payload.data if isinstance(payload.data, dict) else {}
        self._latch_stream_contract(
            request_id,
            state,
            num_codebooks=data.get(
                "num_codebooks", state.num_codebooks or _NUM_CODEBOOKS
            ),
            codebook_size=data.get(
                "codebook_size", state.codebook_size or _CODEBOOK_VOCAB
            ),
            source="payload",
        )
        params = (
            payload.request.params if isinstance(payload.request.params, dict) else None
        )
        self._latch_initial_codec_chunk_frames_from_mapping(state, params)

    def on_stream_chunk(self, request_id: str, item: Any) -> list[OutgoingMessage]:
        """Append the ``[32]`` int64 row from ``item.data`` to the request's
        ``rows`` and run :meth:`_decode_delta`.

        Returns:
            Zero or more audio-chunk ``OutgoingMessage``s (raw PCM payloads of
            exactly ``new_frames * 1920`` samples each).
        """
        state = self._stream_states.setdefault(request_id, CsmStreamState())
        self._latch_stream_metadata(request_id, state, item.metadata)

        row = item.data
        if not isinstance(row, torch.Tensor):
            raise TypeError(
                f"CSM stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(row).__name__}"
            )
        row = row.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        num_codebooks = state.num_codebooks or _NUM_CODEBOOKS
        if int(row.shape[0]) != num_codebooks:
            raise ValueError(
                f"CSM stream chunk has {int(row.shape[0])} codebooks, "
                f"expected {num_codebooks}"
            )
        state.rows.append(row)
        return self._decode_delta(request_id, state)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        """Final flush (emit all held-back frames, trimmed at the first
        all-32-zero frame — fence #3) + slim terminal result with usage
        (prompt_tokens, completion_frames, engine_time_s). Zero-chunk streams
        (frame-0 EOS) produce a well-formed empty/near-empty
        audio response matching HF ``cutoff_idx=0``."""
        payload = self._stream_payloads[request_id]
        state = self._stream_states.setdefault(request_id, CsmStreamState())
        messages = self._decode_delta(request_id, state, final=True)
        if not messages and state.frames_emitted == 0:
            # Zero-chunk stream (frame-0 EOS): fall back to the engine payload's
            # output_frames with exact HF trim parity — empty when cb31 == 0,
            # one 80 ms frame when cb31 != 0.
            output = self._audio_payload_from_stage_payload(request_id, state, payload)
            if output is not None:
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=output,
                        metadata={"modality": "audio"},
                    )
                )
        state.done = True

        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = self._build_usage(load_state(payload, CsmTtsState))
        if usage is not None:
            final_data["usage"] = usage
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        """Drop the per-request :class:`CsmStreamState` INCLUDING any Mimi
        ``past_key_values`` — the cross-request state-leak guard (abort path
        calls this too)."""
        self._stream_states.pop(request_id, None)

    # --- CUDA-graph warmup (MOSS #798/#886 pattern) -----------------------------

    def warmup_now(self) -> None:
        """Boot-time capture: build the runner, capture every T in
        :meth:`_cuda_graph_capture_frames`, seal, attach to the codec — never
        lazily in the serving loop (MOSS discipline: warmup → capture →
        seal). Attempted at most once; a no-op when the toggle is off or the
        codec is not a CUDA :class:`CsmMimiCodec` (CPU boxes and unit-test
        mock codecs serve eager). Capture failures degrade to eager, never
        crash boot."""
        if self._cuda_graph_warmup_attempted:
            return
        self._cuda_graph_warmup_attempted = True
        if not self._cuda_graph or not self._codec_on_cuda():
            return
        from sglang_omni.models.csm_tts.vocoder_cuda_graph import (
            CsmMimiVocoderCudaGraphRunner,
        )

        frames = self._cuda_graph_capture_frames()
        runner = CsmMimiVocoderCudaGraphRunner(
            self._codec.model,
            num_codebooks=_NUM_CODEBOOKS,
            max_frames=max(frames),
            min_free_gb=self._cuda_graph_min_free_gb,
        )
        try:
            runner.warmup(frames)
        except Exception:
            logger.exception(
                "CSM vocoder CUDA-graph warmup failed; serving eager"
            )
            return
        if not runner.captured_frames():
            # Nothing captured (low VRAM / all captures failed): do not
            # attach, so serving skips the wasted per-decode replay probe.
            logger.warning(
                "CSM vocoder CUDA graphs: nothing captured; serving eager"
            )
            return
        self._codec.set_cuda_graph_runner(runner)

    def _codec_on_cuda(self) -> bool:
        """True only for a real CUDA-resident :class:`CsmMimiCodec` (unit
        tests pass mock codecs without ``device``/``model``)."""
        if not torch.cuda.is_available():
            return False
        device = getattr(self._codec, "device", None)
        return (
            getattr(device, "type", None) == "cuda"
            and getattr(self._codec, "model", None) is not None
            and callable(getattr(self._codec, "set_cuda_graph_runner", None))
        )

    def _cuda_graph_capture_frames(self) -> list[int]:
        """Window lengths T to capture. ``cuda_graph_frames`` overrides;
        the default is the CONTIGUOUS ``1..max_streaming_window`` (post-#886
        discipline: no in-range streaming step silently goes eager)."""
        if self._cuda_graph_frames:
            return list(self._cuda_graph_frames)
        return list(range(1, self._max_streaming_window() + 1))

    def _max_streaming_window(self) -> int:
        """Upper bound on every decode window ``_decode_delta`` can produce
        (T = emit_until - window_start; rows arrive one per engine step, so
        emission fires exactly when ``available == need``):

        - first default chunk: ``stride - holdback`` (window starts at 0);
        - first-chunk TTFA knob: ``initial <= stride - 1``;
        - steady chunk: ``followup + overlap``;
        - post-initial chunk: ``<= followup + overlap``;
        - final flush, nothing emitted: ``available <= stride - 1`` (one
          more row would have emitted) → ``T <= stride - 1``;
        - final flush after emitting: ``available <= emitted + followup +
          holdback - 1`` → ``T <= followup + holdback + overlap - 1``.

        Non-streaming full-utterance decodes (up to the 125-frame cap) are
        NOT bounded by this and intentionally serve eager."""
        return max(
            self._stream_stride - self._stream_holdback_frames,
            self._stream_stride - 1,
            self._stream_followup_stride + self._stream_overlap_frames,
            self._stream_followup_stride
            + self._stream_holdback_frames
            + self._stream_overlap_frames
            - 1,
            1,
        )

    # --- latch helpers ----------------------------------------------------------

    def _latch_stream_metadata(
        self,
        request_id: str,
        state: CsmStreamState,
        metadata: dict[str, Any] | None,
    ) -> None:
        if not isinstance(metadata, dict):
            # Engine chunks always carry stream_metadata; tolerate absence by
            # falling back to the CSM constants (latched-once discipline kept).
            self._latch_stream_contract(
                request_id,
                state,
                num_codebooks=state.num_codebooks or _NUM_CODEBOOKS,
                codebook_size=state.codebook_size or _CODEBOOK_VOCAB,
                source="stream metadata defaults",
            )
            return
        if metadata.get("modality") not in (None, "audio_codes"):
            raise ValueError(
                f"CSM stream chunk modality must be audio_codes, got "
                f"{metadata.get('modality')!r}"
            )
        if metadata.get("stream") is not True:
            raise RuntimeError(
                f"CSM stream chunk for {request_id!r} must include "
                "metadata['stream'] == True"
            )
        self._latch_stream_contract(
            request_id,
            state,
            num_codebooks=metadata.get(
                "num_codebooks", state.num_codebooks or _NUM_CODEBOOKS
            ),
            codebook_size=metadata.get(
                "codebook_size", state.codebook_size or _CODEBOOK_VOCAB
            ),
            source="stream metadata",
        )
        if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
            self._latch_initial_codec_chunk_frames_from_mapping(state, metadata)

    @staticmethod
    def _latch_stream_contract(
        request_id: str,
        state: CsmStreamState,
        *,
        num_codebooks: Any,
        codebook_size: Any,
        source: str,
    ) -> None:
        try:
            num_codebooks_i = int(num_codebooks)
            codebook_size_i = int(codebook_size)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"CSM {source} for {request_id!r} must include integer "
                "num_codebooks and codebook_size"
            ) from exc
        if num_codebooks_i != _NUM_CODEBOOKS or codebook_size_i != _CODEBOOK_VOCAB:
            raise ValueError(
                f"CSM {source} for {request_id!r} has invalid contract "
                f"num_codebooks={num_codebooks_i}, codebook_size={codebook_size_i}; "
                f"expected {_NUM_CODEBOOKS}/{_CODEBOOK_VOCAB}"
            )
        if state.num_codebooks is not None and state.num_codebooks != num_codebooks_i:
            raise ValueError(
                f"CSM stream num_codebooks changed for {request_id!r}: "
                f"{state.num_codebooks} -> {num_codebooks_i}"
            )
        if state.codebook_size is not None and state.codebook_size != codebook_size_i:
            raise ValueError(
                f"CSM stream codebook_size changed for {request_id!r}: "
                f"{state.codebook_size} -> {codebook_size_i}"
            )
        state.num_codebooks = num_codebooks_i
        state.codebook_size = codebook_size_i

    def _latch_initial_codec_chunk_frames_from_mapping(
        self,
        state: CsmStreamState,
        params: Mapping[str, Any] | None,
    ) -> None:
        # No delay pattern: the steady first chunk IS stream_stride frames.
        # Only overwrite when the mapping carries the key — chunks (and their
        # metadata latch) may legally arrive BEFORE the engine payload.
        if params is not None and INITIAL_CODEC_CHUNK_FRAMES_PARAM in params:
            state.initial_codec_chunk_frames = resolve_initial_codec_chunk_frames(
                params, steady_chunk_frames=self._stream_stride
            )
        elif state.initial_codec_chunk_frames is None:
            state.initial_codec_chunk_frames = 0

    # --- decode internals -------------------------------------------------------

    def _decode_delta(
        self, request_id: str, state: CsmStreamState, *, final: bool = False
    ) -> list[OutgoingMessage]:
        """Stateless overlap-trim incremental decode:
        when enough new frames are available per stride/first-chunk knobs,
        re-decode ``rows[emitted - overlap : available - holdback]`` through
        :func:`sanitize_for_mimi` + ``codec.decode``, trim ``overlap * 1920``
        leading samples, emit exactly ``new_frames * 1920``.

        Returns:
            Audio-chunk messages (possibly empty when below stride).
        """
        available = len(state.rows)
        if available == 0:
            return []

        if final:
            # Fence #3: never decode past the first all-32-zero frame.
            rows_F32 = torch.stack(state.rows, dim=0)
            emit_until = int(trim_at_first_all_zero_frame(rows_F32).shape[0])
        else:
            initial = state.initial_codec_chunk_frames or 0
            has_emitted = state.frames_emitted > 0
            if not has_emitted:
                use_initial = 0 < initial < self._stream_stride
                need = initial if use_initial else self._stream_stride
                if available < need:
                    return []
                emit_until = (
                    initial if use_initial else available - self._stream_holdback_frames
                )
            else:
                need = (
                    state.frames_emitted
                    + self._stream_followup_stride
                    + self._stream_holdback_frames
                )
                if available < need:
                    return []
                emit_until = available - self._stream_holdback_frames
        if emit_until <= state.frames_emitted:
            return []

        window_start = max(0, state.frames_emitted - self._stream_overlap_frames)
        codes = torch.stack(state.rows[window_start:emit_until], dim=0)  # fresh copy
        codes, num_clamped = sanitize_for_mimi(codes)
        if num_clamped:
            state.clamp_count += num_clamped
            self._clamp_count_total += num_clamped
            logger.warning(
                "CSM vocoder stream %s clamped %d OOB codes (fence #2 should "
                "never fire at temp=0)",
                request_id,
                num_clamped,
            )
        audio = self._codec.decode(codes).reshape(-1)  # [decoded_frames * 1920]

        trim_samples = (state.frames_emitted - window_start) * self._samples_per_frame
        if final:
            delta = audio[trim_samples:]
        else:
            new_frames = emit_until - state.frames_emitted
            delta = audio[
                trim_samples : trim_samples + new_frames * self._samples_per_frame
            ]
        delta = delta.contiguous()
        if delta.numel() == 0:
            return []

        state.frames_emitted = emit_until
        state.samples_emitted += int(delta.numel())
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=self._audio_payload(delta, source_hint="CSM TTS streaming"),
                metadata={"modality": "audio"},
            )
        ]

    def _audio_payload_from_stage_payload(
        self, request_id: str, state: CsmStreamState, payload: StagePayload
    ) -> dict[str, Any] | None:
        """Full-payload decode fallback for streams that never emitted —
        exact HF trim semantics over ``state.output_frames`` (incl. the EOS
        frame)."""
        if not isinstance(payload.data, dict):
            return None
        codes = self._frames_to_codes(load_state(payload, CsmTtsState).output_frames)
        if codes is None:
            return None
        codes, num_clamped = sanitize_for_mimi(codes)
        if num_clamped:
            state.clamp_count += num_clamped
            self._clamp_count_total += num_clamped
            logger.warning(
                "CSM vocoder fallback decode for %s clamped %d OOB codes",
                request_id,
                num_clamped,
            )
        return self._audio_payload(
            self._codec.decode(codes), source_hint="CSM TTS streaming"
        )

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        """Non-streaming single decode (delegates to :meth:`_vocode_payloads`)."""
        return self._vocode_payloads([payload])[0]

    def _vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        """Non-streaming batch decode over ``state.output_frames`` via
        ``codec.decode_batch`` (exact-frame-count buckets), trimming at the
        first all-32-zero frame (exact HF trim parity — the EOS frame IS in
        ``output_frames``)."""
        items: list[tuple[CsmTtsState, torch.Tensor | None]] = []
        for payload in payloads:
            state = load_state(payload, CsmTtsState)
            codes = self._frames_to_codes(state.output_frames)
            if codes is not None:
                codes, num_clamped = sanitize_for_mimi(codes)
                if num_clamped:
                    self._clamp_count_total += num_clamped
                    logger.warning(
                        "CSM vocoder batch decode for %s clamped %d OOB codes",
                        payload.request_id,
                        num_clamped,
                    )
            items.append((state, codes))

        valid = [(i, codes) for i, (_, codes) in enumerate(items) if codes is not None]
        waveforms: list[torch.Tensor | None] = [None] * len(items)
        if valid:
            indices, codes_list = zip(*valid)
            wavs = self._codec.decode_batch(list(codes_list))
            if len(wavs) != len(valid):
                raise RuntimeError(
                    f"CSM vocoder decode_batch returned {len(wavs)} audios "
                    f"for {len(valid)} requests"
                )
            for idx, wav in zip(indices, wavs):
                waveforms[idx] = wav

        results: list[StagePayload] = []
        for payload, (state, _), waveform in zip(payloads, items, waveforms):
            data = self._audio_payload(
                waveform if waveform is not None else [],
                source_hint="CSM TTS vocoder",
            )
            usage = self._build_usage(state)
            if usage is not None:
                data["usage"] = usage
            payload.data = data
            results.append(payload)
        return results

    @staticmethod
    def _frames_to_codes(frames: list[Any] | None) -> torch.Tensor | None:
        """``output_frames`` rows → trimmed int64 ``[F, 32]`` (or None when
        nothing decodable remains — e.g. a frame-0 EOS with cb31 == 0)."""
        if not frames:
            return None
        rows = [torch.as_tensor(row, dtype=torch.long).reshape(-1) for row in frames]
        codes = torch.stack(rows, dim=0)
        codes = trim_at_first_all_zero_frame(codes)
        if codes.shape[0] == 0:
            return None
        return codes

    def _audio_payload(self, audio: Any, *, source_hint: str) -> dict[str, Any]:
        from sglang_omni.utils.audio_payload import audio_waveform_payload

        return audio_waveform_payload(
            audio,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint=source_hint,
        )

    @staticmethod
    def _build_usage(state: CsmTtsState) -> dict[str, Any] | None:
        """Shared PipelineStateBase usage dict (#807) plus CSM's
        model-native ``completion_frames`` (== completion_tokens; 1 frame =
        1 backbone position)."""
        usage = build_usage(state)
        if usage is not None:
            usage["completion_frames"] = state.completion_frames
        return usage
