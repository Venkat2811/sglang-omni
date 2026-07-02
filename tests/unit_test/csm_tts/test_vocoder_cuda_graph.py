# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the CSM Mimi vocoder CUDA-graph adoption (MOSS
#798/#886 pattern): toggle plumbing (default-on, env off), eager-fallback
selection, capture-bucket derivation, and the codec<->runner replay contract
with fake runners.

NOTHING here mocks ``torch.cuda`` — a mocked "bit-identity" test would be a
lie. The REAL bit-identity gate (``torch.equal`` graph-vs-eager on the actual
Mimi weights) is the CUDA-gated test at the bottom and runs on the RTX 3060
with ``CSM_CKPT=<sesame/csm-1b dir>``.

Post-3060 regression tier: the first GPU run found transformers' Mimi decode
aborting capture (unpinned H2D copy in the RVQ zero-init; D2H sync from the
int64-buffer padding math in ``F.pad``) — the graph sealed EMPTY and replay
was a no-op while the CPU mocked-replay tests stayed green. Two nets below:
the runner's boot-time ``_replay_self_check`` (no-op / non-bit-identical
replay must raise → that T drops to eager) and CPU value-identity checks of
the two capture-legality patches against real transformers modules. The
capture-legality violations THEMSELVES only manifest under real stream
capture, so proving capture works stays GPU-tier-only.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.csm_tts.audio_codec import CsmMimiCodec
from sglang_omni.models.csm_tts.vocoder_cuda_graph import (
    CsmMimiVocoderCudaGraphRunner,
)
from sglang_omni.models.csm_tts.vocoder_scheduler import (
    CsmStreamingVocoderScheduler,
    _env_cuda_graph_enabled,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem

SPF = CsmMimiCodec.SAMPLES_PER_FRAME  # 1920
_ENV = "CSM_VOCODER_CUDA_GRAPH"
_STREAM_META = {
    "modality": "audio_codes",
    "stream": True,
    "num_codebooks": 32,
    "codebook_size": 2051,
}


class _FakeMimiModel(torch.nn.Module):
    """Deterministic stand-in for ``transformers.MimiModel``: frame f of the
    decoded audio is 1920 samples all equal to float(cb0 code of f), so every
    output sample is traceable to its source frame. Records the T of every
    decode call (bucket-coverage assertions)."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(sampling_rate=24000, frame_rate=12.5)
        self.decode_calls = 0
        self.seen_ts: list[int] = []

    def decode(self, codes_B32F: torch.Tensor, **kwargs):
        assert codes_B32F.ndim == 3 and codes_B32F.shape[1] == 32
        self.decode_calls += 1
        self.seen_ts.append(int(codes_B32F.shape[-1]))
        ids = codes_B32F[:, 0, :].to(torch.float32)  # [B, F]
        audio = ids.repeat_interleave(SPF, dim=-1).unsqueeze(1)  # [B, 1, F*1920]
        return SimpleNamespace(audio_values=audio)


class _FakeRunner:
    """Scripted runner double for the codec<->runner replay contract."""

    def __init__(self, audio: torch.Tensor | None = None, exc: Exception | None = None):
        self.calls: list[torch.Tensor] = []
        self._audio = audio
        self._exc = exc

    def decode(self, codes_B32F: torch.Tensor) -> torch.Tensor | None:
        self.calls.append(codes_B32F.detach().clone())
        if self._exc is not None:
            raise self._exc
        return self._audio


def _codec() -> CsmMimiCodec:
    return CsmMimiCodec(_FakeMimiModel(), device="cpu", dtype=torch.float32)


def _codes(num_frames: int, value: int = 7) -> torch.Tensor:
    return torch.full((num_frames, 32), value, dtype=torch.long)


# --- toggle plumbing (default-on, env off) -----------------------------------


def test_toggle_default_on(monkeypatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    sched = CsmStreamingVocoderScheduler(_codec())
    assert sched._cuda_graph is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", " FALSE "])
def test_toggle_env_escape_hatch(monkeypatch, value) -> None:
    monkeypatch.setenv(_ENV, value)
    assert _env_cuda_graph_enabled() is False
    sched = CsmStreamingVocoderScheduler(_codec())
    assert sched._cuda_graph is False


def test_toggle_env_on_values(monkeypatch) -> None:
    for value in ("1", "true", "on"):
        monkeypatch.setenv(_ENV, value)
        assert _env_cuda_graph_enabled() is True


def test_toggle_ctor_off_beats_env_on(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    sched = CsmStreamingVocoderScheduler(_codec(), cuda_graph=False)
    assert sched._cuda_graph is False


def test_invalid_cuda_graph_knobs_raise() -> None:
    with pytest.raises(ValueError):
        CsmStreamingVocoderScheduler(_codec(), cuda_graph_frames=[])
    with pytest.raises(ValueError):
        CsmStreamingVocoderScheduler(_codec(), cuda_graph_frames=[0, 5])
    with pytest.raises(ValueError):
        CsmStreamingVocoderScheduler(_codec(), cuda_graph_min_free_gb=-1.0)


# --- eager-fallback selection on CPU ------------------------------------------


def test_cpu_boot_attaches_no_runner_and_serves_eager() -> None:
    """Default-on boot with a CPU codec must not attach a runner (also holds
    on a CUDA box: the CODEC device gates, not the host) and decode must run
    the eager model path."""
    codec = _codec()
    CsmStreamingVocoderScheduler(codec)
    assert codec.has_cuda_graph_runner() is False
    out = codec.decode(_codes(3))
    assert out.shape == (1, 3 * SPF)
    assert codec.model.decode_calls == 1
    assert codec.cg_decode_hits == 0 and codec.cg_decode_misses == 0


def test_warmup_now_attempted_once() -> None:
    sched = CsmStreamingVocoderScheduler(_codec())
    assert sched._cuda_graph_warmup_attempted is True  # ctor already ran it
    sched.warmup_now()  # second call is a no-op, not a re-probe
    assert sched._cuda_graph_warmup_attempted is True


def test_runner_on_cpu_seals_empty_and_declines() -> None:
    """A runner built off-CUDA seals with zero graphs and declines every
    decode (None -> caller decodes eager)."""
    runner = CsmMimiVocoderCudaGraphRunner(_FakeMimiModel(), max_frames=12)
    runner.warmup([1, 2, 3])
    assert runner._sealed is True
    assert runner.captured_frames() == []
    assert runner.decode(torch.zeros(1, 32, 3, dtype=torch.long)) is None


def test_runner_frame_count_support_bounds() -> None:
    runner = CsmMimiVocoderCudaGraphRunner(_FakeMimiModel(), max_frames=12)
    assert runner._is_supported_frame_count(1)
    assert runner._is_supported_frame_count(12)
    assert not runner._is_supported_frame_count(0)
    assert not runner._is_supported_frame_count(13)


# --- bucket-selection logic ---------------------------------------------------


def test_default_capture_frames_contiguous_1_to_12() -> None:
    """Default knobs (stride 13 / followup 6 / overlap 4 / holdback 2) must
    yield the CONTIGUOUS T=1..12 set (#886 discipline: no in-range streaming
    step silently goes eager)."""
    sched = CsmStreamingVocoderScheduler(_codec())
    assert sched._cuda_graph_capture_frames() == list(range(1, 13))


def test_capture_frames_override_dedup_sorted() -> None:
    sched = CsmStreamingVocoderScheduler(_codec(), cuda_graph_frames=[5, 3, 5])
    assert sched._cuda_graph_capture_frames() == [3, 5]


def test_max_streaming_window_alt_knobs() -> None:
    # holdback=0: the first default chunk decodes the full stride.
    sched = CsmStreamingVocoderScheduler(_codec(), stream_holdback_frames=0)
    assert sched._max_streaming_window() == 13
    # tiny strides never go below 1
    sched = CsmStreamingVocoderScheduler(
        _codec(),
        stream_stride=1,
        stream_followup_stride=1,
        stream_overlap_frames=0,
        stream_holdback_frames=0,
    )
    assert sched._max_streaming_window() == 1


def _drive_stream(sched, rid, total, initial) -> None:
    meta = dict(_STREAM_META)
    if initial is not None:
        meta["initial_codec_chunk_frames"] = initial
    for i in range(total):
        row = torch.full((32,), (i % 2000) + 1, dtype=torch.long)
        sched.on_stream_chunk(rid, StreamItem(i, row, "tts_engine", meta))
    state = sched._stream_states[rid]
    sched._decode_delta(rid, state, final=True)
    sched.clear_stream_state(rid)


def test_every_streaming_window_fits_default_buckets() -> None:
    """Brute-force the #886 no-silent-eager property: every decode window
    _decode_delta produces — first chunk, TTFA initial-chunk knob, steady
    chunks, final flush — across stream lengths 1..40 and initial knob
    values, must fall inside the default contiguous capture set."""
    model = _FakeMimiModel()
    codec = CsmMimiCodec(model, device="cpu", dtype=torch.float32)
    sched = CsmStreamingVocoderScheduler(codec)
    buckets = set(sched._cuda_graph_capture_frames())
    for initial in (None, 1, 3, 5, 12):
        for total in range(1, 41):
            _drive_stream(sched, f"req-{initial}-{total}", total, initial)
    assert model.seen_ts, "no decode windows were produced"
    outside = sorted({t for t in model.seen_ts if t not in buckets})
    assert not outside, (
        f"streaming windows {outside} fall outside the default capture set "
        f"{sorted(buckets)} — those steps would silently serve eager"
    )


# --- codec <-> runner replay contract (fake runners; no CUDA) -----------------


def test_codec_replay_path_uses_runner_output_and_sanitized_codes() -> None:
    """A runner hit must (a) receive the SANITIZED [1, 32, F] long codes and
    (b) fully determine the decode output (static-buffer copy semantics)."""
    codec = _codec()
    frames = 4
    replay_audio = (
        torch.arange(frames * SPF, dtype=torch.float32).reshape(1, 1, -1) / 7.0
    )
    runner = _FakeRunner(audio=replay_audio)
    codec.set_cuda_graph_runner(runner)

    codes = _codes(frames, value=5)
    codes[0, 0] = 2050  # OOB: sanitize must clamp to 2047 before the runner
    out = codec.decode(codes)

    assert torch.equal(out, replay_audio.squeeze(0))
    assert codec.model.decode_calls == 0  # eager path never ran
    assert len(runner.calls) == 1
    seen = runner.calls[0]
    assert seen.shape == (1, 32, frames) and seen.dtype == torch.long
    assert int(seen.max()) <= 2047
    assert int(codes[0, 0]) == 2050  # caller's tensor never mutated
    assert codec.cg_decode_hits == 1 and codec.cg_decode_misses == 0


def test_codec_none_from_runner_falls_back_eager() -> None:
    """None (uncaptured shape) selects the eager path; output must equal a
    runner-less codec's output for the same codes."""
    reference = _codec().decode(_codes(6))
    codec = _codec()
    runner = _FakeRunner(audio=None)
    codec.set_cuda_graph_runner(runner)
    out = codec.decode(_codes(6))
    assert torch.equal(out, reference)
    assert codec.model.decode_calls == 1
    assert codec.has_cuda_graph_runner() is True  # None is not a failure
    assert codec.cg_decode_hits == 0 and codec.cg_decode_misses == 1


def test_replay_failure_disables_runner_permanently_and_serves_eager() -> None:
    """A replay exception permanently disables the runner; the SAME call
    retries eager (stateless decode => value-safe, unlike MOSS's stateful
    session) and later calls never touch the runner again."""
    reference = _codec().decode(_codes(5))
    codec = _codec()
    runner = _FakeRunner(exc=RuntimeError("simulated replay failure"))
    codec.set_cuda_graph_runner(runner)

    out = codec.decode(_codes(5))
    assert torch.equal(out, reference)
    assert codec.has_cuda_graph_runner() is False
    assert len(runner.calls) == 1

    codec.decode(_codes(5))
    assert len(runner.calls) == 1  # disabled: never probed again
    assert codec.model.decode_calls == 2
    assert codec.cg_decode_hits == 0 and codec.cg_decode_misses == 1


def test_streaming_decode_never_touches_runner() -> None:
    """The OPTIONAL stateful decoder_past_key_values path is never graphed
    (HF DynamicCache grows across calls); it must bypass the runner."""

    class _StatefulFake(_FakeMimiModel):
        def decode(self, codes_B32F, decoder_past_key_values=None, **kwargs):
            out = super().decode(codes_B32F)
            out.decoder_past_key_values = object()
            return out

    codec = CsmMimiCodec(_StatefulFake(), device="cpu", dtype=torch.float32)
    runner = _FakeRunner(audio=torch.zeros(1, 1, 2 * SPF))
    codec.set_cuda_graph_runner(runner)
    wave, _pkv = codec.streaming_decode(_codes(2), None)
    assert wave.shape == (1, 2 * SPF)
    assert runner.calls == []  # stateful path stayed eager by design


# --- capture hygiene (source contract, mirrors the MOSS test) -----------------


def test_capture_uses_thread_local_error_mode() -> None:
    source = textwrap.dedent(
        inspect.getsource(CsmMimiVocoderCudaGraphRunner._capture_frame_count)
    )
    tree = ast.parse(source)
    graph_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "graph"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "torch"
    ]
    assert graph_calls, "CSM Mimi vocoder CUDA graph capture call not found"
    assert any(
        keyword.arg == "capture_error_mode"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "thread_local"
        for call in graph_calls
        for keyword in call.keywords
    ), "CSM Mimi vocoder CUDA graph capture must use thread-local error mode"


# --- boot-time replay self-check (RTX 3060 empty-capture regression) ----------


def test_replay_self_check_catches_empty_noop_replay() -> None:
    """Regression net for the observed 3060 failure mode: capture aborts
    inside the model (capture_end warns "The CUDA Graph is empty"), replay
    is a NO-OP, and the static output keeps stale garbage. The boot-time
    self-check poisons the output first, so a no-op replay must raise and
    warmup drops that T to eager instead of shipping corrupt audio."""
    static = torch.zeros(1, 1, 4 * SPF)
    ref = torch.randn(1, 1, 4 * SPF)
    with pytest.raises(RuntimeError, match="recorded no work"):
        CsmMimiVocoderCudaGraphRunner._replay_self_check(lambda: None, static, ref)


def test_replay_self_check_rejects_non_bit_identical_replay() -> None:
    static = torch.zeros(1, 1, SPF)
    ref = torch.randn(1, 1, SPF)

    def replay() -> None:
        static.copy_(ref + 1e-3)

    with pytest.raises(RuntimeError, match="not bit-identical"):
        CsmMimiVocoderCudaGraphRunner._replay_self_check(replay, static, ref)


def test_replay_self_check_passes_bit_identical_replay() -> None:
    static = torch.zeros(1, 1, SPF)
    ref = torch.randn(1, 1, SPF)

    def replay() -> None:
        static.copy_(ref)

    CsmMimiVocoderCudaGraphRunner._replay_self_check(replay, static, ref)


# --- capture-legality patches: value identity on real transformers (CPU) ------


def test_patch_mimi_codec_value_identity_and_idempotence() -> None:
    """The two capture-legality patches (RVQ device-side zero init;
    MimiConv1d host-int padding math) must be bitwise NO-OPS for eager —
    patched outputs torch.equal unpatched outputs on real transformers
    modules — and re-patching must not double-wrap."""
    pytest.importorskip("transformers")
    from transformers.models.mimi.configuration_mimi import MimiConfig
    from transformers.models.mimi.modeling_mimi import (
        MimiConv1d,
        MimiResidualVectorQuantizer,
    )

    from sglang_omni.models.csm_tts.vocoder_cuda_graph import (
        patch_mimi_codec_for_cuda_graph,
    )

    torch.manual_seed(0)
    config = MimiConfig()
    conv = MimiConv1d(config, 4, 8, kernel_size=7, stride=2).eval()
    rvq = MimiResidualVectorQuantizer(config, num_quantizers=2).eval()
    for layer in rvq.layers:  # default embed_sum is all-zero -> randomize
        layer.codebook.embed_sum.normal_()
    root = torch.nn.ModuleDict({"conv": conv, "rvq": rvq})

    x = torch.randn(1, 4, 50)
    codes = torch.randint(0, config.codebook_size, (1, 2, 10))
    with torch.no_grad():
        conv_ref = conv(x)
        rvq_ref = rvq.decode(codes)

    patch_mimi_codec_for_cuda_graph(root)
    assert hasattr(conv, "_sglang_omni_original_conv_forward")
    assert hasattr(rvq, "_sglang_omni_original_rvq_decode")
    with torch.no_grad():
        assert torch.equal(conv(x), conv_ref), "patched MimiConv1d diverged"
        assert torch.equal(rvq.decode(codes), rvq_ref), "patched RVQ diverged"

    decode_before, forward_before = rvq.decode, conv.forward
    patch_mimi_codec_for_cuda_graph(root)  # idempotent: no re-wrap
    assert rvq.decode is decode_before
    assert conv.forward is forward_before


# --- THE bit-identity gate (CUDA + real weights; RTX 3060) --------------------

_HAS_CUDA = torch.cuda.is_available()
_CSM_CKPT = os.environ.get("CSM_CKPT", "")


@pytest.mark.skipif(
    not (_HAS_CUDA and _CSM_CKPT),
    reason="RTX 3060 gate: needs CUDA and CSM_CKPT=<sesame/csm-1b checkpoint dir>",
)
def test_bit_identity_graph_vs_eager_real_mimi() -> None:
    """THE bit-identity gate — REAL Mimi weights, REAL CUDA graphs, zero
    mocks. Runs on the RTX 3060:

        CSM_CKPT=~/models/sesame-csm-1b python -m pytest \
            tests/unit_test/csm_tts/test_vocoder_cuda_graph.py -k bit_identity -q

    For every captured bucket T the replayed audio must ``torch.equal`` the
    eager audio for the same codes (max|delta| == 0); an uncaptured T must
    fall back to eager and still decode."""
    codec = CsmMimiCodec.from_pretrained(_CSM_CKPT, device="cuda", dtype=torch.float32)
    CsmStreamingVocoderScheduler(codec)
    assert codec.has_cuda_graph_runner(), (
        "no vocoder graphs captured on a CUDA box (VRAM guard tripped? "
        "check the CSM Mimi vocoder CG warmup logs)"
    )
    runner = codec._cg_runner
    captured = runner.captured_frames()
    assert captured == list(range(1, 13)), captured

    torch.manual_seed(0)
    for t in captured:
        codes = torch.randint(0, 2048, (t, 32), dtype=torch.long)
        graphed = codec.decode(codes)
        codec.set_cuda_graph_runner(None)  # force eager
        eager = codec.decode(codes)
        codec.set_cuda_graph_runner(runner)
        assert torch.equal(graphed, eager), (
            f"T={t}: graph decode not bit-identical to eager, "
            f"max|delta|={(graphed - eager).abs().max().item():.3e}"
        )
    assert codec.cg_decode_hits == len(captured)

    # Uncaptured T (non-streaming full utterance) -> eager fallback still works.
    long_codes = torch.randint(0, 2048, (125, 32), dtype=torch.long)
    misses_before = codec.cg_decode_misses
    out = codec.decode(long_codes)
    assert out.shape == (1, 125 * SPF)
    assert codec.cg_decode_misses == misses_before + 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
