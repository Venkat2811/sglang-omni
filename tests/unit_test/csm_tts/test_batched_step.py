# SPDX-License-Identifier: Apache-2.0
"""Batched decode-step parity + generation-batch-policy tests (S4, #843).

CSM analogue of higgs test_batched_step.py. CSM has no delay window and no
EOC wind-down, so the higgs phase matrix collapses to: running rows /
done-row freeze (STOP sentinel) / padding-row isolation / finished-row skip.
Parity is asserted greedily (temp=0 collapses every draw to argmax, per the
higgs note: per-row and batched modes draw stochastically in different
orders, so stochastic parity is NOT asserted; greedy + state-machine parity
is exact). The reference for every assertion is per-request independence:
a request's frames must not depend on batch size, batch composition,
padding rows, or which pool row it landed on.

CPU-only; models are the real CsmTTSModel decode tail (cb0 head + depth
decoder + sampler pool + CG buffers) built at test scale via ``__new__`` —
the SGLang Llama backbone is not part of the tail under test, so per-step
backbone hiddens are supplied as fixed per-request inputs (identical in
batched and solo runs).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.csm_tts import modeling
from sglang_omni.models.csm_tts.model import CsmGenParams, CsmTTSModel
from sglang_omni.models.csm_tts.model_runner import CsmTTSModelRunner
from sglang_omni.models.csm_tts.sampler import (
    K_MAX,
    NO_SEED,
    STAGING_WIDTH,
    CsmBatchedSamplerState,
)
from sglang_omni.models.csm_tts.utils import NUM_CODEBOOKS, STOP_CODE

# Real CSM-1B depth rope (same values as test_modeling's tiny config).
_DEPTH_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 0.001953125,
    "high_freq_factor": 0.0078125,
    "original_max_position_embeddings": 16,
}


def _tiny_depth_cfg(
    num_codebooks: int = NUM_CODEBOOKS, vocab_size: int = 51
) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=33,
        backbone_hidden_size=32,
        vocab_size=vocab_size,
        num_codebooks=num_codebooks,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        rope_theta=500000.0,
        rope_scaling=dict(_DEPTH_ROPE_SCALING),
    )


def _randomize(module: torch.nn.Module, std: float = 0.05) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, std)


def _make_tail_model(pool_rows: int = 4, cfg: SimpleNamespace | None = None):
    """The real CsmTTSModel decode tail at test scale (CPU, no backbone).

    Attaches exactly the members the tail methods + runner touch: cb0 head,
    depth decoder (with the tied frame-embedding table), sampler pool with
    the reserved padding row, and every ``_cg_*`` buffer, shaped as in
    ``CsmTTSModel.__init__``.
    """
    cfg = cfg or _tiny_depth_cfg()
    m = CsmTTSModel.__new__(CsmTTSModel)
    nn.Module.__init__(m)
    m.config = cfg
    m._num_codebooks = int(cfg.num_codebooks)
    m._codebook_vocab = int(cfg.vocab_size)
    m.frame_embedding = modeling.CsmFrameEmbedding(
        cfg.num_codebooks, cfg.vocab_size, cfg.backbone_hidden_size
    )
    m.codebook0_head = nn.Linear(cfg.backbone_hidden_size, cfg.vocab_size, bias=False)
    m._max_batch_size = int(pool_rows)
    pool_size = pool_rows + 1
    m.depth_decoder = modeling.CsmDepthDecoder(
        cfg, m.frame_embedding, max_slots=pool_size
    )
    m.codebooks_head = m.depth_decoder.codebooks_head
    m._sampler_pool = CsmBatchedSamplerState(pool_size, device="cpu")
    m._padding_row = pool_rows
    m._rid_to_row = {}
    m._free_rows = list(range(pool_rows))
    m._output_codes = {}
    m._cg_row_indices = torch.zeros(pool_size, dtype=torch.long)
    m._cg_temperature = torch.ones(pool_size)
    m._cg_top_p = torch.ones(pool_size)
    m._cg_top_k_buf = torch.full((pool_size,), K_MAX, dtype=torch.long)
    m._cg_depth_temperature = torch.ones(pool_size)
    m._cg_depth_top_k_buf = torch.full((pool_size,), K_MAX, dtype=torch.long)
    m._cg_codes_BN = torch.zeros(pool_size, NUM_CODEBOOKS, dtype=torch.long)
    m._cg_collect_staging = torch.zeros(pool_size, STAGING_WIDTH, dtype=torch.long)
    m._cg_was_done = torch.zeros(pool_size, dtype=torch.bool)
    m._cg_active_generation_done = torch.zeros(pool_size, dtype=torch.bool)
    m._cg_active_last_codes = torch.zeros(pool_size, NUM_CODEBOOKS, dtype=torch.long)
    m._cg_active_seeds = torch.full((pool_size,), NO_SEED, dtype=torch.long)
    m._cg_active_step_count = torch.zeros(pool_size, dtype=torch.long)
    _randomize(m)
    return m


def _make_runner(model) -> CsmTTSModelRunner:
    tp_worker = SimpleNamespace(gpu_id=0, model_runner=SimpleNamespace(model=model))
    return CsmTTSModelRunner(tp_worker, output_processor=None)


class _FakeReq:
    def __init__(self) -> None:
        self.is_chunked = 0
        self.finished_reason = None
        self.output_ids: list[int] = []

    def finished(self) -> bool:
        return self.finished_reason is not None


def _sched_req(rid: str):
    data = SimpleNamespace(
        req=_FakeReq(),
        output_frames=[],
        generation_done=False,
        stream_metadata=None,
        depth_temperature=0.0,  # greedy depth
        depth_top_k=50,
    )
    return SimpleNamespace(request_id=rid, data=data)


def _greedy_gen_params() -> CsmGenParams:
    return CsmGenParams(temperature=0.0, depth_temperature=0.0)


def _result(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(n, 4)),
        next_token_ids=None,
    )


def _prefill(model, runner, requests, hidden) -> list[int]:
    """Frame 0: eager pool-indexed sample + the runner's prefill collect."""
    with torch.no_grad():
        model.decode_codebooks_batch(
            hidden,
            [r.request_id for r in requests],
            [_greedy_gen_params() for _ in requests],
        )
    result = _result(len(requests))
    runner._collect_step_outputs(result, requests)
    return result.next_token_ids.tolist()


def _decode_step(model, runner, requests, hidden_bs) -> list[int]:
    """One scheduler-visible decode step: populate CG buffers -> tail ->
    pack + collect. ``hidden_bs`` is [bs, H]; bs > len(requests) exercises
    the padding row."""
    n = len(requests)
    bs = int(hidden_bs.shape[0])
    fb = SimpleNamespace(
        batch_size=bs,
        sampling_info=SimpleNamespace(
            temperatures=torch.zeros(n),  # greedy cb0
            top_ps=torch.ones(n),
            top_ks=torch.full((n,), K_MAX, dtype=torch.long),
        ),
    )
    runner.before_decode(fb, None, requests)
    with torch.no_grad():
        model.decode_codebooks_batch_cg(hidden_bs)
    result = _result(n)
    runner._collect_step_outputs_cg(result, fb, requests)
    return result.next_token_ids.tolist()


# ---------------------------------------------------------------------------
# generate_frame: batched == per-row (greedy)
# ---------------------------------------------------------------------------


def test_generate_frame_batched_matches_per_row_greedy() -> None:
    """The 31-step depth loop at bs=N produces, per row, exactly the frame
    the same inputs produce at bs=1 (request independence: batching is an
    implementation detail, not a semantic input)."""
    torch.manual_seed(0)
    cfg = _tiny_depth_cfg()
    fe = modeling.CsmFrameEmbedding(
        cfg.num_codebooks, cfg.vocab_size, cfg.backbone_hidden_size
    )
    dd = modeling.CsmDepthDecoder(cfg, fe, max_slots=5)
    _randomize(dd)
    B = 4
    h = torch.randn(B, cfg.backbone_hidden_size)
    cb0 = torch.randint(0, cfg.vocab_size, (B,))
    temps = torch.zeros(5)  # pool-sized on purpose; greedy
    topks = torch.full((5,), cfg.vocab_size, dtype=torch.long)

    with torch.no_grad():
        batched = dd.generate_frame(h, cb0, temps, topks, bs=B)
        for i in range(B):
            solo = dd.generate_frame(
                h[i : i + 1], cb0[i : i + 1], temps, topks, bs=1
            )
            assert torch.equal(batched[i], solo[0]), f"row {i} differs at bs={B}"


# ---------------------------------------------------------------------------
# decode tail: done-row freeze + STOP sentinel, neighbours unaffected
# ---------------------------------------------------------------------------


def test_decode_tail_done_row_freezes_and_emits_stop(
    cpu_multinomial_with_seed,
) -> None:
    """A row whose generation_done latched BEFORE this frame emits STOP_CODE,
    keeps its last_codes, and does not advance its frame index; the running
    neighbour's frame equals its solo (bs=1) run."""
    torch.manual_seed(1)
    model = _make_tail_model()
    H = model.config.backbone_hidden_size
    hidden = torch.randn(2, H)
    last_running = torch.randint(0, model._codebook_vocab, (NUM_CODEBOOKS,))
    last_done = torch.randint(0, model._codebook_vocab, (NUM_CODEBOOKS,))

    def _load_active(rows_hidden: torch.Tensor, done_flags, last_rows, steps):
        bs = rows_hidden.shape[0]
        model._cg_temperature[:bs] = 0.0  # greedy cb0
        model._cg_depth_temperature[:bs] = 0.0  # greedy depth
        model._cg_active_generation_done[:bs] = torch.tensor(done_flags)
        model._cg_active_last_codes[:bs] = torch.stack(last_rows)
        model._cg_active_seeds[:bs] = NO_SEED
        model._cg_active_step_count[:bs] = torch.tensor(steps, dtype=torch.long)

    # Batched: row 0 running (frame index 3), row 1 already done.
    _load_active(hidden, [False, True], [last_running, last_done], [3, 5])
    with torch.no_grad():
        out = model.decode_codebooks_batch_cg(hidden)

    assert (out[1] == STOP_CODE).all(), "done row must emit the STOP sentinel"
    assert bool(model._cg_was_done[1])
    assert torch.equal(model._cg_active_last_codes[1], last_done), "frozen row mutated"
    assert int(model._cg_active_step_count[1]) == 5, "frozen row advanced its frame idx"
    assert int(model._cg_active_step_count[0]) == 4, "running row must advance"
    assert not bool(model._cg_was_done[0])

    # Solo reference for the running row: same initial state at bs=1.
    _load_active(hidden[0:1], [False], [last_running], [3])
    with torch.no_grad():
        solo = model.decode_codebooks_batch_cg(hidden[0:1])
    assert torch.equal(out[0], solo[0]), "running row affected by its done neighbour"


# ---------------------------------------------------------------------------
# full runner steps: batched (with padding row) == solo, per request
# ---------------------------------------------------------------------------


def test_runner_decode_steps_batched_match_solo_with_padding(
    cpu_multinomial_with_seed,
) -> None:
    """Prefill + 2 decode steps for two requests batched at bs=3 (one live
    padding row fed garbage hidden) produce, per request, the exact frames,
    done flags and published cb0 stream of the same requests run alone at
    bs=1 on an identical model — padding rows and batch composition must be
    unobservable."""
    torch.manual_seed(2)
    model_a = _make_tail_model()
    model_b = _make_tail_model()
    model_b.load_state_dict(model_a.state_dict())
    runner_a = _make_runner(model_a)
    runner_b = _make_runner(model_b)
    H = model_a.config.backbone_hidden_size

    rids = ["r0", "r1"]
    h_prefill = {rid: torch.randn(1, H) for rid in rids}
    h_steps = {rid: [torch.randn(1, H) for _ in range(2)] for rid in rids}
    garbage = [torch.randn(1, H) for _ in range(3)]  # padding-row hidden

    # --- batched run (bs=3: 2 real + 1 padding) ---
    reqs_a = [_sched_req(rid) for rid in rids]
    cb0_a = [_prefill(model_a, runner_a, reqs_a, torch.cat([h_prefill[r] for r in rids]))]
    for step in range(2):
        hidden_bs = torch.cat(
            [h_steps[r][step] for r in rids] + [garbage[step]], dim=0
        )
        cb0_a.append(_decode_step(model_a, runner_a, reqs_a, hidden_bs))

    # --- solo runs on the identical model ---
    frames_solo: dict[str, list[list[int]]] = {}
    done_solo: dict[str, bool] = {}
    cb0_solo: dict[str, list[int]] = {}
    for rid in rids:
        req = _sched_req(rid)
        cb0s = _prefill(model_b, runner_b, [req], h_prefill[rid])
        for step in range(2):
            cb0s += _decode_step(model_b, runner_b, [req], h_steps[rid][step])
        frames_solo[rid] = [f.tolist() for f in req.data.output_frames]
        done_solo[rid] = req.data.generation_done
        cb0_solo[rid] = cb0s

    for i, rid in enumerate(rids):
        frames_batched = [f.tolist() for f in reqs_a[i].data.output_frames]
        assert frames_batched == frames_solo[rid], f"{rid}: frames differ vs solo"
        assert reqs_a[i].data.generation_done == done_solo[rid]
        assert [row[i] for row in cb0_a] == cb0_solo[rid], f"{rid}: cb0 stream differs"


def test_runner_collect_skips_finished_rows(cpu_multinomial_with_seed) -> None:
    """A request that already finished (e.g. a length-cap set by the
    scheduler) contributes no frame and publishes cb0=0, while its live
    neighbour keeps generating — the finished()-skip in the decode collect."""
    torch.manual_seed(3)
    model = _make_tail_model()
    runner = _make_runner(model)
    H = model.config.backbone_hidden_size
    reqs = [_sched_req("r-live"), _sched_req("r-capped")]

    _prefill(model, runner, reqs, torch.randn(2, H))
    frames_before = [len(r.data.output_frames) for r in reqs]

    reqs[1].data.req.finished_reason = "length-cap"  # any non-None finish
    cb0 = _decode_step(model, runner, reqs, torch.randn(2, H))

    assert len(reqs[0].data.output_frames) == frames_before[0] + 1
    assert len(reqs[1].data.output_frames) == frames_before[1], (
        "finished request must not receive an overrun frame"
    )
    assert cb0[1] == 0, "finished rows publish cb0=0"


# ---------------------------------------------------------------------------
# #843 generation batch policy + CUDA-graph buffer probe
# ---------------------------------------------------------------------------


def test_generation_batch_policy_accepts_csm_stage_defaults() -> None:
    """The overrides the CSM stage factory builds (mrr=8, cuda_graph_max_bs=8,
    eager default) pass validate_generation_batch_policy against the model's
    sampler-pool capacity, and the CG bucket list is explicit with
    max(cuda_graph_bs) == cuda_graph_max_bs."""
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
        validate_generation_batch_policy,
    )

    overrides = build_generation_batch_overrides(
        max_running_requests=8,
        cuda_graph_max_bs=8,
        disable_cuda_graph=True,
        mem_fraction_static=0.5,
        chunked_prefill_size=2048,
        dtype="bfloat16",
    )
    assert overrides["cuda_graph_bs"] == [1, 2, 4, 8]
    assert max(overrides["cuda_graph_bs"]) == overrides["cuda_graph_max_bs"]

    model = _make_tail_model(pool_rows=8)
    server_args = SimpleNamespace(enable_torch_compile=False, **overrides)
    validate_generation_batch_policy(
        model_name="CSM TTS",
        server_args=server_args,
        model_buffer_bs=model.sampler_pool_max_running_requests,
    )

    # The property equals the number of actually acquirable rows: the pool
    # serves exactly that many concurrent requests, then fails loudly.
    for i in range(model.sampler_pool_max_running_requests):
        model.acquire_row(f"req-{i}")
    with pytest.raises(RuntimeError, match="sampler pool exhausted"):
        model.acquire_row("req-overflow")


def test_generation_batch_policy_rejects_undersized_pool_and_buckets() -> None:
    """Negative paths: a sampler pool smaller than max_running_requests and a
    CG bucket list whose max mismatches cuda_graph_max_bs both raise."""
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
        validate_generation_batch_policy,
    )

    overrides = build_generation_batch_overrides(
        max_running_requests=8, cuda_graph_max_bs=8, disable_cuda_graph=True
    )
    server_args = SimpleNamespace(enable_torch_compile=False, **overrides)
    model = _make_tail_model(pool_rows=4)  # 4 < mrr=8
    with pytest.raises(ValueError, match="model_buffer_bs must cover"):
        validate_generation_batch_policy(
            model_name="CSM TTS",
            server_args=server_args,
            model_buffer_bs=model.sampler_pool_max_running_requests,
        )

    cg_args = SimpleNamespace(
        enable_torch_compile=False,
        max_running_requests=8,
        disable_cuda_graph=False,
        cuda_graph_max_bs=16,
        cuda_graph_bs=[1, 2, 4, 8],  # max != cuda_graph_max_bs
    )
    with pytest.raises(ValueError, match=r"max\(cuda_graph_bs\) must match"):
        validate_generation_batch_policy(
            model_name="CSM TTS", server_args=cg_args, model_buffer_bs=8
        )


def test_cuda_graph_buffer_probe_resolves_csm_pool() -> None:
    """The #843 live-runner audit can read CsmTTSModel's per-request buffers:
    the registered probe resolves the pool first-dim (= max_batch_size + 1
    incl. the padding row) and the sizing evaluation passes/fails on it."""
    from sglang_omni.utils.cuda_graph_batch_validator import (
        evaluate_cuda_graph_batch_sizing,
        read_model_buffer_capacity,
    )

    model = _make_tail_model(pool_rows=8)
    capacity, source = read_model_buffer_capacity(model)
    assert capacity == 9  # 8 rows + 1 reserved padding row
    assert "_sampler_pool.seeds" in source or "_cg_" in source

    ok = evaluate_cuda_graph_batch_sizing(
        stage="tts_engine (CsmTTSModel)",
        max_running_requests=8,
        cuda_graph_max_bs=8,
        captured_bs=[1, 2, 4, 8],
        request_slots=None,
        buffer_capacity=capacity,
        buffer_source=source,
    )
    assert ok.is_valid, ok.format()

    undersized = evaluate_cuda_graph_batch_sizing(
        stage="tts_engine (CsmTTSModel)",
        max_running_requests=16,
        cuda_graph_max_bs=16,
        captured_bs=[1, 2, 4, 8, 16],
        request_slots=None,
        buffer_capacity=capacity,
        buffer_source=source,
    )
    assert not undersized.is_valid, "9-row pool must fail a 16-request target"
