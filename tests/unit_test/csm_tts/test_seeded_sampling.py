# SPDX-License-Identifier: Apache-2.0
"""#824 unified sampling seed consumption for csm_tts (CPU).

The determinism of ``multinomial_with_seed`` itself is upstream's contract
(Triton kernel, CUDA-only; exercised by higgs test_seeded_sampling on CUDA).
These tests pin csm_tts's THREADING of that contract on CPU via the conftest
reference implementation (``cpu_multinomial_with_seed`` fixture): per-row
seeds, per-(frame, codebook) positions, NO_SEED legacy passthrough, and
reproducibility through the full 31-step depth loop.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.csm_tts import modeling, sampler
from sglang_omni.models.csm_tts.sampler import NO_SEED, K_MAX

B = 6
V = 101

# Real CSM-1B depth rope (same values as test_modeling's tiny config).
_DEPTH_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 0.001953125,
    "high_freq_factor": 0.0078125,
    "original_max_position_embeddings": 16,
}


def _tiny_depth_cfg(num_codebooks: int = 8, vocab_size: int = 17) -> SimpleNamespace:
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


def _sample(logits, seeds, positions, temp: float = 0.9):
    return sampler.sample_codes_batched(
        logits,
        torch.full((logits.shape[0],), temp),
        torch.full((logits.shape[0],), K_MAX, dtype=torch.long),
        seeds_B=None if seeds is None else torch.as_tensor(seeds, dtype=torch.long),
        positions_B=(
            None if positions is None else torch.as_tensor(positions, dtype=torch.long)
        ),
    )


# ---------------------------------------------------------------------------
# sample_codes_batched: seed contract at temp 0.9
# ---------------------------------------------------------------------------


def test_same_seed_same_position_reproducible(cpu_multinomial_with_seed) -> None:
    """A seeded draw is a pure function of (distribution, seed, position):
    identical inputs reproduce identical codes regardless of the global
    torch RNG state (which only feeds the unseeded fallback draw)."""
    logits = torch.randn(B, V) * 2.0
    positions = list(range(B))
    torch.manual_seed(1)
    a = _sample(logits, [123] * B, positions)
    torch.manual_seed(999)  # perturb the global RNG; seeded rows must not care
    b = _sample(logits, [123] * B, positions)
    assert torch.equal(a, b)


def test_different_seeds_differ(cpu_multinomial_with_seed) -> None:
    logits = torch.randn(B, V)
    positions = [4] * B
    a = _sample(logits, [1] * B, positions)
    b = _sample(logits, [2] * B, positions)
    assert not torch.equal(a, b)


def test_position_decorrelates_frames(cpu_multinomial_with_seed) -> None:
    """Same seed at different (frame, codebook) positions draws
    independently — consecutive frames don't repeat codes in lockstep."""
    logits = torch.randn(B, V)
    a = _sample(logits, [9] * B, [0] * B)
    b = _sample(logits, [9] * B, [32] * B)  # next frame, same codebook
    assert not torch.equal(a, b)


def test_row_independence_batched_vs_solo(cpu_multinomial_with_seed) -> None:
    """A row's seeded draw depends only on its own (seed, position), not on
    its batch neighbours: batched == solo per row."""
    logits = torch.randn(3, V)
    seeds = [11, 22, 33]
    positions = [5, 5, 5]
    full = _sample(logits, seeds, positions)
    for i in range(3):
        solo = _sample(logits[i : i + 1], seeds[i : i + 1], positions[i : i + 1])
        assert torch.equal(full[i], solo[0]), f"row {i} depends on neighbours"


def test_no_seed_rows_keep_legacy_multinomial(cpu_multinomial_with_seed) -> None:
    """NO_SEED rows are byte-identical to the seeds_B=None path (the #824
    sentinel keeps unseeded decode unchanged)."""
    logits = torch.randn(B, V)
    torch.manual_seed(7)
    legacy = _sample(logits, None, None)
    torch.manual_seed(7)
    sentinel = _sample(logits, [NO_SEED] * B, list(range(B)))
    assert torch.equal(legacy, sentinel)


def test_greedy_rows_ignore_seeds(cpu_multinomial_with_seed) -> None:
    """The greedy short-circuit (temp<=1e-5) stays argmax over RAW logits,
    seeded or not."""
    logits = torch.randn(B, V) * 4.0
    out = sampler.sample_codes_batched(
        logits,
        torch.zeros(B),
        torch.full((B,), K_MAX, dtype=torch.long),
        seeds_B=torch.arange(1, B + 1, dtype=torch.long),
        positions_B=torch.zeros(B, dtype=torch.long),
    )
    assert torch.equal(out, logits.argmax(dim=-1))


# ---------------------------------------------------------------------------
# generate_frame: seed threading through the full depth loop
# ---------------------------------------------------------------------------


def _tiny_decoder(cfg):
    fe = modeling.CsmFrameEmbedding(
        cfg.num_codebooks, cfg.vocab_size, cfg.backbone_hidden_size
    )
    dd = modeling.CsmDepthDecoder(cfg, fe, max_slots=4)
    _randomize(dd)
    return dd


def test_generate_frame_same_seed_reproducible(cpu_multinomial_with_seed) -> None:
    """Full depth loop at temp 0.9: the same request seed reproduces the
    same 32-code frame across runs with different global RNG states."""
    torch.manual_seed(0)
    cfg = _tiny_depth_cfg()
    dd = _tiny_decoder(cfg)
    bs = 2
    h = torch.randn(bs, cfg.backbone_hidden_size)
    cb0 = torch.randint(0, cfg.vocab_size, (bs,))
    temps = torch.full((4,), 0.9)
    topks = torch.full((4,), K_MAX, dtype=torch.long)
    seeds = torch.tensor([77, 77, NO_SEED, NO_SEED], dtype=torch.long)
    step = torch.zeros(4, dtype=torch.long)

    with torch.no_grad():
        torch.manual_seed(3)
        f1 = dd.generate_frame(h, cb0, temps, topks, bs=bs, seeds_B=seeds, step_B=step)
        torch.manual_seed(4242)
        f2 = dd.generate_frame(h, cb0, temps, topks, bs=bs, seeds_B=seeds, step_B=step)
        other = dd.generate_frame(
            h,
            cb0,
            temps,
            topks,
            bs=bs,
            seeds_B=torch.tensor([78, 78, NO_SEED, NO_SEED]),
            step_B=step,
        )
    assert torch.equal(f1, f2), "same seed must reproduce the frame"
    assert not torch.equal(f1, other), "a different seed must change the frame"
    # A later frame index draws at fresh (frame, codebook) positions.
    with torch.no_grad():
        f_next = dd.generate_frame(
            h, cb0, temps, topks, bs=bs, seeds_B=seeds, step_B=step + 1
        )
    assert not torch.equal(f1, f_next)


def test_generate_frame_row_isolation(cpu_multinomial_with_seed) -> None:
    """Changing only the NEIGHBOURS' seeds leaves a row's frame untouched
    (per-row reproducibility under batching)."""
    torch.manual_seed(1)
    cfg = _tiny_depth_cfg()
    dd = _tiny_decoder(cfg)
    bs = 3
    h = torch.randn(bs, cfg.backbone_hidden_size)
    cb0 = torch.randint(0, cfg.vocab_size, (bs,))
    temps = torch.full((4,), 0.9)
    topks = torch.full((4,), K_MAX, dtype=torch.long)
    step = torch.zeros(4, dtype=torch.long)

    with torch.no_grad():
        a = dd.generate_frame(
            h, cb0, temps, topks, bs=bs,
            seeds_B=torch.tensor([50, 60, 70]), step_B=step,
        )
        b = dd.generate_frame(
            h, cb0, temps, topks, bs=bs,
            seeds_B=torch.tensor([50, 61, 71]), step_B=step,
        )
    assert torch.equal(a[0], b[0]), "row 0 changed when only neighbours' seeds did"
    assert not torch.equal(a[1:], b[1:])


# ---------------------------------------------------------------------------
# pool + runner threading
# ---------------------------------------------------------------------------


def test_pool_seed_fields_reset() -> None:
    """reset_row wipes the #824 fields so the next row owner can't inherit a
    previous request's seed or frame index."""
    pool = sampler.CsmBatchedSamplerState(4, device="cpu")
    assert pool.seeds.dtype == torch.int64
    assert pool.step_count.dtype == torch.int64
    assert (pool.seeds == NO_SEED).all()  # unseeded by default
    pool.seeds[1] = 7
    pool.step_count[1] = 9
    pool.reset_row(1)
    assert int(pool.seeds[1]) == NO_SEED
    assert int(pool.step_count[1]) == 0


def test_before_prefill_pins_request_seeds() -> None:
    """The runner forwards each request's SamplingParams.sampling_seed to
    model.set_request_seed at prefill (the #824 ingestion->consumption
    hand-off; None stays None -> unseeded row)."""
    from sglang_omni.models.csm_tts.model_runner import CsmTTSModelRunner

    pinned: list[tuple[str, int | None]] = []
    fake_model = SimpleNamespace(
        set_request_seed=lambda rid, seed: pinned.append((rid, seed)),
    )
    tp_worker = SimpleNamespace(
        gpu_id=0, model_runner=SimpleNamespace(model=fake_model)
    )
    runner = CsmTTSModelRunner(tp_worker, output_processor=None)

    def _req(rid: str, seed: int | None):
        data = SimpleNamespace(
            req=SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=seed)),
            context_codes=None,
            depth_temperature=0.9,
            depth_top_k=50,
        )
        return SimpleNamespace(request_id=rid, data=data)

    forward_batch = SimpleNamespace(input_ids=torch.tensor([1, 2]))
    runner.before_prefill(
        forward_batch, None, [_req("r-seeded", 42), _req("r-unseeded", None)]
    )

    assert pinned == [("r-seeded", 42), ("r-unseeded", None)]
    # No context audio in the batch -> no embed overlay.
    assert forward_batch.input_embeds is None
