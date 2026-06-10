# SPDX-License-Identifier: Apache-2.0
"""Pipeline-level unit tests for csm_tts (CPU, no GPU, no sglang engine —
higgs monkeypatch pattern; the fixtures double as upstream API-drift
tripwires per PLAN §8.R5).

PLAN §6.1.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="csm_tts skeleton — pipeline not implemented yet (PLAN §6.1)"
)


def test_registry_smoke():
    """``PIPELINE_CONFIG_REGISTRY.get_config("CsmForConditionalGeneration")``
    resolves to ``CsmTtsPipelineConfig`` after the pkgutil scan."""
    raise NotImplementedError("PLAN §6.1 test_pipeline: registry smoke")


def test_config_topology():
    """Stage topology: next= chain preprocessing→audio_encoder→tts_engine→
    vocoder, ``tts_engine.stream_to=["vocoder"]``, vocoder ``terminal=True`` +
    ``can_accept_stream_before_payload=True``, all ``process="pipeline"``."""
    raise NotImplementedError("PLAN §6.1 test_pipeline: config topology")


def test_engine_factory_defaults():
    """With monkeypatched ``build_sglang_server_args`` +
    ``create_sglang_infrastructure``: asserts context_length=2048,
    disable_overlap_schedule=True, max_running_requests=8,
    chunked_prefill_size=2048, dtype bfloat16, NO truncate_rope_to_bf16."""
    raise NotImplementedError("PLAN §6.1 test_pipeline: engine factory defaults")


def test_prompt_builder_matches_hub_chat_template():
    """Prompt builder vs the hub chat template token-for-token (golden
    token-id fixtures incl. 128000 / 128001 / 128002×F / 128003 framing,
    literal [spk] text, add_special_tokens=False)."""
    raise NotImplementedError("PLAN §6.1 test_pipeline: prompt golden fixtures")


def test_vocoder_framing_math():
    """Delay-free framing at 12.5 Hz: stride/overlap/holdback emission
    windows, first-chunk knob, clamp counter, all-zero-frame trim, and the
    EOS frame is NEVER emitted."""
    raise NotImplementedError("PLAN §6.1 test_pipeline: vocoder framing math")


def test_prefill_overlay_paste_positions():
    """Overlay pastes context-frame embeds at real-id 128002 positions and
    the all-zeros-frame embed at 128003, including across chunked-prefill
    boundaries via the num_ctx_codes_consumed cursor."""
    raise NotImplementedError("PLAN §6.1 test_pipeline: overlay paste positions")


def test_runner_finish_marking_on_synthetic_model():
    """Runner finish-marking (FINISH_MATCHED_TOKEN with matched=0) on a
    synthetic model, incl. the frame-0-EOS zero-decode lifecycle."""
    raise NotImplementedError("PLAN §6.1 test_pipeline: runner finish marking")
