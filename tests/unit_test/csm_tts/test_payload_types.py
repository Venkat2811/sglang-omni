# SPDX-License-Identifier: Apache-2.0
"""PipelineStateBase contract tests for CsmTtsState (#807 adoption).

Mirrors tests/unit_test/scheduling/test_pipeline_state.py: subclass +
inherited base fields + to_dict/from_dict round-trips that preserve the
payload (including the CSM-specific ``finish_reason`` and
``completion_frames`` fields).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.scheduling.pipeline_state import PipelineStateBase, build_usage

_BASE_FIELDS = {"sample_rate", "prompt_tokens", "completion_tokens", "engine_time_s"}


def test_state_subclasses_pipeline_state_base() -> None:
    """CsmTtsState rides the shared base: issubclass, all four base usage
    fields present as dataclass fields, and to_dict/from_dict overridden
    (the base stubs raise NotImplementedError)."""
    assert issubclass(CsmTtsState, PipelineStateBase)
    field_names = {f.name for f in dataclasses.fields(CsmTtsState)}
    missing = _BASE_FIELDS - field_names
    assert not missing, f"CsmTtsState missing base fields: {missing}"
    assert CsmTtsState.to_dict is not PipelineStateBase.to_dict
    assert CsmTtsState.from_dict.__func__ is not PipelineStateBase.from_dict.__func__


def test_defaults_match_csm_contract() -> None:
    """Mimi runs at 24 kHz (matches the base default); usage counters start
    at zero; an empty payload rebuilds to pure defaults."""
    state = CsmTtsState.from_dict({})
    assert state.sample_rate == 24000
    assert state.prompt_tokens == 0
    assert state.completion_tokens == 0
    assert state.completion_frames == 0
    assert state.engine_time_s == 0.0
    assert state.finish_reason is None
    assert build_usage(state) is None  # empty usage stays omitted


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def _assert_round_trip_preserves_payload(state: CsmTtsState) -> None:
    before = state.to_dict()
    restored = CsmTtsState.from_dict(before)
    after = restored.to_dict()
    assert set(after) == set(before)
    assert _normalize(after) == _normalize(before)


def test_round_trip_preserves_payload_fields() -> None:
    """to_dict → from_dict → to_dict is lossless for a fully-populated state,
    for BOTH finish_reason values ("stop" natural EOS / "length" cap hit)."""
    for finish_reason in ("stop", "length"):
        state = CsmTtsState(
            text="hello",
            speaker_id=1,
            context=[{"speaker_id": 0, "text": "ref"}],
            prompt_ids=[1, 2, 3],
            context_codes=[[[1] * 32, [2] * 32]],
            num_ctx_codes_consumed=2,
            max_new_tokens=10,
            temperature=0.7,
            top_k=40,
            top_p=0.95,
            depth_temperature=0.8,
            depth_top_k=30,
            seed=42,
            stream=True,
            output_frames=[[5] * 32, [0] * 32],
            completion_frames=2,
            finish_reason=finish_reason,
            sample_rate=24000,
            prompt_tokens=3,
            completion_tokens=2,
            engine_time_s=0.125,
        )
        _assert_round_trip_preserves_payload(state)
        restored = CsmTtsState.from_dict(state.to_dict())
        assert restored.finish_reason == finish_reason
        assert restored.completion_frames == 2
        assert restored.completion_tokens == 2


def test_usage_fields_ride_the_shared_base_serialization() -> None:
    """Truthy usage fields serialize through append_usage_fields (base
    contract: ints/floats, omitted when zero) and feed the shared
    build_usage total."""
    state = CsmTtsState(
        text="x",
        prompt_ids=[1],
        completion_frames=4,
        prompt_tokens=7,
        completion_tokens=4,
        engine_time_s=1.23456789,
    )
    data = state.to_dict()
    assert data["prompt_tokens"] == 7
    assert data["completion_tokens"] == 4
    assert data["completion_frames"] == 4
    assert data["engine_time_s"] == 1.23456789

    usage = build_usage(state)
    assert usage == {
        "prompt_tokens": 7,
        "completion_tokens": 4,
        "total_tokens": 11,
        "engine_time_s": 1.234568,
    }
