# SPDX-License-Identifier: Apache-2.0
"""Unit tests for csm_tts.request_builders.

PLAN §6.1.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="csm_tts skeleton — request_builders not implemented yet (PLAN §1.10/§6.1)"
)


def test_sampling_params_normalize_called():
    """``sampling_params.normalize(tokenizer=None)`` is invoked (stop_strs
    crash upstream otherwise) and seed rides the ``sampling_seed`` kwarg."""
    raise NotImplementedError("PLAN §6.1 test_request_builders: normalize(None)")


def test_max_new_tokens_clamp_vs_ctx_budget():
    """Clamp is ``min(cap, BACKBONE_CTX - 1 - len(prompt_ids))`` — against
    2047 (max_req_len), not 2048 (review A4/B18)."""
    raise NotImplementedError("PLAN §6.1 test_request_builders: ctx clamp")


def test_context_fingerprint_sensitivity():
    """Any context code or speaker change → new extra_key; no context →
    None (shared radix subtree)."""
    raise NotImplementedError("PLAN §6.1 test_request_builders: fingerprint")
