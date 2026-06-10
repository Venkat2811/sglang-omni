# SPDX-License-Identifier: Apache-2.0
"""HF config wrapper + synthetic sub-configs for CSM-1B (``sesame/csm-1b``).

CSM's native ``transformers.CsmConfig`` stores the backbone fields FLAT (not as
a nested ``text_config``) and its ``get_text_config()`` returns ``self`` with
``vocab_size=2051`` — useless for SGLang's ``ModelConfig.from_server_args``,
which reads head counts / hidden size / layers / vocab through
``get_text_config()``. This module synthesizes a real ``LlamaConfig`` for the
backbone and a small namespace config for the depth decoder.

Must stay sglang/CUDA-free: imported by ``csm_tts/__init__.py`` during the
registry pkgutil scan (contract §1).

PLAN §1.1.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import transformers
from transformers import CsmConfig, LlamaConfig


def build_backbone_llama_config(csm_cfg: Any) -> LlamaConfig:
    """Synthesize the SGLang-loadable backbone ``LlamaConfig``.

    Reads the FLAT backbone fields off the HF ``CsmConfig`` and realizes:

    - ``hidden_size=2048``, ``num_hidden_layers=16``, ``num_attention_heads=32``,
      ``num_key_value_heads=8``, ``head_dim=64``, ``intermediate_size=8192``
    - ``max_position_embeddings=2048`` (hard model ceiling)
    - ``rope_theta=500000``,
      ``rope_scaling={"rope_type": "llama3", "factor": 32.0,
      "low_freq_factor": 0.125, "high_freq_factor": 0.5,
      "original_max_position_embeddings": 1024}``
    - ``vocab_size=128256`` (text vocab IS the backbone embedding)
    - ``tie_word_embeddings=True`` — sglang ``LlamaForCausalLM``'s lm_head then
      aliases ``embed_tokens``, avoiding a dead 525 MB bf16 text head that we
      never use (cb0 comes from ``codebook0_head``, not the backbone lm_head).

    Args:
        csm_cfg: the (possibly wrapped) HF ``CsmConfig`` for ``sesame/csm-1b``.

    Returns:
        A concrete ``transformers.LlamaConfig``.
    """
    raise NotImplementedError("skeleton — PLAN §1.1 build_backbone_llama_config")


def build_depth_decoder_config(csm_cfg: Any) -> SimpleNamespace:
    """Synthesize the depth-decoder config namespace.

    Depth decoder: 4 layers, ``hidden_size=1024``, 8 heads with EXPLICIT
    ``head_dim=128`` (8*128 != 1024 — do not derive head_dim from hidden),
    2 KV heads (GQA), ``intermediate_size=8192`` (SwiGLU),
    ``max_position_embeddings=33`` (1 backbone hidden + 32 codebook positions),
    depth rope ``theta=500000`` llama3-scaled with ``factor=32.0``,
    ``low_freq_factor=0.001953125``, ``high_freq_factor=0.0078125``,
    ``original_max_position_embeddings=16``.

    Args:
        csm_cfg: the HF ``CsmConfig``.

    Returns:
        ``SimpleNamespace`` (or ``PretrainedConfig``) consumed by
        :class:`sglang_omni.models.csm_tts.modeling.CsmDepthDecoder`.
    """
    raise NotImplementedError("skeleton — PLAN §1.1 build_depth_decoder_config")


class CsmTtsHfConfig(CsmConfig):
    """``CsmConfig`` subclass whose ``get_text_config()`` returns the realized
    synthetic backbone ``LlamaConfig``.

    Subclassing the native class keeps ``isinstance`` and field compatibility
    for HF-side tooling, and inherits ``model_type="csm"`` so
    ``AutoConfig.register("csm", ..., exist_ok=True)`` passes the model-type
    consistency check (override semantics: ``_LazyConfigMapping._extra_content``
    is consulted BEFORE the native mapping — the override wins in-process;
    confirm in the container at M1 via ``AutoConfig.for_model("csm")``).

    PLAN §1.1, §1.2, §8.R7.
    """

    def get_text_config(self, decoder: bool = False) -> transformers.PretrainedConfig:
        """Return the synthetic backbone ``LlamaConfig`` (NOT ``self``).

        Native ``CsmConfig.get_text_config()`` would return ``self`` with
        ``vocab_size=2051``; SGLang reads head counts / hidden / layers / vocab
        through this hook, so it must see the 2048-hidden / 16-layer / 128256-
        vocab backbone view.
        """
        raise NotImplementedError("skeleton — PLAN §1.1 CsmTtsHfConfig.get_text_config")


__all__ = [
    "CsmTtsHfConfig",
    "build_backbone_llama_config",
    "build_depth_decoder_config",
]
