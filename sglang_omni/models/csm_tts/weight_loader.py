# SPDX-License-Identifier: Apache-2.0
"""Checkpoint weight-name remapping for CSM-1B.

The ship artifact is the HF-transformers export: shards
``transformers-00001-of-00002.safetensors`` / ``transformers-00002-of-00002
.safetensors`` indexed by ``transformers.safetensors.index.json`` — 538
tensors, ALL fp32 (cast to backbone bf16 on copy). NEVER read the orig-format
``ckpt.pt`` / ``model.safetensors`` in the same repo: their Q/K are
un-permuted-RoPE (PLAN §8.R2).

Remap table (PLAN §1.6):

==============================================================  =====================================
checkpoint name                                                 destination
==============================================================  =====================================
``embed_text_tokens.weight``                                    ``backbone.model.embed_tokens.weight``
``backbone_model.embed_tokens.embed_audio_tokens.weight``       ``frame_embedding.weight``
``backbone_model.layers.{i}.*``                                 ``backbone.model.layers.{i}.*``
``backbone_model.norm.weight``                                  ``backbone.model.norm.weight``
``lm_head.weight`` (shape ``[2051, 2048]``)                     ``codebook0_head.weight`` (NOT a text head!)
``depth_decoder.model.inputs_embeds_projector.weight``          ``depth_decoder.inputs_embeds_projector.weight``
``depth_decoder.model.layers.{0..3}.*``                         ``depth_decoder.layers.{i}.*`` (manual copies)
``depth_decoder.model.norm.weight``                             depth final norm
``depth_decoder.codebooks_head.weight``                         ``codebooks_head.weight``
``depth_decoder.model.embed_tokens.weight``                     SKIP — tied alias of the audio table
``codec_model.*``                                               SKIP — ``audio_codec.py`` loads that subtree
==============================================================  =====================================

The depth tied-embed duplicate IS present in the 538-tensor census (verified
vs ``tf_index.json``) so the skip path ALWAYS fires; the M1 zero-unexpected
census counts it as deliberately-skipped, not missing (PLAN F13). Llama
qkv/gate_up stacking on the backbone side is handled by
``LlamaForCausalLM.load_weights``; depth layers use their own dense modules
(no stacking).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CsmWeightMapper:
    """Weight-name remapper for the CSM HF-transformers shards.

    ``map`` routes each checkpoint tensor name to (dest_name | None-to-skip);
    the model's ``load_weights`` splits the stream into backbone-vs-own params
    and casts everything to ``backbone_dtype`` on copy.
    """

    backbone_dtype: torch.dtype = torch.bfloat16

    def map(self, name: str) -> str | None:
        """Map a checkpoint tensor name to its destination name.

        Args:
            name: checkpoint key (e.g. ``"backbone_model.layers.3.self_attn.q_proj.weight"``).

        Returns:
            Destination parameter name, or ``None`` to deliberately skip
            (``depth_decoder.model.embed_tokens.weight`` tied alias and the
            whole ``codec_model.*`` subtree).
        """
        raise NotImplementedError("skeleton — PLAN §1.6 CsmWeightMapper.map")

    def census(self, names: list[str]) -> dict[str, int]:
        """M1-gate tensor census over all checkpoint keys.

        Returns:
            ``{"mapped": int, "skipped": int, "unexpected": int}`` — must be a
            538/0-unexpected split for the ship artifact (PLAN §7 M1, §8.R2).
        """
        raise NotImplementedError("skeleton — PLAN §1.6 CsmWeightMapper.census")


__all__ = ["CsmWeightMapper"]
