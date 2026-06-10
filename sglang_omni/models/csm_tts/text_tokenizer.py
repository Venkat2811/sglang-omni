# SPDX-License-Identifier: Apache-2.0
"""CSM prompt assembly over the checkpoint's Llama-3 tokenizer.

Mirrors the hub chat template EXACTLY (parity is hostage to token-exact
prompt assembly — PLAN §8.R3):

- per context message:
  ``128000 ⊕ tok("[<spk>]<text>") ⊕ 128001 ⊕ 128002×F ⊕ 128003``
- final (target) message:
  ``128000 ⊕ tok("[<spk>]<text>") ⊕ 128001``

``add_special_tokens=False`` everywhere; the speaker tag is LITERAL text
``[0]``. Unlike higgs's ``-100`` sentinel, the ``<|AUDIO|>`` positions keep
their REAL id 128002 (in-vocab) — no masking needed; the prefill overlay
simply overwrites those embedding rows.

The tokenizer is loaded from the raw ``tokenizer.json`` (higgs trick — dodges
the transformers-v5 tokenizer-metadata incompatibility).

PLAN §1.9.
"""

from __future__ import annotations

from sglang_omni.models.csm_tts.utils import (
    AUDIO_EOS_TOKEN_ID,
    AUDIO_TOKEN_ID,
    BOS,
    EOT,
)


class CsmPromptBuilder:
    """Token-exact CSM prompt builder."""

    def __init__(self, checkpoint_dir: str) -> None:
        """Load the Llama-3 tokenizer from ``<checkpoint_dir>/tokenizer.json``
        via ``tokenizers.Tokenizer`` wrapped in ``PreTrainedTokenizerFast``
        (higgs text_tokenizer pattern)."""
        raise NotImplementedError("skeleton — PLAN §1.9 CsmPromptBuilder.__init__")

    def build_prompt(
        self,
        *,
        text: str,
        speaker_id: int,
        context: list[dict] | None = None,
        context_frame_counts: list[int] | None = None,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Assemble the full prompt id sequence + per-segment placeholder spans.

        Args:
            text: target utterance text.
            speaker_id: speaker tag rendered as literal ``[<spk>]`` text.
            context: ordered context segments ``{speaker_id, text}``.
            context_frame_counts: F per context segment — MUST come from the
                actual Mimi encode on the raw-waveform path (deferred prompt
                assembly, PLAN §1.14 / F14) or ``codes.shape[0]`` on the
                pre-encoded fast path.

        Returns:
            ``(prompt_ids, placeholder_spans)`` where ``placeholder_spans`` is
            one ``(start, length)`` per context segment covering its
            ``128002×F`` run (the ``128003`` position immediately after gets
            the all-zeros-frame embed at overlay time — HF
            ``_merge_input_ids_with_input_values`` parity, PLAN §2.4).
        """
        raise NotImplementedError("skeleton — PLAN §1.9 CsmPromptBuilder.build_prompt")

    def tokenize_segment_text(self, speaker_id: int, text: str) -> list[int]:
        """``tok("[<spk>]<text>", add_special_tokens=False)`` — body tokens
        only (no BOS/EOT framing).

        Returns:
            list[int] token ids in the 128256 text vocab.
        """
        raise NotImplementedError("skeleton — PLAN §1.9 CsmPromptBuilder.tokenize_segment_text")


__all__ = [
    "AUDIO_EOS_TOKEN_ID",
    "AUDIO_TOKEN_ID",
    "BOS",
    "EOT",
    "CsmPromptBuilder",
]
