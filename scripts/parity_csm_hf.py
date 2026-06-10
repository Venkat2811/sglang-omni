# SPDX-License-Identifier: Apache-2.0
"""Golden-parity harness: sglang-omni CSM engine vs HF transformers — THE
correctness gate for M2 (PLAN §6.2).

Protocol: N=20 prompts (10 text-only, 10 with 3-10 s context audio), fixed
seeds, temp=0 at BOTH levels (cb0 greedy + depth greedy via
``depth_decoder_do_sample=False``) so the run is deterministic. The HF
reference (``CsmForConditionalGeneration.generate(output_audio=False)`` frame
sequences) runs IN A SUBPROCESS — isolating our in-process
``AutoConfig.register("csm", ...)`` override (PLAN §8.R7).

Gate:
- engine fp32: EXACT integer frame-sequence match (incl. the EOS frame), 20/20;
- engine bf16 (ship config): exact match through >= the first 25 frames AND
  >= 99% total code match, divergence only downstream of a first sampling
  tie-break;
- Mimi-clamp guard: ``sanitize_for_mimi`` counter == 0 across the whole run
  (valid ONLY because the run is greedy — under sampling it is telemetry,
  never a gate, PLAN §4.2/F11);
- prompt-token parity: our ``prompt_ids`` == HF processor output, 20/20;
- streamed-waveform comparisons exclude the final frame (PLAN §2.5 deviation:
  HF decodes a cb31!=0 EOS frame, we never stream it); non-streaming WAVs are
  exact-trim-parity.

Note: HF's effective top_k=50 default arrives via
``_get_default_generation_params()`` at d557ef5 (GenerationConfig's attribute
default is None) — any transformers bump touching that function is a
parity-relevant event; re-run this gate (PLAN §6.2/F10).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def build_prompt_set(num_prompts: int, context_audio_dir: str | None) -> list[dict[str, Any]]:
    """Assemble the fixed prompt set: half text-only, half with 3-10 s
    context audio; seeds pinned per prompt.

    Returns:
        list of prompt spec dicts consumed by both generators.
    """
    raise NotImplementedError("skeleton — PLAN §6.2 build_prompt_set")


def generate_hf_reference_subprocess(
    model_path: str, prompts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run HF ``CsmForConditionalGeneration.generate(output_audio=False)`` in
    a SUBPROCESS (clean transformers, no AutoConfig override) and collect
    integer frame sequences + processor prompt_ids per prompt.

    Returns:
        list of ``{"frames": [[32 ints] ...], "prompt_ids": [int ...]}``.
    """
    raise NotImplementedError("skeleton — PLAN §6.2 generate_hf_reference_subprocess")


def generate_engine_frames(
    model_path: str, prompts: list[dict[str, Any]], engine_dtype: str
) -> list[dict[str, Any]]:
    """Drive the sglang-omni CSM pipeline (temp=0 both levels) and collect
    ``output_frames`` (incl. EOS frame), our ``prompt_ids``, and the
    cumulative ``sanitize_for_mimi`` clamp counter.

    Returns:
        list of ``{"frames": ..., "prompt_ids": ..., "clamp_count": int}``.
    """
    raise NotImplementedError("skeleton — PLAN §6.2 generate_engine_frames")


def compare_runs(
    reference: list[dict[str, Any]],
    engine: list[dict[str, Any]],
    engine_dtype: str,
) -> dict[str, Any]:
    """Apply the §6.2 gate (fp32 exact / bf16 relaxed >=25-frame prefix +
    >=99% codes / clamp==0 / prompt parity).

    Returns:
        ``{"passed": bool, "per_prompt": [...], "summary": {...}}``.
    """
    raise NotImplementedError("skeleton — PLAN §6.2 compare_runs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="sesame/csm-1b")
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--engine-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--context-audio-dir", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    prompts = build_prompt_set(args.num_prompts, args.context_audio_dir)
    reference = generate_hf_reference_subprocess(args.model_path, prompts)
    engine = generate_engine_frames(args.model_path, prompts, args.engine_dtype)
    report = compare_runs(reference, engine, args.engine_dtype)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report.get("summary", {}), indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
