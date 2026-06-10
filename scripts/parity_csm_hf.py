#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Golden-parity harness: sglang-omni CSM engine vs HF transformers — THE
correctness gate for M2 (PLAN §6.2).

Protocol:
- Reference leg: HF ``CsmForConditionalGeneration.generate(output_audio=
  False)`` at ``do_sample=False`` AND ``depth_decoder_do_sample=False``
  (greedy at BOTH levels — deterministic), run IN A SUBPROCESS so it sees a
  clean transformers without our in-process ``AutoConfig.register("csm", …)``
  override (PLAN §8.R7). Import-guarded on transformers>=4.52.1 (native CSM).
- Ours leg: OUR ``CsmTTSModel`` + ``CsmTTSModelRunner`` driven DIRECTLY in
  eager B=1 — no server, no OmniScheduler: ``create_sglang_infrastructure``
  → ``PrefillManager.schedule_next_batch`` (prefill samples frame 0, PLAN
  §1.7) → manual ``prepare_for_decode`` + ``runner.execute`` loop, greedy
  everywhere (temperature 0 / top_k 1 on cb0 AND the depth set).
- Compare frame-by-frame: EXACT 32-code match over the first N frames
  (``--frames``, default 10); on mismatch report the first divergence
  (frame, codebook, ours-vs-ref ids; per-step logits aren't surfaced through
  the runner drive path, so logit gap is reported as null). Also gated:
  prompt-token parity (ours vs HF processor) and the Mimi-clamp guard
  (zero codes >= 2048 on stream-eligible frames — valid ONLY because the run
  is greedy; under sampling it is telemetry, never a gate, PLAN §4.2/F11).

Exit codes: 0 = parity, 1 = divergence, 2 = setup error.
``--device`` requires CUDA; pass ``--allow-cpu`` to override (the engine leg
runs the real sglang stack and is not expected to work on CPU).

Full PLAN §6.2 gate = ``--num-prompts 20 --max-frames 125 --frames 125``
with ``--context-audio-dir`` supplying 3-10 s 24 kHz mono WAVs (half the
prompts become context-audio prompts). Streamed-waveform comparisons exclude
the final frame (PLAN §2.5 deviation: HF decodes a cb31!=0 EOS frame, we
never stream it); this harness compares code matrices incl. the EOS frame
(HF ``sequences`` parity).

Note: HF's effective top_k=50 default arrives via
``_get_default_generation_params()`` at d557ef5 — irrelevant under greedy,
but any transformers bump touching that function is a parity-relevant event;
re-run this gate (PLAN §6.2/F10).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import torch

EXIT_PARITY = 0
EXIT_DIVERGENCE = 1
EXIT_SETUP = 2

# Fixed prompt pool (PLAN §6.2: fixed prompts, speaker 0, pinned seeds —
# seeds are inert under greedy but recorded for the sampled-mode benches).
_FIXED_TEXTS = [
    "Hello from Sesame.",
    "The quick brown fox jumps over the lazy dog.",
    "Continuous batching keeps every stream under eighty milliseconds.",
    "We hold these truths to be self evident.",
    "Please leave a message after the tone.",
    "It was the best of times, it was the worst of times.",
    "Boarding for flight seven forty seven begins at gate twelve.",
    "The mitochondria is the powerhouse of the cell.",
    "Turn left in four hundred feet, then merge onto the highway.",
    "Thanks for calling; how can I help you today?",
]
_CONTEXT_TEXT = "Here is a short sample of my voice."


def build_prompt_set(
    num_prompts: int, context_audio_dir: str | None
) -> list[dict[str, Any]]:
    """Assemble the fixed prompt set; with ``context_audio_dir`` the second
    half of the prompts carries one 24 kHz mono context-audio segment each.

    Returns:
        list of prompt spec dicts consumed by both generators.
    """
    if num_prompts < 1:
        raise ValueError(f"--num-prompts must be >= 1, got {num_prompts}")
    wavs: list[str] = []
    if context_audio_dir:
        wavs = sorted(str(p) for p in Path(context_audio_dir).glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(
                f"no .wav files in --context-audio-dir {context_audio_dir!r}"
            )

    prompts: list[dict[str, Any]] = []
    text_only_count = num_prompts - num_prompts // 2 if wavs else num_prompts
    for i in range(num_prompts):
        spec: dict[str, Any] = {
            "id": f"parity-{i:02d}",
            "text": _FIXED_TEXTS[i % len(_FIXED_TEXTS)],
            "speaker": 0,
            "seed": 1000 + i,
        }
        if wavs and i >= text_only_count:
            spec["context"] = [
                {
                    "speaker": 0,
                    "text": _CONTEXT_TEXT,
                    "audio_path": wavs[(i - text_only_count) % len(wavs)],
                }
            ]
        prompts.append(spec)
    return prompts


# Runs in a fresh interpreter: clean transformers, no AutoConfig override,
# and its GPU memory is fully released before the engine leg starts.
_HF_CHILD_SOURCE = r"""
import json, sys

spec_path, out_path = sys.argv[1], sys.argv[2]
with open(spec_path) as f:
    spec = json.load(f)

import torch

try:
    import transformers
except ImportError as exc:
    sys.stderr.write(f"transformers is required for the HF reference leg: {exc}\n")
    raise SystemExit(2)

from packaging import version

if version.parse(transformers.__version__) < version.parse("4.52.1"):
    sys.stderr.write(
        "parity_csm_hf: transformers>=4.52.1 is required for native CSM "
        f"support (found {transformers.__version__}); "
        "pip install -U 'transformers>=4.52'\n"
    )
    raise SystemExit(2)

from transformers import CsmForConditionalGeneration, CsmProcessor

device = spec["device"]
processor = CsmProcessor.from_pretrained(spec["model_path"])
# fp32 = checkpoint dtype = the parity ground truth.
model = CsmForConditionalGeneration.from_pretrained(
    spec["model_path"], torch_dtype=torch.float32
)
model.to(device).eval()


def _load_audio_24k(path):
    import torchaudio

    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    return wav[0].numpy()


results = []
for prompt in spec["prompts"]:
    conversation = []
    for seg in prompt.get("context", []):
        conversation.append(
            {
                "role": str(int(seg.get("speaker", 0))),
                "content": [
                    {"type": "text", "text": seg.get("text", "")},
                    {"type": "audio", "audio": _load_audio_24k(seg["audio_path"])},
                ],
            }
        )
    conversation.append(
        {
            "role": str(int(prompt.get("speaker", 0))),
            "content": [{"type": "text", "text": prompt["text"]}],
        }
    )
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True
    ).to(device)
    prompt_ids = inputs["input_ids"][0].tolist()
    torch.manual_seed(int(prompt.get("seed", 0)))
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            depth_decoder_do_sample=False,
            output_audio=False,
            max_new_tokens=int(spec["max_frames"]),
        )
    seqs = out.sequences if hasattr(out, "sequences") else out
    frames = seqs[0].tolist()
    if not all(isinstance(row, list) and len(row) == 32 for row in frames):
        raise RuntimeError(
            f"expected an [F, 32] frame matrix from HF generate, got "
            f"{type(seqs)} with first row "
            f"{frames[0] if frames else None}"
        )
    results.append(
        {"id": prompt["id"], "frames": frames, "prompt_ids": prompt_ids}
    )

with open(out_path, "w") as f:
    json.dump(results, f)
"""


def generate_hf_reference_subprocess(
    model_path: str,
    prompts: list[dict[str, Any]],
    *,
    device: str,
    max_frames: int,
) -> list[dict[str, Any]]:
    """Run HF ``CsmForConditionalGeneration.generate(output_audio=False)`` in
    a SUBPROCESS (clean transformers, no AutoConfig override) and collect
    integer frame sequences + processor prompt_ids per prompt.

    Returns:
        list of ``{"id", "frames": [[32 ints] ...], "prompt_ids": [int ...]}``.
    """
    spec = {
        "model_path": model_path,
        "device": device,
        "max_frames": int(max_frames),
        "prompts": prompts,
    }
    with tempfile.TemporaryDirectory(prefix="csm_parity_") as tmpdir:
        spec_path = os.path.join(tmpdir, "spec.json")
        out_path = os.path.join(tmpdir, "reference.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        proc = subprocess.run(
            [sys.executable, "-c", _HF_CHILD_SOURCE, spec_path, out_path],
            capture_output=True,
            text=True,
        )
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"HF reference subprocess failed (rc={proc.returncode}); "
                "see stderr above"
            )
        with open(out_path) as f:
            return json.load(f)


def _run_engine_request(
    *,
    runner: Any,
    prefill_mgr: Any,
    tree_cache: Any,
    model: Any,
    state: Any,
    request_id: str,
) -> list[list[int]]:
    """Drive one request through OUR model+runner with no scheduler:
    prefill (samples frame 0, PLAN §1.7) then manual ``prepare_for_decode``
    + ``runner.execute`` steps until EOS finish or the frame cap.

    Returns:
        ``[F][32]`` int frame matrix incl. the EOS frame.
    """
    from sglang_omni.models.csm_tts.request_builders import build_sglang_csm_request
    from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest

    data = build_sglang_csm_request(state, request_id=request_id)
    req = data.req
    req._omni_data = data  # OmniScheduler stamps this; the runner reads it

    def _step(batch: Any) -> None:
        sched_output = SchedulerOutput(
            requests=[SchedulerRequest(request_id=request_id, data=data)],
            batch_data=batch,
        )
        runner.execute(sched_output)
        # Minimal upstream bookkeeping: append cb0 so Req length/state stays
        # honest (process_batch_result normally does this).
        out_ids = batch.output_ids
        token = int(out_ids[0].item()) if torch.is_tensor(out_ids) else int(out_ids[0])
        req.output_ids.append(token)

    prefill_mgr.add_one_request(req)
    batch = prefill_mgr.schedule_next_batch(None, num_allocatable_reqs=1)
    if batch is None:
        raise RuntimeError(f"prefill admission failed for {request_id!r}")
    _step(batch)
    # Prompts here are < chunked_prefill_size, but stay correct if chunked.
    while prefill_mgr.chunked_req is not None:
        batch = prefill_mgr.schedule_next_batch(None, num_allocatable_reqs=1)
        if batch is None:
            raise RuntimeError(f"chunked prefill stalled for {request_id!r}")
        _step(batch)

    # Decode loop: 1 frame = 1 backbone position = 1 step. EOS finish comes
    # from the runner's _mark_sampler_finished (frame-0 EOS at prefill is a
    # legal zero-decode lifecycle, PLAN §1.7/F7).
    max_frames = int(state.max_new_tokens)
    safety = max_frames + 2
    while (
        not req.finished()
        and len(data.output_frames) < max_frames
        and safety > 0
    ):
        batch.prepare_for_decode()
        _step(batch)
        safety -= 1

    frames = [
        [int(c) for c in torch.as_tensor(row).reshape(-1).tolist()]
        for row in data.output_frames
    ]
    # Best-effort release so the next prompt starts from a clean pool (the
    # harness bypasses process_batch_result, which normally does this).
    try:
        tree_cache.cache_finished_req(req)
    except Exception as exc:  # noqa: BLE001 — release is best-effort here
        print(
            f"parity_csm_hf: warning: KV release failed for {request_id}: {exc}",
            file=sys.stderr,
        )
    model.reset_request(request_id)
    return frames


def generate_engine_frames(
    model_path: str,
    prompts: list[dict[str, Any]],
    engine_dtype: str,
    *,
    device: str,
    max_frames: int,
) -> list[dict[str, Any]]:
    """Drive the sglang-omni CSM model+runner (greedy at both levels, eager
    B=1) and collect ``output_frames`` (incl. the EOS frame), our
    ``prompt_ids``, and the Mimi-clamp guard count per prompt.

    Returns:
        list of ``{"id", "frames", "prompt_ids", "clamp_count"}``.
    """
    # Lazy imports: this leg needs the real sglang stack (CUDA).
    from sglang_omni.models.csm_tts.model_runner import CsmTTSModelRunner
    from sglang_omni.models.csm_tts.payload_types import CsmTtsState
    from sglang_omni.models.csm_tts.text_tokenizer import CsmPromptBuilder
    from sglang_omni.models.csm_tts.utils import (
        BACKBONE_CTX,
        MIMI_CODEBOOK_SIZE,
        SAMPLE_RATE,
        get_or_load_codec,
        load_audio_to_24k,
        resolve_checkpoint,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = resolve_checkpoint(model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=BACKBONE_CTX,
        chunked_prefill_size=BACKBONE_CTX,
        max_running_requests=4,
        mem_fraction_static=0.5,
        disable_cuda_graph=True,  # the M2 gate runs the eager baseline
        dtype=engine_dtype,
    )
    server_args.disable_overlap_schedule = True

    (
        model_worker,
        tree_cache,
        _req_to_token_pool,
        _token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        _model_config,
    ) = create_sglang_infrastructure(server_args, gpu_id)

    def _no_retract(req: Any) -> None:
        # PLAN §8.R1a: the frame sampler has no rollback; at B=1 against a
        # 45x-overprovisioned pool this must be structurally unreachable.
        raise RuntimeError(
            f"unexpected retraction of {getattr(req, 'rid', '?')!r} during "
            "the B=1 parity run"
        )

    decode_mgr.on_retract = _no_retract

    model = model_worker.model_runner.model
    runner = CsmTTSModelRunner(
        model_worker,
        SGLangOutputProcessor(
            capture_hidden=False, capture_hidden_layers=None, model=model
        ),
    )
    builder = CsmPromptBuilder(checkpoint_dir)

    codec = None
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        context_entries: list[dict[str, Any]] = []
        context_codes: list[torch.Tensor] = []
        frame_counts: list[int] = []
        for seg in prompt.get("context", []):
            if codec is None:
                # fp32 Mimi (conv stability) shared encode-side, like stages.
                codec = get_or_load_codec(checkpoint_dir, device, "float32")
            wav_np, sr = load_audio_to_24k(seg["audio_path"])
            wav = torch.from_numpy(wav_np)
            if sr != SAMPLE_RATE:
                import torchaudio.functional as F_audio

                wav = F_audio.resample(wav, sr, SAMPLE_RATE)
            codes = codec.encode_reference(
                wav.reshape(1, 1, -1).contiguous().float(), SAMPLE_RATE
            )
            context_entries.append(
                {
                    "speaker_id": int(seg.get("speaker", 0)),
                    "text": seg.get("text", ""),
                }
            )
            context_codes.append(codes.to(torch.int32).cpu())
            frame_counts.append(int(codes.shape[0]))

        prompt_ids, _spans = builder.build_prompt(
            text=prompt["text"],
            speaker_id=int(prompt.get("speaker", 0)),
            context=context_entries,
            context_frame_counts=frame_counts,
        )
        state = CsmTtsState(
            text=prompt["text"],
            speaker_id=int(prompt.get("speaker", 0)),
            context=context_entries,
            prompt_ids=prompt_ids,
            context_codes=context_codes or None,
            max_new_tokens=int(max_frames),
            # Greedy at BOTH levels (temp=0 short-circuits to argmax over RAW
            # logits in sample_codes_batched; top_k=1 is belt-and-suspenders).
            temperature=0.0,
            top_k=1,
            top_p=None,
            depth_temperature=0.0,
            depth_top_k=1,
            seed=int(prompt.get("seed", 0)),
            stream=False,
        )
        frames = _run_engine_request(
            runner=runner,
            prefill_mgr=prefill_mgr,
            tree_cache=tree_cache,
            model=model,
            state=state,
            request_id=prompt["id"],
        )
        # Mimi-clamp guard over stream-eligible frames: the EOS frame (cb0..30
        # all zero) never reaches the vocoder (PLAN §2.5), so it is excluded.
        streamed = (
            frames[:-1]
            if frames and all(c == 0 for c in frames[-1][:31])
            else frames
        )
        clamp_count = sum(
            1 for row in streamed for c in row if c >= MIMI_CODEBOOK_SIZE
        )
        results.append(
            {
                "id": prompt["id"],
                "frames": frames,
                "prompt_ids": [int(t) for t in prompt_ids],
                "clamp_count": clamp_count,
            }
        )
    return results


def _first_divergence(
    ref_frames: list[list[int]],
    our_frames: list[list[int]],
    frames_to_check: int,
) -> dict[str, Any] | None:
    """First (frame, codebook) where the legs disagree within the horizon;
    a leg ending early inside the horizon is a length divergence."""
    horizon = min(frames_to_check, max(len(ref_frames), len(our_frames)))
    for f in range(horizon):
        if f >= len(ref_frames) or f >= len(our_frames):
            return {
                "kind": "length",
                "frame": f,
                "codebook": None,
                "ours": our_frames[f] if f < len(our_frames) else None,
                "ref": ref_frames[f] if f < len(ref_frames) else None,
                "logit_gap": None,
            }
        for k in range(32):
            if int(ref_frames[f][k]) != int(our_frames[f][k]):
                return {
                    "kind": "code",
                    "frame": f,
                    "codebook": k,
                    "ours": int(our_frames[f][k]),
                    "ref": int(ref_frames[f][k]),
                    # Per-step logits aren't surfaced through the runner
                    # drive path; rerun with instrumented modeling to get gaps.
                    "logit_gap": None,
                }
    return None


def compare_runs(
    reference: list[dict[str, Any]],
    engine: list[dict[str, Any]],
    engine_dtype: str,
    frames_to_check: int = 10,
) -> dict[str, Any]:
    """Apply the gate: exact first-N-frame code match + prompt-token parity
    + clamp counter == 0 (greedy run), reporting the first divergence.

    Returns:
        ``{"passed": bool, "per_prompt": [...], "summary": {...}}``.
    """
    if len(reference) != len(engine):
        raise ValueError(
            f"leg length mismatch: {len(reference)} reference vs "
            f"{len(engine)} engine results"
        )
    per_prompt: list[dict[str, Any]] = []
    all_passed = True
    for ref, eng in zip(reference, engine):
        entry: dict[str, Any] = {"id": eng.get("id") or ref.get("id")}
        ref_ids = [int(t) for t in ref["prompt_ids"]]
        our_ids = [int(t) for t in eng["prompt_ids"]]
        prompt_match = ref_ids == our_ids
        entry["prompt_token_parity"] = prompt_match
        if not prompt_match:
            idx = next(
                (
                    i
                    for i in range(min(len(ref_ids), len(our_ids)))
                    if ref_ids[i] != our_ids[i]
                ),
                min(len(ref_ids), len(our_ids)),
            )
            entry["prompt_first_mismatch"] = {
                "index": idx,
                "ref": ref_ids[idx] if idx < len(ref_ids) else None,
                "ours": our_ids[idx] if idx < len(our_ids) else None,
                "ref_len": len(ref_ids),
                "ours_len": len(our_ids),
            }
        divergence = _first_divergence(
            ref["frames"], eng["frames"], frames_to_check
        )
        entry["ref_frames"] = len(ref["frames"])
        entry["ours_frames"] = len(eng["frames"])
        entry["divergence"] = divergence
        entry["clamp_count"] = int(eng.get("clamp_count", 0))
        passed = prompt_match and divergence is None and entry["clamp_count"] == 0
        entry["passed"] = passed
        all_passed = all_passed and passed
        per_prompt.append(entry)

    summary = {
        "passed": all_passed,
        "engine_dtype": engine_dtype,
        "frames_checked": frames_to_check,
        "num_prompts": len(per_prompt),
        "num_failed": sum(1 for p in per_prompt if not p["passed"]),
        "note": (
            "exact-match gate over the first N frames at greedy/temp=0 both "
            "levels (PLAN 6.2; the bf16 ship gate tolerates divergence only "
            "past frame 25 — raise --frames to probe it)"
        ),
    }
    return {"passed": all_passed, "per_prompt": per_prompt, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", default="sesame/csm-1b")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="exact-match horizon in frames (default 10)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=24,
        help="frames to GENERATE per leg (raise to 125 with --frames 125 "
        "for the full PLAN 6.2 gate)",
    )
    parser.add_argument(
        "--engine-dtype", choices=["float32", "bfloat16"], default="bfloat16"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda[:N] required; cpu refused without --allow-cpu",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--context-audio-dir",
        default=None,
        help="dir of 24 kHz mono WAVs; second half of the prompts become "
        "context-audio prompts",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if args.device.startswith("cpu") and not args.allow_cpu:
        print(
            "parity_csm_hf: this gate needs CUDA — the ours-leg runs the real "
            "sglang stack. Pass --allow-cpu to try anyway (unsupported).",
            file=sys.stderr,
        )
        return EXIT_SETUP
    max_frames = max(args.max_frames, args.frames)

    try:
        prompts = build_prompt_set(args.num_prompts, args.context_audio_dir)
        reference = generate_hf_reference_subprocess(
            args.model_path, prompts, device=args.device, max_frames=max_frames
        )
        engine = generate_engine_frames(
            args.model_path,
            prompts,
            args.engine_dtype,
            device=args.device,
            max_frames=max_frames,
        )
        report = compare_runs(
            reference, engine, args.engine_dtype, frames_to_check=args.frames
        )
    except Exception:
        traceback.print_exc()
        return EXIT_SETUP

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    for entry in report["per_prompt"]:
        if not entry["passed"]:
            print(json.dumps(entry, indent=2))
    return EXIT_PARITY if report["passed"] else EXIT_DIVERGENCE


if __name__ == "__main__":
    sys.exit(main())
