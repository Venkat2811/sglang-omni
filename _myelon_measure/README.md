# `_myelon_measure/` — CSM ⟷ myelon IPC measurement (experiment branch)

This directory + the branch `exp/csm-myelon-wire-measure` hold the **measurement
harness and findings** for "does myelon's intranode SHM/MMAP help SGLang-Omni CSM?"
It is an **experiment snapshot, not a merge target** — the model PR
(`feat/csm-tts`) and optimization PR (`feat/csm-optimize`) stay clean of this.

## What's instrumented
- `sglang_omni/pipeline/wire_trace.py` — env-gated (`OMNI_WIRE_TRACE_DIR`) tracer.
  One JSONL/process: `{t, pid, plane:control|data, dir:tx|rx, kind, n:bytes, us, op}`.
- `sglang_omni/pipeline/control_plane.py` — `serialize_message`/`deserialize_message`
  + `PushSocket.send` timing hooks (call into `wire_trace`).
- `sglang_omni/models/csm_tts/config.py` — vocoder split into its **own process**
  (`process="vocoder"` + per-stage `runtime.resources.total_gpu_memory_fraction`),
  to force the `tts_engine → vocoder` relay data-plane to fire (it is short-circuited
  when stages are colocated threads).
- `examples/csm_tts/csm-1b-rtx3060-split.yaml` — eager config, `mem_fraction_static 0.45`
  for the 2-process split (fits 12 GB).

## Harness
- `csm_ttfa_probe.py N` — sequential c=1 TTFA probe (`/v1/audio/speech`, PCM stream).
- `csm_load.py N CONC WARMUP` — concurrent load driver.
- `wt_analyze.py <wire_dir>` — per-plane/kind size/count/timing rollup + data-plane focus.

## Boot (in the csm_m2 container on the RTX 3060 box)
```bash
# colocated (default ship): OMNI_WIRE_TRACE_DIR=... python3 -m sglang_omni.cli serve \
#   --config examples/csm_tts/csm-1b-rtx3060-ship.yaml --port 8200
# split (this experiment): ... --config examples/csm_tts/csm-1b-rtx3060-split.yaml
OMNI_CONTROL_TRANSPORT={zmq|myelon}   # carrier flag (ad41507)
OMNI_WIRE_TRACE_DIR=/tmp/.../wt       # enables the tracer
```

## Findings (RTX 3060, measured 2026-06-15/16)
- **Control plane (ZMQ): ~0.008 % of wall.** Audio rides inline in 46 KB StreamMessages;
  ~9–26 msg/s; ZMQ ipc sustains 42k msg/s / ~2 GB/s → 4 orders of magnitude headroom.
- **Colocated:** big tensors pass in-process (threads) → zero data-plane traffic.
- **Split:** `tts_engine→vocoder` relay carries **256 B/frame** (32 codes), ~68/req,
  ~49 µs/frame protocol overhead = **~0.06 % of wall**. No bulk hop; MMAP irrelevant.
- **Myelon relay-ring microbench** (in `myelon-playground @ feat/sgl-omni-relay-ring-bench`,
  `experiments/sgl_omni_relay_ring/`): one SHM-ring push (payload inline) vs the relay's
  msgpack-`DataReadyMessage`+ZMQ+blob protocol — **256 B: 4.7 µs vs 19.4 µs p50 (4.1×),
  7.9 vs 116 µs p99 (~15×), 1.37M vs 341k msg/s (4×)**.

**Verdict:** CSM-on-one-GPU is GPU-compute-bound; the per-frame IPC win is real (4–15×)
but ~0.06 % of wall here. Myelon's wall-time win surface is KV-transfer disaggregation
(LMCache/Mooncake), not streaming-TTS on a single GPU.
