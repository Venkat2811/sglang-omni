# SPDX-License-Identifier: Apache-2.0
# Env-gated wire tracer: control-plane vs data-plane traffic + timing.
# Active only when OMNI_WIRE_TRACE_DIR set. One JSONL per process:
#   {t:monotonic_ns, pid, plane, dir, kind, n:bytes, us:duration, op:codec|send|data}
from __future__ import annotations
import json
import os
import threading
import time

_dir = None
_checked = False
_fh = None
_lock = threading.Lock()


def _enabled():
    global _checked, _dir
    if not _checked:
        _checked = True
        d = os.environ.get("OMNI_WIRE_TRACE_DIR", "").strip()
        _dir = d or None
    return _dir is not None


def _fh_get():
    global _fh
    if _fh is None:
        os.makedirs(_dir, exist_ok=True)
        _fh = open(os.path.join(_dir, "wire-" + str(os.getpid()) + ".jsonl"), "a", buffering=1)
    return _fh


def _emit(plane, direction, kind, n, us=None, op="codec"):
    rec = {"t": time.monotonic_ns(), "pid": os.getpid(), "plane": plane,
           "dir": direction, "kind": kind, "n": int(n), "op": op}
    if us is not None:
        rec["us"] = round(us, 2)
    line = json.dumps(rec)
    with _lock:
        _fh_get().write(line + "\n")


def record_control(d, n, direction, us=None):
    if not _enabled():
        return
    try:
        kind = d.get("type", "?") if isinstance(d, dict) else "?"
        _emit("control", direction, kind, n, us, "codec")
        if direction == "tx" and kind == "data_ready" and isinstance(d, dict):
            sm = d.get("shm_metadata") or {}
            if isinstance(sm, dict):
                if sm.get("_ipc"):
                    tb = sm.get("tensor_bytes")
                    _emit("data", "tx", "cuda_ipc_handle", len(tb) if tb is not None else 0, None, "data")
                else:
                    ri = sm.get("relay_info") if isinstance(sm.get("relay_info"), dict) else {}
                    ti = ri.get("transfer_info") if isinstance(ri.get("transfer_info"), dict) else {}
                    _emit("data", "tx", "relay_blob", ti.get("size", 0), None, "data")
    except Exception:
        pass


def record_send(kind, n, us):
    if not _enabled():
        return
    try:
        _emit("control", "tx", kind, n, us, "send")
    except Exception:
        pass
