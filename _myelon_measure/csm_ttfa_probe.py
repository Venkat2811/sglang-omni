#!/usr/bin/env python3
"""OSS-native c=1 TTFA probe against sglang-omni CSM /v1/audio/speech (streaming PCM).
Audio-byte-honest: TTFA = time from request send to FIRST non-empty PCM chunk received.
Sequential (concurrency=1). Prints per-request + p50/p90/min/max summary as JSON."""
import sys, time, json, urllib.request

BASE = "http://127.0.0.1:8200"
MODEL = "csm-1b-rtx3060-ship"
TEXT = "Hello, this is a short test of the streaming text to speech system."
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
WARMUP = 2

def one_request():
    body = json.dumps({
        "model": MODEL, "input": TEXT, "voice": "0",
        "response_format": "pcm", "stream": True, "stream_format": "audio",
    }).encode()
    req = urllib.request.Request(BASE + "/v1/audio/speech", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttfa = None
    total_bytes = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            if ttfa is None and len(chunk) > 0:
                ttfa = (time.perf_counter() - t0) * 1000.0
            total_bytes += len(chunk)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return ttfa, total_ms, total_bytes

results = []
for i in range(WARMUP + N):
    ttfa, total_ms, nbytes = one_request()
    tag = "warmup" if i < WARMUP else "timed"
    print(f"[{tag} {i}] ttfa={ttfa:.1f}ms total={total_ms:.1f}ms bytes={nbytes}", flush=True)
    if i >= WARMUP:
        results.append({"ttfa_ms": ttfa, "total_ms": total_ms, "bytes": nbytes})

ttfas = sorted(r["ttfa_ms"] for r in results)
def pct(xs, p):
    if not xs: return None
    k = max(0, min(len(xs)-1, int(round((p/100.0)*(len(xs)-1)))))
    return xs[k]
summary = {
    "n": len(results), "concurrency": 1, "warmup": WARMUP,
    "ttfa_ms": {"min": min(ttfas), "p50": pct(ttfas,50), "p90": pct(ttfas,90),
                "max": max(ttfas), "mean": sum(ttfas)/len(ttfas)},
    "all_ttfa_ms": ttfas,
    "min_bytes": min(r["bytes"] for r in results),
}
print("SUMMARY_JSON " + json.dumps(summary))
