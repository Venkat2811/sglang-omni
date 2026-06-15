#!/usr/bin/env python3
# Concurrent load driver for sglang-omni CSM /v1/audio/speech (streaming PCM).
# usage: csm_load.py N CONC WARMUP
import sys, time, json, threading, urllib.request

BASE = "http://127.0.0.1:8200"
MODEL = "csm-1b-rtx3060-ship"
TEXTS = [
    "Hello, this is a short test of the streaming text to speech system.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "In a world of continuous batching, every millisecond of latency counts.",
    "She sells sea shells by the sea shore on a bright and sunny morning.",
]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
WARMUP = int(sys.argv[3]) if len(sys.argv) > 3 else 2

def one_request(text):
    body = json.dumps({"model": MODEL, "input": text, "voice": "0",
                       "response_format": "pcm", "stream": True,
                       "stream_format": "audio"}).encode()
    req = urllib.request.Request(BASE + "/v1/audio/speech", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttfa = None; total = 0
    with urllib.request.urlopen(req, timeout=180) as resp:
        while True:
            ch = resp.read(4096)
            if not ch:
                break
            if ttfa is None:
                ttfa = (time.perf_counter() - t0) * 1000.0
            total += len(ch)
    return ttfa, (time.perf_counter() - t0) * 1000.0, total

for i in range(WARMUP):
    one_request(TEXTS[i % len(TEXTS)])
    print("[warmup %d] done" % i, flush=True)

results = []; lock = threading.Lock(); idx = [0]
def worker():
    while True:
        with lock:
            i = idx[0]
            if i >= N:
                return
            idx[0] += 1
        ttfa, total_ms, nb = one_request(TEXTS[i % len(TEXTS)])
        with lock:
            results.append({"ttfa_ms": ttfa, "total_ms": total_ms, "bytes": nb})
        print("[req %d] ttfa=%.1f total=%.1f bytes=%d" % (i, ttfa, total_ms, nb), flush=True)

t_start = time.perf_counter()
threads = [threading.Thread(target=worker) for _ in range(CONC)]
for t in threads: t.start()
for t in threads: t.join()
wall = time.perf_counter() - t_start
ttfas = sorted(r["ttfa_ms"] for r in results)
print("SUMMARY_JSON " + json.dumps({
    "n": len(results), "conc": CONC, "wall_s": round(wall, 2),
    "ttfa_p50_ms": ttfas[len(ttfas)//2] if ttfas else None,
    "total_bytes": sum(r["bytes"] for r in results),
}))
