#!/usr/bin/env python3
# Analyze a wire-trace dir: control vs data plane, per-kind size/count/timing.
import json, glob, sys, statistics
d = sys.argv[1]
rows = []
for f in glob.glob(d + "/*.jsonl"):
    for ln in open(f):
        ln = ln.strip()
        if ln:
            try: rows.append(json.loads(ln))
            except: pass
def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs)-1, int(p/100.0*len(xs)))] if xs else 0
print("dir=%s  events=%d  pids=%s" % (d, len(rows), sorted(set(r['pid'] for r in rows))))
reqs = len([r for r in rows if r['kind']=='submit' and r['dir']=='tx'])
ts = sorted(r['t'] for r in rows); span=(ts[-1]-ts[0])/1e9 if len(ts)>1 else 0
print("requests(incl warmup)=%d  span=%.1fs" % (reqs, span))
print("\n%-8s %-18s %-5s %6s %12s %9s %9s %8s" % ("plane","kind","dir","count","totB","medB","maxB","medUs"))
agg = {}
for r in rows:
    k=(r['plane'], r['kind'], r['dir'], r.get('op','codec'))
    agg.setdefault(k, []).append(r)
for k in sorted(agg, key=lambda x:-sum(r['n'] for r in agg[x])):
    rs=agg[k]; sizes=[r['n'] for r in rs]; us=[r['us'] for r in rs if 'us' in r]
    print("%-8s %-18s %-5s %6d %12d %9d %9d %8s  op=%s" % (
        k[0],k[1],k[2],len(rs),sum(sizes),int(statistics.median(sizes)),max(sizes),
        ("%.1f"%pct(us,50)) if us else "-", k[3]))
# DATA PLANE focus (the myelon target)
dp=[r for r in rows if r['plane']=='data']
print("\n*** DATA-PLANE events: %d ***" % len(dp))
for kind in set(r['kind'] for r in dp):
    sub=[r for r in dp if r['kind']==kind]; sizes=[r['n'] for r in sub]
    print("  %s: count=%d  med=%dB  max=%dB  total=%.3f MB  (%.2f MB/s)" % (
        kind, len(sub), int(statistics.median(sizes)), max(sizes), sum(sizes)/1e6,
        sum(sizes)/1e6/span if span else 0))
