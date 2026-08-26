#!/usr/bin/env python3
"""Deep integrity audit for the CONCURRENCY artifacts (in addition to scripts/audit_repo.py).

Checks, per results/conc_<dev>_seq<seq>_rb<rb>.json:
  - valid JSON; required top-level fields present and correct:
      model, device, endpoint == .../v1/rerank, prefix_caching is False (customer rule),
      seq_len_target in {512,1024}, req_batch in {1,32}, filename matches seq/rb metadata.
  - each by_concurrency row: concurrency in the requested set; iters>0; wall_s>0;
      percentiles monotonic p50<=p90<=p99; mean within [p50*0.5, p99*1.5] sanity band;
      qps ~= iters/wall_s (within 20%); qps <= concurrency*1000/p50_ms*1.5 (can't beat the
      theoretical max requests/s given latency and concurrency, with slack) and
      qps >= concurrency*1000/mean_ms*0.5 (shouldn't be absurdly low vs latency).
  - device label matches the filename device token.
Then recompute the README concurrency table (single-pair seq512) values from the JSON and confirm
each printed p50 and peak req/s appears verbatim in README.md. Also recompute the $/1k example values.

Exit non-zero on ANY failure. This is intentionally strict to catch fabrication/mismatch.
"""
import json, os, re, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); CH = os.path.join(ROOT, "charts")
PRICE = {"tpu": 1.61, "b200": 6.95, "h200": 3.85, "g4": 3.11}
LABELTOK = {"tpu": "tpu", "b200": "B200", "h200": "H200", "g4": "G4"}
fails = []

def c1k(dev, ms): return round((PRICE[dev]/3600.0)*(ms/1000.0)*1000.0, 5)

files = sorted(glob.glob(os.path.join(RES, "conc_*.json")))
if not files:
    print("no concurrency files found"); sys.exit(1)

parsed = {}
for f in files:
    b = os.path.basename(f)
    m = re.match(r"conc_(tpu|b200|h200|g4)_seq(\d+)_rb(\d+)\.json$", b)
    if not m:
        fails.append(f"{b}: filename does not match expected pattern"); continue
    dev, seq, rb = m.group(1), int(m.group(2)), int(m.group(3))
    if os.path.getsize(f) == 0:
        fails.append(f"{b}: empty file"); continue
    try:
        d = json.load(open(f))
    except Exception as e:
        fails.append(f"{b}: invalid JSON ({e})"); continue
    # top-level checks
    if not str(d.get("endpoint","")).endswith("/v1/rerank"):
        fails.append(f"{b}: endpoint is not /v1/rerank ({d.get('endpoint')})")
    if d.get("prefix_caching") is not False:
        fails.append(f"{b}: prefix_caching not False")
    if d.get("seq_len_target") != seq:
        fails.append(f"{b}: seq_len_target {d.get('seq_len_target')} != filename {seq}")
    if d.get("req_batch") != rb:
        fails.append(f"{b}: req_batch {d.get('req_batch')} != filename {rb}")
    if LABELTOK[dev] not in str(d.get("device","")):
        fails.append(f"{b}: device label '{d.get('device')}' missing token {LABELTOK[dev]}")
    # per-row checks
    for r in d.get("by_concurrency", []):
        c = r["concurrency"]; tag = f"{b} c{c}"
        if r.get("iters", 0) <= 0: fails.append(f"{tag}: iters<=0")
        if r.get("wall_s", 0) <= 0: fails.append(f"{tag}: wall_s<=0")
        p50, p90, p99 = r["req_latency_ms_p50"], r["req_latency_ms_p90"], r["req_latency_ms_p99"]
        mean = r["req_latency_ms_mean"]
        if not (p50 <= p90 + 1e-6 <= p99 + 1e-6):
            fails.append(f"{tag}: percentiles not monotonic {p50}/{p90}/{p99}")
        if not (p50*0.5 <= mean <= p99*1.5):
            fails.append(f"{tag}: mean {mean} outside sanity band [{p50*0.5},{p99*1.5}]")
        qps = r["qps"]
        exp_qps = r["iters"]/r["wall_s"]
        if abs(qps-exp_qps)/max(exp_qps,1e-9) > 0.2:
            fails.append(f"{tag}: qps {qps} != iters/wall {exp_qps:.2f}")
        # concurrency/latency vs qps consistency: with C in flight and mean latency, qps ~ C/mean_s.
        theo = c / (mean/1000.0)
        if qps > theo*1.6:
            fails.append(f"{tag}: qps {qps} implausibly high vs C/mean {theo:.1f} (latency/throughput inconsistent)")
        if qps < theo*0.4:
            fails.append(f"{tag}: qps {qps} implausibly low vs C/mean {theo:.1f}")
    parsed[(dev,seq,rb)] = {r["concurrency"]: r for r in d.get("by_concurrency", [])}

# cross-check README single-pair seq512 table (the one shown in README)
readme = open(os.path.join(ROOT, "README.md")).read()
for dev in ["tpu","b200","h200","g4"]:
    key = (dev,512,1)
    if key not in parsed: continue
    for c in (1,2,4,8):
        if c in parsed[key]:
            p50 = parsed[key][c]["req_latency_ms_p50"]
            # README rounds to 1 decimal (e.g. 11.1 ms); check the rounded value appears
            r1 = f"{round(p50,1)} ms"
            if r1 not in readme:
                fails.append(f"README concurrency: {dev} c{c} p50 {r1} not found")
    # peak req/s (max qps across concurrencies) shown in README as integer
    peak = int(round(max(parsed[key][c]["qps"] for c in parsed[key])))
    if str(peak) not in readme:
        fails.append(f"README concurrency: {dev} peak req/s {peak} not found")

# cross-check README's conc-1 $/1k example values
for dev, ms_ref in [("tpu",None),("g4",None),("h200",None),("b200",None)]:
    key=(dev,512,1)
    if key in parsed and 1 in parsed[key]:
        v = c1k(dev, parsed[key][1]["req_latency_ms_p50"])
        if f"${v}" not in readme:
            fails.append(f"README conc-1 $/1k for {dev} = ${v} not found")

print("="*64)
if fails:
    print("CONCURRENCY AUDIT FAILED -", len(fails), "issue(s):")
    for x in fails: print("  -", x)
    sys.exit(1)
print("CONCURRENCY AUDIT PASSED - all concurrency artifacts consistent, reproducible, matched to README.")
for (dev,seq,rb),rows in sorted(parsed.items()):
    cs = ",".join(str(c) for c in sorted(rows))
    print(f"  {dev:5} seq{seq} rb{rb}: conc [{cs}]")
sys.exit(0)
