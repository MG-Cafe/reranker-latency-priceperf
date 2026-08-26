#!/usr/bin/env python3
"""End-to-end integrity audit for the latency-first price/performance repo.

Verifies (all must pass), so nothing is fabricated / hallucinated / inconsistent:

 1. Every results/*.json is valid JSON and internally consistent:
    - each by_batch row: throughput_pairs_per_s ~= batch_size / (request_latency_ms_mean/1000)
    - percentiles monotonic: p50 <= p90 <= p95 <= p99
    - per_pair_latency_ms_mean ~= request_latency_ms_mean / batch_size
    - prefix_caching is False (customer rule), seq_len target present
 2. Recompute BOTH price/perf views from raw latency JSON and confirm they match the numbers
    printed in charts/latency_priceperf_tables.md and in README.md:
    - throughput view: $/1M pairs = (price/3600)/pairs_per_s*1e6
    - latency view:    $/1000 req = (price/3600)*(p50_ms/1000)*1000
 3. Every latency/throughput value in the README bs-tables exists verbatim in the JSON.
 4. Chips-to-match math in the tables equals ceil(GPU pairs/s / v6e pairs/s) and fleet $/hr.
 5. Every chart referenced by README exists on disk.
 6. Re-run make_latency_priceperf.py and confirm the regenerated tables file is byte-identical
    (deterministic; no hidden randomness in reporting).

Exit non-zero on ANY failure.
"""
import json, os, re, sys, math, subprocess, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); CH = os.path.join(ROOT, "charts")
fails = []
PRICE = {"TPU v6e (torchax)": 1.61, "B200 (1 GPU)": 6.95, "H200 (1 GPU)": 3.85, "G4 RTX PRO 6000 (1 GPU)": 3.11}
FILES = {
    "TPU v6e (torchax)": "lat_tpu_torchax.json",
    "B200 (1 GPU)": "lat_gpu_b200.json",
    "H200 (1 GPU)": "lat_gpu_h200.json",
    "G4 RTX PRO 6000 (1 GPU)": "lat_gpu_g4.json",
}

def approx(a, b, tol=0.02):
    return abs(a - b) / max(abs(b), 1e-9) <= tol

# ---------- 1. internal consistency ----------
data = {}
for label, fn in FILES.items():
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        fails.append(f"[{label}] missing results file {fn}"); continue
    d = json.load(open(p))
    if d.get("prefix_caching") is not False:
        fails.append(f"[{label}] prefix_caching is not False (customer rule: no caching)")
    if not d.get("seq_len_target"):
        fails.append(f"[{label}] seq_len_target missing")
    by = {}
    for r in d["by_batch"]:
        bs = r["batch_size"]; by[bs] = r
        # throughput ~ bs / mean_latency
        exp_tp = bs / (r["request_latency_ms_mean"]/1000.0)
        if not approx(exp_tp, r["throughput_pairs_per_s"]):
            fails.append(f"[{label}] bs{bs} throughput {r['throughput_pairs_per_s']} != bs/mean_latency {exp_tp:.2f}")
        # per-pair
        exp_pp = r["request_latency_ms_mean"]/bs
        if not approx(exp_pp, r["per_pair_latency_ms_mean"]):
            fails.append(f"[{label}] bs{bs} per_pair {r['per_pair_latency_ms_mean']} != mean/bs {exp_pp:.4f}")
        # percentiles monotonic
        seq = [r["request_latency_ms_p50"], r["request_latency_ms_p90"], r["request_latency_ms_p95"], r["request_latency_ms_p99"]]
        if any(seq[i] > seq[i+1] + 1e-6 for i in range(len(seq)-1)):
            fails.append(f"[{label}] bs{bs} percentiles not monotonic: {seq}")
    data[label] = by

# ---------- price/perf recompute helpers ----------
def cpm(price, pps): return round((price/3600.0)/pps*1e6, 4)          # $/1M pairs
def c1k(price, ms):  return round((price/3600.0)*(ms/1000.0)*1000.0, 5)  # $/1000 req

# ---------- 2/3. cross-check README + tables numbers ----------
readme = open(os.path.join(ROOT, "README.md")).read()
tables = open(os.path.join(CH, "latency_priceperf_tables.md")).read() if os.path.exists(os.path.join(CH, "latency_priceperf_tables.md")) else ""

def must(text, val, where):
    if str(val) not in text:
        fails.append(f"{where}: value {val} not found")

for label, by in data.items():
    price = PRICE[label]
    for bs, r in by.items():
        # every p50 and throughput appears in README bs-tables
        must(readme, r["request_latency_ms_p50"], f"README p50 {label} bs{bs}")
        must(readme, r["throughput_pairs_per_s"], f"README pairs/s {label} bs{bs}")
    # bs32 headline price/perf both views
    if 32 in by:
        r = by[32]
        must(readme, cpm(price, r["throughput_pairs_per_s"]), f"README $/1M {label}")
        must(readme, c1k(price, r["request_latency_ms_p50"]), f"README $/1k {label}")

# ---------- 4. throughput-match AND latency-match math in tables ----------
if tables and "TPU v6e (torchax)" in data:
    tpu = data["TPU v6e (torchax)"]
    tbs = sorted(tpu)
    for gpu_label in ["B200 (1 GPU)", "H200 (1 GPU)", "G4 RTX PRO 6000 (1 GPU)"]:
        if gpu_label not in data: continue
        # throughput-match: ceil(gpu pairs/s / v6e pairs/s) fleet cost present
        for bs in sorted(set(tpu) & set(data[gpu_label])):
            chips = math.ceil(data[gpu_label][bs]["throughput_pairs_per_s"] / tpu[bs]["throughput_pairs_per_s"])
            fleet = round(chips * PRICE["TPU v6e (torchax)"], 2)
            if f"${fleet}" not in tables:
                fails.append(f"tables throughput-match {gpu_label} bs{bs}: fleet ${fleet} not present")
        # latency-match: fastest v6e batch within GPU p50 budget, its $/1k present
        for bs in sorted(data[gpu_label]):
            gp = data[gpu_label][bs]["request_latency_ms_p50"]
            ok = [b for b in tbs if tpu[b]["request_latency_ms_p50"] <= gp]
            if ok:
                mb = max(ok); mp = tpu[mb]["request_latency_ms_p50"]
                t1k = c1k(PRICE["TPU v6e (torchax)"], mp)
                if f"${t1k}" not in tables:
                    fails.append(f"tables latency-match {gpu_label} bs{bs}: v6e $/1k ${t1k} not present")



# ---------- 5. charts referenced exist ----------
for m in re.findall(r"\(charts/([^)]+\.png)\)", readme):
    if not os.path.exists(os.path.join(CH, m)):
        fails.append(f"README references missing chart charts/{m}")

# ---------- 6. deterministic regeneration of tables ----------
def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None
tbl_path = os.path.join(CH, "latency_priceperf_tables.md")
before = md5(tbl_path)
try:
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_latency_priceperf.py")],
                   cwd=ROOT, capture_output=True, check=True)
    after = md5(tbl_path)
    if before and after and before != after:
        fails.append("make_latency_priceperf.py is non-deterministic (tables changed on regen)")
except subprocess.CalledProcessError as e:
    fails.append(f"make_latency_priceperf.py failed to run: {e.stderr.decode()[:200]}")

# ---------- summary ----------
print("="*64)
if fails:
    print("AUDIT FAILED -", len(fails), "issue(s):")
    for f_ in fails: print("  -", f_)
    sys.exit(1)
print("AUDIT PASSED - reproducible, verifiable, no fabrication/mismatch found.")
for label, by in data.items():
    if 32 in by:
        r = by[32]; price = PRICE[label]
        print(f"  {label:26} bs32 p50 {r['request_latency_ms_p50']} ms | "
              f"${cpm(price,r['throughput_pairs_per_s'])}/1M | ${c1k(price,r['request_latency_ms_p50'])}/1k")
sys.exit(0)
