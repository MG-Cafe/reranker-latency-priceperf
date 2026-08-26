#!/usr/bin/env python3
"""Concurrency sweep report for the Qwen3-Reranker (served vLLM /v1/rerank, no caching, seq 512/1024).

Consumes results/conc_<device>_seq<seq>_rb<n>.json (produced by scripts/reranker_conc_bench.py) and
produces charts + a markdown tables file. Reports p50/p90/p99 request latency and achieved QPS at
each concurrency, plus latency-based price/perf (USD per 1000 requests = price/hr/3600 * p50_s * 1000)
and the conc<=4 operating point meeting p90 < 100 ms.

Prices per chip-hour: TPU v6e $1.61, B200 $6.95, H200 $3.85, G4 $3.11.
"""
import json, os, glob, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); CH = os.path.join(ROOT, "charts"); os.makedirs(CH, exist_ok=True)
PRICE = {"b200": 6.95, "h200": 3.85, "g4": 3.11, "tpu": 1.61}
NAME = {"b200": "B200 (1 GPU)", "h200": "H200 (1 GPU)", "g4": "G4 RTX PRO 6000 (1 GPU)", "tpu": "TPU v6e (torchax)"}
COLOR = {"b200": "#76b900", "h200": "#1a73e8", "g4": "#9334e6", "tpu": "#e37400"}
ORDER = ["tpu", "b200", "h200", "g4"]

def load(dev, seq, rb):
    p = os.path.join(RES, f"conc_{dev}_seq{seq}_rb{rb}.json")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    d = json.load(open(p))
    return {r["concurrency"]: r for r in d["by_concurrency"]}

def c1k(dev, ms): return round((PRICE[dev]/3600.0)*(ms/1000.0)*1000.0, 5)

L = []
L.append("# Concurrency sweep: served vLLM reranker (real-time, no caching)\n")
L.append("Served via `vllm serve` /v1/rerank; prefix caching OFF; fresh randomized content per request "
         "so KV/prefix reuse cannot help. Each request is a single (query,document) pair (rb=1) or a "
         "batch of 32 pairs (rb=32). Concurrency = simultaneous in-flight requests. Prices per chip-hour: "
         "TPU v6e $1.61, B200 $6.95, H200 $3.85, G4 $3.11.\n")

def section(seq, rb, title, concs=(1,2,4,8)):
    data = {dev: load(dev, seq, rb) for dev in ORDER}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return
    L.append(f"## {title} (seq {seq}, {'1 pair/request' if rb==1 else str(rb)+' pairs/request'})\n")
    L.append("| Conc | " + " | ".join(f"{NAME[d]} p50 / p90 / p99 ms | req/s | $/1k" for d in data) + " |")
    L.append("|---:|" + "|".join(["---"]*len(data)) + "|")
    for c in concs:
        cells = []
        for d in data:
            r = data[d].get(c)
            if r:
                cells.append(f"{r['req_latency_ms_p50']}/{r['req_latency_ms_p90']}/{r['req_latency_ms_p99']} | {r['qps']} | ${c1k(d, r['req_latency_ms_p50'])}")
            else:
                cells.append("- | - | -")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")
    # chart: p50 vs concurrency
    plt.figure(figsize=(8,5))
    for d in data:
        xs = sorted(data[d]); ys = [data[d][c]["req_latency_ms_p50"] for c in xs]
        plt.plot(xs, ys, marker="o", color=COLOR[d], label=NAME[d])
    plt.xlabel("Concurrency (in-flight requests)"); plt.ylabel("Request latency p50 (ms)")
    plt.title(f"{title} - p50 latency vs concurrency (seq {seq}, rb{rb}, no cache)")
    plt.grid(True, ls=":", alpha=0.5); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(CH, f"concurrency_seq{seq}_rb{rb}_p50.png"), dpi=150); plt.close()
    # chart: QPS vs concurrency
    plt.figure(figsize=(8,5))
    for d in data:
        xs = sorted(data[d]); ys = [data[d][c]["qps"] for c in xs]
        plt.plot(xs, ys, marker="s", color=COLOR[d], label=NAME[d])
    plt.xlabel("Concurrency (in-flight requests)"); plt.ylabel("Throughput (requests/s)")
    plt.title(f"{title} - throughput vs concurrency (seq {seq}, rb{rb}, no cache)")
    plt.grid(True, ls=":", alpha=0.5); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(CH, f"concurrency_seq{seq}_rb{rb}_qps.png"), dpi=150); plt.close()

section(512, 1, "Single-pair requests")
section(512, 32, "Batch-32 requests")
section(1024, 32, "Batch-32 requests, long context", concs=(1,2,4,8))

# p90 < 100ms operating point at conc 4 (customer's rough target)
L.append("## Operating point: p90 < 100 ms (rough SLA)\n")
L.append("Highest concurrency at which each device keeps p90 < 100 ms, per workload:\n")
def best_conc_under_p90(dev, seq, rb, limit=100.0):
    d = load(dev, seq, rb)
    if not d: return None
    ok = [c for c in sorted(d) if d[c]["req_latency_ms_p90"] < limit]
    return max(ok) if ok else None
L.append("| Workload | " + " | ".join(NAME[d] for d in ORDER) + " |")
L.append("|---|" + "|".join(["---"]*len(ORDER)) + "|")
for seq, rb, lbl in [(512,1,"single pair, seq512"), (512,32,"bs32, seq512"), (1024,32,"bs32, seq1024")]:
    cells = []
    for d in ORDER:
        c = best_conc_under_p90(d, seq, rb)
        cells.append(f"conc {c}" if c else "n/a")
    L.append(f"| {lbl} | " + " | ".join(cells) + " |")
L.append("")

open(os.path.join(CH, "concurrency_tables.md"), "w").write("\n".join(L)+"\n")
print("\n".join(L))
print("\nConcurrency charts + tables written to", CH)
