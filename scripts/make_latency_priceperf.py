#!/usr/bin/env python3
"""Latency-first price/performance for Qwen3-Reranker-0.6B across accelerators.

Real-time serving, prefix caching OFF, synthetic seq-len 512, fresh content per request.

Chips (prices per chip-hour): TPU v6e $1.61, B200 $6.95, H200 $3.85, G4 (1x RTX PRO 6000) $3.11.
A device is included only if its results/lat_gpu_*.json (or lat_tpu_*.json) file exists.


Two price/perf views:
  1) Throughput-based: USD per 1M pairs = (price_per_hr/3600)/pairs_per_s*1e6.
  2) Latency-based: cost per 1000 requests = price_per_hr/3600 * request_latency_s * 1000.

Outputs (charts/): latency_vs_batch.png, priceperf_latency_bs32.png, priceperf_throughput_bs32.png,
                   latency_priceperf_tables.md
"""
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); CH = os.path.join(ROOT, "charts"); os.makedirs(CH, exist_ok=True)

# (label, results file, price/chip-hr, color)
DEVICES = [
    ("TPU v6e (torchax)", "lat_tpu_torchax.json", 1.61, "#e37400"),
    ("B200 (1 GPU)",      "lat_gpu_b200.json",    6.95, "#76b900"),
    ("H200 (1 GPU)",      "lat_gpu_h200.json",    3.85, "#1a73e8"),
    ("G4 RTX PRO 6000 (1 GPU)", "lat_gpu_g4.json", 3.11, "#9334e6"),
]

loaded = []
for label, fn, price, color in DEVICES:
    p = os.path.join(RES, fn)
    if os.path.exists(p):
        d = json.load(open(p))
        by = {r["batch_size"]: r for r in d["by_batch"]}
        loaded.append({"label": label, "price": price, "color": color, "by": by, "data": d})
present = {d["label"] for d in loaded}
G4_PENDING = "G4 RTX PRO 6000 (1 GPU)" not in present

def cpm(price, pps): return round((price/3600.0)/pps*1e6, 4)          # USD/1M pairs
def cost_per_1k(price, lat_ms): return round((price/3600.0)*(lat_ms/1000.0)*1000.0, 5)  # USD/1000 req

# batch sizes union
allbs = sorted({bs for d in loaded for bs in d["by"]})

# ---------- chart: request latency p50 vs batch ----------
plt.figure(figsize=(8.5,5.5))
for d in loaded:
    xs = [bs for bs in allbs if bs in d["by"]]
    ys = [d["by"][bs]["request_latency_ms_p50"] for bs in xs]
    plt.plot(xs, ys, marker="o", color=d["color"], label=d["label"])
plt.xscale("log", base=2); plt.xlabel("Batch size (pairs per real-time request)")
plt.ylabel("Request latency p50 (ms)")
plt.title("Real-time request latency vs batch (seq 512, no prefix cache)")
plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(CH, "latency_vs_batch.png"), dpi=150); plt.close()

BS = 32
have_bs32 = [d for d in loaded if BS in d["by"]]

# ---------- latency-based price/perf bar @bs32 ----------
labels = [d["label"] for d in have_bs32]
colors = [d["color"] for d in have_bs32]
lat_vals = [cost_per_1k(d["price"], d["by"][BS]["request_latency_ms_p50"]) for d in have_bs32]
plt.figure(figsize=(8,5)); bars = plt.bar(labels, lat_vals, color=colors)
for b, v in zip(bars, lat_vals): plt.annotate(f"${v:.4f}", (b.get_x()+b.get_width()/2, v), textcoords="offset points", xytext=(0,4), ha="center", fontsize=8)
plt.ylabel("USD per 1000 requests (bs=32, lower=better)")
plt.title("Latency-based price/perf at bs=32 (cost = price/hr x request latency)")
plt.xticks(rotation=12, ha="right"); plt.grid(True, axis="y", ls=":", alpha=0.5); plt.tight_layout()
plt.savefig(os.path.join(CH, "priceperf_latency_bs32.png"), dpi=150); plt.close()

# ---------- throughput-based price/perf bar @bs32 ----------
tp_vals = [cpm(d["price"], d["by"][BS]["throughput_pairs_per_s"]) for d in have_bs32]
plt.figure(figsize=(8,5)); bars = plt.bar(labels, tp_vals, color=colors)
for b, v in zip(bars, tp_vals): plt.annotate(f"${v:.4f}", (b.get_x()+b.get_width()/2, v), textcoords="offset points", xytext=(0,4), ha="center", fontsize=8)
plt.ylabel("USD per 1M pairs (bs=32, lower=better)")
plt.title("Throughput-based price/perf at bs=32")
plt.xticks(rotation=12, ha="right"); plt.grid(True, axis="y", ls=":", alpha=0.5); plt.tight_layout()
plt.savefig(os.path.join(CH, "priceperf_throughput_bs32.png"), dpi=150); plt.close()

# ---------- tables ----------
L = []
L.append("# Latency-first price/performance across accelerators\n")
L.append("Real-time serving; prefix caching OFF; synthetic seq-len 512; fresh content per request "
         "(no KV/prefix reuse). Prices per chip-hour: TPU v6e $1.61, B200 $6.95, H200 $3.85, G4 $3.11.")
if G4_PENDING:
    L.append("G4 (1x RTX PRO 6000): results not available in this run; not shown in the tables below.")

L.append("")

L.append("## Request latency p50/p99 (ms) by batch size\n")
head = "| Batch | " + " | ".join(f"{d['label']} p50" for d in loaded) + " |"
sep = "|------:|" + "|".join(["---:"]*len(loaded)) + "|"
L.append(head); L.append(sep)
for bs in allbs:
    row = f"| {bs} | " + " | ".join((str(d["by"][bs]["request_latency_ms_p50"]) if bs in d["by"] else "-") for d in loaded) + " |"
    L.append(row)
L.append("")

L.append("## Throughput (pairs/s) by batch size\n")
L.append(head.replace("p50", "pairs/s")); L.append(sep)
for bs in allbs:
    row = f"| {bs} | " + " | ".join((str(d["by"][bs]["throughput_pairs_per_s"]) if bs in d["by"] else "-") for d in loaded) + " |"
    L.append(row)
L.append("")

L.append(f"## Both price/perf views at batch size {BS} (customer operating point)\n")
L.append("| Device | p50 latency (ms) | p99 latency (ms) | pairs/s | $/1M pairs (throughput) | $/1000 req (latency) |")
L.append("|---|---:|---:|---:|---:|---:|")
for d in have_bs32:
    r = d["by"][BS]
    L.append(f"| {d['label']} | {r['request_latency_ms_p50']} | {r['request_latency_ms_p99']} | "
             f"{r['throughput_pairs_per_s']} | ${cpm(d['price'], r['throughput_pairs_per_s'])} | "
             f"${cost_per_1k(d['price'], r['request_latency_ms_p50'])} |")
L.append("")

# chips-to-match latency/throughput vs each GPU, using TPU v6e as the scalable unit
tpu = next((d for d in loaded if d["label"].startswith("TPU")), None)
if tpu:
    for gpu in [d for d in loaded if d["label"].startswith(("B200","H200"))]:
        L.append(f"## Chips-to-match: v6e vs {gpu['label']}\n")
        L.append("| Batch | GPU pairs/s | 1x v6e pairs/s | v6e chips to match | v6e fleet $/hr | GPU $/hr | TPU fleet cheaper? |")
        L.append("|------:|-----------:|---------------:|-------------------:|---------------:|--------:|:--|")
        for bs in allbs:
            if bs in gpu["by"] and bs in tpu["by"]:
                gpps = gpu["by"][bs]["throughput_pairs_per_s"]; tpps = tpu["by"][bs]["throughput_pairs_per_s"]
                chips = math.ceil(gpps / tpps)
                fleet = round(chips * tpu["price"], 2)
                L.append(f"| {bs} | {gpps} | {tpps} | {chips} | ${fleet} | ${gpu['price']} | {'yes' if fleet < gpu['price'] else 'no'} |")
        L.append("")

open(os.path.join(CH, "latency_priceperf_tables.md"), "w").write("\n".join(L)+"\n")
print("\n".join(L))
print("\nDevices included:", ", ".join(present))
print("G4 pending:", G4_PENDING)
print("Charts + tables written to", CH)
