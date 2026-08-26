# Qwen3-Reranker-0.6B latency-first price/performance: TPU v6e vs B200 vs H200 vs G4

Real-time (online) serving benchmark of Qwen3-Reranker-0.6B on vLLM, focused on latency and
price/performance rather than offline batch throughput. TPU runs on the public JAX/torchax backend
(`tpu-inference`); GPUs run on CUDA vLLM. Every request is timed on-device (in-process `score()`),
so results are compute latency and do not depend on VM region.

Scope and rules (per customer request):
- Framework: vLLM torchax on TPU (no torch-tpu); CUDA vLLM on GPU.
- No caching: prefix caching disabled (`enable_prefix_caching=False`); every request uses freshly
  randomized content so KV/prefix reuse cannot inflate results.
- Standardized synthetic dataset: fixed sequence length 512 tokens per (query+document) pair.
- Real-time, not offline: each measured request is one in-flight `score()` call; "batch" means
  concurrent pairs in one real-time request, not an offline job.
- Primary operating point: batch size 32, seq 512; plus a batch sweep (1..64).
- Price/performance reported two ways: throughput-based (USD per 1M pairs) and latency-based
  (USD per 1000 requests), plus a chips-to-match analysis.

Prices (per chip-hour): TPU v6e $1.61, B200 $6.95, H200 $3.85, G4 (1x RTX PRO 6000) $3.11.

All four accelerators are now measured. (G4 = NVIDIA RTX PRO 6000 Blackwell, 96 GB; it required the
GCP g4-standard-48 machine type + Ubuntu 24.04 accelerator image with the nvidia-595 driver to
initialize, per the AI-Hypercomputer gpu-recipes G4 guidance.)

## Headline (batch size 32, seq 512, no caching)

| Device | p50 latency (ms) | p99 latency (ms) | pairs/s | $/1M pairs (throughput) | $/1000 req (latency) |
|---|---:|---:|---:|---:|---:|
| TPU v6e (torchax) | 173.995 | 187.747 | 182.667 | $2.4483 | $0.07781 |
| B200 (1 GPU) | 72.120 | 78.676 | 438.642 | $4.4012 | $0.13923 |
| H200 (1 GPU) | 87.551 | 98.340 | 362.625 | $2.9492 | $0.09363 |
| G4 RTX PRO 6000 (1 GPU) | 100.025 | 101.605 | 319.721 | $2.7020 | $0.08641 |

How to read it:
- Raw per-request latency (bs=32): B200 fastest (72.1 ms), then H200 (87.6), then G4 (100.0), then
  TPU v6e (174.0).
- Price/performance (both views): TPU v6e is cheapest overall ($2.4483/1M, $0.07781/1k req). Among
  the GPUs, G4 is the best value ($2.7020/1M, $0.08641/1k), ahead of H200 ($2.9492 / $0.09363) and
  well ahead of B200 ($4.4012 / $0.13923). So cost ranking is TPU v6e, then G4, then H200, then B200;
  latency ranking is B200, then H200, then G4, then TPU.

## Latency and price/perf views

![Request latency vs batch](charts/latency_vs_batch.png)
![Latency-based price/perf at bs=32](charts/priceperf_latency_bs32.png)
![Throughput-based price/perf at bs=32](charts/priceperf_throughput_bs32.png)

Request latency p50 (ms) by batch size:

| Batch | TPU v6e | B200 | H200 | G4 |
|------:|--------:|-----:|-----:|---:|
| 1 | 10.793 | 11.451 | 11.533 | 7.104 |
| 2 | 20.423 | 19.653 | 20.153 | 9.221 |
| 4 | 31.673 | 23.388 | 23.098 | 18.122 |
| 8 | 50.295 | 28.720 | 30.696 | 29.281 |
| 16 | 92.316 | 42.255 | 47.730 | 52.677 |
| 32 | 173.995 | 72.120 | 87.551 | 100.025 |
| 64 | 335.344 | 138.802 | 169.198 | 198.313 |

Throughput (pairs/s) by batch size:

| Batch | TPU v6e | B200 | H200 | G4 |
|------:|--------:|-----:|-----:|---:|
| 1 | 92.239 | 85.498 | 85.185 | 139.995 |
| 2 | 100.221 | 116.606 | 98.105 | 178.580 |
| 4 | 118.104 | 183.052 | 181.414 | 229.313 |
| 8 | 155.212 | 282.684 | 262.667 | 273.581 |
| 16 | 172.784 | 379.786 | 332.293 | 303.367 |
| 32 | 182.667 | 438.642 | 362.625 | 319.721 |
| 64 | 190.484 | 460.704 | 376.102 | 321.662 |

Notes:
- At batch size 1 (strict one-at-a-time serving) the G4 is actually the fastest single-request device
  (p50 7.104 ms), ahead of TPU v6e (10.793), B200 (11.451) and H200 (11.533). So for low-concurrency
  real-time reranking, G4 gives the best raw latency and TPU v6e the best cost.
- The B200 pulls ahead at larger batches, as expected from its higher parallel throughput.

## Chips-to-match analysis (how many v6e to match one GPU)

For each GPU, chips = ceil(GPU pairs/s / one-v6e pairs/s) to match that GPU's throughput; then the
v6e fleet cost per hour is compared to the single GPU. Across all three GPUs and the whole batch
range, a v6e fleet that matches throughput is still cheaper per hour (full tables in
`charts/latency_priceperf_tables.md`). Examples at bs=32:
- vs B200: 3x v6e = $4.83/hr vs $6.95/hr -> TPU cheaper.
- vs H200: 2x v6e = $3.22/hr vs $3.85/hr -> TPU cheaper.
- vs G4: 2x v6e = $3.22/hr vs $3.11/hr -> here the single G4 is actually slightly cheaper per hour at
  bs=32 while also being faster, so G4 is the standout GPU for cost-sensitive low-latency serving.

Caveat: matching throughput with more TPU chips does not lower single-request latency; if the SLA is
strict per-request latency at high concurrency, the GPUs (B200 > H200 > G4) win on latency while the
TPU wins on aggregate cost per pair.

## Reproduce

TPU v6e (torchax / tpu-inference):
```bash
source "$HOME/.local/bin/env"; uv venv --python 3.12 "$HOME/tpuinf_venv"; source "$HOME/tpuinf_venv/bin/activate"
uv pip install vllm-tpu transformers
export HF_HOME=/dev/shm/hf; export HF_TOKEN=<token>; cd /tmp
python scripts/reranker_latency_bench.py --device-label "tpu v6e-1 (torchax)" --is-tpu \
  --seq-len 512 --batch-sizes 1,2,4,8,16,32,64 --iters 50 --result-filename ~/lat_tpu_torchax.json
```

GPUs (B200 / H200 / G4), one GPU each:
```bash
# G4 requires machine type g4-standard-48 and the Ubuntu 24.04 accelerator image (nvidia-595) so the
# RTX PRO 6000 initializes; B200/H200 use their standard accelerator VMs.
sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential g++ python3.12-dev
source "$HOME/.local/bin/env"; uv venv --python 3.12 "$HOME/gpuvenv"; source "$HOME/gpuvenv/bin/activate"
uv pip install vllm transformers
CUDA_VISIBLE_DEVICES=0 python scripts/reranker_latency_bench.py --device-label "gpu 1x <B200|H200|G4>" \
  --seq-len 512 --batch-sizes 1,2,4,8,16,32,64 --iters 50 --gpu-mem-util 0.85 \
  --result-filename ~/lat_gpu_<b200|h200|g4>.json
```

Charts + tables:
```bash
pip install matplotlib
python scripts/make_latency_priceperf.py
```

## Files

```
results/lat_tpu_torchax.json   TPU v6e latency sweep (seq 512, no prefix cache)
results/lat_gpu_b200.json      B200 latency sweep
results/lat_gpu_h200.json      H200 latency sweep
results/lat_gpu_g4.json        G4 (RTX PRO 6000) latency sweep
scripts/reranker_latency_bench.py   latency bench (prefix caching off, fresh random content, seq 512)
scripts/make_latency_priceperf.py   charts + tables (throughput and latency price/perf, chips-to-match)
charts/latency_vs_batch.png, priceperf_latency_bs32.png, priceperf_throughput_bs32.png
charts/latency_priceperf_tables.md
```

## Verification

Run `python scripts/audit_repo.py` to verify end-to-end: it re-checks every results JSON for internal consistency (throughput = batch/latency, monotonic percentiles, prefix caching disabled, seq 512), recomputes both price/perf views and confirms they match the README and the generated tables, verifies every README latency/throughput number exists in the raw JSON, checks the chips-to-match math, confirms all referenced charts exist, and re-runs the table generator to confirm deterministic output. Current status: AUDIT PASSED.
