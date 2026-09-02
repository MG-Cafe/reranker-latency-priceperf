# TensorRT-LLM re-benchmark - handoff for next session

## Why
Customer reconciled the earlier latency gap: their fast B200 number comes from **TensorRT-LLM**, not vLLM.
Their table (same workload, different engine):

| GPU | vLLM p50 | TRT-LLM p50 | speedup |
|---|---:|---:|---:|
| B200 | 102.94 ms | 23.70 ms | 4.34x |
| H200 | 98.12 ms | 40.13 ms | 2.45x |
| RTX PRO 6000 (G4) | 121.69 ms | 66.06 ms | 1.84x |

Our vLLM numbers agree with their vLLM numbers; the discrepancy is purely engine (vLLM vs TRT-LLM).

## Decision (agreed with user)
- Re-benchmark the GPUs on **TensorRT-LLM** and compare to **TPU v6e on vLLM torchax** (TRT-LLM has NO TPU backend).
- **Precision = BF16 on all chips** (BF16 works on B200/H200/G4 AND TPU v6e) for a fair comparison. (GPU FP8 has no native TPU equivalent; TPU low-precision option would be INT8, not FP8.)
- Same workload as before: Qwen3-Reranker-0.6B, bs32 seq512, prefix caching OFF, fresh content per request, p50/p90/p99; ideally also the batch sweep 1..64 to mirror the vLLM study.
- Keep existing vLLM baselines; present vLLM vs TRT side by side; recompute price/perf ($/1M pairs, $/1000 req) on TRT numbers. Prices/chip-hr: v6e 1.61, B200 6.95, H200 3.85, G4 3.11.

## Key gotcha discovered
- **a3-ultragpu-8g comes up as H200 in this project, NOT B200.** The real B200 machine type is **a4-highgpu-8g** (Blackwell). Always verify with `nvidia-smi --query-gpu=name` before trusting a "B200" VM.
- TRT-LLM pip install fails building a prerelease from source ("No such file or directory: 'cc'"). Use NVIDIA's **prebuilt container** `nvcr.io/nvidia/tensorrt-llm/release:<tag>` (install NVIDIA Container Toolkit first), or install gcc/cc + pin a stable (non-prerelease) tensorrt-llm.
- TRT-LLM is generation-oriented; Qwen3-Reranker-0.6B is a seq-cls/scoring head -> need to build a TRT engine and wire the score/rerank path. Non-trivial; budget setup time per GPU.

## Provisioning helpers (in ~/)
- `b200_hunt_a4.sh` - hunts a GENUINE B200 via a4-highgpu-8g across all zones (running in background; writes ~/b200_real.zone on success, VM name `b200-real`).
- `gpu_hunt_trt.sh <b200|h200|g4> <name>` - generic worldwide hunter.
- Current VMs (verify names/zones/GPU with nvidia-smi at session start; Spot may have preempted):
  - G4 (RTX PRO 6000): `g4-trt` (asia-east1-a), `g4-bench` (us-central1-b)
  - TPU v6e: `torchtpu-1` (asia-northeast1-b) - vLLM torchax already set up
  - B200: pending real a4-highgpu-8g (hunter running)
  - H200: recreate via a3-ultragpu-8g when needed (comes up as H200)

## Steps next session
1. Confirm/obtain: real B200 (a4-highgpu-8g, verify nvidia-smi=B200), H200 (a3-ultragpu-8g), G4 (g4-standard-48). TPU stays vLLM.
2. On each GPU: install NVIDIA Container Toolkit, pull TRT-LLM container, build BF16 engine for Qwen3-Reranker-0.6B, run bs32 seq512 (no cache, fresh content), capture p50/p90/p99 + throughput; also batch sweep if time.
3. Record exact versions (TRT-LLM, TensorRT, CUDA, driver) for the report.
4. Add results to repo: results/trt_*.json, charts (vLLM vs TRT), update `scripts/make_latency_priceperf.py` or a new `make_trt_compare.py`, extend README + audits.
5. Regenerate the DOCX (`scripts/make_report_docx.py`) with a TRT section (GPUs=TRT-LLM BF16 vs TPU=vLLM torchax BF16) and refresh the Downloads copy.

## READY-TO-GO INVENTORY (verified this session, 2026-09-01)
All chips are provisioned, verified, and RUNNING/READY, start TRT work immediately next session:
- **B200 (real, a4-highgpu-8g)**: VM `b200-real`, zone `europe-west1-b`, nvidia-smi = "NVIDIA B200, 183359 MiB", driver 580.173.02. RUNNING (Spot).
- **G4 (RTX PRO 6000 Blackwell)**: VM `g4-trt`, zone `asia-east1-a`, verified. RUNNING (Spot). Backup: `g4-bench` (us-central1-b) RUNNING.
- **TPU v6e**: `torchtpu-1`, zone `asia-northeast1-b`, READY (stays on vLLM torchax, BF16).
- H200: recreate via a3-ultragpu-8g when needed (that SKU = H200 in this project).
Background hunters have been stopped (we have what we need). NOTE: these are Spot, re-verify status + `nvidia-smi` at the start of the next session (they may have been preempted).

## OUTPUT (per user): create a NEW repo + NEW report, do NOT overwrite the existing ones
- New GitHub repo under MG-Cafe (e.g. `reranker-trt-vs-vllm` or similar), fresh git history, credential-free, neutral wording.
- New DOCX report (results-only) into ~/Downloads for Google Docs upload.
- Present vLLM baseline vs TensorRT-LLM (BF16) for B200/H200/G4, plus TPU v6e (vLLM torchax BF16), with both price/perf views and the same audits.

## Reminder

All results-only, neutral wording (no customer references) in the public repo/report; keep the "no caching, fresh content, seq 512, bs32" definition identical for apples-to-apples.
