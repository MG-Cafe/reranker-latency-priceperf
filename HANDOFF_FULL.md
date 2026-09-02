# FULL HANDOFF - Qwen3-Reranker-0.6B accelerator benchmarking (for a fresh AI session)

Last updated: 2026-09-02. Read this top-to-bottom; it contains everything done so far, exact state,
gotchas, and the precise next steps. A ready-to-paste kickoff PROMPT is at the very bottom.

--------------------------------------------------------------------------------
## 0. TL;DR of what this project is
Benchmark **Qwen3-Reranker-0.6B** for real-time serving across accelerators and report
**latency-first price/performance**. Chips: **TPU v6e** (Google), **NVIDIA B200, H200, G4 (RTX PRO 6000
Blackwell)**. Everything is single-chip. Rules that MUST hold for every run:
- Framework so far: **vLLM** (torchax on TPU, CUDA vLLM on GPUs).
- **No caching**: prefix caching disabled AND fresh randomized content per request (no KV/prefix reuse).
- Standardized synthetic data: **512 tokens per (query+doc) pair** (plus a 1024 long-context sweep).
- **Real-time, not offline**: each request = one in-flight `score()`/rerank call. "bs32" = ONE request
  containing 32 pairs. "concurrency" = simultaneous in-flight requests (served HTTP).
- Prices per chip-hour used everywhere: **TPU v6e $1.61, B200 $6.95, H200 $3.85, G4 $3.11**
  (H200 was 1-yr CUD per the requester).

Two price/perf definitions (memorize):
- Throughput view: **USD per 1M pairs = (price_per_hr/3600)/pairs_per_s * 1e6**.
- Latency view: **USD per 1000 requests = (price_per_hr/3600) * (p50_ms/1000) * 1000**.
- Throughput (pairs/s) = batch_size / (mean_request_latency_s).

--------------------------------------------------------------------------------
## 1. Existing repo (DONE, pushed, public, audited) - DO NOT overwrite
GitHub: **https://github.com/MG-Cafe/reranker-latency-priceperf** (org MG-Cafe; commits authored as
"MG-Cafe <mg-cafe@users.noreply.github.com>").
Local: **/Users/emgi/reranker-latency-priceperf** (this file lives here).

Contents (all vLLM results, credential-free, neutral wording - NO customer references anywhere):
- results/ : lat_tpu_torchax.json, lat_gpu_b200.json, lat_gpu_h200.json, lat_gpu_g4.json (single-request
  latency sweeps bs 1..64, seq512), and conc_{tpu,b200,h200,g4}_seq512_rb1.json, _seq512_rb32.json,
  _seq1024_rb32.json (served concurrency sweeps).
- scripts/ : reranker_latency_bench.py (in-process latency bench), reranker_conc_bench.py (served
  /v1/rerank concurrency bench), make_latency_priceperf.py (charts+tables incl per-batch charts and
  throughput_vs_batch), make_concurrency.py (concurrency charts+tables), audit_repo.py + audit_concurrency.py
  (strict audits; BOTH PASS), g4_provision.sh, make_report_docx.py (the DOCX report generator).
- charts/ : latency_vs_batch.png, throughput_vs_batch.png, priceperf_latency_bs32.png,
  priceperf_throughput_bs32.png, latency_priceperf_tables.md, concurrency_tables.md, concurrency_*_p50/qps.png,
  and charts/by_batch/*.png (28 per-batch charts). README embeds ALL charts inline.
- README.md : full write-up: metrics/formulas, headline bs32 table, batch sweep + charts, throughput-match
  + latency-match, concurrency sweep tables+charts, sequence-length (512 vs 1024) section, reproduce, verification.

Headline vLLM numbers (bs32, seq512, no cache), already in the repo:
| Device | p50 ms | p99 ms | pairs/s | $/1M pairs | $/1000 req |
|---|---:|---:|---:|---:|---:|
| TPU v6e (torchax) | 173.995 | 187.747 | 182.667 | $2.4483 | $0.07781 |
| B200 (1 GPU)      | 72.120  | 78.676  | 438.642 | $4.4012 | $0.13923 |
| H200 (1 GPU)      | 87.551  | 98.340  | 362.625 | $2.9492 | $0.09363 |
| G4 RTX PRO 6000   | 100.025 | 101.605 | 319.721 | $2.7020 | $0.08641 |
Rankings (vLLM): latency B200<H200<G4<TPU; price/perf TPU<G4<H200<B200 (TPU cheapest). At bs1 G4 is
fastest single request (7.1ms). 512->1024 tokens ~doubles latency, halves throughput, ranking unchanged.

## 1a. DOCX report already delivered (vLLM)
- Generator: scripts/make_report_docx.py. Output committed at report/Qwen3-Reranker-0.6B_Accelerator_Benchmark_Report.docx
  and a copy is in **~/Downloads/Qwen3-Reranker-0.6B_Accelerator_Benchmark_Report.docx** (for Google Docs upload).
- It is results-only, neutral, with charts+tables+analysis.

--------------------------------------------------------------------------------
## 2. WHY we are doing a second round (TensorRT-LLM)
The requester reported much lower B200 latency and shared this (same workload, different engine):
| GPU | vLLM p50 | TRT-LLM p50 | speedup |
|---|---:|---:|---:|
| B200 | 102.94 ms | 23.70 ms | 4.34x |
| H200 | 98.12 ms  | 40.13 ms | 2.45x |
| RTX PRO 6000 (G4) | 121.69 ms | 66.06 ms | 1.84x |
Conclusion: the gap is purely **engine** (our vLLM vs their TensorRT-LLM). Our vLLM numbers agree with
their vLLM numbers. So we will re-benchmark the GPUs on **TensorRT-LLM** and compare against **TPU v6e on
vLLM torchax** (TRT-LLM has NO TPU backend - cannot run on v6e).

Precision decision (agreed): **BF16 on ALL chips** for a fair comparison (BF16 works on B200/H200/G4 and
TPU). GPU FP8 has no native TPU equivalent; the TPU low-precision option would be INT8 (different format).
NOTE: their 23.7ms is likely FP8; BF16 TRT will be faster than vLLM but won't fully reach 23.7ms - that's
expected and honest. Keep vLLM baselines; present vLLM vs TRT side by side; recompute both price/perf views.

--------------------------------------------------------------------------------
## 3. CURRENT LIVE INFRA (verified 2026-09-02) - GCP project **diesel-patrol-382622**
All RUNNING/READY right now (Spot GPUs - RE-VERIFY with nvidia-smi at session start; may be preempted):
- **B200 (REAL)**: VM `b200-real`, zone `europe-west1-b`, machine `a4-highgpu-8g`, nvidia-smi="NVIDIA B200"
  x8, 183GB each, driver 580.173.02. Docker + NVIDIA Container Toolkit INSTALLED; GPU-in-Docker VERIFIED.
- **G4 (RTX PRO 6000 Blackwell)**: VM `g4-trt`, zone `asia-east1-a`, machine `g4-standard-48`. Backup:
  `g4-bench`, zone `us-central1-b`.
- **TPU v6e**: TPU VM `torchtpu-1`, zone `asia-northeast1-b`, state READY (has vLLM torchax venv from round 1).
- **H200**: NOT currently up (we deleted the mislabeled ones). Recreate with `a3-ultragpu-8g`.

### CRITICAL GOTCHA: machine types
- **a3-ultragpu-8g comes up as H200 in this project (NOT B200).** ALWAYS `nvidia-smi --query-gpu=name` to confirm.
- **The real B200 is `a4-highgpu-8g` (Blackwell).**
- G4 (RTX PRO 6000) = `g4-standard-48` + Ubuntu 24.04 accelerator image `ubuntu-accelerator-2404-amd64-with-nvidia-595-*`
  (needs python3.12-dev; earlier the GRID/vGPU images failed - use g4-standard-48 + 2404/595, that worked).

### Provisioning helper scripts in ~/ (reuse if a VM is gone):
- `b200_hunt_a4.sh` (correct B200 via a4-highgpu-8g, sweeps all zones spot->on-demand, writes ~/b200_real.zone).
- `gpu_hunt_trt.sh <b200|h200|g4> <name>` (generic worldwide hunter).
- `h200_provision.sh` (a3-ultragpu-8g H200), `g4_recipe.sh`/`g4_recipe2.sh` (G4), `b200_provision.sh`.
- vLLM latency bench: ~/reranker_latency_bench.py ; concurrency bench: ~/reranker_conc_bench.py.

--------------------------------------------------------------------------------
## 4. TRT-LLM setup state on B200 (what worked / what to do)
- Docker + NVIDIA Container Toolkit installed on b200-real; `docker run --gpus all ... nvidia-smi` shows 8x B200. GOOD.
- **BLOCKER 1**: `pip install tensorrt-llm` on the bare VM tried to build a prerelease from source and failed:
  "No such file or directory: 'cc'" (no toolchain). Do NOT install TRT-LLM on the bare VM.
- **BLOCKER 2**: `nvcr.io/nvidia/tensorrt-llm/release:latest` tag does NOT exist ("not found").
- **WORKING APPROACH (in progress)**: pull public NGC PyTorch container **`nvcr.io/nvidia/pytorch:25.01-py3`**
  (has full CUDA toolchain incl gcc/cc), run it with `--gpus all`, then `pip install tensorrt_llm` INSIDE it.
  The pull was started in background on b200-real (log: ~/ngc_pull.log; look for DONE_PULL / PULL_RC=0).
  Alternative valid images to try if needed: a specific dated TRT-LLM release tag from NGC (check
  `nvcr.io/nvidia/tensorrt-llm/release:<YY.MM-pyN>`), or `nvcr.io/nvidia/tritonserver:*-trtllm-python-py3`.

--------------------------------------------------------------------------------
## 5. EXACT NEXT STEPS (do these in order)
1. Re-verify all VMs + `nvidia-smi` names. Recreate any preempted Spot VM with the helper scripts. Recreate H200
   via a3-ultragpu-8g (confirm nvidia-smi=H200).
2. On b200-real: confirm `~/ngc_pull.log` shows the pytorch container pulled. Then run the container:
   `sudo docker run --rm --gpus all -v $HOME:/work -w /work nvcr.io/nvidia/pytorch:25.01-py3 bash -lc "..."`.
   Inside: `pip install tensorrt_llm` (or the pinned wheel), verify `python -c "import tensorrt_llm,tensorrt;print(...)"`,
   capture EXACT versions (tensorrt_llm, tensorrt, cuda, driver) for the report.
3. Build a **BF16** TRT-LLM engine for **Qwen/Qwen3-Reranker-0.6B**. This model is a seq-cls/scoring head
   (hf_overrides: architectures=["Qwen3ForSequenceClassification"], classifier_from_token=["no","yes"],
   is_original_qwen3_reranker=true). TRT-LLM is generation-oriented, so you must wire the scoring/logit path
   (or use TRT-LLM's rerank/pooling support if the installed version has it). Match the exact workload:
   bs32, seq512, prefix caching OFF, FRESH random content per request, 50 iters, report p50/p90/p99 + pairs/s.
   Also do the batch sweep 1..64 to mirror vLLM. (Concurrency + seq1024 optional if time.)
4. Save raw JSON as results/trt_gpu_b200.json (+ h200, g4) in the SAME schema as lat_gpu_*.json so the
   existing generators/audits can consume them.
5. Repeat on G4 (g4-trt) and H200. TPU v6e stays vLLM torchax (reuse round-1 numbers, label engine clearly).
6. Recompute both price/perf views on the TRT numbers (same prices). Make vLLM-vs-TRT comparison charts+tables.

--------------------------------------------------------------------------------
## 6. OUTPUT for round 2 (per the requester) - create NEW, do NOT overwrite round 1
- **NEW GitHub repo** under MG-Cafe, e.g. `reranker-trt-vs-vllm` (fresh `git init`, single clean commit,
  credential-free, neutral wording - NO customer/company references, no internal project IDs; scrub like round 1).
  Include: results/ (trt_*.json + copy the vLLM lat_*.json baselines), scripts (bench + a make_trt_compare.py
  generator + audits), charts, README embedding all charts, and a Verification section with an audit that PASSES.
- **NEW DOCX report** (results-only, neutral) generated by a make_report_docx.py variant; copy to ~/Downloads.
  Present per chip: vLLM (BF16) vs TensorRT-LLM (BF16) latency + both price/perf views, with clear engine labels
  (GPUs=TRT-LLM, TPU=vLLM torchax), plus the vLLM-vs-TRT speedup table and the (their) reference numbers noted as
  "reported" if you cite them.
- Mirror round-1 quality: strict audit scripts (recompute every number from JSON, verify README matches,
  deterministic chart regen), all charts embedded in README, and no fabrication.

--------------------------------------------------------------------------------
## 7. Guardrails / lessons learned (important)
- Be honest and precise: never change bs/seq or turn caching on to "improve" a number; label engine & precision.
- SSH to these VMs intermittently returns code 255 (IAP/backend) right after boot or when CorpSSH/gcert is stale
  -> retry after ~60-120s, and if it persists ask the user to run `gcert`. `reset` the instance as a last resort.
- Long inline SSH heredocs sometimes 255; prefer scp-ing a script then running it. Use setsid + nohup for long jobs,
  write progress to a log file, and poll the log (VMs' long runs outlive a single ssh call).
- The strict audit in round 1 CAUGHT a contaminated data cell (G4 seq1024 conc-1 had wall/qps inconsistent with
  latency due to a mid-run stall) - re-run and replace such cells; keep audits strict (qps ~ iters/wall and ~ C/mean).
- Cost: these are Spot GPUs and they bill while up. If pausing between sessions, offer to stop/delete and re-provision.
- Repo authorship uses `git -c user.email=... -c user.name="MG-Cafe"`; push with `gh`/`git push` (gh is authenticated).

--------------------------------------------------------------------------------
## 8. Quick status checklist for round 2
- [ ] All 4 chips up + nvidia-smi verified (B200 a4, H200 a3, G4, TPU)
- [ ] TRT-LLM available in NGC pytorch container on each GPU; versions captured
- [ ] BF16 TRT engine for Qwen3-Reranker-0.6B; scoring path validated (sane relevance scores)
- [ ] TRT bench: bs32 seq512 no-cache + batch sweep; results/trt_gpu_{b200,h200,g4}.json
- [ ] Recompute price/perf; vLLM-vs-TRT charts+tables; README; audit PASSES
- [ ] NEW GitHub repo pushed (fresh history, credential-free)
- [ ] NEW DOCX report in ~/Downloads
- [ ] Tear down or hand back VM state; report to user

================================================================================
## KICKOFF PROMPT for the next session (paste this, then attach this HANDOFF_FULL.md)
================================================================================
You are resuming a benchmarking project on GCP project diesel-patrol-382622. Read the attached
HANDOFF_FULL.md fully before acting. Summary of the job: I already published a vLLM latency + price/perf
benchmark of Qwen3-Reranker-0.6B across TPU v6e, B200, H200, and G4 (repo:
github.com/MG-Cafe/reranker-latency-priceperf, results-only, audited, credential-free). Now I need a SECOND
study re-benchmarking the NVIDIA GPUs (B200, H200, G4) on **TensorRT-LLM** (BF16) using the identical
workload (Qwen3-Reranker-0.6B, bs32, seq512, prefix caching OFF, fresh random content per request,
p50/p90/p99 + throughput, batch sweep 1..64), and compare against TPU v6e on vLLM torchax (BF16; TRT-LLM
cannot run on TPU). Keep the vLLM baselines and recompute both price/perf views (USD/1M pairs and
USD/1000 requests) with prices v6e $1.61, B200 $6.95, H200 $3.85, G4 $3.11.

Do this:
1) Verify/recreate the VMs (real B200 = a4-highgpu-8g; H200 = a3-ultragpu-8g; G4 = g4-standard-48 + Ubuntu
   2404/595; TPU torchtpu-1). Always nvidia-smi to confirm the actual GPU (a3 wrongly reports as H200 -
   that's expected; a4 is the real B200). Helper scripts are in ~/.
2) Install TensorRT-LLM inside the NGC PyTorch container (nvcr.io/nvidia/pytorch:25.01-py3, has the CUDA
   toolchain) - do NOT pip-install on the bare VM (fails: no 'cc'); the tensorrt-llm/release:latest tag does
   not exist. Capture exact TRT-LLM/TensorRT/CUDA/driver versions.
3) Build a BF16 TRT-LLM engine for Qwen3-Reranker-0.6B (seq-cls scoring head: classifier_from_token=["no","yes"],
   is_original_qwen3_reranker=true), wire the scoring/rerank path, validate sane relevance scores, then benchmark
   the exact workload above. Save results/trt_gpu_{b200,h200,g4}.json in the same schema as the existing
   lat_gpu_*.json.
4) Recompute price/perf on the TRT numbers; generate vLLM-vs-TRT comparison charts + tables; embed all charts in
   a README; add a STRICT audit script that recomputes every number from JSON and confirms README matches
   (must PASS). No fabrication; label engine (GPUs=TRT-LLM, TPU=vLLM) and precision (BF16) clearly; neutral
   wording, no customer/company references, no internal project IDs.
5) Create a NEW GitHub repo under MG-Cafe (e.g. reranker-trt-vs-vllm) with fresh single-commit history, and a
   NEW results-only DOCX report copied to ~/Downloads (for Google Docs). Do NOT modify the existing round-1 repo
   or report.
Be honest about any gap vs the requester's reported ~23.7ms B200 (likely their FP8 vs our BF16); do not change
the workload or enable caching to chase a number. Work in background with logs for long steps, handle SSH 255 by
retrying and asking me to run gcert if needed, and tell me before deleting anything. The VMs are Spot and billing
while up.
