#!/usr/bin/env python3
"""LATENCY-first reranker benchmark (real-time, no caching) - device-agnostic (TPU torchax + GPU).

Benchmark rules baked in:
  * Real-time serving, not offline batch: we time each score() call as one in-flight request.
  * NO caching to avoid inflating results: prefix caching disabled (enable_prefix_caching=False),
    and every request uses FRESH randomized content so KV/prefix reuse cannot help.
  * Standardized synthetic dataset: fixed sequence length ~512 tokens per (query+document) pair.
  * Primary metric: per-request LATENCY (p50/p90/p95/p99) at batch size 32, seq 512.
  * Also sweep batch sizes for a latency-vs-batch curve; report throughput for context.

Batch here = number of pairs submitted together in one real-time score() call (concurrent request
group), NOT an offline job. Each measured "request" is one such call; we report call latency and
per-pair latency.

Usage (from a NON-source dir), env set up:
  python reranker_latency_bench.py --device-label "tpu v6e-1 (torchax)" --is-tpu \
      --seq-len 512 --batch-sizes 1,2,4,8,16,32,64 --iters 50 --result-filename ~/lat.json
"""
import argparse, json, os, random, statistics, time


def build_pair(tokenizer, seq_len, rng):
    """Make one (query, document) whose combined length ~= seq_len tokens, with random tokens
    so no two requests share a prefix (defeats prefix/KV caching)."""
    vocab = tokenizer.vocab_size
    # random token ids -> text; query gets ~1/4, doc ~3/4 of the budget
    q_ids = [rng.randrange(1000, min(vocab, 30000)) for _ in range(max(4, seq_len // 4))]
    d_ids = [rng.randrange(1000, min(vocab, 30000)) for _ in range(seq_len - len(q_ids))]
    q = tokenizer.decode(q_ids, skip_special_tokens=True)
    d = tokenizer.decode(d_ids, skip_special_tokens=True)
    return q, d


def pctl(s, p):
    if not s: return None
    k = min(len(s)-1, int(round(p/100.0*(len(s)-1))))
    return round(s[k], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-label", required=True)
    ap.add_argument("--is-tpu", action="store_true")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    ap.add_argument("--iters", type=int, default=50, help="timed requests per batch size")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--result-filename", required=True)
    args = ap.parse_args()

    from vllm import LLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")

    # NO CACHING: disable prefix caching. max_model_len covers seq_len with headroom.
    mml = max(1024, args.seq_len + 64)
    extra = {"enable_prefix_caching": False, "max_model_len": mml,
             "max_num_batched_tokens": max(8192, mml * 8)}
    if not args.is_tpu:
        extra["gpu_memory_utilization"] = args.gpu_mem_util

    print(f"[lat] loading Qwen3-Reranker-0.6B (hf_overrides, no prefix cache) extra={extra}", flush=True)
    t0 = time.perf_counter()
    llm = LLM(model="Qwen/Qwen3-Reranker-0.6B", runner="pooling",
              hf_overrides={"architectures": ["Qwen3ForSequenceClassification"],
                            "classifier_from_token": ["no", "yes"],
                            "is_original_qwen3_reranker": True},
              **extra)
    load_s = time.perf_counter() - t0
    print(f"[lat] model load: {load_s:.2f}s", flush=True)

    rng = random.Random(1234)
    sizes = [int(x) for x in args.batch_sizes.split(",")]

    result = {"model": "Qwen/Qwen3-Reranker-0.6B", "load_mode": "orig_override",
              "device": args.device_label, "runner": "pooling (vLLM score API)",
              "prefix_caching": False, "seq_len_target": args.seq_len, "max_model_len": mml,
              "iters_per_bs": args.iters, "model_load_s": round(load_s, 3),
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "by_batch": []}

    # measure actual pair token length once
    q, d = build_pair(tok, args.seq_len, rng)
    pair_tokens = len(tok(q).input_ids) + len(tok(d).input_ids)
    result["pair_tokens_actual"] = pair_tokens

    sample = None
    for bs in sizes:
        # warmup with fresh random content
        for _ in range(args.warmup):
            qs = []; ds = []
            for _ in range(bs):
                a, b = build_pair(tok, args.seq_len, rng); qs.append(a); ds.append(b)
            try:
                llm.score(qs, ds)
            except Exception as e:
                print(f"[lat] bs={bs} warmup failed: {e}", flush=True)
                break
        lat = []
        ok = True
        for _ in range(args.iters):
            qs = []; ds = []
            for _ in range(bs):
                a, b = build_pair(tok, args.seq_len, rng); qs.append(a); ds.append(b)
            t = time.perf_counter()
            try:
                outs = llm.score(qs, ds)
            except Exception as e:
                print(f"[lat] bs={bs} error: {e}", flush=True); ok = False; break
            lat.append((time.perf_counter() - t) * 1000.0)  # ms per request (the whole bs call)
            if sample is None and outs:
                sample = [round(float(o.outputs.score), 4) for o in outs[:4]]
        if not ok or not lat:
            continue
        lat.sort()
        mean_ms = statistics.mean(lat)
        row = {"batch_size": bs,
               "request_latency_ms_p50": pctl(lat, 50),
               "request_latency_ms_p90": pctl(lat, 90),
               "request_latency_ms_p95": pctl(lat, 95),
               "request_latency_ms_p99": pctl(lat, 99),
               "request_latency_ms_mean": round(mean_ms, 3),
               "per_pair_latency_ms_mean": round(mean_ms / bs, 4),
               "throughput_pairs_per_s": round(bs / (mean_ms / 1000.0), 3)}
        result["by_batch"].append(row)
        print(f"[lat] bs={bs}: req p50 {row['request_latency_ms_p50']}ms p99 {row['request_latency_ms_p99']}ms "
              f"| per-pair {row['per_pair_latency_ms_mean']}ms | {row['throughput_pairs_per_s']} pairs/s", flush=True)

    result["sample_scores"] = sample
    with open(args.result_filename, "w") as f:
        json.dump(result, f, indent=2)
    print("[lat] wrote", args.result_filename, flush=True)


if __name__ == "__main__":
    main()
