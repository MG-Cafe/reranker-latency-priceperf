#!/usr/bin/env python3
"""HTTP client/server CONCURRENCY sweep for the Qwen3-Reranker on vLLM (real-time serving).

Starts nothing itself; it drives an already-running `vllm serve` reranker via the OpenAI-style
/v1/score (or /score) endpoint. Each request is one (query, document) pair (bs=1 request) OR a
batch of N pairs (--req-batch), and we fire C concurrent requests continuously for a fixed number
of iterations, measuring per-request latency (p50/p90/p99) and achieved QPS at each concurrency.

No caching: prefix caching is disabled at server launch, AND every request uses freshly randomized
content so KV/prefix reuse cannot help. Sequence length fixed via --seq-len (512 or 1024).

Usage (client side, after `vllm serve` is up on :8000):
  python reranker_conc_bench.py --device-label "gpu 1x B200" --base-url http://localhost:8000 \
      --seq-len 512 --req-batch 1 --concurrencies 1,2,4,8 --iters 300 \
      --result-filename ~/conc_gpu_b200_seq512.json
"""
import argparse, asyncio, json, os, random, statistics, time
import urllib.request

def build_doc(approx_tokens, tokenizer, rng):
    vocab = min(tokenizer.vocab_size, 30000)
    ids = [rng.randrange(1000, vocab) for _ in range(approx_tokens)]
    return tokenizer.decode(ids, skip_special_tokens=True)

def pctl(s, p):
    if not s: return None
    k = min(len(s)-1, int(round(p/100.0*(len(s)-1))))
    return round(s[k], 3)

async def one_request(session_url, model, q, docs):
    import aiohttp  # local import so the file imports even without aiohttp for --help
    payload = {"model": model, "query": q, "documents": docs}
    t = time.perf_counter()
    async with _SESSION.post(session_url, json=payload) as r:
        body = await r.read()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {body[:200]!r}")
    return (time.perf_counter() - t) * 1000.0


_SESSION = None

async def run_conc(url, model, tok, seq_len, req_batch, conc, iters, rng):
    import aiohttp
    global _SESSION
    q_tokens = max(4, seq_len // 4); d_tokens = seq_len - q_tokens
    lat = []
    sem = asyncio.Semaphore(conc)
    async def worker():
        # fresh random content each call (defeats caching)
        q = build_doc(q_tokens, tok, rng)
        docs = [build_doc(d_tokens, tok, rng) for _ in range(req_batch)]
        async with sem:
            ms = await one_request(url, model, q, docs)
            lat.append(ms)
    # warmup
    await asyncio.gather(*[worker() for _ in range(conc*2)])
    lat.clear()
    t0 = time.perf_counter()
    # keep `conc` in flight for `iters` total requests
    tasks = set()
    done = 0
    async def spawn():
        nonlocal done
        while done < iters:
            if len(tasks) < conc:
                tsk = asyncio.create_task(worker()); tasks.add(tsk); done += 1
            await asyncio.sleep(0)
            for t in list(tasks):
                if t.done(): tasks.discard(t)
    await spawn()
    if tasks: await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    lat.sort()
    return {"concurrency": conc, "iters": iters, "wall_s": round(wall,3),
            "qps": round(len(lat)/wall,3),
            "req_latency_ms_p50": pctl(lat,50), "req_latency_ms_p90": pctl(lat,90),
            "req_latency_ms_p99": pctl(lat,99), "req_latency_ms_mean": round(statistics.mean(lat),3)}

async def main_async(args):
    import aiohttp
    from transformers import AutoTokenizer
    global _SESSION
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
    rng = random.Random(1234)
    url = args.base_url.rstrip("/") + "/v1/rerank"   # reranker endpoint: {model, query, documents[]}
    model = args.model
    # validate one real 200 response with a relevance score before timing anything
    import aiohttp
    async with aiohttp.ClientSession() as s:
        vq = build_doc(max(4, args.seq_len//4), tok, rng)
        vd = [build_doc(args.seq_len - max(4, args.seq_len//4), tok, rng) for _ in range(args.req_batch)]
        async with s.post(url, json={"model": model, "query": vq, "documents": vd}) as r:
            vb = await r.read()
            if r.status != 200 or b"relevance_score" not in vb:
                raise SystemExit(f"validation failed: HTTP {r.status}: {vb[:200]!r}")
    print("[conc] endpoint validated (200 with relevance_score):", url, flush=True)

    result = {"model": model, "device": args.device_label, "endpoint": url,
              "prefix_caching": False, "seq_len_target": args.seq_len, "req_batch": args.req_batch,
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "by_concurrency": []}
    timeout = aiohttp.ClientTimeout(total=600)

    _SESSION = aiohttp.ClientSession(timeout=timeout)
    try:
        for c in [int(x) for x in args.concurrencies.split(",")]:
            row = await run_conc(url, model, tok, args.seq_len, args.req_batch, c, args.iters, rng)
            result["by_concurrency"].append(row)
            print(f"[conc] c={c}: p50 {row['req_latency_ms_p50']}ms p90 {row['req_latency_ms_p90']}ms "
                  f"p99 {row['req_latency_ms_p99']}ms | {row['qps']} req/s", flush=True)
    finally:
        await _SESSION.close()
    with open(args.result_filename, "w") as f:
        json.dump(result, f, indent=2)
    print("[conc] wrote", args.result_filename, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-label", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="tomaarsen/Qwen3-Reranker-0.6B-seq-cls")

    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--req-batch", type=int, default=1, help="pairs per request (documents list length)")
    ap.add_argument("--concurrencies", default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--result-filename", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
