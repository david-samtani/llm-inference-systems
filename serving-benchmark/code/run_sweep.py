#!/usr/bin/env python3
"""
sweep driver - tests every combination of - prompt lengths x concurrency x reps for one system per run

  python3 code/run_sweep.py --system minimal_b1
  python3 code/run_sweep.py --system vllm_paged --url http://localhost:8000
  python3 code/run_sweep.py --system mooncake   --url http://localhost:8000
  python3 code/run_sweep.py --system vllm_paged --url http://localhost:8000 --wildchat

"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, median

import requests
from transformers import AutoConfig, AutoTokenizer

from export_results import REPO_ROOT, kv_bytes, load_json, tbt_ms_per_step, ttft_ms, write_json

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
RESULTS = REPO_ROOT / "report/src/data/results.json"

# TinyLlama's context is 2048 tokens total - so the longest prompt leaves room for 64 new tokens
PROMPT_TARGETS = [128, 512, 1984]
CONCURRENCY = [1, 4]
REPS = 3
WARMUP = 1
MAX_NEW_TOKENS = 64
WILDCHAT_SHARD = "data/train-00000-of-00014.parquet"

FILLER = (
    "The history of computing is a story of moving data closer to where it is used. "
    "Early machines stored programs on punched cards, later on magnetic drums, and "
    "eventually in fast semiconductor memory. Each step reduced the time a processor "
    "spent waiting. Modern accelerators repeat the pattern: arithmetic is cheap, but "
    "moving bytes between memory and compute units dominates cost. Summarize the key "
    "ideas in this passage and explain why memory bandwidth matters for inference. "
)

def templated_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, return_dict=True
    )["input_ids"]


class PromptBuilder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.filler_ids = tokenizer(FILLER * 60, add_special_tokens=False)["input_ids"]

    def make(self, target: int, tag: str) -> tuple[str, list[int]]:
        n = target
        for _ in range(20):
            text = f"[{tag}] " + self.tokenizer.decode(self.filler_ids[:n])
            ids = templated_ids(self.tokenizer, text)
            if len(ids) == target:
                return text, ids
            n += target - len(ids)
        raise ValueError(f"could not hit {target} prompt tokens (got {len(ids)})")


def wildchat_prompts(tokenizer, count: int, max_tokens: int) -> list[tuple[str, list[int]]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("allenai/WildChat-1M", WILDCHAT_SHARD, repo_type="dataset")
    out = []
    for batch in pq.ParquetFile(path).iter_batches(columns=["language", "toxic", "conversation"]):
        for row in batch.to_pylist():
            if row["language"] != "English" or row["toxic"]:
                continue
            text = row["conversation"][0]["content"]
            ids = templated_ids(tokenizer, text)
            if 16 <= len(ids) <= max_tokens:
                out.append((text, ids))
                if len(out) == count:
                    return out
    raise ValueError(f"only found {len(out)} WildChat prompts")


def synthetic_cells(builder: PromptBuilder, args, salt: int):
    for target in args.prompts:
        for n in args.concurrency:
            for rep in range(-WARMUP, args.reps):
                yield target, n, rep, [
                    builder.make(target, f"{salt}-{target}-{n}-{rep}-{c}") for c in range(n)
                ]


def wildchat_cells(tokenizer, args):
    need = sum(n * (WARMUP + args.reps) for n in args.concurrency)
    pool = iter(wildchat_prompts(tokenizer, need, max(PROMPT_TARGETS)))
    for n in args.concurrency:
        for rep in range(-WARMUP, args.reps):
            prompts = [next(pool) for _ in range(n)]
            yield int(median(len(ids) for _, ids in prompts)), n, rep, prompts


# systems -----------------------------------------------------------------

class Baseline:
    """minimal_full_server in-process. One request at a time, so a burst of N queues."""

    def __init__(self, device: str | None):
        import minimal_full_server as mfs

        self.mfs = mfs
        self.device = device or mfs.default_device()
        self.tokenizer, self.model = mfs.load_model(MODEL, self.device)

    def burst(self, prompts: list[tuple[str, list[int]]], max_new: int) -> list[dict]:
        self.mfs.sync(self.device)
        t_burst = time.perf_counter()
        clients = []
        for text, _ in prompts:
            queued_ms = (time.perf_counter() - t_burst) * 1000.0
            r = self.mfs.generate(self.tokenizer, self.model, text, max_new, self.device)
            clients.append({**r, "ttft_ms": queued_ms + r["ttft_ms"]})
        return clients


class OpenAIServer:
    def __init__(self, url: str):
        self.url = url.rstrip("/") + "/v1/completions"
        cfg = AutoConfig.from_pretrained(MODEL)
        self.cfg = cfg
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    def one(self, ids: list[int], max_new: int, barrier: threading.Barrier) -> dict:
        barrier.wait()
        t_start = time.perf_counter()
        resp = requests.post(
            self.url,
            json={
                "model": MODEL,
                "prompt": ids,
                "max_tokens": max_new,
                "ignore_eos": True,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            stream=True,
            timeout=600,
        )
        resp.raise_for_status()
        arrivals: list[float] = []
        new_tokens = None
        for line in resp.iter_lines():
            if not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: "):]
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("choices"):
                arrivals.append(time.perf_counter())
            if chunk.get("usage"):
                new_tokens = chunk["usage"]["completion_tokens"]
        new_tokens = new_tokens or len(arrivals)
        steps = [b - a for a, b in zip(arrivals, arrivals[1:])]
        seq_len = len(ids) + new_tokens - 1
        return {
            "prompt_tokens": len(ids),
            "ttft_ms": ttft_ms(t_start, arrivals[0]),
            **tbt_ms_per_step(steps),
            "kv_bytes": kv_bytes(
                self.cfg.num_hidden_layers, self.cfg.num_key_value_heads, seq_len, self.head_dim
            ),
            "kv_seq_len": seq_len,
            "new_tokens": new_tokens,
        }

    def burst(self, prompts: list[tuple[str, list[int]]], max_new: int) -> list[dict]:
        barrier = threading.Barrier(len(prompts))
        with ThreadPoolExecutor(len(prompts)) as pool:
            futures = [pool.submit(self.one, ids, max_new, barrier) for _, ids in prompts]
            return [f.result() for f in futures]


# rows --------------------------------------------------------------------

def to_row(
    system_id: str, target: int, n: int, rep: int, clients: list[dict], warm: bool, source: str
) -> dict:
    ttft = [c["ttft_ms"] for c in clients]
    tbt = [c["tbt_ms_p50"] for c in clients]
    kvb = [int(c["kv_bytes"]) for c in clients]
    row = {
        "system_id": system_id,
        "prompt_tokens": target,
        "concurrency": n,
        "rep": rep,
        "ttft_ms": round(median(ttft), 3),
        "tbt_ms_p50": round(median(tbt), 3),
        "tbt_ms_mean": round(mean(c["tbt_ms_mean"] for c in clients), 3),
        "kv_bytes": int(median(kvb)),
        "kv_seq_len": int(median(c["kv_seq_len"] for c in clients)),
        "new_tokens": min(c["new_tokens"] for c in clients),
        "prompt_source": source,
    }
    if n > 1:
        row["ttft_ms_samples"] = [round(v, 3) for v in ttft]
        row["tbt_ms_samples"] = [round(v, 3) for v in tbt]
        row["kv_bytes_samples"] = kvb
    if warm:
        row["prefix_reuse"] = "warm"
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark sweep for one system")
    p.add_argument("--system", required=True, help="system_id, e.g. minimal_b1 | vllm_paged | mooncake")
    p.add_argument("--url", help="OpenAI-compatible server URL; omit to run the in-process baseline")
    p.add_argument("--device", help="baseline device (default: cuda > mps > cpu)")
    p.add_argument("--warm", action="store_true", help="send each burst once untimed first (prefix reuse)")
    p.add_argument("--prompts", type=int, nargs="+", default=PROMPT_TARGETS)
    p.add_argument("--concurrency", type=int, nargs="+", default=CONCURRENCY)
    p.add_argument("--reps", type=int, default=REPS)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--wildchat", action="store_true", help="real WildChat-1M prompts instead of the synthetic grid")
    p.add_argument("--dry-run", action="store_true", help="print rows, do not write JSON")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    system = OpenAIServer(args.url) if args.url else Baseline(args.device)

    doc = load_json(RESULTS)
    reuse = "warm" if args.warm else None
    source = "wildchat" if args.wildchat else "synthetic"
    key = (args.system, reuse, source)
    kept = [
        r for r in doc["runs"]
        if (r["system_id"], r.get("prefix_reuse"), r.get("prompt_source", "synthetic")) != key
    ]
    salt = int(time.time()) % 100000
    cells = (
        wildchat_cells(tokenizer, args) if args.wildchat
        else synthetic_cells(PromptBuilder(tokenizer), args, salt)
    )
    rows: list[dict] = []

    for target, n, rep, prompts in cells:
        if args.warm:
            system.burst(prompts, args.max_new_tokens)
        clients = system.burst(prompts, args.max_new_tokens)
        if rep < 0:
            continue
        row = to_row(args.system, target, n, rep, clients, args.warm, source)
        rows.append(row)
        print(
            f"{args.system} {source} p={target} N={n} rep={rep}  "
            f"TTFT={row['ttft_ms']:.1f}ms  TBT p50={row['tbt_ms_p50']:.2f}ms  "
            f"KV={row['kv_bytes']:,}B"
        )
        if not args.dry_run:
            doc["runs"] = kept + rows
            write_json(RESULTS, doc)

    if not args.dry_run and not args.wildchat:
        doc["source"]["workload"].update(
            prompt_token_targets=args.prompts,
            concurrency_grid=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            reps_per_cell=args.reps,
            warmup_runs=WARMUP,
        )
        write_json(RESULTS, doc)
        print(f"Wrote {len(rows)} rows for {args.system} to {RESULTS}")


if __name__ == "__main__":
    main()
