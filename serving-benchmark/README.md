# LLM Serving Benchmark: Minimal Server vs. vLLM vs. Mooncake

A controlled comparison of three ways to serve the same model
(TinyLlama-1.1B-Chat) on one NVIDIA A10 (Lambda Cloud), from a single-request
Hugging Face loop to production engines:

| System | What changes |
|---|---|
| `minimal_b1`: minimal single-request server | Contiguous per-request `DynamicCache`, no batching or scheduler. Concurrent requests queue. |
| `vllm_paged`: vLLM | PagedAttention (fixed-size KV blocks) plus continuous, iteration-level batching. |
| `mooncake` / `mooncake_warm`: vLLM + Mooncake store | KV blocks saved to an external, hash-addressed KV store and fetched on prefix hits instead of recomputed (cold vs. warm prefix). |

The workload is 128 / 512 / 1984-token prompts × 1 and 4 concurrent clients × 3
repetitions with 64 new tokens each, plus a 20-prompt sanity check on real
[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) first turns.
Every run records TTFT, per-step TBT (p50 and mean), and KV bytes, plus
per-client samples at concurrency > 1 so tail latency is visible, not just
medians.

## Key results

- **Batching is the big win under load.** At 4 concurrent clients the baseline
  queues requests: median TTFT is ~1.5 s and the last client waits 3.0–3.4 s.
  vLLM serves the same burst in 26 / 72 / 191 ms (128 / 512 / 1984-token prompts).
- **vLLM decodes ~2.9× faster per user** (TBT 5.5–6.0 ms vs. ~15.8 ms), and TBT
  rises only 0.1–0.4 ms from 1 to 4 clients, because decode is
  memory-bandwidth bound and one read of the weights serves every sequence in
  the batch.
- **Mooncake prefix reuse pays off on long prompts.** At 1984 tokens warm TTFT
  is 43 ms vs. 77 ms cold (1 client) and 142 ms vs. 377 ms cold (4 clients),
  beating plain vLLM (73 / 191 ms) with the lowest tail (p95 182 vs. 299 ms).
- **Reuse is not free.** Cold Mooncake is slower than plain vLLM under load
  (377 vs. 191 ms) because every KV block is also written to the store, and on
  short real chat prompts warm is no better than cold.

The full write-up (interactive line chart, per-client TTFT/TBT boxplots,
measurement setup, discussion, and limitations) is the React report in
`report/`.

## Layout

```text
code/
  run_sweep.py            # sweep driver: prompts x concurrency x reps for one system per run
  minimal_full_server.py  # baseline server, called in-process
  export_results.py       # metric helpers (TTFT, TBT, KV bytes) + results JSON validator
  lambda/                 # scripts used to launch vLLM / Mooncake and run each sweep, plus server logs
report/
  src/data/results.json   # every measured run (96 rows) + systems, takeaways, plot config
  src/App.jsx             # report page and discussion
```

## Reproducing

Measurements need Python 3.10+ and a CUDA GPU. The scripts in `code/lambda/`
are what I ran on a fresh Lambda Cloud A10 instance (clone this repo to
`~/llm-inference-systems` first):

```bash
bash code/lambda/run_all.sh        # baseline + vLLM synthetic sweep
bash code/lambda/run_mooncake.sh   # Mooncake cold + warm (expects ~/mooncake_config.json, see code/lambda/)
bash code/lambda/run_wildchat.sh   # WildChat sanity check for all systems
```

Each `run_sweep.py` invocation appends rows to `report/src/data/results.json`.
Validate the file with:

```bash
python3 code/export_results.py
```

View the report locally (Node 20.19+ or 22.12+):

```bash
npm run setup && npm run dev
```

The `Deploy serving-benchmark report to GitHub Pages` workflow builds and
publishes the report; it is manual (`workflow_dispatch`) so it only runs once
Pages is enabled for the repo.

Versions: vLLM 0.32.0, torch 2.13.0, transformers 5.19.0,
mooncake-transfer-engine 0.3.13.post1.

## Starting point

The baseline server (`minimal_full_server.py`, which I refactored into a
reusable `load_model()` / `generate()` API with TTFT/TBT instrumentation), the
metric helpers and validator in `export_results.py`, and the report scaffold
(`MetricsPlot`, `DistributionBoxplot`, styling, and the Pages workflow) were
provided as starter code. My contributions are the sweep driver
(`run_sweep.py`), the vLLM and Mooncake deployment scripts, all measurements in
`results.json`, and the analysis and discussion in the report.
