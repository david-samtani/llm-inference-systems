# LLM Inference Systems

Two hands-on projects on the memory and scheduling side of LLM inference, both
built around TinyLlama-1.1B-Chat.

## [KV Cache Compression](kv-cache-compression/)

Three KV cache compression methods implemented inside a layer-by-layer
TinyLlama inference loop: sink-aware sliding-window eviction,
cumulative-attention importance eviction, and symmetric int8 quantization.
Each is measured against an uncompressed baseline on persistent KV bytes,
decode latency, and output divergence.

- Int8 quantization halved KV state with near-identical output (token edit
  distance ≤ 0.016) at a 14–19% decode-latency cost from dequantization.
- A budget-8 sliding window cut state by 91.5% with no latency cost, but the
  generated text diverged almost immediately.

![KV cache footprint by method](kv-cache-compression/plots/cache_memory.png)

## [LLM Serving Benchmark: Minimal Server vs. vLLM vs. Mooncake](serving-benchmark/)

A controlled benchmark of a single-request Hugging Face server, vLLM
(PagedAttention + continuous batching), and vLLM with Mooncake's external KV
store (cold vs. warm prefix) on an NVIDIA A10, sweeping prompt length and
concurrency and reporting TTFT / TBT distributions per client.

- At 4 concurrent clients, continuous batching cuts median TTFT from ~1.5 s
  (queueing) to 26–191 ms, and vLLM decodes ~2.9× faster per user.
- Mooncake prefix reuse nearly halves TTFT on 1984-token prompts, but cold runs
  pay a store-write penalty and short chat prompts see no benefit.

Results are presented in an interactive React report in `serving-benchmark/report/`.

---

Each project README lists which parts started from provided baseline code and
which parts are my own work.
