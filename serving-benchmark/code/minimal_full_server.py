#!/usr/bin/env python3
"""
Minimal single-request inference server: the baseline the benchmark compares against.

  1) load model
  2) receive prompt
  3) prefill + KV cache
  4) autoregressive decode
  5) exit

Split into load_model(), which runs once, and generate(), which can be called
many times and returns TTFT, per-step TBT, and KV bytes.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from export_results import kv_bytes, kv_seq_len, tbt_ms_per_step, ttft_ms


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# --- 1) Load model -once ---
def load_model(model_id: str, device: str):
    dtype = torch.float16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def model_kv_bytes(model, seq_len: int) -> int:
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return kv_bytes(
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        itemsize=model.dtype.itemsize,
    )


def generate(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    device: str,
    stop_at_eos: bool = False,
    verbose: bool = False,
) -> dict:
    sync(device)
    t_request = time.perf_counter()

    # --- 2) Tokenize the prompt ---
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"].to(device)
    prompt_tokens = input_ids.shape[1]

    # --- 3) Prefill + KV cache ---
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    # past_key_values is a handle to the on-device cache
    kv_cache = outputs.past_key_values
    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(device)
    t_first = time.perf_counter()
    generated = [next_token_id.item()]
    if verbose:
        print(f"    {prompt_tokens} prompt tokens, KV seq_len={kv_seq_len(kv_cache)}")
        print(f"    first token={tokenizer.decode(generated)!r}")

    # --- 4) Autoregressive decode ---
    step_s: list[float] = []
    for step in range(max_new_tokens - 1):
        if stop_at_eos and generated[-1] == tokenizer.eos_token_id:
            break
        t_step = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=next_token_id,
                past_key_values=kv_cache,
                use_cache=True,
                return_dict=True,
            )
        kv_cache = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sync(device)
        step_s.append(time.perf_counter() - t_step)
        generated.append(next_token_id.item())
        if verbose:
            piece = tokenizer.decode(generated[-1:], skip_special_tokens=False)
            print(f"    step {step + 1:2d}: piece={piece!r}  kv_len={kv_seq_len(kv_cache)}")

    seq_len = kv_seq_len(kv_cache)
    return {
        "prompt_tokens": prompt_tokens,
        "ttft_ms": ttft_ms(t_request, t_first),
        **tbt_ms_per_step(step_s),
        "decode_step_ms": [s * 1000.0 for s in step_s],
        "kv_bytes": model_kv_bytes(model, seq_len),
        "kv_seq_len": seq_len,
        "new_tokens": len(generated),
        "text": tokenizer.decode(generated, skip_special_tokens=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal full-server LLM inference")
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model id (same default as dc_server.py)",
    )
    parser.add_argument("--device", default=default_device(), help="cuda | cpu | mps")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Input prompt (if omitted, read from stdin)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="New tokens to generate (first comes from prefill)",
    )
    args = parser.parse_args()

    prompt = args.prompt
    if prompt is None:
        prompt = input("Prompt: ").strip()
    if not prompt:
        print("Empty prompt.", file=sys.stderr)
        sys.exit(1)

    print("\n[1] Load model")
    t0 = time.perf_counter()
    tokenizer, model = load_model(args.model, args.device)
    print(f"    model={args.model}  device={args.device}  dtype={model.dtype}")
    print(f"    load time={(time.perf_counter() - t0) * 1000:.0f} ms")

    print("\n[2-4] Prompt -> prefill -> decode")
    result = generate(
        tokenizer, model, prompt, args.max_tokens, args.device,
        stop_at_eos=True, verbose=True,
    )

    print("\n[5] Exit")
    print(f"    TTFT={result['ttft_ms']:.1f} ms")
    print(f"    TBT p50={result['tbt_ms_p50']:.2f} ms  mean={result['tbt_ms_mean']:.2f} ms")
    print(f"    KV bytes={result['kv_bytes']:,}  (seq_len={result['kv_seq_len']})")
    print(f"    output: {result['text']!r}")


if __name__ == "__main__":
    main()