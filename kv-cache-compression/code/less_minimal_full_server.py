#!/usr/bin/env python3
"""
Five-step TinyLlama inference with pluggable KV cache compression.

--verbose unwraps model() into RMSNorm, Q/K/V, RoPE, GQA, attention, SwiGLU.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def kv_seq_len(kv_cache) -> int:
    if hasattr(kv_cache, "get_seq_length"):
        return kv_cache.get_seq_length()
    return kv_cache[0][0].shape[2]


def kv_bytes(n_layers, n_kv_heads, seq_len, head_dim, batch, itemsize) -> int:
    return 2 * itemsize * n_layers * n_kv_heads * seq_len * head_dim * batch


def decode_answer(tokenizer, token_ids) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    for tag in ("<|system|>", "<|user|>", "<|assistant|>"):
        text = text.replace(tag, "")
    return text.strip()


def apply_sink_window(k, v, budget, sink_tokens):
    """Keep fixed initial sink tokens plus the newest tokens."""
    if k.shape[2] <= budget:
        return k, v

    recent_tokens = budget - sink_tokens

    if sink_tokens == 0:
        return (
            k[:, :, -recent_tokens:, :].contiguous(),
            v[:, :, -recent_tokens:, :].contiguous(),
        )

    k_kept = torch.cat(
        (k[:, :, :sink_tokens, :], k[:, :, -recent_tokens:, :]),
        dim=2,
    )
    v_kept = torch.cat(
        (v[:, :, :sink_tokens, :], v[:, :, -recent_tokens:, :]),
        dim=2,
    )
    return k_kept, v_kept


def apply_importance_eviction(k, v, importance, budget):
    """Keep the highest-scoring tokens, always protecting the newest token."""
    seq_len = k.shape[2]
    if seq_len <= budget:
        return k, v, importance

    if budget == 1:
        keep = torch.tensor([seq_len - 1], device=k.device)
    else:
        # Reserve one slot for the newest token, which has not had time to
        # accumulate as much attention as older cached positions.
        heavy_hitters = torch.topk(
            importance[:-1],
            k=budget - 1,
            largest=True,
            sorted=False,
        ).indices
        heavy_hitters = heavy_hitters.sort().values
        newest = torch.tensor([seq_len - 1], device=k.device)
        keep = torch.cat((heavy_hitters, newest))

    return (
        k.index_select(2, keep).contiguous(),
        v.index_select(2, keep).contiguous(),
        importance.index_select(0, keep).contiguous(),
    )


def quantize_int8(tensor, scale=None):
    """Symmetrically quantize a tensor to int8 with one float32 scale."""
    if scale is None:
        max_abs = tensor.abs().amax().float()
        scale = torch.clamp(max_abs / 127.0, min=1e-8)
    quantized = torch.round(tensor.float() / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize_int8(tensor, scale, dtype):
    """Reconstruct a floating-point tensor for attention."""
    return tensor.to(dtype) * scale.to(dtype)


class QuantizedCacheLayer:
    def __init__(self):
        self.keys = None
        self.values = None
        self.key_scale = None
        self.value_scale = None


class QuantizedKVCache:
    """Minimal DynamicCache-like cache that stores K/V as symmetric int8."""

    def __init__(self):
        self.layers = []

    def _get_layer(self, layer_idx):
        while len(self.layers) <= layer_idx:
            self.layers.append(QuantizedCacheLayer())
        return self.layers[layer_idx]

    def update(self, key_states, value_states, layer_idx):
        layer = self._get_layer(layer_idx)

        if layer.keys is None:
            layer.keys, layer.key_scale = quantize_int8(key_states)
            layer.values, layer.value_scale = quantize_int8(value_states)
        else:
            # Reuse the prefill scales so old cache entries never need to be
            # requantized; future outliers are clipped to the int8 range.
            new_keys, _ = quantize_int8(key_states, layer.key_scale)
            new_values, _ = quantize_int8(value_states, layer.value_scale)
            layer.keys = torch.cat((layer.keys, new_keys), dim=2)
            layer.values = torch.cat((layer.values, new_values), dim=2)

        return (
            dequantize_int8(layer.keys, layer.key_scale, key_states.dtype),
            dequantize_int8(layer.values, layer.value_scale, value_states.dtype),
        )

    def get_seq_length(self):
        if not self.layers or self.layers[0].keys is None:
            return 0
        return self.layers[0].keys.shape[2]


def actual_cache_bytes(kv_cache) -> int:
    total = 0
    for layer in kv_cache.layers:
        for tensor in (layer.keys, layer.values):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def tensor_list_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def quantization_scale_bytes(kv_cache) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in kv_cache.layers
        for tensor in (layer.key_scale, layer.value_scale)
        if tensor is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal full-server LLM inference")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Total new tokens, including the one chosen during prefill "
        "(the decode loop runs at most this many minus 1)",
    )
    parser.add_argument("--verbose", action="store_true", help="Unwrap model() layer by layer")
    parser.add_argument(
        "--compression",
        choices=("none", "window", "importance", "quant"),
        default="none",
    )
    parser.add_argument("--cache-budget", type=int, default=64)
    parser.add_argument("--sink-tokens", type=int, default=4)
    args = parser.parse_args()
    if args.compression != "none" and not args.verbose:
        parser.error("cache compression currently requires --verbose")

    if args.compression in ("window", "importance"):
        if args.cache_budget <= 0:
            parser.error("--cache-budget must be positive")
    if args.compression == "window":
        if not 0 <= args.sink_tokens < args.cache_budget:
            parser.error("--sink-tokens must be between 0 and cache-budget - 1")

    prompt = args.prompt
    if prompt is None:
        prompt = input("Prompt: ").strip()
    if not prompt:
        print("Empty prompt.", file=sys.stderr)
        sys.exit(1)

    dtype = torch.float16 if args.device != "cpu" else torch.float32

    # --- 1) Load model ---
    print("\n[1] Load model")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device)
    model.eval()

    cfg = model.config
    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = getattr(cfg, "head_dim", d_model // num_heads)
    attn_scale = 1.0 / math.sqrt(head_dim)
    kv_itemsize = 2 if dtype == torch.float16 else 4

    print(f"    model={args.model}")
    print(f"    device={args.device}, dtype={dtype}")
    print(f"    d_model={d_model}, layers={n_layers}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"    GQA: {num_heads} query heads share {num_kv_heads} KV heads ({num_heads // num_kv_heads}x smaller KV cache than MHA)")
    print(f"    load time={(time.perf_counter() - t0) * 1000:.0f} ms")

    # --- 2) Receive prompt ---
    print("\n[2] Receive prompt")
    print(f"    raw: {prompt!r}")
    messages = [{"role": "user", "content": prompt}]
    chat_kwargs = {"add_generation_prompt": True}
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, **chat_kwargs)
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", **chat_kwargs
    ).to(args.device)
    print(f"    templated: {formatted!r}")
    print(f"    {input_ids.shape[1]} tokens")

    # --- 3) Prefill + KV cache ---
    print("\n[3] Prefill + KV cache")
    t_prefill = time.perf_counter()

    if not args.verbose:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        kv_cache = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    else:
        print("    --- unwrap model(): layer-by-layer prefill ---")
        with torch.no_grad():
            # 1. Embed token IDs
            hidden_states = model.get_input_embeddings()(input_ids)
            batch_size, seq_len, _ = hidden_states.shape
            print(f"    * Token IDs {tuple(input_ids.shape)} -> embeddings {tuple(hidden_states.shape)}")

            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=args.device)).bool()
            position_ids = torch.arange(seq_len, device=args.device).unsqueeze(0)
            kv_cache = QuantizedKVCache() if args.compression == "quant" else DynamicCache()
            importance_scores = [None] * n_layers

            # 2. LOOP THROUGH ALL LAYERS MANUALLY
            for layer_idx, layer in enumerate(model.model.layers):
                attn, mlp = layer.self_attn, layer.mlp

                # A. Pre-Attention Normalization
                x_norm = layer.input_layernorm(hidden_states)

                # B. Attention Computations
                # Q, K, V Projections
                q = attn.q_proj(x_norm).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = attn.k_proj(x_norm).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                v = attn.v_proj(x_norm).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

                # Apply Rotary Position Embeddings (RoPE)
                cos, sin = model.model.rotary_emb(v, position_ids)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                # Append K and V into the DynamicCache object
                k_cached, v_cached = kv_cache.update(k, v, layer_idx)

                if args.compression == "window":
                    k_cached, v_cached = apply_sink_window(
                        k_cached,
                        v_cached,
                        args.cache_budget,
                        args.sink_tokens,
                    )
                    kv_cache.layers[layer_idx].keys = k_cached
                    kv_cache.layers[layer_idx].values = v_cached

                # Handle Grouped-Query Attention (GQA) expansion
                if num_kv_heads < num_heads:
                    k_exp = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
                    v_exp = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
                else:
                    k_exp, v_exp = k, v

                # Scaled Dot-Product Attention
                scores = torch.matmul(q, k_exp.transpose(-2, -1)) * attn_scale
                scores = scores.masked_fill(~causal_mask, float("-inf"))
                attn_weights = torch.softmax(scores, dim=-1)
                attn_output = torch.matmul(attn_weights, v_exp)

                if args.compression == "importance":
                    # One cumulative score per cached position in this layer.
                    layer_importance = attn_weights.float().sum(dim=(0, 1, 2))
                    k_cached, v_cached, layer_importance = apply_importance_eviction(
                        k_cached,
                        v_cached,
                        layer_importance,
                        args.cache_budget,
                    )
                    kv_cache.layers[layer_idx].keys = k_cached
                    kv_cache.layers[layer_idx].values = v_cached
                    importance_scores[layer_idx] = layer_importance

                # Output Projection + Residual Add
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
                hidden_states = hidden_states + attn.o_proj(attn_output)

                # C. SwiGLU MLP + Residual Add
                x_mid = layer.post_attention_layernorm(hidden_states)
                hidden_states = hidden_states + mlp.down_proj(mlp.act_fn(mlp.gate_proj(x_mid)) * mlp.up_proj(x_mid))

                if layer_idx == 0:
                    print(f"    * Layer 0 Q {tuple(q.shape)}  K cached {tuple(k.shape)}")
                    print(f"    * Layer 0 scores {tuple(scores.shape)} (pre-softmax)  weights {tuple(attn_weights.shape)}")

            print(f"    * Ran all {n_layers} layers manually")

            # 3. Final RMSNorm + LM head
            logits = model.lm_head(model.model.norm(hidden_states))
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_prefill) * 1000

    if args.verbose:  # sanity check, outside the prefill timer
        with torch.no_grad():
            ref_id = model(input_ids=input_ids, use_cache=False, return_dict=True).logits[:, -1, :].argmax(dim=-1)
        print(f"    * First token matches HuggingFace model(): {ref_id.item() == next_token_id.item()}")
    seq = kv_seq_len(kv_cache)
    actual_bytes = actual_cache_bytes(kv_cache)
    dense_bytes = kv_bytes(
        n_layers,
        num_kv_heads,
        input_ids.shape[1],
        head_dim,
        1,
        kv_itemsize,
    )
    first_text = tokenizer.decode(next_token_id[0], skip_special_tokens=False)

    print(f"    prefill time={prefill_ms:.1f} ms")
    print(f"    KV cache seq_len={seq}")
    print(f"    actual KV cache bytes={actual_bytes}")
    print(f"    uncompressed dense baseline bytes={dense_bytes}")
    if args.compression == "importance":
        score_bytes = tensor_list_bytes(importance_scores)
        print(f"    importance score bytes={score_bytes}")
        print(f"    total cache state bytes={actual_bytes + score_bytes}")
    elif args.compression == "quant":
        scale_bytes = quantization_scale_bytes(kv_cache)
        print(f"    quantization scale bytes={scale_bytes}")
        print(f"    total cache state bytes={actual_bytes + scale_bytes}")
    print(f"    first output token id={next_token_id.item()} piece={first_text!r}")

    # --- 4) Autoregressive decode ---
    print("\n[4] Autoregressive decode")
    generated = [next_token_id.item()]
    # Absolute position of the next token. Track it separately: after eviction,
    # the cache length no longer equals the token's true position (RoPE).
    cur_pos = input_ids.shape[1]
    t_decode = time.perf_counter()
    eos_id = tokenizer.eos_token_id
    print(
        f"    prefill chose id={generated[0]:6d}  "
        f"text so far={tokenizer.decode(generated, skip_special_tokens=True)!r}"
    )

    # --max-tokens includes the prefill token, so decode emits at most one fewer.
    for step in range(args.max_tokens - 1):
        with torch.no_grad():
            if not args.verbose:
                outputs = model(input_ids=next_token_id, past_key_values=kv_cache, use_cache=True, return_dict=True)
                kv_cache = outputs.past_key_values
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                # Same block as prefill, but Q has seq=1 and attends to the full cache
                position_ids = torch.tensor([[cur_pos]], device=args.device)
                hidden_states = model.get_input_embeddings()(next_token_id)
                batch_size = hidden_states.shape[0]

                # LOOP THROUGH ALL LAYERS MANUALLY
                for layer_idx, layer in enumerate(model.model.layers):
                    attn, mlp = layer.self_attn, layer.mlp

                    # A. Pre-Attention Normalization
                    x_norm = layer.input_layernorm(hidden_states)

                    # B. Attention Computations
                    # Q, K, V Projections
                    q = attn.q_proj(x_norm).view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
                    k_new = attn.k_proj(x_norm).view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)
                    v_new = attn.v_proj(x_norm).view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)

                    # Apply Rotary Position Embeddings (RoPE)
                    cos, sin = model.model.rotary_emb(v_new, position_ids)
                    q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

                    # Append the current token to this layer's cache.
                    k_full, v_full = kv_cache.update(k_new, v_new, layer_idx)

                    # Apply sink + recent-window eviction.
                    if args.compression == "window":
                        k_full, v_full = apply_sink_window(
                            k_full,
                            v_full,
                            args.cache_budget,
                            args.sink_tokens,
                        )
                        kv_cache.layers[layer_idx].keys = k_full
                        kv_cache.layers[layer_idx].values = v_full

                    # Handle Grouped-Query Attention (GQA) expansion
                    if num_kv_heads < num_heads:
                        k_exp = k_full.repeat_interleave(num_heads // num_kv_heads, dim=1)
                        v_exp = v_full.repeat_interleave(num_heads // num_kv_heads, dim=1)
                    else:
                        k_exp, v_exp = k_full, v_full

                    # Scaled Dot-Product Attention (no mask: the new token may attend to all cached positions)
                    scores = torch.matmul(q, k_exp.transpose(-2, -1)) * attn_scale
                    attn_weights = torch.softmax(scores, dim=-1)
                    attn_output = torch.matmul(attn_weights, v_exp)

                    if args.compression == "importance":
                        # Add the new position, accumulate attention received
                        # this step, and evict the lowest-scoring positions.
                        layer_importance = torch.cat(
                            (
                                importance_scores[layer_idx],
                                torch.zeros(1, device=args.device, dtype=torch.float32),
                            )
                        )
                        layer_importance += attn_weights.float().sum(dim=(0, 1, 2))
                        k_full, v_full, layer_importance = apply_importance_eviction(
                            k_full,
                            v_full,
                            layer_importance,
                            args.cache_budget,
                        )
                        kv_cache.layers[layer_idx].keys = k_full
                        kv_cache.layers[layer_idx].values = v_full
                        importance_scores[layer_idx] = layer_importance

                    # Output Projection + Residual Add
                    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, d_model)
                    hidden_states = hidden_states + attn.o_proj(attn_output)

                    # C. SwiGLU MLP + Residual Add
                    x_mid = layer.post_attention_layernorm(hidden_states)
                    hidden_states = hidden_states + mlp.down_proj(mlp.act_fn(mlp.gate_proj(x_mid)) * mlp.up_proj(x_mid))

                    if step == 0 and layer_idx == 0:
                        print(f"    * decode step 1 layer 0: Q seq={q.shape[2]}  K seq={k_full.shape[2]}")

                # Final RMSNorm + LM head
                logits = model.lm_head(model.model.norm(hidden_states))
                next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        cur_pos += 1
        token_id = next_token_id.item()
        generated.append(token_id)
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        so_far = tokenizer.decode(generated, skip_special_tokens=True)
        print(
            f"    step {step + 1:2d}: id={token_id:6d}  "
            f"piece={piece!r}  text so far={so_far!r}  kv_len={kv_seq_len(kv_cache)}"
        )
        if token_id == eos_id:
            print("    (EOS — stopping decode)")
            break
        so_far = tokenizer.decode(generated, skip_special_tokens=False)
        if "<|user|>" in so_far or "<|system|>" in so_far:
            print("    (chat role tag — stopping decode)")
            break

    decode_ms = (time.perf_counter() - t_decode) * 1000

    # --- 5) Exit ---
    print("\n[5] Exit")
    print(f"    decode time={decode_ms:.1f} ms ({len(generated) - 1} new tokens)")
    final_cache_bytes = actual_cache_bytes(kv_cache)
    print(f"    final KV cache bytes={final_cache_bytes}")
    print(f"    final KV cache length={kv_seq_len(kv_cache)}")
    if args.compression == "importance":
        score_bytes = tensor_list_bytes(importance_scores)
        print(f"    final importance score bytes={score_bytes}")
        print(f"    final total cache state bytes={final_cache_bytes + score_bytes}")
    elif args.compression == "quant":
        scale_bytes = quantization_scale_bytes(kv_cache)
        print(f"    final quantization scale bytes={scale_bytes}")
        print(f"    final total cache state bytes={final_cache_bytes + scale_bytes}")
    print(f"    output: {decode_answer(tokenizer, generated)!r}")
    print("    done.")


if __name__ == "__main__":
    main()
