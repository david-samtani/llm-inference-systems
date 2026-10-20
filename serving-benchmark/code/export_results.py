#!/usr/bin/env python3
"""
Validate the results JSON consumed by report/src/results.js, and provide the
shared metric helpers (TTFT, TBT, KV bytes) used by the sweep driver.

  python code/export_results.py report/src/data/results.json

Relative paths resolve against the project root, so this works from any directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str | Path) -> Path:
    """Absolute paths and paths that exist from the CWD win; otherwise resolve from repo root."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def kv_seq_len(kv_cache) -> int:
    """Same helper as minimal_full_server.py / less_minimal_full_server.py."""
    if hasattr(kv_cache, "get_seq_length"):
        return kv_cache.get_seq_length()
    return kv_cache[0][0].shape[2]


def kv_bytes(
    n_layers: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
    batch: int = 1,
    itemsize: int = 2,
) -> int:
    """Dense GQA KV footprint (K + V), as in kv-cache-compression/code/less_minimal_full_server.py."""
    return 2 * itemsize * n_layers * n_kv_heads * seq_len * head_dim * batch


def ttft_ms(t_request_start: float, t_first_token: float) -> float:
    """Time to first token: wall clock from request start to first generated token."""
    return (t_first_token - t_request_start) * 1000.0


def percentile_ms(samples_ms: list[float], p: float) -> float:
    """Linear-interpolation percentile on milliseconds (0 <= p <= 1)."""
    s = sorted(v for v in samples_ms if math.isfinite(v))
    if not s:
        return float("nan")
    if len(s) == 1:
        return float(s[0])
    i = (len(s) - 1) * p
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (i - lo))


def summarize_client_samples(samples_ms: list[float]) -> dict[str, float]:
    """Collapse per-client TTFT or TBT samples into report scalars + tail stats."""
    s = [v for v in samples_ms if math.isfinite(v)]
    if not s:
        return {
            "p50": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
        }
    return {
        "p50": percentile_ms(s, 0.5),
        "p95": percentile_ms(s, 0.95),
        "max": float(max(s)),
        "mean": float(sum(s) / len(s)),
    }


def tbt_ms_per_step(decode_step_durations_s: list[float]) -> dict[str, float]:
    """
    Time-between-tokens for decode steps (exclude prefill).
    Returns p50 and mean in milliseconds.
    """
    if not decode_step_durations_s:
        return {"tbt_ms_p50": float("nan"), "tbt_ms_mean": float("nan")}
    ms = [d * 1000.0 for d in decode_step_durations_s]
    return {"tbt_ms_p50": float(median(ms)), "tbt_ms_mean": float(sum(ms) / len(ms))}


@dataclass
class RunRow:
    system_id: str
    prompt_tokens: int
    concurrency: int
    rep: int
    ttft_ms: float
    tbt_ms_p50: float
    tbt_ms_mean: float
    kv_bytes: int
    kv_seq_len: int
    new_tokens: int
    extra: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        row = {
            "system_id": self.system_id,
            "prompt_tokens": self.prompt_tokens,
            "concurrency": self.concurrency,
            "rep": self.rep,
            "ttft_ms": round(self.ttft_ms, 3),
            "tbt_ms_p50": round(self.tbt_ms_p50, 3),
            "tbt_ms_mean": round(self.tbt_ms_mean, 3),
            "kv_bytes": int(self.kv_bytes),
            "kv_seq_len": int(self.kv_seq_len),
            "new_tokens": int(self.new_tokens),
        }
        row.update(self.extra)
        return row


REQUIRED_RUN_KEYS = {
    "system_id",
    "prompt_tokens",
    "concurrency",
    "rep",
    "ttft_ms",
    "tbt_ms_p50",
    "tbt_ms_mean",
    "kv_bytes",
    "kv_seq_len",
    "new_tokens",
}


def validate_document(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("meta", "source", "systems", "runs", "takeaways", "summary", "plot", "boxplot"):
        if key not in doc:
            errors.append(f"missing top-level key: {key}")
    if not isinstance(doc.get("takeaways"), list) or len(doc["takeaways"]) < 3:
        errors.append("takeaways must be a list with at least 3 bullets")
    systems = doc.get("systems") or []
    if len(systems) < 3:
        errors.append(
            f"systems must contain at least 3 entries (baseline + 2 engines); got {len(systems)}"
        )
    for i, system in enumerate(systems):
        if not (system.get("novelty") or "").strip():
            errors.append(f"systems[{i}] must include non-empty novelty (the mechanism under test)")
    plot = doc.get("plot") or {}
    for pk in ("title", "metric", "x_field", "series_field"):
        if pk not in plot:
            errors.append(f"plot.{pk} is required")
    metric = plot.get("metric")
    if metric not in ("ttft_ms", "tbt_ms_p50", "tbt_ms_mean", "kv_bytes"):
        errors.append(f"plot.metric must be one of ttft_ms, tbt_ms_p50, tbt_ms_mean, kv_bytes (got {metric!r})")
    runs = doc.get("runs") or []
    if len(runs) < 6:
        errors.append("runs must contain at least 6 measurement rows for a minimal sweep")
    system_ids = {s.get("id") for s in systems}
    for i, run in enumerate(runs):
        missing = REQUIRED_RUN_KEYS - set(run.keys())
        if missing:
            errors.append(f"runs[{i}] missing keys: {sorted(missing)}")
        if run.get("system_id") not in system_ids:
            errors.append(f"runs[{i}] system_id {run.get('system_id')!r} is not in systems[]")
        n = run.get("concurrency") or 1
        if n > 1:
            for field in ("ttft_ms_samples", "tbt_ms_samples"):
                arr = run.get(field)
                if not isinstance(arr, list) or len(arr) != n:
                    got = len(arr) if isinstance(arr, list) else "missing"
                    errors.append(f"runs[{i}] {field} must have length {n} (concurrency); got {got}")
            kvs = run.get("kv_bytes_samples")
            if kvs is not None and (not isinstance(kvs, list) or len(kvs) != n):
                errors.append(f"runs[{i}] kv_bytes_samples, if present, must have length {n}")
    workload = (doc.get("source") or {}).get("workload") or {}
    for wk, kind in (
        ("prompt_token_targets", list),
        ("concurrency_grid", list),
        ("max_new_tokens", int),
        ("reps_per_cell", int),
    ):
        if not isinstance(workload.get(wk), kind):
            errors.append(f"source.workload.{wk} is required ({kind.__name__}); the report page reads it")
    bp = doc.get("boxplot") or {}
    for pk in ("title", "metric", "filter", "group_field"):
        if pk not in bp:
            errors.append(f"boxplot.{pk} is required")
    max_conc = max((r.get("concurrency") or 1) for r in runs) if runs else 1
    if max_conc > 1:
        has_samples = any(
            isinstance(r.get("ttft_ms_samples"), list)
            and len(r["ttft_ms_samples"]) >= 2
            for r in runs
            if (r.get("concurrency") or 1) > 1
        )
        if not has_samples:
            errors.append(
                "concurrent runs must include ttft_ms_samples[] "
                "(length >= 2 per burst) on at least one row"
            )
    else:
        errors.append(
            "measurement matrix requires concurrency > 1 for at least some runs "
            "(needed for required boxplot)"
        )
    return errors


def matrix_warnings(doc: dict[str, Any]) -> list[str]:
    """Soft check: is every (prompt, concurrency) cell covered with reps_per_cell reps per system?"""
    workload = (doc.get("source") or {}).get("workload") or {}
    prompts = workload.get("prompt_token_targets") or []
    grid = workload.get("concurrency_grid") or []
    reps = workload.get("reps_per_cell") or 1
    warnings: list[str] = []
    for system in doc.get("systems") or []:
        sid = system.get("id")
        short = []
        for pt in prompts:
            for n in grid:
                got = sum(
                    1 for r in doc.get("runs") or []
                    if r.get("system_id") == sid
                    and r.get("prompt_tokens") == pt
                    and (r.get("concurrency") or 1) == n
                )
                if got < reps:
                    short.append(f"p={pt},N={n}: {got}/{reps}")
        if short:
            total = len(prompts) * len(grid)
            warnings.append(
                f"{sid}: {len(short)} of {total} cells below {reps} reps ({'; '.join(short)})"
            )
    return warnings


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, allow_nan=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark results JSON")
    parser.add_argument(
        "path",
        nargs="?",
        default="report/src/data/results.json",
        help="JSON file to validate (default: report/src/data/results.json)",
    )
    parser.add_argument(
        "--write",
        metavar="OUT",
        help="Copy validated input to OUT (after validation)",
    )
    args = parser.parse_args()

    path = resolve_path(args.path)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    doc = load_json(path)
    errors = validate_document(doc)
    if errors:
        print("Validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {path} ({len(doc['runs'])} runs)")
    warnings = matrix_warnings(doc)
    if warnings:
        print("Warning: measurement matrix incomplete (expected every cell):")
        for w in warnings:
            print(f"  - {w}")
    if args.write:
        dest = resolve_path(args.write)
        write_json(dest, doc)
        print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
