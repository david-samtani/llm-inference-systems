#!/usr/bin/env python3
"""Parse compression run logs, calculate metrics, and generate report plots."""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def require(pattern: str, text: str, source: Path) -> re.Match[str]:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find {pattern!r} in {source}")
    return match


def token_edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_token in enumerate(left, start=1):
        current = [row]
        for column, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def parse_result(path: Path) -> dict:
    text = path.read_text()
    mode, setting_text = path.stem.split("-", maxsplit=1)
    setting = int(setting_text)

    first_token = int(require(r"prefill chose id=\s*(\d+)", text, path).group(1))
    step_tokens = [
        int(token)
        for token in re.findall(r"^\s*step\s+\d+: id=\s*(\d+)", text, re.MULTILINE)
    ]
    decode_match = require(
        r"decode time=([0-9.]+) ms \((\d+) new tokens\)", text, path
    )
    decode_ms = float(decode_match.group(1))
    decode_steps = int(decode_match.group(2))
    kv_bytes = int(require(r"final KV cache bytes=(\d+)", text, path).group(1))
    total_match = re.search(r"final total cache state bytes=(\d+)", text)
    total_bytes = int(total_match.group(1)) if total_match else kv_bytes
    cache_length = int(
        require(r"final KV cache length=(\d+)", text, path).group(1)
    )
    output = ast.literal_eval(require(r"^\s*output: (.+)$", text, path).group(1))

    return {
        "name": path.stem,
        "mode": mode,
        "setting": setting,
        "setting_type": "budget" if mode in {"window", "importance"} else "max_tokens",
        "generated_tokens": len(step_tokens) + 1,
        "token_ids": [first_token, *step_tokens],
        "decode_ms": decode_ms,
        "decode_steps": decode_steps,
        "ms_per_token": decode_ms / decode_steps,
        "tokens_per_second": 1000.0 * decode_steps / decode_ms,
        "cache_length": cache_length,
        "kv_bytes": kv_bytes,
        "metadata_bytes": total_bytes - kv_bytes,
        "total_bytes": total_bytes,
        "output": output,
    }


def add_quality_metrics(results: dict[str, dict]) -> None:
    for name, result in results.items():
        if result["mode"] in {"none", "quant"}:
            baseline = results[f"none-{result['setting']}"]
        else:
            baseline = results["none-64"]

        distance = token_edit_distance(result["token_ids"], baseline["token_ids"])
        comparison_length = max(
            len(result["token_ids"]), len(baseline["token_ids"])
        )
        matching_prefix = 0
        for token, baseline_token in zip(result["token_ids"], baseline["token_ids"]):
            if token != baseline_token:
                break
            matching_prefix += 1

        result["edit_distance"] = distance
        result["normalized_edit_distance"] = distance / comparison_length
        result["matching_prefix_tokens"] = matching_prefix


def annotate_points(axis, xs, ys, formatter, offset=7) -> None:
    for x_value, y_value in zip(xs, ys):
        axis.annotate(
            formatter(y_value),
            (x_value, y_value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )


def plot_two_panel(
    results: dict[str, dict],
    output_path: Path,
    metric: str,
    y_label: str,
    title: str,
    formatter,
) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.2), constrained_layout=True
    )

    for series_index, (mode, label, marker) in enumerate((
        ("none", "Uncompressed", "o"),
        ("quant", "Int8 quantization", "s"),
    )):
        rows = sorted(
            (row for row in results.values() if row["mode"] == mode),
            key=lambda row: row["setting"],
        )
        xs = [row["setting"] for row in rows]
        ys = [row[metric] for row in rows]
        axes[0].plot(xs, ys, marker=marker, linewidth=2, label=label)
        annotate_points(
            axes[0], xs, ys, formatter, offset=7 if series_index == 0 else -14
        )

    axes[0].set_title("Sequence-length sweep")
    axes[0].set_xlabel("Maximum generated tokens")
    axes[0].set_ylabel(y_label)
    axes[0].set_xticks([16, 32, 64])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    for series_index, (mode, label, marker) in enumerate((
        ("window", "Sink + sliding window", "o"),
        ("importance", "Importance eviction", "s"),
    )):
        rows = sorted(
            (row for row in results.values() if row["mode"] == mode),
            key=lambda row: row["setting"],
        )
        xs = [row["setting"] for row in rows]
        ys = [row[metric] for row in rows]
        axes[1].plot(xs, ys, marker=marker, linewidth=2, label=label)
        annotate_points(
            axes[1], xs, ys, formatter, offset=7 if series_index == 0 else -14
        )

    axes[1].set_title("Cache-budget sweep (64 generated tokens)")
    axes[1].set_xlabel("Cache budget (tokens)")
    axes[1].set_ylabel(y_label)
    axes[1].set_xticks([8, 16, 24])
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(results: dict[str, dict], output_path: Path) -> None:
    fields = [
        "name",
        "mode",
        "setting",
        "setting_type",
        "generated_tokens",
        "decode_ms",
        "decode_steps",
        "ms_per_token",
        "tokens_per_second",
        "cache_length",
        "kv_bytes",
        "metadata_bytes",
        "total_bytes",
        "edit_distance",
        "normalized_edit_distance",
        "matching_prefix_tokens",
        "output",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in sorted(results):
            writer.writerow({field: results[name][field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--plots", type=Path, default=Path("plots"))
    args = parser.parse_args()

    results = {
        path.stem: parse_result(path) for path in sorted(args.results.glob("*.txt"))
    }
    expected = {
        f"{mode}-{setting}"
        for mode, settings in {
            "none": (16, 32, 64),
            "quant": (16, 32, 64),
            "window": (8, 16, 24),
            "importance": (8, 16, 24),
        }.items()
        for setting in settings
    }
    missing = expected - results.keys()
    if missing:
        raise ValueError(f"Missing result files: {sorted(missing)}")

    add_quality_metrics(results)
    args.plots.mkdir(parents=True, exist_ok=True)
    write_summary(results, args.results / "summary.csv")

    plot_two_panel(
        results,
        args.plots / "cache_memory.png",
        metric="total_bytes",
        y_label="Total persistent cache state (bytes)",
        title="KV-cache storage after generation",
        formatter=lambda value: f"{value / 1024:.0f} KiB",
    )
    plot_two_panel(
        results,
        args.plots / "decode_latency.png",
        metric="ms_per_token",
        y_label="Decode latency (ms/token)",
        title="Decode latency by compression setting",
        formatter=lambda value: f"{value:.1f}",
    )
    plot_two_panel(
        results,
        args.plots / "quality_degradation.png",
        metric="normalized_edit_distance",
        y_label="Normalized token edit distance (0–1)",
        title="Generated-token divergence from uncompressed baseline",
        formatter=lambda value: f"{value:.3f}",
    )

    print(f"Parsed {len(results)} runs")
    print(f"Wrote {args.results / 'summary.csv'}")
    for name in ("cache_memory.png", "decode_latency.png", "quality_degradation.png"):
        print(f"Wrote {args.plots / name}")


if __name__ == "__main__":
    main()
