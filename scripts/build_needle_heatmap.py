#!/usr/bin/env python3
"""Build Figure-4-style Needle heatmap artifacts from JSONL runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


POSITION_TO_DEPTH = {
    "start": 0,
    "middle": 50,
    "end": 100,
}

PAPER_SCORES = {
    "Full-Precision": 0.997,
    "TurboQuant": 0.997,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def infer_token_limit(row: dict[str, Any]) -> int:
    target = row.get("target_prompt_tokens")
    if target is not None:
        return int(target)
    prompt_tokens = int(row.get("prompt_tokens") or 0)
    # Bucket to the closest paper-style power-of-two-ish limit used in Figure 4.
    for bucket in [4096, 8192, 16384, 32768, 65536, 104000]:
        if prompt_tokens <= bucket:
            return bucket
    return prompt_tokens


def infer_depth_percent(row: dict[str, Any]) -> int:
    position = row.get("needle_position")
    if position in POSITION_TO_DEPTH:
        return POSITION_TO_DEPTH[position]
    return -1


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cache_ratios = [float(row["cache_storage_ratio"]) for row in rows if row.get("cache_storage_ratio") is not None]
    return {
        "num_examples": len(rows),
        "score": mean([float(row["answer_contains"]) for row in rows]),
        "avg_prompt_tokens": mean([float(row["prompt_tokens"]) for row in rows]),
        "avg_latency_seconds": mean([float(row["latency_seconds"]) for row in rows]),
        "avg_cache_storage_ratio": mean(cache_ratios),
    }


def build_run(method: str, kv_size: float, source: Path) -> dict[str, Any]:
    rows = load_jsonl(source)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(infer_token_limit(row), infer_depth_percent(row))].append(row)

    cells = []
    for (token_limit, depth_percent), group_rows in sorted(grouped.items()):
        cells.append(
            {
                "token_limit": token_limit,
                "depth_percent": depth_percent,
                **summarize_group(group_rows),
            }
        )

    token_limits = sorted({cell["token_limit"] for cell in cells})
    depth_percents = sorted({cell["depth_percent"] for cell in cells})
    score_matrix = [
        [
            next(
                (
                    cell["score"]
                    for cell in cells
                    if cell["token_limit"] == token_limit and cell["depth_percent"] == depth_percent
                ),
                None,
            )
            for token_limit in token_limits
        ]
        for depth_percent in depth_percents
    ]
    return {
        "method": method,
        "kv_size": kv_size,
        "paper_score": PAPER_SCORES.get(method),
        "source": str(source),
        "overall": summarize_group(rows),
        "token_limits": token_limits,
        "depth_percents": depth_percents,
        "score_matrix": score_matrix,
        "cells": cells,
    }


def write_csv(runs: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "method",
        "kv_size",
        "token_limit",
        "depth_percent",
        "num_examples",
        "score",
        "avg_prompt_tokens",
        "avg_latency_seconds",
        "avg_cache_storage_ratio",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for cell in run["cells"]:
                writer.writerow(
                    {
                        "method": run["method"],
                        "kv_size": run["kv_size"],
                        "source": run["source"],
                        **cell,
                    }
                )


def write_markdown(runs: list[dict[str, Any]], path: Path, title: str, description: str) -> None:
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "## Overall",
        "",
        "| Method | KV Size | Examples | Score | Paper Score | Avg Prompt Tokens | Avg Latency | Cache Ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        overall = run["overall"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run["method"],
                    f"{run['kv_size']:.1f}",
                    str(overall["num_examples"]),
                    format_optional(overall["score"], 4),
                    format_optional(run.get("paper_score"), 3),
                    format_optional(overall["avg_prompt_tokens"], 2),
                    format_optional(overall["avg_latency_seconds"], 2),
                    format_optional(overall["avg_cache_storage_ratio"], 4),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Cells", ""])
    for run in runs:
        lines.extend(
            [
                f"### {run['method']} {run['kv_size']:.1f}",
                "",
                "| Token Limit | Depth Percent | Examples | Score | Avg Prompt Tokens | Avg Latency | Cache Ratio |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for cell in run["cells"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(cell["token_limit"]),
                        str(cell["depth_percent"]),
                        str(cell["num_examples"]),
                        format_optional(cell["score"], 4),
                        format_optional(cell["avg_prompt_tokens"], 2),
                        format_optional(cell["avg_latency_seconds"], 2),
                        format_optional(cell["avg_cache_storage_ratio"], 4),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(["## Sources", ""])
    for run in runs:
        lines.append(f"- {run['method']}: `{run['source']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_plot(runs: list[dict[str, Any]], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), squeeze=False, constrained_layout=True)
    for ax, run in zip(axes[0], runs):
        token_limits = run["token_limits"]
        depth_percents = run["depth_percents"]
        matrix = [
            [float(value) if value is not None else float("nan") for value in row]
            for row in run["score_matrix"]
        ]
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto", origin="lower")
        ax.set_title(f"{run['method']} ({run['kv_size']:.1f})")
        ax.set_xticks(range(len(token_limits)))
        ax.set_xticklabels([str(value) for value in token_limits], rotation=45, ha="right")
        ax.set_yticks(range(len(depth_percents)))
        ax.set_yticklabels([str(value) for value in depth_percents])
        ax.set_xlabel("Token Limit")
        ax.set_ylabel("Depth Percent")
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                if value == value:
                    ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.suptitle(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Recall")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=3, metavar=("METHOD", "KV_SIZE", "JSONL"), required=True)
    parser.add_argument("--output-prefix", default="reproduce/runs/figure4_needle_heatmap_local")
    parser.add_argument("--title", default="Figure 4 Needle Local Heatmap")
    parser.add_argument(
        "--description",
        default=(
            "This heatmap is built from available local Needle JSONL runs. Token limits and depth "
            "percents are inferred from run metadata, so generated/cropped datasets should be treated "
            "as diagnostics unless true paper-length grids are available."
        ),
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    runs = [build_run(method, float(kv_size), Path(path)) for method, kv_size, path in args.run]
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    png_path = prefix.with_suffix(".png")

    json_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(runs, csv_path)
    write_markdown(runs, md_path, args.title, args.description)
    outputs = {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}
    if args.plot:
        write_plot(runs, png_path, args.title)
        outputs["png"] = str(png_path)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
