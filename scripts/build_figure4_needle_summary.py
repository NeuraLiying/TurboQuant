#!/usr/bin/env python3
"""Build a compact Figure-4-style Needle summary from aggregate JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAPER_SCORES = {
    "Full-Precision": 0.997,
    "TurboQuant": 0.997,
}


def load_run(method: str, kv_size: float, path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "method": method,
        "kv_size": kv_size,
        "source": str(path),
        "paper_score": PAPER_SCORES.get(method),
        "overall": data["overall"],
        "by_position": data.get("by_position", {}),
        "by_distractor_lang": data.get("by_distractor_lang", {}),
    }
    return row


def metric_row(summary: dict, *, method: str, kv_size: float, source: str, level: str, group: str) -> dict:
    return {
        "level": level,
        "group": group,
        "method": method,
        "kv_size": kv_size,
        "num_examples": summary.get("num_examples"),
        "answer_contains_accuracy": summary.get("answer_contains_accuracy"),
        "avg_answer_f1": summary.get("avg_answer_f1"),
        "avg_answer_sentence_f1": summary.get("avg_answer_sentence_f1"),
        "avg_prompt_tokens": summary.get("avg_prompt_tokens"),
        "avg_generated_tokens": summary.get("avg_generated_tokens"),
        "avg_latency_seconds": summary.get("avg_latency_seconds"),
        "avg_cache_storage_ratio": summary.get("avg_cache_storage_ratio"),
        "source": source,
    }


def flatten_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        common = {
            "method": run["method"],
            "kv_size": run["kv_size"],
            "source": run["source"],
        }
        rows.append(metric_row(run["overall"], level="overall", group="all", **common))
        for position, summary in sorted(run["by_position"].items()):
            rows.append(metric_row(summary, level="position", group=position, **common))
        for lang, summary in sorted(run["by_distractor_lang"].items()):
            rows.append(metric_row(summary, level="distractor_lang", group=lang, **common))
    return rows


def format_optional(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(runs: list[dict], path: Path, title: str, description: str) -> None:
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "## Overall",
        "",
        "| Method | KV Size | Examples | Local score | Paper score | Avg prompt tokens | Avg latency | Cache ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        overall = run["overall"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run["method"],
                    format_optional(run["kv_size"], 1),
                    str(overall["num_examples"]),
                    format_optional(overall["answer_contains_accuracy"], 4),
                    format_optional(run["paper_score"], 3),
                    format_optional(overall["avg_prompt_tokens"], 2),
                    format_optional(overall["avg_latency_seconds"], 2),
                    format_optional(overall["avg_cache_storage_ratio"], 4),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## By Position",
            "",
            "| Method | Position | Examples | Local score | Avg latency | Cache ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in runs:
        for position, summary in sorted(run["by_position"].items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        run["method"],
                        position,
                        str(summary["num_examples"]),
                        format_optional(summary["answer_contains_accuracy"], 4),
                        format_optional(summary["avg_latency_seconds"], 2),
                        format_optional(summary["avg_cache_storage_ratio"], 4),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Sources",
            "",
        ]
    )
    for run in runs:
        lines.append(f"- {run['method']}: `{run['source']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=3, metavar=("METHOD", "KV_SIZE", "AGGREGATE_JSON"), required=True)
    parser.add_argument("--output-prefix", default="reproduce/runs/figure4_needle_16k_local_summary")
    parser.add_argument("--title", default="Figure 4 Needle 16k Local Summary")
    parser.add_argument(
        "--description",
        default=(
            "This is a local 16k grid summary for the currently cached Needle data. "
            "It is not the full paper Figure 4 heatmap because the local cache does not include "
            "the 4k-104k length grid."
        ),
    )
    args = parser.parse_args()

    runs = [load_run(method, float(kv_size), Path(path)) for method, kv_size, path in args.run]
    rows = flatten_rows(runs)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")

    json_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(runs, md_path, args.title, args.description)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
