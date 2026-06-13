#!/usr/bin/env python3
"""Build a Table-1-shaped LongBench summary from reproduction JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.longbench_metrics import TABLE1_CATEGORIES, category_for_dataset, normalize_dataset_name


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize_run(path: Path, *, model: str, method: str, kv_size: float) -> dict:
    dataset_scores = defaultdict(list)
    storage_ratios = []
    for row in load_records(path):
        if row.get("longbench_score") is None:
            continue
        dataset = normalize_dataset_name(row.get("longbench_dataset") or row.get("task") or row.get("dataset"))
        if not dataset:
            continue
        score = float(row["longbench_score"]) * 100.0
        dataset_scores[dataset].append(score)
        if row.get("cache_storage_ratio") is not None:
            storage_ratios.append(float(row["cache_storage_ratio"]))

    dataset_score_means = {name: sum(scores) / len(scores) for name, scores in sorted(dataset_scores.items())}
    category_means = {
        category: (
            sum(dataset_score_means[dataset] for dataset in datasets if dataset in dataset_score_means)
            / len([dataset for dataset in datasets if dataset in dataset_score_means])
            if any(dataset in dataset_score_means for dataset in datasets)
            else None
        )
        for category, datasets in TABLE1_CATEGORIES.items()
    }
    available_categories = [value for value in category_means.values() if value is not None]
    average = sum(available_categories) / len(available_categories) if available_categories else None

    return {
        "model": model,
        "method": method,
        "kv_size": kv_size,
        "source": str(path),
        "categories": category_means,
        "average_available_categories": average,
        "dataset_scores": dataset_score_means,
        "coverage": {
            category: {
                "available": [dataset for dataset in datasets if dataset in dataset_scores],
                "missing": [dataset for dataset in datasets if dataset not in dataset_scores],
            }
            for category, datasets in TABLE1_CATEGORIES.items()
        },
        "avg_cache_storage_ratio": sum(storage_ratios) / len(storage_ratios) if storage_ratios else None,
    }


def format_value(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["Model", "Method", "KV Size", *TABLE1_CATEGORIES.keys(), "Average", "Source"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Model": row["model"],
                    "Method": row["method"],
                    "KV Size": row["kv_size"],
                    **{category: format_value(row["categories"][category]) for category in TABLE1_CATEGORIES},
                    "Average": format_value(row["average_available_categories"]),
                    "Source": row["source"],
                }
            )


def write_md(rows: list[dict], path: Path) -> None:
    headers = ["Model", "Method", "KV Size", *TABLE1_CATEGORIES.keys(), "Average"]
    lines = [
        "# Table 1 Llama Reproduction Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["model"],
            row["method"],
            str(row["kv_size"]),
            *[format_value(row["categories"][category]) for category in TABLE1_CATEGORIES],
            format_value(row["average_available_categories"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Scores are LongBench-style percentages. Empty cells mean the required datasets are not present in the input JSONL files.",
            "",
            "## Coverage",
        ]
    )
    for row in rows:
        lines.append("")
        lines.append(f"### {row['method']} KV={row['kv_size']}")
        for category, coverage in row["coverage"].items():
            available = ", ".join(coverage["available"]) or "none"
            missing = ", ".join(coverage["missing"]) or "none"
            lines.append(f"- {category}: available={available}; missing={missing}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=4, metavar=("METHOD", "KV_SIZE", "MODEL", "JSONL"), required=True)
    parser.add_argument("--output-prefix", default="reproduce/runs/table1_llama_partial")
    args = parser.parse_args()

    rows = [
        summarize_run(Path(jsonl), model=model, method=method, kv_size=float(kv_size))
        for method, kv_size, model, jsonl in args.run
    ]

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(rows, csv_path)
    write_md(rows, md_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
