#!/usr/bin/env python3
"""Build Table-1 results for the runner-validated unified regular-gain gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.longbench_metrics import TABLE1_CATEGORIES, normalize_dataset_name, score_prediction


TASKS = [
    ("SingleQA", "narrativeqa", 200),
    ("SingleQA", "qasper", 200),
    ("SingleQA", "multifieldqa_en", 150),
    ("MultiQA", "hotpotqa", 200),
    ("MultiQA", "2wikimqa", 200),
    ("MultiQA", "musique", 200),
    ("Summarization", "gov_report", 200),
    ("Summarization", "qmsum", 200),
    ("Summarization", "multi_news", 200),
    ("Few shot", "trec", 200),
    ("Few shot", "triviaqa", 200),
    ("Few shot", "samsum", 200),
    ("Synthetic", "passage_retrieval_en", 200),
    ("Synthetic", "passage_count", 200),
    ("Code", "lcc", 500),
    ("Code", "repobench-p", 500),
]
BITS = [("2p5", "2_5", 2.5), ("3p5", "3_5", 3.5)]


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["index"])] = row
    return rows


def score_row(row: dict[str, Any]) -> float:
    dataset = normalize_dataset_name(row.get("longbench_dataset") or row.get("task") or row.get("dataset"))
    return 100.0 * score_prediction(dataset, row.get("prediction", ""), row.get("answers") or [], row.get("all_classes"))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def active(row: dict[str, Any]) -> bool:
    return row.get("effective_key_quantizer") == "regular_gain_mse" or row.get("effective_value_quantizer") == "regular_gain_mse"


def summarize_bit(
    *,
    run_dir: Path,
    baseline_dir: Path,
    method_name: str,
    bit_tag: str,
    baseline_bit_tag: str,
    kv_bits: float,
) -> dict[str, Any]:
    task_rows = []
    for category, dataset, expected in TASKS:
        baseline_path = baseline_dir / f"longbench_{dataset}_turboquant_{baseline_bit_tag}bit_all.jsonl"
        candidate_path = run_dir / f"{method_name}_{dataset}_turboquant_{bit_tag}_full.jsonl"
        baseline_rows = load_rows(baseline_path)
        candidate_rows = load_rows(candidate_path)
        common = sorted(set(baseline_rows) & set(candidate_rows))
        complete = len(candidate_rows) == expected and len(common) == expected
        baseline_score = mean([score_row(baseline_rows[index]) for index in common]) if common else None
        candidate_score = mean([score_row(candidate_rows[index]) for index in common]) if common else None
        ratio_values = [
            float(candidate_rows[index]["cache_storage_ratio"])
            for index in common
            if candidate_rows[index].get("cache_storage_ratio") is not None
        ]
        activation_values = [float(active(candidate_rows[index])) for index in common]
        task_rows.append(
            {
                "category": category,
                "dataset": dataset,
                "expected": expected,
                "baseline_records": len(baseline_rows),
                "candidate_records": len(candidate_rows),
                "common_records": len(common),
                "complete": complete,
                "baseline_score": baseline_score if complete else None,
                "candidate_score": candidate_score if complete else None,
                "delta": candidate_score - baseline_score
                if complete and baseline_score is not None and candidate_score is not None
                else None,
                "candidate_cache_ratio": mean(ratio_values) if complete else None,
                "activation_rate": mean(activation_values) if complete else None,
                "candidate_jsonl": str(candidate_path),
                "baseline_jsonl": str(baseline_path),
            }
        )

    category_rows = []
    for category, datasets in TABLE1_CATEGORIES.items():
        rows = [row for row in task_rows if row["dataset"] in datasets]
        complete = all(row["complete"] for row in rows)
        category_rows.append(
            {
                "category": category,
                "complete": complete,
                "baseline_score": mean([row["baseline_score"] for row in rows if row["baseline_score"] is not None])
                if complete
                else None,
                "candidate_score": mean([row["candidate_score"] for row in rows if row["candidate_score"] is not None])
                if complete
                else None,
                "delta": mean([row["delta"] for row in rows if row["delta"] is not None]) if complete else None,
                "candidate_cache_ratio": mean(
                    [row["candidate_cache_ratio"] for row in rows if row["candidate_cache_ratio"] is not None]
                )
                if complete
                else None,
                "activation_rate": mean([row["activation_rate"] for row in rows if row["activation_rate"] is not None])
                if complete
                else None,
            }
        )

    complete_categories = [row for row in category_rows if row["complete"]]
    all_complete = len(complete_categories) == len(TABLE1_CATEGORIES)
    return {
        "method_name": method_name,
        "kv_bits": kv_bits,
        "bit_tag": bit_tag,
        "complete": all_complete,
        "num_complete_tasks": sum(1 for row in task_rows if row["complete"]),
        "num_tasks": len(task_rows),
        "category_rows": category_rows,
        "task_rows": task_rows,
        "average": {
            "baseline_score": mean([row["baseline_score"] for row in complete_categories]) if all_complete else None,
            "candidate_score": mean([row["candidate_score"] for row in complete_categories]) if all_complete else None,
            "delta": mean([row["delta"] for row in complete_categories]) if all_complete else None,
            "candidate_cache_ratio": mean([row["candidate_cache_ratio"] for row in complete_categories])
            if all_complete
            else None,
            "activation_rate": mean([row["activation_rate"] for row in complete_categories]) if all_complete else None,
        },
    }


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["KV bits", "Level", "Name", "TurboQuant", "Unified Gate", "Delta", "Cache ratio", "Activation", "Complete"]
        )
        for result in results:
            for row in result["category_rows"]:
                writer.writerow(
                    [
                        result["kv_bits"],
                        "category",
                        row["category"],
                        fmt(row["baseline_score"]),
                        fmt(row["candidate_score"]),
                        fmt(row["delta"]),
                        fmt(row["candidate_cache_ratio"]),
                        fmt(row["activation_rate"]),
                        row["complete"],
                    ]
                )
            avg = result["average"]
            writer.writerow(
                [
                    result["kv_bits"],
                    "average",
                    "Average",
                    fmt(avg["baseline_score"]),
                    fmt(avg["candidate_score"]),
                    fmt(avg["delta"]),
                    fmt(avg["candidate_cache_ratio"]),
                    fmt(avg["activation_rate"]),
                    result["complete"],
                ]
            )
            for row in result["task_rows"]:
                writer.writerow(
                    [
                        result["kv_bits"],
                        "task",
                        row["dataset"],
                        fmt(row["baseline_score"]),
                        fmt(row["candidate_score"]),
                        fmt(row["delta"]),
                        fmt(row["candidate_cache_ratio"]),
                        fmt(row["activation_rate"]),
                        row["complete"],
                    ]
                )


def write_md(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Unified Regular-Gain Gate Runner Results",
        "",
        "Runner-validated results for one prompt-only gate at both 2.5-bit and 3.5-bit. The gate uses TurboQuant MSE by default and switches K/V to `regular_gain_mse` when the prompt is not code-like, has at most 12 Passage-style passages, and has at most 20 question marks.",
        "",
    ]
    for result in results:
        avg = result["average"]
        lines.extend(
            [
                f"## KV {result['kv_bits']}",
                "",
                f"Complete tasks: {result['num_complete_tasks']} / {result['num_tasks']}",
                "",
                "| Category | TurboQuant | Unified Gate | Delta | Cache ratio | Activation | Complete |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in result["category_rows"]:
            lines.append(
                f"| {row['category']} | {fmt(row['baseline_score'])} | {fmt(row['candidate_score'])} | "
                f"{fmt(row['delta'])} | {fmt(row['candidate_cache_ratio'])} | {fmt(row['activation_rate'])} | "
                f"{'yes' if row['complete'] else 'no'} |"
            )
        lines.append(
            f"| Average | {fmt(avg['baseline_score'])} | {fmt(avg['candidate_score'])} | "
            f"{fmt(avg['delta'])} | {fmt(avg['candidate_cache_ratio'])} | {fmt(avg['activation_rate'])} | "
            f"{'yes' if result['complete'] else 'no'} |"
        )
        lines.extend(["", "| Category | Task | Records | TurboQuant | Unified Gate | Delta | Activation | Complete |"])
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in result["task_rows"]:
            records = f"{row['candidate_records']}/{row['expected']}"
            lines.append(
                f"| {row['category']} | `{row['dataset']}` | {records} | {fmt(row['baseline_score'])} | "
                f"{fmt(row['candidate_score'])} | {fmt(row['delta'])} | {fmt(row['activation_rate'])} | "
                f"{'yes' if row['complete'] else 'no'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="unified_regular_gain_gate")
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "reproduce/runs/incremental"))
    parser.add_argument("--baseline-dir", default=str(PROJECT_ROOT / "reproduce/runs/table1_official"))
    parser.add_argument(
        "--output-prefix",
        default=str(PROJECT_ROOT / "reproduce/incremental/unified_regular_gain_gate_table1_runner"),
    )
    args = parser.parse_args()

    results = [
        summarize_bit(
            run_dir=Path(args.run_dir),
            baseline_dir=Path(args.baseline_dir),
            method_name=args.method_name,
            bit_tag=bit_tag,
            baseline_bit_tag=baseline_bit_tag,
            kv_bits=kv_bits,
        )
        for bit_tag, baseline_bit_tag, kv_bits in BITS
    ]
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(results, csv_path)
    write_md(results, md_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
