#!/usr/bin/env python3
"""Build a Table-1-style comparison for one incremental method."""

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

from turboquant.longbench_metrics import normalize_dataset_name, score_prediction


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
CATEGORIES = ["SingleQA", "MultiQA", "Summarization", "Few shot", "Synthetic", "Code"]
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
                "candidate_jsonl": str(candidate_path),
                "baseline_jsonl": str(baseline_path),
            }
        )

    category_rows = []
    for category in CATEGORIES:
        rows = [row for row in task_rows if row["category"] == category]
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
            }
        )

    complete_categories = [row for row in category_rows if row["complete"]]
    all_complete = len(complete_categories) == len(CATEGORIES)
    return {
        "baseline": "TurboQuant",
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
        },
    }


def write_csv(results: list[dict[str, Any]], path: Path, *, display_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["KV bits", "Level", "Name", "TurboQuant", display_name, "Delta", "Cache ratio", "Complete"])
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
                        row["complete"],
                    ]
                )


def write_md(results: list[dict[str, Any]], path: Path, *, display_name: str, description: str) -> None:
    lines = [
        f"# {display_name} Table 1 Comparison",
        "",
        description,
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
                f"| Category | TurboQuant | {display_name} | Delta | Cache ratio | Complete |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in result["category_rows"]:
            lines.append(
                f"| {row['category']} | {fmt(row['baseline_score'])} | {fmt(row['candidate_score'])} | "
                f"{fmt(row['delta'])} | {fmt(row['candidate_cache_ratio'])} | {'yes' if row['complete'] else 'no'} |"
            )
        lines.append(
            f"| Average | {fmt(avg['baseline_score'])} | {fmt(avg['candidate_score'])} | "
            f"{fmt(avg['delta'])} | {fmt(avg['candidate_cache_ratio'])} | {'yes' if result['complete'] else 'no'} |"
        )
        lines.extend(["", f"| Category | Task | Records | TurboQuant | {display_name} | Delta | Complete |"])
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for row in result["task_rows"]:
            records = f"{row['candidate_records']}/{row['expected']}"
            lines.append(
                f"| {row['category']} | `{row['dataset']}` | {records} | {fmt(row['baseline_score'])} | "
                f"{fmt(row['candidate_score'])} | {fmt(row['delta'])} | {'yes' if row['complete'] else 'no'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument(
        "--description",
        default="Candidate method evaluated against reproduced TurboQuant baselines on LongBench Table 1 tasks.",
    )
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "reproduce/runs/incremental"))
    parser.add_argument("--baseline-dir", default=str(PROJECT_ROOT / "reproduce/runs/table1_official"))
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    baseline_dir = Path(args.baseline_dir)
    results = [
        summarize_bit(
            run_dir=run_dir,
            baseline_dir=baseline_dir,
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
    json_path.write_text(
        json.dumps(
            {
                "method_name": args.method_name,
                "display_name": args.display_name,
                "description": args.description,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(results, csv_path, display_name=args.display_name)
    write_md(results, md_path, display_name=args.display_name, description=args.description)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
