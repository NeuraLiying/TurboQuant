#!/usr/bin/env python3
"""Build the official local Table 1 comparison for Full Cache and TurboQuant."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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

METHODS = [
    {
        "name": "Full Cache",
        "method_stem": "full_cache",
        "kv_size": 16.0,
        "paper": {
            "SingleQA": 45.29,
            "MultiQA": 45.16,
            "Summarization": 26.55,
            "Few shot": 68.38,
            "Synthetic": 59.54,
            "Code": 46.28,
            "Average": 50.06,
        },
    },
    {
        "name": "TurboQuant",
        "method_stem": "turboquant_2_5bit",
        "kv_size": 2.5,
        "paper": {
            "SingleQA": 44.16,
            "MultiQA": 44.96,
            "Summarization": 24.80,
            "Few shot": 68.01,
            "Synthetic": 59.65,
            "Code": 45.76,
            "Average": 49.44,
        },
    },
    {
        "name": "TurboQuant",
        "method_stem": "turboquant_3_5bit",
        "kv_size": 3.5,
        "paper": {
            "SingleQA": 45.01,
            "MultiQA": 45.31,
            "Summarization": 26.00,
            "Few shot": 68.63,
            "Synthetic": 59.95,
            "Code": 46.17,
            "Average": 50.06,
        },
    },
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_indexes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indexes = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            indexes.add(int(json.loads(line)["index"]))
    return indexes


def format_score(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def summarize_method(run_root: Path, method: dict) -> dict:
    task_rows = []
    category_values: dict[str, list[float]] = {category: [] for category in CATEGORIES}
    complete = True
    for category, dataset, expected in TASKS:
        stem = f"longbench_{dataset}_{method['method_stem']}_all"
        jsonl_path = run_root / f"{stem}.jsonl"
        aggregate_path = run_root / f"{stem}.aggregate.json"
        indexes = jsonl_indexes(jsonl_path)
        aggregate = read_json(aggregate_path) if aggregate_path.exists() else None
        score = float(aggregate["longbench_score"]) if aggregate and "longbench_score" in aggregate else None
        task_complete = len(indexes) == expected and score is not None
        complete = complete and task_complete
        if task_complete and score is not None:
            category_values[category].append(score)
        task_rows.append(
            {
                "category": category,
                "dataset": dataset,
                "expected": expected,
                "unique_records": len(indexes),
                "records": sum(1 for line in jsonl_path.open("r", encoding="utf-8") if line.strip())
                if jsonl_path.exists()
                else 0,
                "complete": task_complete,
                "score": score if task_complete else None,
                "jsonl": str(jsonl_path),
                "aggregate": str(aggregate_path),
            }
        )

    category_scores = {}
    for category in CATEGORIES:
        values = category_values[category]
        expected_count = sum(1 for task_category, _, _ in TASKS if task_category == category)
        category_scores[category] = sum(values) / len(values) if len(values) == expected_count else None
    available = [score for score in category_scores.values() if score is not None]
    average = sum(available) / len(available) if len(available) == len(CATEGORIES) else None
    paper = method["paper"]
    return {
        "method": method["name"],
        "method_stem": method["method_stem"],
        "kv_size": method["kv_size"],
        "complete": complete,
        "category_scores": category_scores,
        "average": average,
        "paper": paper,
        "tasks": task_rows,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    headers = ["Method", "KV Size", "Source", *CATEGORIES, "Average"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            local = {
                "Method": row["method"],
                "KV Size": row["kv_size"],
                "Source": "local",
                **{category: format_score(row["category_scores"][category]) for category in CATEGORIES},
                "Average": format_score(row["average"]),
            }
            paper = {
                "Method": row["method"],
                "KV Size": row["kv_size"],
                "Source": "paper",
                **{category: format_score(row["paper"][category]) for category in CATEGORIES},
                "Average": format_score(row["paper"]["Average"]),
            }
            writer.writerows([paper, local])


def write_md(rows: list[dict], path: Path) -> None:
    lines = [
        "# Table 1 Official Local Comparison",
        "",
        "Scope: `meta-llama/Llama-3.1-8B-Instruct`, LongBench-V1 full task splits, Full Cache and TurboQuant only.",
        "",
        "Only complete task files with the expected unique example indexes are used for category and average scores.",
        "",
        "## Category Comparison",
        "",
        "| Method | KV Size | Source | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average | Complete |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    str(row["kv_size"]),
                    "paper",
                    *[format_score(row["paper"][category]) for category in CATEGORIES],
                    format_score(row["paper"]["Average"]),
                    "yes",
                ]
            )
            + " |"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    str(row["kv_size"]),
                    "local",
                    *[format_score(row["category_scores"][category]) for category in CATEGORIES],
                    format_score(row["average"]),
                    "yes" if row["complete"] else "no",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Task Scores", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['method']} KV={row['kv_size']}",
                "",
                "| Category | Dataset | Unique Records | Records | Score | Complete |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for task in row["tasks"]:
            lines.append(
                f"| {task['category']} | `{task['dataset']}` | "
                f"{task['unique_records']}/{task['expected']} | {task['records']} | "
                f"{format_score(task['score'])} | {'yes' if task['complete'] else 'no'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="reproduce/runs/table1_official")
    parser.add_argument("--output-prefix", default="reproduce/TABLE1_OFFICIAL_COMPARISON")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = [summarize_method(run_root, method) for method in METHODS]
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
