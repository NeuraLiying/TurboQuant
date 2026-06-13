#!/usr/bin/env python3
"""Summarize official LongBench-V1 Table 1 progress without using partial runs."""

from __future__ import annotations

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


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_score(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("longbench_score")


def main() -> None:
    run_root = Path("reproduce/runs/table1_official")
    rows = []
    by_category: dict[str, list[float]] = {}
    for category, dataset, expected in TASKS:
        stem = f"longbench_{dataset}_full_cache_all"
        jsonl_path = run_root / f"{stem}.jsonl"
        aggregate_path = run_root / f"{stem}.aggregate.json"
        records = count_jsonl(jsonl_path)
        complete = records == expected
        score = load_score(aggregate_path) if complete else None
        if score is not None:
            by_category.setdefault(category, []).append(float(score))
        rows.append(
            {
                "category": category,
                "dataset": dataset,
                "expected_examples": expected,
                "records": records,
                "status": "complete" if complete else "partial" if records else "missing",
                "longbench_score": score,
                "jsonl": str(jsonl_path),
                "aggregate": str(aggregate_path),
            }
        )

    category_scores = {name: sum(values) / len(values) for name, values in by_category.items()}
    completed_categories = {category for category, datasets in {
        "SingleQA": ["narrativeqa", "qasper", "multifieldqa_en"],
        "MultiQA": ["hotpotqa", "2wikimqa", "musique"],
        "Summarization": ["gov_report", "qmsum", "multi_news"],
        "Few shot": ["trec", "triviaqa", "samsum"],
        "Synthetic": ["passage_retrieval_en", "passage_count"],
        "Code": ["lcc", "repobench-p"],
    }.items() if all(row["status"] == "complete" for row in rows if row["category"] == category)}
    completed_category_scores = {k: v for k, v in category_scores.items() if k in completed_categories}
    average_completed_categories = (
        sum(completed_category_scores.values()) / len(completed_category_scores) if completed_category_scores else None
    )

    summary = {
        "run_root": str(run_root),
        "num_complete_tasks": sum(1 for row in rows if row["status"] == "complete"),
        "num_partial_tasks": sum(1 for row in rows if row["status"] == "partial"),
        "num_missing_tasks": sum(1 for row in rows if row["status"] == "missing"),
        "category_scores_available_subsets": category_scores,
        "completed_category_scores": completed_category_scores,
        "average_completed_categories": average_completed_categories,
        "tasks": rows,
    }

    out_json = run_root / "table1_full_cache_progress.json"
    out_md = run_root / "table1_full_cache_progress.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Table 1 Full Cache Progress",
        "",
        f"Complete tasks: {summary['num_complete_tasks']} / {len(TASKS)}",
        f"Partial tasks: {summary['num_partial_tasks']}",
        f"Missing tasks: {summary['num_missing_tasks']}",
        "",
        "## Category Scores",
        "",
        "| Category | Score | Status |",
        "| --- | ---: | --- |",
    ]
    for category in ["SingleQA", "MultiQA", "Summarization", "Few shot", "Synthetic", "Code"]:
        score = category_scores.get(category)
        status = "complete" if category in completed_categories else "partial" if score is not None else "missing"
        lines.append(f"| {category} | {score:.2f} | {status} |" if score is not None else f"| {category} |  | {status} |")
    lines.extend(["", "## Tasks", "", "| Category | Dataset | Records | Score | Status |", "| --- | --- | ---: | ---: | --- |"])
    for row in rows:
        score = row["longbench_score"]
        score_text = f"{score:.2f}" if score is not None else ""
        lines.append(
            f"| {row['category']} | `{row['dataset']}` | {row['records']}/{row['expected_examples']} | {score_text} | {row['status']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "md": str(out_md), "complete": summary["num_complete_tasks"]}, indent=2))


if __name__ == "__main__":
    main()
