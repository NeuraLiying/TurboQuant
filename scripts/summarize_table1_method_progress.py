#!/usr/bin/env python3
"""Summarize LongBench-V1 Table 1 progress for one local method stem.

This is the method-general version of the Full Cache progress helper. It only
uses complete task files for category scores, so resumable partial JSONL files
cannot accidentally become official Table 1 numbers.
"""

from __future__ import annotations

import argparse
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

CATEGORIES = {
    "SingleQA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "MultiQA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_retrieval_en", "passage_count"],
    "Code": ["lcc", "repobench-p"],
}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_score(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    score = data.get("longbench_score")
    return float(score) if score is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="reproduce/runs/table1_official")
    parser.add_argument("--method-stem", required=True, help="File stem between dataset and all, e.g. full_cache or turboquant_2_5bit")
    parser.add_argument("--title", default=None)
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = []
    by_category: dict[str, list[float]] = {}
    for category, dataset, expected in TASKS:
        stem = f"longbench_{dataset}_{args.method_stem}_all"
        jsonl_path = run_root / f"{stem}.jsonl"
        aggregate_path = run_root / f"{stem}.aggregate.json"
        records = count_jsonl(jsonl_path)
        complete = records == expected
        score = load_score(aggregate_path) if complete else None
        if score is not None:
            by_category.setdefault(category, []).append(score)
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

    completed_categories = {
        category
        for category, datasets in CATEGORIES.items()
        if all(row["status"] == "complete" for row in rows if row["dataset"] in datasets)
    }
    category_scores = {name: sum(values) / len(values) for name, values in by_category.items()}
    completed_category_scores = {k: v for k, v in category_scores.items() if k in completed_categories}
    average_completed_categories = (
        sum(completed_category_scores.values()) / len(completed_category_scores) if completed_category_scores else None
    )

    summary = {
        "run_root": str(run_root),
        "method_stem": args.method_stem,
        "num_complete_tasks": sum(1 for row in rows if row["status"] == "complete"),
        "num_partial_tasks": sum(1 for row in rows if row["status"] == "partial"),
        "num_missing_tasks": sum(1 for row in rows if row["status"] == "missing"),
        "category_scores_available_subsets": category_scores,
        "completed_category_scores": completed_category_scores,
        "average_completed_categories": average_completed_categories,
        "tasks": rows,
    }

    output_stem = args.output_stem or f"table1_{args.method_stem}_progress"
    out_json = run_root / f"{output_stem}.json"
    out_md = run_root / f"{output_stem}.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    title = args.title or f"Table 1 {args.method_stem} Progress"
    lines = [
        f"# {title}",
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
    for category in CATEGORIES:
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
