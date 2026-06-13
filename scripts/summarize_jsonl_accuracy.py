#!/usr/bin/env python3
"""Summarize JSONL records produced by LongBench evaluation scripts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.longbench_metrics import TABLE1_CATEGORIES, category_for_dataset, normalize_dataset_name, score_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--recompute-longbench-score",
        action="store_true",
        help="Ignore stored longbench_score values and score saved predictions with the current metric implementation.",
    )
    args = parser.parse_args()

    path = Path(args.jsonl)
    total = 0
    correct = 0
    prompt_tokens = []
    generated_tokens = []
    latencies = []
    cache_storage_ratios = []
    cache_storage_nbytes = []
    cache_materialized_nbytes = []
    longbench_scores = []
    dataset_scores = defaultdict(list)
    category_scores = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            correct += int(bool(row.get("contains_answer")))
            if "prompt_tokens" in row:
                prompt_tokens.append(row["prompt_tokens"])
            if "generated_tokens" in row:
                generated_tokens.append(row["generated_tokens"])
            if "latency_seconds" in row:
                latencies.append(row["latency_seconds"])
            if row.get("cache_storage_ratio") is not None:
                cache_storage_ratios.append(row["cache_storage_ratio"])
            if row.get("cache_storage_nbytes") is not None:
                cache_storage_nbytes.append(row["cache_storage_nbytes"])
            if row.get("cache_materialized_nbytes") is not None:
                cache_materialized_nbytes.append(row["cache_materialized_nbytes"])
            if args.recompute_longbench_score:
                dataset_name = normalize_dataset_name(row.get("longbench_dataset") or row.get("task") or row.get("dataset"))
                if not dataset_name or row.get("prediction") is None:
                    continue
                score = score_prediction(
                    dataset_name,
                    row["prediction"],
                    row.get("answers") or [],
                    row.get("all_classes"),
                )
            elif row.get("longbench_score") is not None:
                score = float(row["longbench_score"])
            else:
                continue
            if score is not None:
                longbench_scores.append(score)
                dataset_name = normalize_dataset_name(row.get("longbench_dataset") or row.get("task") or row.get("dataset"))
                if dataset_name:
                    dataset_scores[dataset_name].append(score)
                    category = row.get("longbench_category") or category_for_dataset(dataset_name)
                    if category:
                        category_scores[category].append(score)

    summary = {
        "path": str(path),
        "num_examples": total,
        "contains_answer_accuracy": correct / total if total else 0.0,
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
        "avg_generated_tokens": sum(generated_tokens) / len(generated_tokens) if generated_tokens else None,
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "avg_cache_storage_ratio": sum(cache_storage_ratios) / len(cache_storage_ratios) if cache_storage_ratios else None,
        "avg_cache_storage_nbytes": sum(cache_storage_nbytes) / len(cache_storage_nbytes) if cache_storage_nbytes else None,
        "avg_cache_materialized_nbytes": (
            sum(cache_materialized_nbytes) / len(cache_materialized_nbytes) if cache_materialized_nbytes else None
        ),
        "longbench_score": 100 * sum(longbench_scores) / len(longbench_scores) if longbench_scores else None,
        "longbench_dataset_scores": {
            name: 100 * sum(scores) / len(scores) for name, scores in sorted(dataset_scores.items())
        },
        "longbench_category_scores": {
            name: 100 * sum(scores) / len(scores) for name, scores in sorted(category_scores.items())
        },
        "table1_category_coverage": {
            category: {
                "available": [dataset for dataset in datasets if dataset in dataset_scores],
                "missing": [dataset for dataset in datasets if dataset not in dataset_scores],
            }
            for category, datasets in TABLE1_CATEGORIES.items()
        },
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
