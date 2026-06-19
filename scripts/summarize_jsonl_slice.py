#!/usr/bin/env python3
"""Summarize a JSONL subset by dataset indexes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.longbench_metrics import category_for_dataset, normalize_dataset_name, score_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--recompute-longbench-score", action="store_true")
    args = parser.parse_args()

    total = 0
    correct = 0
    scores = []
    dataset_scores = defaultdict(list)
    category_scores = defaultdict(list)
    prompt_tokens = []
    cache_storage_ratios = []
    path = Path(args.jsonl)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["index"])
            if idx < args.start_index or idx >= args.end_index:
                continue
            total += 1
            correct += int(bool(row.get("contains_answer")))
            if row.get("prompt_tokens") is not None:
                prompt_tokens.append(float(row["prompt_tokens"]))
            if row.get("cache_storage_ratio") is not None:
                cache_storage_ratios.append(float(row["cache_storage_ratio"]))
            dataset_name = normalize_dataset_name(row.get("longbench_dataset") or row.get("task") or row.get("dataset"))
            if args.recompute_longbench_score:
                score = score_prediction(dataset_name, row.get("prediction") or "", row.get("answers") or [], row.get("all_classes"))
            else:
                score = row.get("longbench_score")
            if score is not None:
                score = float(score)
                scores.append(score)
                dataset_scores[dataset_name].append(score)
                category = row.get("longbench_category") or category_for_dataset(dataset_name)
                if category:
                    category_scores[category].append(score)

    summary = {
        "path": str(path),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "num_examples": total,
        "contains_answer_accuracy": correct / total if total else None,
        "longbench_score": 100 * sum(scores) / len(scores) if scores else None,
        "longbench_dataset_scores": {
            name: 100 * sum(values) / len(values) for name, values in sorted(dataset_scores.items())
        },
        "longbench_category_scores": {
            name: 100 * sum(values) / len(values) for name, values in sorted(category_scores.items())
        },
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
        "avg_cache_storage_ratio": sum(cache_storage_ratios) / len(cache_storage_ratios) if cache_storage_ratios else None,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
