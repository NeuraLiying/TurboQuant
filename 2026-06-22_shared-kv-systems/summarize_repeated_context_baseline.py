#!/usr/bin/env python3
"""Summarize repeated-context baseline JSONL outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.longbench_metrics import category_for_dataset, normalize_dataset_name, score_prediction


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_row(row: dict[str, Any], recompute: bool) -> float | None:
    if not recompute and row.get("longbench_score") is not None:
        return float(row["longbench_score"])
    dataset = normalize_dataset_name(row.get("longbench_dataset") or row.get("task") or row.get("dataset"))
    prediction = row.get("prediction", row.get("pred"))
    if not dataset or prediction is None:
        return None
    return score_prediction(dataset, prediction, row.get("answers") or [], row.get("all_classes"))


def prefix_identity(row: dict[str, Any]) -> str:
    if row.get("repeat_group_id"):
        return str(row["repeat_group_id"])
    if row.get("shared_prefix_cache_entry_id"):
        return str(row["shared_prefix_cache_entry_id"])
    return str(row.get("index"))


def summarize(rows: list[dict[str, Any]], *, expected: int | None, recompute: bool) -> dict[str, Any]:
    task_rows: dict[str, list[dict[str, Any]]] = {}
    scores = []
    latencies = []
    for row in rows:
        dataset = normalize_dataset_name(row.get("longbench_dataset") or row.get("task") or row.get("dataset"))
        task_rows.setdefault(dataset, []).append(row)
        score = score_row(row, recompute)
        if score is not None:
            scores.append(score)
        if row.get("latency_seconds") is not None:
            latencies.append(float(row["latency_seconds"]))

    def storage_sum(key: str, scoped_rows: list[dict[str, Any]], *, unique_prefix: bool = False) -> int | None:
        values = []
        seen = set()
        for row in scoped_rows:
            if unique_prefix:
                ident = prefix_identity(row)
                if ident in seen:
                    continue
                seen.add(ident)
            if row.get(key) is not None:
                values.append(int(row[key]))
        return sum(values) if values else None

    task_summaries = []
    for task, scoped in sorted(task_rows.items()):
        task_scores = [score_row(row, recompute) for row in scoped]
        task_scores = [score for score in task_scores if score is not None]
        task_latencies = [float(row["latency_seconds"]) for row in scoped if row.get("latency_seconds") is not None]
        prefix_storage = storage_sum("shared_prefix_table_storage_nbytes", scoped, unique_prefix=True)
        prefix_materialized = storage_sum("shared_prefix_materialized_cache_nbytes", scoped, unique_prefix=True)
        request_storage = storage_sum("request_storage_nbytes", scoped)
        shared_total = (
            prefix_storage + request_storage
            if prefix_storage is not None and request_storage is not None
            else None
        )
        task_summaries.append(
            {
                "task": task,
                "records": len(scoped),
                "groups": len({str(row.get("repeat_group_id") or row.get("index")) for row in scoped}),
                "score": mean(task_scores),
                "score_percent": 100.0 * mean(task_scores) if task_scores else None,
                "avg_latency_seconds": mean(task_latencies),
                "shared_prefix_table_storage_nbytes": prefix_storage,
                "shared_prefix_materialized_cache_nbytes": prefix_materialized,
                "request_storage_nbytes": request_storage,
                "shared_total_storage_nbytes": shared_total,
                "category": category_for_dataset(task),
            }
        )

    prefix_storage = storage_sum("shared_prefix_table_storage_nbytes", rows, unique_prefix=True)
    prefix_materialized = storage_sum("shared_prefix_materialized_cache_nbytes", rows, unique_prefix=True)
    request_storage = storage_sum("request_storage_nbytes", rows)
    shared_total = (
        prefix_storage + request_storage
        if prefix_storage is not None and request_storage is not None
        else None
    )
    complete = expected is None or len(rows) == expected
    return {
        "records": len(rows),
        "expected": expected,
        "complete": complete,
        "groups": len({str(row.get("repeat_group_id") or row.get("index")) for row in rows}),
        "score": mean(scores),
        "score_percent": 100.0 * mean(scores) if scores else None,
        "avg_latency_seconds": mean(latencies),
        "shared_prefix_table_storage_nbytes": prefix_storage,
        "shared_prefix_materialized_cache_nbytes": prefix_materialized,
        "request_storage_nbytes": request_storage,
        "shared_total_storage_nbytes": shared_total,
        "task_summaries": task_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected", type=int, default=2400)
    parser.add_argument("--use-stored-score", action="store_true")
    args = parser.parse_args()

    rows = []
    for jsonl in args.jsonl:
        rows.extend(load_jsonl(Path(jsonl)))
    summary = summarize(rows, expected=args.expected, recompute=not args.use_stored_score)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
