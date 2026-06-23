#!/usr/bin/env python3
"""Summarize CacheGen repeated-context shared-prefix shard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dedupe_key(row: dict[str, Any]) -> tuple[str, str]:
    source_dataset = str(row.get("source_dataset_key") or row.get("task") or "")
    if row.get("source_index") is not None:
        return source_dataset, str(row["source_index"])
    return source_dataset, str(row.get("group_id"))


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def summarize_rows(rows: list[dict[str, Any]], *, expected_groups: int | None) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (str(row.get("source_dataset_key") or ""), int(row.get("source_index") or -1)))
    def sum_int(scoped_rows: list[dict[str, Any]], key: str) -> int:
        return sum(int(row.get(key) or 0) for row in scoped_rows)

    def scoped_summary(scoped_rows: list[dict[str, Any]]) -> dict[str, Any]:
        shared_total = sum_int(scoped_rows, "shared_total_storage_nbytes")
        fp16_total = sum_int(scoped_rows, "fp16_shared_total_storage_nbytes")
        cachegen_prefix = sum_int(scoped_rows, "cachegen_prefix_bytes")
        fp16_prefix = sum_int(scoped_rows, "fp16_prefix_bytes")
        return {
            "groups": len(scoped_rows),
            "records": sum_int(scoped_rows, "records"),
            "shared_prefix_table_storage_nbytes": cachegen_prefix,
            "fp16_prefix_table_storage_nbytes": fp16_prefix,
            "request_storage_nbytes": sum_int(scoped_rows, "request_storage_nbytes"),
            "shared_total_storage_nbytes": shared_total,
            "fp16_shared_total_storage_nbytes": fp16_total,
            "shared_total_savings_vs_fp16_shared": 1.0 - shared_total / fp16_total if fp16_total else None,
            "prefix_table_savings_vs_fp16_prefix": 1.0 - cachegen_prefix / fp16_prefix if fp16_prefix else None,
            "mean_prefix_storage_saving": mean(scoped_rows, "prefix_storage_saving"),
            "mean_prefill_s": mean(scoped_rows, "prefill_s"),
            "mean_encode_s": mean(scoped_rows, "encode_s"),
            "mean_decode_s": mean(scoped_rows, "decode_s"),
            "mean_rel_l2": mean(scoped_rows, "rel_l2"),
            "max_rel_l2": max((float(row["rel_l2"]) for row in scoped_rows if row.get("rel_l2") is not None), default=None),
            "mean_max_abs": mean(scoped_rows, "max_abs"),
        }

    duplicate_count = 0
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = dedupe_key(row)
        if key in unique:
            duplicate_count += 1
            continue
        unique[key] = row

    unique_rows = list(unique.values())
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in unique_rows:
        by_task.setdefault(str(row.get("task") or row.get("source_dataset_key") or ""), []).append(row)

    summary = scoped_summary(unique_rows)
    summary.update(
        {
            "expected_groups": expected_groups,
            "complete": expected_groups is None or len(unique_rows) == expected_groups,
            "duplicate_rows_skipped": duplicate_count,
            "input_rows": len(rows),
            "task_summaries": {
                task: scoped_summary(task_rows)
                for task, task_rows in sorted(by_task.items())
            },
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected-groups", type=int, default=600)
    args = parser.parse_args()

    rows = []
    for path_text in args.jsonl:
        rows.extend(load_jsonl(Path(path_text)))
    summary = summarize_rows(rows, expected_groups=args.expected_groups)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
