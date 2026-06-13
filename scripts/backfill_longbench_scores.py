#!/usr/bin/env python3
"""Backfill LongBench-style scores into existing generation JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.longbench_metrics import category_for_dataset, score_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.jsonl)
    output_path = Path(args.output) if args.output else input_path
    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dataset_name = row.get("longbench_dataset") or row.get("task") or row.get("dataset")
            row["longbench_dataset"] = dataset_name[:-2] if isinstance(dataset_name, str) and dataset_name.endswith("_e") else dataset_name
            row["longbench_category"] = category_for_dataset(row["longbench_dataset"])
            row["longbench_score"] = score_prediction(
                row["longbench_dataset"],
                row["prediction"],
                row["answers"],
                row.get("all_classes"),
            )
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input": str(input_path), "output": str(output_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
