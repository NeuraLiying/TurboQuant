#!/usr/bin/env python3
"""Merge JSONL shards by their integer `index` field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(paths: list[Path]) -> tuple[dict[int, dict], dict[int, int]]:
    records: dict[int, dict] = {}
    counts: dict[int, int] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "index" not in record:
                    raise KeyError(f"{path}:{line_number} does not contain an `index` field")
                index = int(record["index"])
                records[index] = record
                counts[index] = counts.get(index, 0) + 1
    return records, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-start", type=int, default=None)
    parser.add_argument("--expected-end", type=int, default=None)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.inputs]
    records, counts = load_records(input_paths)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index in sorted(records):
            handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")

    duplicates = {index: count for index, count in sorted(counts.items()) if count > 1}
    missing: list[int] | None = None
    if args.expected_start is not None and args.expected_end is not None:
        present = set(records)
        missing = [index for index in range(args.expected_start, args.expected_end) if index not in present]

    print(
        json.dumps(
            {
                "output": str(output_path),
                "inputs": [str(path) for path in input_paths],
                "num_unique_records": len(records),
                "duplicates": duplicates,
                "missing": missing,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
