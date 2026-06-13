#!/usr/bin/env python3
"""Summarize TurboQuant KV compression metadata from JSONL runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def extend_counter(counter: Counter, values: list[Any] | None) -> None:
    for value in values or []:
        counter[str(value)] += 1


def summarize_run(label: str, path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    rows_with_summary = [row for row in rows if row.get("cache_compression_summary")]
    effective_bits = []
    storage_ratios = []
    outlier_policy = Counter()
    outlier_counts = Counter()
    regular_bits = Counter()
    outlier_bits = Counter()
    segment_counts = []
    packed_segments = []
    raw_segments = []

    for row in rows_with_summary:
        summary = row["cache_compression_summary"]
        if summary.get("avg_effective_index_bits") is not None:
            effective_bits.append(float(summary["avg_effective_index_bits"]))
        if row.get("cache_storage_ratio") is not None:
            storage_ratios.append(float(row["cache_storage_ratio"]))
        if summary.get("outlier_policy") is not None:
            outlier_policy[str(summary["outlier_policy"])] += 1
        extend_counter(outlier_counts, summary.get("outlier_counts"))
        extend_counter(regular_bits, summary.get("regular_bits"))
        extend_counter(outlier_bits, summary.get("outlier_bits"))
        if summary.get("num_segments") is not None:
            segment_counts.append(float(summary["num_segments"]))
        if summary.get("num_packed_segments") is not None:
            packed_segments.append(float(summary["num_packed_segments"]))
        if summary.get("num_raw_segments") is not None:
            raw_segments.append(float(summary["num_raw_segments"]))

    return {
        "label": label,
        "source": str(path),
        "num_records": len(rows),
        "num_records_with_cache_summary": len(rows_with_summary),
        "avg_effective_index_bits": mean(effective_bits),
        "min_effective_index_bits": min(effective_bits) if effective_bits else None,
        "max_effective_index_bits": max(effective_bits) if effective_bits else None,
        "avg_cache_storage_ratio": mean(storage_ratios),
        "outlier_policy_counts": dict(sorted(outlier_policy.items())),
        "outlier_count_values": dict(sorted(outlier_counts.items(), key=lambda item: float(item[0]))),
        "regular_bit_values": dict(sorted(regular_bits.items(), key=lambda item: float(item[0]))),
        "outlier_bit_values": dict(sorted(outlier_bits.items(), key=lambda item: float(item[0]))),
        "avg_num_segments": mean(segment_counts),
        "avg_num_packed_segments": mean(packed_segments),
        "avg_num_raw_segments": mean(raw_segments),
    }


def flatten_for_csv(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        row = {
            key: value
            for key, value in summary.items()
            if key
            not in {
                "outlier_policy_counts",
                "outlier_count_values",
                "regular_bit_values",
                "outlier_bit_values",
            }
        }
        row["outlier_policy_counts"] = json.dumps(summary["outlier_policy_counts"], sort_keys=True)
        row["outlier_count_values"] = json.dumps(summary["outlier_count_values"], sort_keys=True)
        row["regular_bit_values"] = json.dumps(summary["regular_bit_values"], sort_keys=True)
        row["outlier_bit_values"] = json.dumps(summary["outlier_bit_values"], sort_keys=True)
        rows.append(row)
    return rows


def format_optional(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(summaries: list[dict[str, Any]], path: Path, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| Run | Records | With Summary | Effective Bits | Cache Ratio | Outlier Policy | Outlier Counts | Regular Bits | Outlier Bits |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["label"],
                    str(summary["num_records"]),
                    str(summary["num_records_with_cache_summary"]),
                    format_optional(summary["avg_effective_index_bits"], 4),
                    format_optional(summary["avg_cache_storage_ratio"], 4),
                    "`" + json.dumps(summary["outlier_policy_counts"], sort_keys=True) + "`",
                    "`" + json.dumps(summary["outlier_count_values"], sort_keys=True) + "`",
                    "`" + json.dumps(summary["regular_bit_values"], sort_keys=True) + "`",
                    "`" + json.dumps(summary["outlier_bit_values"], sort_keys=True) + "`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Sources", ""])
    for summary in summaries:
        lines.append(f"- {summary['label']}: `{summary['source']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "JSONL"), required=True)
    parser.add_argument("--output-prefix", default="reproduce/runs/kv_compression_summary")
    parser.add_argument("--title", default="TurboQuant KV Compression Summary")
    args = parser.parse_args()

    summaries = [summarize_run(label, Path(path)) for label, path in args.run]
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")

    json_path.write_text(json.dumps({"runs": summaries}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = flatten_for_csv(summaries)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(summaries, md_path, args.title)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
