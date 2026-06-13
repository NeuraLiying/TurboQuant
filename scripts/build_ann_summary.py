#!/usr/bin/env python3
"""Build a compact ANN recall summary from TurboQuant JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for row in json.loads(path.read_text(encoding="utf-8")):
            row = dict(row)
            row["source"] = str(path)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+")
    parser.add_argument("--output-prefix", default="reproduce/runs/figure5_turboquant_dbpedia")
    args = parser.parse_args()

    rows = load_rows([Path(path) for path in args.json])
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = [
        "dataset_key",
        "dimension",
        "num_database",
        "num_queries",
        "bits",
        "topk",
        "recall_1_at_k",
        "quantize_seconds_mean",
        "source",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_key"], row["dimension"], row["bits"])].append(row)

    lines = [
        "# Figure 5 TurboQuant DBpedia Summary",
        "",
        "Recall is `1@k`: whether exact top-1 appears in approximate top-k.",
        "",
    ]
    for (dataset_key, dimension, bits), group_rows in sorted(grouped.items()):
        group_rows = sorted(group_rows, key=lambda item: item["topk"])
        settings = group_rows[0]
        lines.append(f"## {dataset_key}, d={dimension}, {bits}-bit")
        lines.append("")
        lines.append(f"- database: {settings['num_database']}")
        lines.append(f"- queries: {settings['num_queries']}")
        lines.append(f"- quantize_seconds_mean: {settings['quantize_seconds_mean']:.6f}")
        lines.append("")
        lines.append("| k | recall 1@k |")
        lines.append("| ---: | ---: |")
        for row in group_rows:
            lines.append(f"| {row['topk']} | {row['recall_1_at_k']:.4f} |")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
