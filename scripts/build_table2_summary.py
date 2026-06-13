#!/usr/bin/env python3
"""Build a Table 2-style quantization time summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


PAPER_TURBOQUANT = {200: 0.0007, 1536: 0.0013, 3072: 0.0021}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", default="reproduce/runs/table2_quantization_time_smoke.csv", nargs="?")
    parser.add_argument("--output-prefix", default="reproduce/runs/table2_quantization_time_summary")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    rows = []
    for _, row in data.iterrows():
        dimension = int(row["dimension"])
        local = float(row["mean_seconds"])
        paper = PAPER_TURBOQUANT.get(dimension)
        rows.append(
            {
                "dimension": dimension,
                "source": row["source"],
                "num_vectors": int(row["num_vectors"]),
                "local_mean_seconds": local,
                "paper_turboquant_seconds": paper,
                "local_minus_paper_seconds": None if paper is None else local - paper,
                "local_over_paper": None if paper is None else local / paper,
            }
        )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Table 2 TurboQuant Quantization Time Summary",
        "",
        "| d | Source | Vectors | Local mean seconds | Paper seconds | Local / paper |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        ratio = "" if row["local_over_paper"] is None else f"{row['local_over_paper']:.2f}"
        paper = "" if row["paper_turboquant_seconds"] is None else f"{row['paper_turboquant_seconds']:.6f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dimension"]),
                    row["source"],
                    str(row["num_vectors"]),
                    f"{row['local_mean_seconds']:.6f}",
                    paper,
                    ratio,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Local timing measures `TurboQuantMSE.quantize()` after warmup. It excludes codebook construction, rotation construction, data loading, and dequantization.",
            "The d=200 local row uses random unit vectors because local GloVe 200d data is not available.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
