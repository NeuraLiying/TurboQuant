#!/usr/bin/env python3
"""Plot DBpedia TurboQuant recall curves for Figure 5 reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", default="reproduce/runs/figure5_turboquant_dbpedia_fullscale.csv", nargs="?")
    parser.add_argument("--output-png", default="reproduce/runs/figure5_turboquant_dbpedia_fullscale.png")
    parser.add_argument("--output-pdf", default="reproduce/runs/figure5_turboquant_dbpedia_fullscale.pdf")
    parser.add_argument("--output-json", default="reproduce/runs/figure5_turboquant_dbpedia_fullscale_plot.json")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    data = data.sort_values(["dimension", "bits", "topk"])

    dimensions = sorted(data["dimension"].unique())
    fig, axes = plt.subplots(1, len(dimensions), figsize=(5.2 * len(dimensions), 3.8), sharey=True)
    if len(dimensions) == 1:
        axes = [axes]

    colors = {2: "#1f77b4", 4: "#d62728"}
    markers = {2: "o", 4: "s"}
    for axis, dimension in zip(axes, dimensions):
        subset = data[data["dimension"] == dimension]
        for bits in sorted(subset["bits"].unique()):
            curve = subset[subset["bits"] == bits]
            axis.plot(
                curve["topk"],
                curve["recall_1_at_k"],
                marker=markers.get(int(bits), "o"),
                color=colors.get(int(bits)),
                linewidth=2.0,
                label=f"TurboQuant {int(bits)} bits",
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8, 16, 32, 64])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.set_ylim(0.84, 1.01)
        axis.set_xlabel("Top-k")
        axis.set_title(f"OpenAI3 - d={int(dimension)}")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Recall 1@k")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.tight_layout()

    output_png = Path(args.output_png)
    output_pdf = Path(args.output_pdf)
    output_json = Path(args.output_json)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)
    output_json.write_text(
        json.dumps(
            {
                "source_csv": args.csv,
                "output_png": str(output_png),
                "output_pdf": str(output_pdf),
                "dimensions": [int(value) for value in dimensions],
                "note": "DBpedia/OpenAI3 panels only; GloVe panel is omitted because local GloVe data is unavailable.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"png": str(output_png), "pdf": str(output_pdf), "json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
