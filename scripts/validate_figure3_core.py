#!/usr/bin/env python3
"""Small Figure 3-style validation for the TurboQuant core implementation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant import TurboQuantMSE, TurboQuantProd


def unit_random(shape: tuple[int, int], *, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    x = torch.randn(shape, generator=generator, device=device)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--num-vectors", type=int, default=512)
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--codebook-grid-size", type=int, default=50_001)
    parser.add_argument("--output", default="reproduce/runs/figure3_core_smoke.csv")
    parser.add_argument("--summary", default="reproduce/runs/figure3_core_smoke_summary.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []

    for seed in args.seeds:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = unit_random((args.num_vectors, args.dimension), generator=gen, device=device)
        y = unit_random((args.num_vectors, args.dimension), generator=gen, device=device)
        exact_ip = (x * y).sum(dim=-1)

        for bits in args.bits:
            mse_q = TurboQuantMSE(
                args.dimension,
                bits,
                seed=seed,
                device=device,
                codebook_grid_size=args.codebook_grid_size,
            )
            x_mse = mse_q(x)
            mse_error = torch.sum((x - x_mse) ** 2, dim=-1)
            mse_ip_error = exact_ip - (x_mse * y).sum(dim=-1)

            prod_q = TurboQuantProd(
                args.dimension,
                bits,
                seed=seed,
                device=device,
                codebook_grid_size=args.codebook_grid_size,
            )
            x_prod = prod_q(x)
            prod_ip_error = exact_ip - (x_prod * y).sum(dim=-1)

            rows.append(
                {
                    "seed": seed,
                    "dimension": args.dimension,
                    "num_vectors": args.num_vectors,
                    "bits": bits,
                    "mse_mean": float(mse_error.mean().cpu()),
                    "mse_std": float(mse_error.std(unbiased=False).cpu()),
                    "mse_ip_error_mean": float(mse_ip_error.mean().cpu()),
                    "mse_ip_error_var": float(mse_ip_error.var(unbiased=False).cpu()),
                    "prod_ip_error_mean": float(prod_ip_error.mean().cpu()),
                    "prod_ip_error_var": float(prod_ip_error.var(unbiased=False).cpu()),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for bits in args.bits:
        selected = [row for row in rows if row["bits"] == bits]
        summary[str(bits)] = {
            "mse_mean": sum(row["mse_mean"] for row in selected) / len(selected),
            "mse_ip_error_mean": sum(row["mse_ip_error_mean"] for row in selected) / len(selected),
            "prod_ip_error_mean": sum(row["prod_ip_error_mean"] for row in selected) / len(selected),
            "prod_ip_error_var": sum(row["prod_ip_error_var"] for row in selected) / len(selected),
        }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
