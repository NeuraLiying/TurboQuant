#!/usr/bin/env python3
"""Benchmark TurboQuant quantization time for Table 2-style reporting."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
import yaml
from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant import TurboQuantMSE


DBPEDIA_COLUMNS = {
    1536: ("dbpedia_openai3_1536", "text-embedding-3-large-1536-embedding"),
    3072: ("dbpedia_openai3_3072", "text-embedding-3-large-3072-embedding"),
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_vectors_from_cache(paths: dict, dimension: int, num_vectors: int) -> torch.Tensor:
    dataset_key, column = DBPEDIA_COLUMNS[dimension]
    data_cfg = paths["datasets"][dataset_key]
    arrow_files = sorted(glob.glob(data_cfg["arrow_glob"]))
    vectors = []
    for arrow_path in arrow_files:
        dataset = Dataset.from_file(arrow_path)
        for row in dataset:
            vectors.append(row[column])
            if len(vectors) >= num_vectors:
                return torch.tensor(vectors, dtype=torch.float32)
    raise RuntimeError(f"not enough cached vectors for d={dimension}")


def random_unit_vectors(num_vectors: int, dimension: int, *, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(num_vectors, dimension, generator=gen)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--dimensions", type=int, nargs="+", default=[200, 1536, 3072])
    parser.add_argument("--num-vectors", type=int, default=2048)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--codebook-grid-size", type=int, default=10_001)
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "reproduce/runs/table2_quantization_time_smoke.json"))
    parser.add_argument("--output-csv", default=str(PROJECT_ROOT / "reproduce/runs/table2_quantization_time_smoke.csv"))
    args = parser.parse_args()

    paths = load_yaml(Path(args.paths))
    device = torch.device(args.device)
    rows = []
    for dimension in args.dimensions:
        if dimension in DBPEDIA_COLUMNS:
            x = load_vectors_from_cache(paths, dimension, args.num_vectors)
            source = "dbpedia_cache"
        else:
            x = random_unit_vectors(args.num_vectors, dimension, seed=args.seed)
            source = "random_unit"
        x = x.to(device)
        quantizer = TurboQuantMSE(
            dimension,
            args.bits,
            seed=args.seed,
            device=device,
            dtype=torch.float32,
            codebook_grid_size=args.codebook_grid_size,
        )
        for _ in range(args.warmup):
            quantizer.quantize(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
            quantizer.quantize(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(perf_counter() - started)
        rows.append(
            {
                "dimension": dimension,
                "bits": args.bits,
                "num_vectors": args.num_vectors,
                "source": source,
                "device": args.device,
                "mean_seconds": sum(timings) / len(timings),
                "min_seconds": min(timings),
                "max_seconds": max(timings),
                "repeats": args.repeats,
                "warmup": args.warmup,
            }
        )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(output_json), "csv": str(output_csv), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
