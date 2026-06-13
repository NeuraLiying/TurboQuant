#!/usr/bin/env python3
"""TurboQuant ANN recall and quantization-time experiments."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant import TurboQuantMSE
from turboquant.core import MSEQuantized


EMBEDDING_COLUMNS = {
    "dbpedia_openai3_1536": "text-embedding-3-large-1536-embedding",
    "dbpedia_openai3_3072": "text-embedding-3-large-3072-embedding",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def arrow_files_for_dataset(data_cfg: dict) -> list[str]:
    if "arrow_files" in data_cfg:
        return data_cfg["arrow_files"]
    return sorted(glob.glob(data_cfg["arrow_glob"]))


def load_embeddings(
    data_cfg: dict,
    *,
    column: str,
    num_database: int,
    num_queries: int,
    shard_limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors: list[list[float]] = []
    needed = num_database + num_queries
    for arrow_path in arrow_files_for_dataset(data_cfg)[:shard_limit]:
        dataset = Dataset.from_file(arrow_path)
        for row in dataset:
            vectors.append(row[column])
            if len(vectors) >= needed:
                break
        if len(vectors) >= needed:
            break
    if len(vectors) < needed:
        raise RuntimeError(f"not enough vectors: needed {needed}, loaded {len(vectors)}")
    tensor = torch.tensor(vectors, dtype=torch.float32)
    return tensor[:num_database], tensor[num_database:needed]


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def topk_recall_1_at_k(exact_top1: torch.Tensor, approx_scores: torch.Tensor, topk: list[int]) -> dict[int, float]:
    max_k = max(topk)
    approx_topk = torch.topk(approx_scores, k=max_k, dim=1).indices
    recalls = {}
    for k in topk:
        hit = (approx_topk[:, :k] == exact_top1.unsqueeze(1)).any(dim=1).to(torch.float32)
        recalls[k] = float(hit.mean().item())
    return recalls


def quantize_in_chunks(
    quantizer: TurboQuantMSE,
    x: torch.Tensor,
    *,
    chunk_size: int,
    device: torch.device,
) -> tuple[MSEQuantized, float]:
    indices = []
    norms = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start : start + chunk_size]
        q = quantizer.quantize(chunk)
        indices.append(q.indices.detach().cpu())
        norms.append(q.norms.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    return MSEQuantized(indices=torch.cat(indices, dim=0), norms=torch.cat(norms, dim=0)), elapsed


def approximate_scores_from_quantized(
    quantizer: TurboQuantMSE,
    q: MSEQuantized,
    queries: torch.Tensor,
    *,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    score_chunks = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    for start in range(0, q.indices.shape[0], chunk_size):
        q_chunk = MSEQuantized(
            indices=q.indices[start : start + chunk_size].to(device),
            norms=q.norms[start : start + chunk_size].to(device),
        )
        database_hat = l2_normalize(quantizer.dequantize(q_chunk))
        score_chunks.append((queries @ database_hat.T).detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    return torch.cat(score_chunks, dim=1), elapsed


def run_one_bitwidth(
    database: torch.Tensor,
    queries: torch.Tensor,
    *,
    bits: int,
    topk: list[int],
    seed: int,
    device: torch.device,
    codebook_grid_size: int,
    repeats: int,
    quantize_chunk_size: int,
    score_chunk_size: int,
) -> tuple[dict[int, float], dict]:
    database = l2_normalize(database).to(device)
    queries = l2_normalize(queries).to(device)
    exact_scores = queries @ database.T
    exact_top1 = exact_scores.argmax(dim=1).detach().cpu()

    quantizer = TurboQuantMSE(
        database.shape[-1],
        bits,
        seed=seed,
        device=device,
        dtype=torch.float32,
        codebook_grid_size=codebook_grid_size,
    )
    # Warm up once so timings focus on the quantization call.
    quantizer.quantize(database[: min(8, database.shape[0])])

    timings = []
    q = None
    for _ in range(repeats):
        q, elapsed = quantize_in_chunks(
            quantizer,
            database,
            chunk_size=quantize_chunk_size,
            device=device,
        )
        timings.append(elapsed)
    assert q is not None

    approx_scores, dequant_seconds = approximate_scores_from_quantized(
        quantizer,
        q,
        queries,
        chunk_size=score_chunk_size,
        device=device,
    )
    recalls = topk_recall_1_at_k(exact_top1, approx_scores, topk)
    timing = {
        "quantize_seconds_mean": sum(timings) / len(timings),
        "quantize_seconds_min": min(timings),
        "quantize_seconds_max": max(timings),
        "dequantize_seconds": dequant_seconds,
        "num_repeats": repeats,
    }
    return recalls, timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--dataset-key", choices=sorted(EMBEDDING_COLUMNS), default="dbpedia_openai3_1536")
    parser.add_argument("--num-database", type=int, default=2048)
    parser.add_argument("--num-queries", type=int, default=128)
    parser.add_argument("--shard-limit", type=int, default=1)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--codebook-grid-size", type=int, default=10_001)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--quantize-chunk-size", type=int, default=8192)
    parser.add_argument("--score-chunk-size", type=int, default=8192)
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "reproduce/runs/ann_turboquant_smoke.json"))
    parser.add_argument("--output-csv", default=str(PROJECT_ROOT / "reproduce/runs/ann_turboquant_smoke.csv"))
    args = parser.parse_args()

    paths = load_yaml(Path(args.paths))
    data_cfg = paths["datasets"][args.dataset_key]
    column = EMBEDDING_COLUMNS[args.dataset_key]
    device = torch.device(args.device)
    database, queries = load_embeddings(
        data_cfg,
        column=column,
        num_database=args.num_database,
        num_queries=args.num_queries,
        shard_limit=args.shard_limit,
    )

    rows = []
    for bits in args.bits:
        recalls, timing = run_one_bitwidth(
            database,
            queries,
            bits=bits,
            topk=args.topk,
            seed=args.seed,
            device=device,
            codebook_grid_size=args.codebook_grid_size,
            repeats=args.timing_repeats,
            quantize_chunk_size=args.quantize_chunk_size,
            score_chunk_size=args.score_chunk_size,
        )
        for k, recall in recalls.items():
            rows.append(
                {
                    "dataset_key": args.dataset_key,
                    "dimension": int(database.shape[-1]),
                    "num_database": args.num_database,
                    "num_queries": args.num_queries,
                    "bits": bits,
                    "topk": k,
                    "recall_1_at_k": recall,
                    **timing,
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
