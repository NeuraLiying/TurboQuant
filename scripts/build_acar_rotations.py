#!/usr/bin/env python3
"""Build orthogonal calibrated TurboQuant rotation matrices from LongBench prompts."""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil, floor
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.longbench.run_full_cache_eval import apply_chat_template_if_needed, build_legacy_prompt, load_yaml, truncate_middle
from turboquant.codebook import lloyd_max_codebook
from turboquant.core import hadamard_orthogonal
from turboquant.longbench_prompts import build_longbench_prompt, dataset_name_from_row


def load_dataset_from_config(data_cfg: dict) -> Dataset:
    arrow_files = data_cfg.get("arrow_files") or []
    parquet_files = data_cfg.get("parquet_files") or []
    if arrow_files:
        datasets = [Dataset.from_file(path) for path in arrow_files]
    elif parquet_files:
        datasets = [Dataset.from_parquet(path) for path in parquet_files]
    else:
        raise ValueError("dataset config must contain either arrow_files or parquet_files")
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def iter_prompts(args, tokenizer, data_cfg, dataset):
    for row in dataset:
        dataset_name = dataset_name_from_row(row, data_cfg.get("requested_config") or data_cfg["config"])
        if args.prompt_mode == "longbench":
            prompt = build_longbench_prompt(row, dataset_name)
        else:
            prompt = build_legacy_prompt(row)
        prompt = apply_chat_template_if_needed(tokenizer, prompt, dataset_name, args.chat_template_mode)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        inputs["input_ids"] = truncate_middle(inputs["input_ids"], args.max_input_tokens)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = truncate_middle(inputs["attention_mask"], args.max_input_tokens)
        yield inputs


def collect_layer_vectors(model, inputs, max_vectors_per_layer: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    attention_mask = attention_mask.to(model.device) if attention_mask is not None else None
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(outputs.hidden_states[0], position_ids)

    key_rows: list[torch.Tensor] = []
    value_rows: list[torch.Tensor] = []
    for layer_idx, layer in enumerate(model.model.layers):
        hidden = outputs.hidden_states[layer_idx].detach()
        hidden_shape = (*hidden.shape[:-1], -1, layer.self_attn.head_dim)
        query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
        key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
        value = layer.self_attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        _, key = apply_rotary_pos_emb(query, key, cos, sin)
        key_flat = key.reshape(-1, key.shape[-1]).detach().to(torch.float32).cpu()
        value_flat = value.reshape(-1, value.shape[-1]).detach().to(torch.float32).cpu()
        if key_flat.shape[0] > max_vectors_per_layer:
            stride = max(1, key_flat.shape[0] // max_vectors_per_layer)
            key_flat = key_flat[::stride][:max_vectors_per_layer]
            value_flat = value_flat[::stride][:max_vectors_per_layer]
        key_rows.append(key_flat)
        value_rows.append(value_flat)
    return key_rows, value_rows


def second_hadamard_rotation(vectors: torch.Tensor, eps: float) -> torch.Tensor:
    unit = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
    moment = unit.T @ unit / max(unit.shape[0], 1)
    eigvals, eigvecs = torch.linalg.eigh(moment.to(torch.float64))
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(eps)
    eigvecs = eigvecs[:, order]
    hadamard = hadamard_orthogonal(vectors.shape[-1], device=vectors.device, dtype=torch.float64, seed=None)
    rotation = hadamard @ eigvecs.T
    return rotation.to(torch.float32)


def resolve_fractional_bits(bits: float, dimension: int) -> tuple[int, int, int]:
    lower_bits = max(1, floor(bits))
    upper_bits = ceil(bits)
    if lower_bits == upper_bits:
        return lower_bits, upper_bits, 0
    outlier_count = int(round((bits - lower_bits) * dimension))
    return lower_bits, upper_bits, max(0, min(dimension, outlier_count))


def nearest_centroids(values: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    distances = torch.abs(values.unsqueeze(-1) - centroids)
    indices = torch.argmin(distances, dim=-1)
    return centroids[indices]


def fractional_rotation_mse(
    vectors: torch.Tensor,
    transform: torch.Tensor,
    *,
    bits: float,
    codebook_grid_size: int,
) -> float:
    dimension = vectors.shape[-1]
    regular_bits, outlier_bits, outlier_count = resolve_fractional_bits(bits, dimension)
    regular_centroids = torch.tensor(
        lloyd_max_codebook(dimension, regular_bits, grid_size=codebook_grid_size).centroids,
        device=vectors.device,
        dtype=vectors.dtype,
    )
    outlier_centroids = torch.tensor(
        lloyd_max_codebook(dimension, outlier_bits, grid_size=codebook_grid_size).centroids,
        device=vectors.device,
        dtype=vectors.dtype,
    )
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(torch.finfo(vectors.dtype).eps)
    unit = vectors / norms
    rotated = unit @ transform.T
    regular_hat = nearest_centroids(rotated, regular_centroids)
    if outlier_count > 0:
        outlier_hat = nearest_centroids(rotated, outlier_centroids)
        gain = (rotated - regular_hat).square() - (rotated - outlier_hat).square()
        outlier_indices = torch.topk(gain.mean(dim=0), k=outlier_count, largest=True, sorted=False).indices
        rotated_hat = regular_hat.clone()
        rotated_hat[:, outlier_indices] = outlier_hat[:, outlier_indices]
    else:
        rotated_hat = regular_hat
    x_hat_unit = rotated_hat @ transform
    numerator = (vectors * x_hat_unit).sum(dim=-1)
    denominator = x_hat_unit.square().sum(dim=-1).clamp_min(torch.finfo(vectors.dtype).eps)
    scales = numerator / denominator
    reconstructed = x_hat_unit * scales.unsqueeze(-1)
    return float((vectors - reconstructed).square().sum(dim=-1).mean().item())


def sign_balanced_hadamard_rotation(
    vectors: torch.Tensor,
    *,
    bits: list[float],
    bank_size: int,
    seed: int,
    codebook_grid_size: int,
    eval_device: torch.device,
) -> tuple[torch.Tensor, dict]:
    vectors = vectors.to(device=eval_device, dtype=torch.float32)
    dimension = vectors.shape[-1]
    hadamard = hadamard_orthogonal(dimension, device=eval_device, dtype=torch.float32, seed=None)
    generator = torch.Generator(device=eval_device)
    generator.manual_seed(seed)

    best_transform: torch.Tensor | None = None
    best_score: float | None = None
    rows = []
    baseline_by_bit: dict[float, float] = {}
    for candidate_idx in range(bank_size):
        if candidate_idx == 0:
            signs = torch.ones(dimension, device=eval_device, dtype=torch.float32)
        else:
            signs = torch.randint(0, 2, (dimension,), generator=generator, device=eval_device, dtype=torch.int8)
            signs = signs.to(dtype=torch.float32).mul_(2).sub_(1)
        transform = hadamard * signs
        by_bit = {
            bit: fractional_rotation_mse(
                vectors,
                transform,
                bits=bit,
                codebook_grid_size=codebook_grid_size,
            )
            for bit in bits
        }
        if candidate_idx == 0:
            baseline_by_bit = dict(by_bit)
        score = sum(by_bit[bit] / max(baseline_by_bit[bit], torch.finfo(torch.float32).eps) for bit in bits) / len(bits)
        rows.append({"candidate": candidate_idx, "score": score, "mse_by_bit": by_bit})
        if best_score is None or score < best_score:
            best_score = score
            best_transform = transform.detach().cpu()
    if best_transform is None or best_score is None:
        raise RuntimeError("sign-balanced Hadamard search produced no candidate")
    return best_transform, {"best_score": best_score, "rows": rows}


def orthogonality_error(matrix: torch.Tensor) -> float:
    identity = torch.eye(matrix.shape[0], dtype=torch.float32)
    return float(torch.linalg.matrix_norm(matrix @ matrix.T - identity).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--dataset-key", default="longbench_2wikimqa")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-vectors-per-layer", type=int, default=512)
    parser.add_argument("--method", choices=["second_hadamard", "sign_balanced_hadamard"], default="second_hadamard")
    parser.add_argument("--objective-bits", type=float, nargs="+", default=[2.5, 3.5])
    parser.add_argument("--sign-bank-size", type=int, default=16)
    parser.add_argument("--codebook-grid-size", type=int, default=10_001)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-mode", choices=["longbench", "legacy"], default="longbench")
    parser.add_argument("--chat-template-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = load_yaml(Path(args.paths))
    cfg = load_yaml(Path(args.config))
    model_cfg = paths["models"][cfg["model_key"]]
    data_cfg = paths["datasets"][args.dataset_key]
    dataset = load_dataset_from_config(data_cfg)
    selected_indices = list(range(args.start_index, min(len(dataset), args.start_index + args.num_examples)))
    dataset = dataset.select(selected_indices)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["snapshot"], local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if cfg["hardware"].get("dtype") == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["snapshot"],
        local_files_only=True,
        torch_dtype=dtype,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.eval()

    num_layers = len(model.model.layers)
    key_rows = [[] for _ in range(num_layers)]
    value_rows = [[] for _ in range(num_layers)]
    for inputs in iter_prompts(args, tokenizer, data_cfg, dataset):
        keys, values = collect_layer_vectors(model, inputs, args.max_vectors_per_layer)
        for layer_idx in range(num_layers):
            key_rows[layer_idx].append(keys[layer_idx])
            value_rows[layer_idx].append(values[layer_idx])

    key_matrices: list[list[list[float]]] = []
    value_matrices: list[list[list[float]]] = []
    diagnostics = []
    eval_device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    for layer_idx in range(num_layers):
        if args.method == "second_hadamard":
            key_matrix = second_hadamard_rotation(torch.cat(key_rows[layer_idx], dim=0), args.eps)
            value_matrix = second_hadamard_rotation(torch.cat(value_rows[layer_idx], dim=0), args.eps)
            key_diag: dict = {}
            value_diag: dict = {}
        else:
            key_matrix, key_diag = sign_balanced_hadamard_rotation(
                torch.cat(key_rows[layer_idx], dim=0),
                bits=args.objective_bits,
                bank_size=args.sign_bank_size,
                seed=10_000 + layer_idx,
                codebook_grid_size=args.codebook_grid_size,
                eval_device=eval_device,
            )
            value_matrix, value_diag = sign_balanced_hadamard_rotation(
                torch.cat(value_rows[layer_idx], dim=0),
                bits=args.objective_bits,
                bank_size=args.sign_bank_size,
                seed=20_000 + layer_idx,
                codebook_grid_size=args.codebook_grid_size,
                eval_device=eval_device,
            )
        key_matrices.append(key_matrix.tolist())
        value_matrices.append(value_matrix.tolist())
        diagnostics.append(
            {
                "layer": layer_idx,
                "key_orthogonality_error": orthogonality_error(key_matrix),
                "value_orthogonality_error": orthogonality_error(value_matrix),
                "key_diagnostics": key_diag,
                "value_diagnostics": value_diag,
            }
        )

    output = {
        "config": {
            "dataset_key": args.dataset_key,
            "selected_indices": selected_indices,
            "max_input_tokens": args.max_input_tokens,
            "max_vectors_per_layer": args.max_vectors_per_layer,
            "method": args.method,
            "objective_bits": args.objective_bits,
            "sign_bank_size": args.sign_bank_size,
            "codebook_grid_size": args.codebook_grid_size,
        },
        "layer_key_rotation_matrices": key_matrices,
        "layer_value_rotation_matrices": value_matrices,
        "diagnostics": diagnostics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "max_orthogonality_error": max(max(row["key_orthogonality_error"], row["value_orthogonality_error"]) for row in diagnostics)}, indent=2))


if __name__ == "__main__":
    main()
