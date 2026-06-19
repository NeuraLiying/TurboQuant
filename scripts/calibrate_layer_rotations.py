#!/usr/bin/env python3
"""Calibrate fixed layer-wise TurboQuant rotation indices on LongBench prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.longbench.run_full_cache_eval import apply_chat_template_if_needed, build_legacy_prompt, load_yaml, truncate_middle
from turboquant.core import TurboQuantMSE
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


def collect_layer_vectors(
    model,
    inputs,
    max_vectors_per_layer: int,
    *,
    query_tokens: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
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
    attention_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for layer_idx, layer in enumerate(model.model.layers):
        hidden = outputs.hidden_states[layer_idx].detach()
        hidden_shape = (*hidden.shape[:-1], -1, layer.self_attn.head_dim)
        query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
        key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
        value = layer.self_attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key_flat = key.reshape(-1, key.shape[-1]).detach().to(torch.float32).cpu()
        value_flat = value.reshape(-1, value.shape[-1]).detach().to(torch.float32).cpu()
        if key_flat.shape[0] > max_vectors_per_layer:
            stride = max(1, key_flat.shape[0] // max_vectors_per_layer)
            key_flat = key_flat[::stride][:max_vectors_per_layer]
            value_flat = value_flat[::stride][:max_vectors_per_layer]
        key_rows.append(key_flat)
        value_rows.append(value_flat)
        q_keep = min(query_tokens, query.shape[-2])
        attention_rows.append(
            (
                query[..., -q_keep:, :].detach().to(torch.float32).cpu(),
                key.detach().to(torch.float32).cpu(),
                value.detach().to(torch.float32).cpu(),
            )
        )
    return key_rows, value_rows, attention_rows


def rotation_mse(vectors: torch.Tensor, *, bits: int, seed: int, grid_size: int, device: torch.device) -> float:
    vectors = vectors.to(device=device, dtype=torch.float32)
    quantizer = TurboQuantMSE(
        vectors.shape[-1],
        bits,
        seed=seed,
        device=device,
        dtype=torch.float32,
        codebook_grid_size=grid_size,
    )
    with torch.no_grad():
        quantized = quantizer.quantize(vectors)
        decoded = quantizer.dequantize(quantized)
    return float((vectors - decoded).square().sum(dim=-1).mean().item())


def quantize_vectors(vectors: torch.Tensor, *, bits: int, seed: int, grid_size: int, device: torch.device) -> torch.Tensor:
    original_shape = vectors.shape
    flat = vectors.reshape(-1, original_shape[-1]).to(device=device, dtype=torch.float32)
    quantizer = TurboQuantMSE(
        original_shape[-1],
        bits,
        seed=seed,
        device=device,
        dtype=torch.float32,
        codebook_grid_size=grid_size,
    )
    with torch.no_grad():
        decoded = quantizer.dequantize(quantizer.quantize(flat))
    return decoded.reshape(original_shape).cpu()


def attention_output_error(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    candidate_key: torch.Tensor | None = None,
    candidate_value: torch.Tensor | None = None,
) -> float:
    key_eval = candidate_key if candidate_key is not None else key
    value_eval = candidate_value if candidate_value is not None else value
    query = query.to(dtype=torch.float32)
    key = key.to(dtype=torch.float32)
    value = value.to(dtype=torch.float32)
    key_eval = key_eval.to(dtype=torch.float32)
    value_eval = value_eval.to(dtype=torch.float32)
    bsz, num_query_heads, query_len, head_dim = query.shape
    num_key_heads = key.shape[1]
    num_groups = num_query_heads // num_key_heads
    query_grouped = query.reshape(bsz, num_key_heads, num_groups, query_len, head_dim)
    scale = head_dim**-0.5
    original_scores = torch.einsum("bhgqd,bhsd->bhgqs", query_grouped, key) * scale
    candidate_scores = torch.einsum("bhgqd,bhsd->bhgqs", query_grouped, key_eval) * scale
    key_len = key.shape[-2]
    query_positions = torch.arange(query_len)
    key_positions = torch.arange(key_len)
    causal_mask = key_positions.view(1, 1, 1, 1, key_len) <= (
        (key_len - query_len + query_positions).view(1, 1, 1, query_len, 1)
    )
    original_probs = torch.softmax(original_scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
    candidate_probs = torch.softmax(candidate_scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
    original_output = torch.einsum("bhgqs,bhsd->bhgqd", original_probs, value)
    candidate_output = torch.einsum("bhgqs,bhsd->bhgqd", candidate_probs, value_eval)
    return float((original_output - candidate_output).square().mean().item())


def choose_rotation_by_reconstruction_mse(
    vectors: torch.Tensor,
    *,
    side: str,
    layer_idx: int,
    bits: list[int],
    rotation_bank_size: int,
    rotation_seed_stride: int,
    codebook_grid_size: int,
    device: torch.device,
) -> tuple[int, list[dict]]:
    per_rotation: dict[int, dict[int, float]] = {}
    baseline_by_bit: dict[int, float] = {}
    rows: list[dict] = []
    for rotation_idx in range(rotation_bank_size):
        seed = 1_000 * layer_idx + (0 if side == "key" else 500) + rotation_seed_stride * rotation_idx
        per_rotation[rotation_idx] = {}
        for bit in bits:
            mse = rotation_mse(
                vectors,
                bits=bit,
                seed=seed,
                grid_size=codebook_grid_size,
                device=device,
            )
            per_rotation[rotation_idx][bit] = mse
            if rotation_idx == 0:
                baseline_by_bit[bit] = mse
            rows.append(
                {
                    "layer": layer_idx,
                    "side": side,
                    "rotation_index": rotation_idx,
                    "bits": bit,
                    "objective": "mse",
                    "objective_error": mse,
                    "mse": mse,
                }
            )
    eps = torch.finfo(torch.float32).eps
    best_idx = min(
        range(rotation_bank_size),
        key=lambda rotation_idx: sum(
            per_rotation[rotation_idx][bit] / max(baseline_by_bit[bit], eps)
            for bit in bits
        )
        / len(bits),
    )
    return best_idx, rows


def choose_rotation_by_attention_output(
    attention_examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    side: str,
    layer_idx: int,
    bits: list[int],
    rotation_bank_size: int,
    rotation_seed_stride: int,
    codebook_grid_size: int,
    device: torch.device,
) -> tuple[int, list[dict]]:
    per_rotation: dict[int, dict[int, float]] = {}
    baseline_by_bit: dict[int, float] = {}
    rows: list[dict] = []
    for rotation_idx in range(rotation_bank_size):
        seed = 1_000 * layer_idx + (0 if side == "key" else 500) + rotation_seed_stride * rotation_idx
        per_rotation[rotation_idx] = {}
        for bit in bits:
            errors: list[float] = []
            for query, key, value in attention_examples:
                if side == "key":
                    candidate_key = quantize_vectors(
                        key,
                        bits=bit,
                        seed=seed,
                        grid_size=codebook_grid_size,
                        device=device,
                    )
                    error = attention_output_error(query, key, value, candidate_key=candidate_key)
                else:
                    candidate_value = quantize_vectors(
                        value,
                        bits=bit,
                        seed=seed,
                        grid_size=codebook_grid_size,
                        device=device,
                    )
                    error = attention_output_error(query, key, value, candidate_value=candidate_value)
                errors.append(error)
            mean_error = float(sum(errors) / len(errors))
            per_rotation[rotation_idx][bit] = mean_error
            if rotation_idx == 0:
                baseline_by_bit[bit] = mean_error
            rows.append(
                {
                    "layer": layer_idx,
                    "side": side,
                    "rotation_index": rotation_idx,
                    "bits": bit,
                    "objective": "attention_output",
                    "objective_error": mean_error,
                    "example_errors": errors,
                }
            )
    eps = torch.finfo(torch.float32).eps
    best_idx = min(
        range(rotation_bank_size),
        key=lambda rotation_idx: sum(
            per_rotation[rotation_idx][bit] / max(baseline_by_bit[bit], eps)
            for bit in bits
        )
        / len(bits),
    )
    return best_idx, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--dataset-key", default="longbench_2wikimqa")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-vectors-per-layer", type=int, default=1024)
    parser.add_argument("--bits", type=int, nargs="+", default=[2])
    parser.add_argument("--objective", choices=["mse", "attention_output"], default="mse")
    parser.add_argument("--query-tokens", type=int, default=8)
    parser.add_argument("--rotation-bank-size", type=int, default=8)
    parser.add_argument("--rotation-seed-stride", type=int, default=100_003)
    parser.add_argument("--codebook-grid-size", type=int, default=10_001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-mode", choices=["longbench", "legacy"], default="longbench")
    parser.add_argument("--chat-template-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.query_tokens <= 0:
        raise ValueError("--query-tokens must be positive")

    paths = load_yaml(Path(args.paths))
    cfg = load_yaml(Path(args.config))
    model_cfg = paths["models"][cfg["model_key"]]
    data_cfg = paths["datasets"][args.dataset_key]
    dataset = load_dataset_from_config(data_cfg)
    end_index = min(len(dataset), args.start_index + args.num_examples)
    selected_indices = list(range(args.start_index, end_index))
    selected_dataset = dataset.select(selected_indices)

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
    attention_rows = [[] for _ in range(num_layers)]
    for inputs in iter_prompts(args, tokenizer, data_cfg, selected_dataset):
        keys, values, attentions = collect_layer_vectors(
            model,
            inputs,
            args.max_vectors_per_layer,
            query_tokens=args.query_tokens,
        )
        for layer_idx in range(num_layers):
            key_rows[layer_idx].append(keys[layer_idx])
            value_rows[layer_idx].append(values[layer_idx])
            attention_rows[layer_idx].append(attentions[layer_idx])

    eval_device = torch.device(args.device)
    layer_key_rotation_indices: list[int] = []
    layer_value_rotation_indices: list[int] = []
    rows: list[dict] = []
    for layer_idx in range(num_layers):
        for side, vectors_by_layer, selected in (
            ("key", key_rows, layer_key_rotation_indices),
            ("value", value_rows, layer_value_rotation_indices),
        ):
            if args.objective == "mse":
                vectors = torch.cat(vectors_by_layer[layer_idx], dim=0)
                best_idx, side_rows = choose_rotation_by_reconstruction_mse(
                    vectors,
                    side=side,
                    layer_idx=layer_idx,
                    bits=args.bits,
                    rotation_bank_size=args.rotation_bank_size,
                    rotation_seed_stride=args.rotation_seed_stride,
                    codebook_grid_size=args.codebook_grid_size,
                    device=eval_device,
                )
            else:
                best_idx, side_rows = choose_rotation_by_attention_output(
                    attention_rows[layer_idx],
                    side=side,
                    layer_idx=layer_idx,
                    bits=args.bits,
                    rotation_bank_size=args.rotation_bank_size,
                    rotation_seed_stride=args.rotation_seed_stride,
                    codebook_grid_size=args.codebook_grid_size,
                    device=eval_device,
                )
            rows.extend(side_rows)
            selected.append(best_idx)

    output = {
        "config": {
            "dataset_key": args.dataset_key,
            "indices": selected_indices,
            "max_input_tokens": args.max_input_tokens,
            "max_vectors_per_layer": args.max_vectors_per_layer,
            "bits": args.bits,
            "objective": args.objective,
            "query_tokens": args.query_tokens,
            "rotation_bank_size": args.rotation_bank_size,
            "rotation_seed_stride": args.rotation_seed_stride,
            "codebook_grid_size": args.codebook_grid_size,
        },
        "layer_key_rotation_indices": layer_key_rotation_indices,
        "layer_value_rotation_indices": layer_value_rotation_indices,
        "runner_args": {
            "layer_key_rotation_indices": ",".join(str(item) for item in layer_key_rotation_indices),
            "layer_value_rotation_indices": ",".join(str(item) for item in layer_value_rotation_indices),
        },
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["runner_args"], indent=2))


if __name__ == "__main__":
    main()
