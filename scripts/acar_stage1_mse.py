#!/usr/bin/env python3
"""Stage-1 MSE probe for ACAR-style calibrated TurboQuant transforms."""

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
from turboquant.core import TurboQuantMSE, hadamard_orthogonal
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


def covariance_transform(vectors: torch.Tensor, mode: str, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    unit = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
    if mode in {"pca", "whiten"}:
        estimator = unit - unit.mean(dim=0, keepdim=True)
        moment = estimator.T @ estimator / max(estimator.shape[0] - 1, 1)
    elif mode == "second_hadamard":
        # TurboQuant quantizes unit vectors.  Use the second moment instead of
        # centered covariance so a non-zero mean direction is also spread.
        moment = unit.T @ unit / max(unit.shape[0], 1)
    else:
        raise ValueError(f"unsupported ACAR transform mode: {mode}")
    eigvals, eigvecs = torch.linalg.eigh(moment.to(torch.float64))
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(eps)
    eigvecs = eigvecs[:, order]
    if mode == "pca":
        transform = eigvecs.T
        inverse = eigvecs
    elif mode == "whiten":
        transform = torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T
        inverse = eigvecs @ torch.diag(torch.sqrt(eigvals))
    elif mode == "second_hadamard":
        hadamard = hadamard_orthogonal(vectors.shape[-1], device=vectors.device, dtype=torch.float64, seed=None)
        transform = hadamard @ eigvecs.T
        inverse = transform.T
    return transform.to(torch.float32), inverse.to(torch.float32)


def mse_for(vectors: torch.Tensor, bits: int, transform: torch.Tensor | None, inverse: torch.Tensor | None) -> float:
    quantizer = TurboQuantMSE(
        vectors.shape[-1],
        bits,
        seed=0,
        dtype=torch.float32,
        codebook_grid_size=10_001,
        transform_matrix=transform,
        inverse_transform_matrix=inverse,
    )
    with torch.no_grad():
        q = quantizer.quantize(vectors)
        recon = quantizer.dequantize(q)
    return float((vectors - recon).square().sum(dim=-1).mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--dataset-key", default="longbench_2wikimqa")
    parser.add_argument("--calib-start-index", type=int, default=0)
    parser.add_argument("--calib-examples", type=int, default=4)
    parser.add_argument("--eval-start-index", type=int, default=4)
    parser.add_argument("--eval-examples", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-vectors-per-layer", type=int, default=512)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--transform-modes", nargs="+", default=["pca", "whiten", "second_hadamard"])
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-mode", choices=["longbench", "legacy"], default="longbench")
    parser.add_argument("--chat-template-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/incremental/acar/stage1_mse.json"))
    args = parser.parse_args()

    paths = load_yaml(Path(args.paths))
    cfg = load_yaml(Path(args.config))
    model_cfg = paths["models"][cfg["model_key"]]
    data_cfg = paths["datasets"][args.dataset_key]
    dataset = load_dataset_from_config(data_cfg)
    calib_indices = list(range(args.calib_start_index, min(len(dataset), args.calib_start_index + args.calib_examples)))
    eval_indices = list(range(args.eval_start_index, min(len(dataset), args.eval_start_index + args.eval_examples)))
    calib_dataset = dataset.select(calib_indices)
    eval_dataset = dataset.select(eval_indices)

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
    calib_key = [[] for _ in range(num_layers)]
    calib_value = [[] for _ in range(num_layers)]
    for inputs in iter_prompts(args, tokenizer, data_cfg, calib_dataset):
        key_rows, value_rows = collect_layer_vectors(model, inputs, args.max_vectors_per_layer)
        for layer_idx in range(num_layers):
            calib_key[layer_idx].append(key_rows[layer_idx])
            calib_value[layer_idx].append(value_rows[layer_idx])

    eval_key = [[] for _ in range(num_layers)]
    eval_value = [[] for _ in range(num_layers)]
    for inputs in iter_prompts(args, tokenizer, data_cfg, eval_dataset):
        key_rows, value_rows = collect_layer_vectors(model, inputs, args.max_vectors_per_layer)
        for layer_idx in range(num_layers):
            eval_key[layer_idx].append(key_rows[layer_idx])
            eval_value[layer_idx].append(value_rows[layer_idx])

    transforms = {mode: {"key": [], "value": []} for mode in args.transform_modes}
    for layer_idx in range(num_layers):
        key_vectors = torch.cat(calib_key[layer_idx], dim=0)
        value_vectors = torch.cat(calib_value[layer_idx], dim=0)
        for mode in args.transform_modes:
            transforms[mode]["key"].append(covariance_transform(key_vectors, mode, args.eps))
            transforms[mode]["value"].append(covariance_transform(value_vectors, mode, args.eps))

    rows = []
    for bits in args.bits:
        for name in ("key", "value"):
            for layer_idx in range(num_layers):
                vectors = torch.cat(eval_key[layer_idx] if name == "key" else eval_value[layer_idx], dim=0)
                random_mse = mse_for(vectors, bits, None, None)
                rows.append(
                    {
                        "bits": bits,
                        "name": name,
                        "layer": layer_idx,
                        "method": "random",
                        "mse": random_mse,
                    }
                )
                for mode in args.transform_modes:
                    transform, inverse = transforms[mode][name][layer_idx]
                    rows.append(
                        {
                            "bits": bits,
                            "name": name,
                            "layer": layer_idx,
                            "method": mode,
                            "mse": mse_for(vectors, bits, transform, inverse),
                        }
                    )

    summary = {}
    for bits in args.bits:
        for name in ("key", "value"):
            random_values = [row["mse"] for row in rows if row["bits"] == bits and row["name"] == name and row["method"] == "random"]
            random_avg = sum(random_values) / len(random_values)
            for method in ["random", *args.transform_modes]:
                values = [row["mse"] for row in rows if row["bits"] == bits and row["name"] == name and row["method"] == method]
                avg = sum(values) / len(values)
                summary[f"{name}_{bits}bit_{method}"] = {
                    "avg_mse": avg,
                    "relative_reduction_vs_random": (random_avg - avg) / random_avg if random_avg else 0.0,
                }

    output = {
        "config": {
            "dataset_key": args.dataset_key,
            "calib_indices": calib_indices,
            "eval_indices": eval_indices,
            "max_input_tokens": args.max_input_tokens,
            "max_vectors_per_layer": args.max_vectors_per_layer,
            "bits": args.bits,
            "transform_modes": args.transform_modes,
        },
        "summary": summary,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
