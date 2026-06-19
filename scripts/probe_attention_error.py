#!/usr/bin/env python3
"""Probe layer/head KV quantization error on LongBench prompts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import repeat_kv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.longbench.run_full_cache_eval import (
    apply_chat_template_if_needed,
    build_legacy_prompt,
    load_yaml,
    truncate_middle,
)
from turboquant.kv_cache import TurboQuantDynamicCache, make_kv_config_from_effective_bits
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


def parse_methods(text: str) -> list[dict]:
    methods = []
    for spec in text.split(","):
        spec = spec.strip()
        if not spec:
            continue
        parts = spec.split(":")
        name = parts[0]
        if name == "turboquant":
            bits = float(parts[1]) if len(parts) > 1 else 2.5
            key_quantizer = parts[2] if len(parts) > 2 else "mse"
            value_quantizer = parts[3] if len(parts) > 3 else key_quantizer
            methods.append(
                {
                    "name": (
                        f"turboquant_{str(bits).replace('.', '_')}bit"
                        if key_quantizer == "mse" and value_quantizer == "mse"
                        else f"turboquant_{str(bits).replace('.', '_')}bit_{key_quantizer}_{value_quantizer}"
                    ),
                    "bits": bits,
                    "key_quantizer": key_quantizer,
                    "value_quantizer": value_quantizer,
                }
            )
        elif name == "naive_int":
            bits = float(parts[1]) if len(parts) > 1 else 4.0
            methods.append(
                {
                    "name": f"naive_int{int(bits)}",
                    "bits": bits,
                    "key_quantizer": "uniform_token",
                    "value_quantizer": "uniform_token",
                }
            )
        elif name == "kivi_style":
            bits = float(parts[1]) if len(parts) > 1 else 4.0
            methods.append(
                {
                    "name": f"kivi_style_int{int(bits)}",
                    "bits": bits,
                    "key_quantizer": "uniform_channel",
                    "value_quantizer": "uniform_token",
                }
            )
        else:
            raise ValueError(f"unsupported method spec: {spec}")
    return methods


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def layer_inputs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> tuple[torch.Tensor, object, object]:
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(outputs.hidden_states[0], position_ids)
    return outputs.hidden_states, position_embeddings, outputs


def attention_metrics(query: torch.Tensor, key: torch.Tensor, key_hat: torch.Tensor, scaling: float) -> dict[str, torch.Tensor]:
    scores = torch.matmul(query.to(torch.float32), key.transpose(-1, -2).to(torch.float32)) * scaling
    scores_hat = torch.matmul(query.to(torch.float32), key_hat.transpose(-1, -2).to(torch.float32)) * scaling
    score_err = scores_hat - scores
    probs = torch.softmax(scores, dim=-1)
    probs_hat = torch.softmax(scores_hat, dim=-1)
    prob_err = probs_hat - probs
    return {
        "score_mae": score_err.abs().mean(dim=(0, 2, 3)),
        "score_rmse": torch.sqrt(score_err.square().mean(dim=(0, 2, 3))),
        "prob_mae": prob_err.abs().mean(dim=(0, 2, 3)),
        "prob_l1": prob_err.abs().sum(dim=-1).mean(dim=(0, 2)),
        "top1_match": (torch.argmax(scores, dim=-1) == torch.argmax(scores_hat, dim=-1)).to(torch.float32).mean(dim=(0, 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--dataset-key", default="longbench_2wikimqa")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--methods", default="turboquant:2.5,turboquant:3.5,naive_int:4,kivi_style:4")
    parser.add_argument("--prompt-mode", choices=["longbench", "legacy"], default="longbench")
    parser.add_argument("--chat-template-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--codebook-grid-size", type=int, default=10_001)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/incremental/attention_error_probe.json"))
    args = parser.parse_args()

    paths = load_yaml(Path(args.paths))
    cfg = load_yaml(Path(args.config))
    model_cfg = paths["models"][cfg["model_key"]]
    data_cfg = paths["datasets"][args.dataset_key]
    methods = parse_methods(args.methods)

    dataset = load_dataset_from_config(data_cfg)
    end_index = min(len(dataset), args.start_index + args.max_examples)
    selected_indices = list(range(args.start_index, end_index))
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

    records = []
    aggregates: dict[str, dict[tuple[int, int], dict[str, list[float]]]] = {
        method["name"]: defaultdict(lambda: defaultdict(list)) for method in methods
    }
    for offset, row in enumerate(dataset):
        idx = selected_indices[offset]
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
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        attention_mask = attention_mask.to(model.device) if attention_mask is not None else None

        with torch.no_grad():
            hidden_states, position_embeddings, _ = layer_inputs(model, input_ids, attention_mask)

        for layer_idx, layer in enumerate(model.model.layers):
            hidden = hidden_states[layer_idx].detach()
            input_shape = hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
            key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
            value = layer.self_attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            query_tail = query[..., -1:, :].detach()
            key = key.detach()
            value = value.detach()
            key_for_attn = repeat_kv(key, layer.self_attn.num_key_value_groups)

            for method in methods:
                kv_cfg = make_kv_config_from_effective_bits(
                    method["bits"],
                    seed=idx,
                    codebook_grid_size=args.codebook_grid_size,
                    key_quantizer=method["key_quantizer"],
                    value_quantizer=method["value_quantizer"],
                )
                cache = TurboQuantDynamicCache(kv_cfg)
                key_hat, value_hat = cache.update(key, value, layer_idx=layer_idx)
                key_hat_for_attn = repeat_kv(key_hat, layer.self_attn.num_key_value_groups)
                key_err = (key_hat.to(torch.float32) - key.to(torch.float32))
                value_err = (value_hat.to(torch.float32) - value.to(torch.float32))
                attn = attention_metrics(query_tail, key_for_attn, key_hat_for_attn, layer.self_attn.scaling)
                key_rel = key_err.square().sum(dim=-1).sqrt() / key.to(torch.float32).square().sum(dim=-1).sqrt().clamp_min(1e-8)
                value_rel = (
                    value_err.square().sum(dim=-1).sqrt()
                    / value.to(torch.float32).square().sum(dim=-1).sqrt().clamp_min(1e-8)
                )
                for head_idx in range(query.shape[1]):
                    kv_head_idx = head_idx // layer.self_attn.num_key_value_groups
                    record = {
                        "example_index": idx,
                        "dataset": dataset_name,
                        "prompt_tokens": int(input_ids.shape[-1]),
                        "method": method["name"],
                        "bits": method["bits"],
                        "key_quantizer": method["key_quantizer"],
                        "value_quantizer": method["value_quantizer"],
                        "layer": layer_idx,
                        "head": head_idx,
                        "kv_head": kv_head_idx,
                        "key_rel_l2": float(key_rel[:, kv_head_idx, :].mean().item()),
                        "value_rel_l2": float(value_rel[:, kv_head_idx, :].mean().item()),
                        **{name: float(value_[head_idx].item()) for name, value_ in attn.items()},
                    }
                    records.append(record)
                    bucket = aggregates[method["name"]][(layer_idx, head_idx)]
                    for metric, value_ in record.items():
                        if isinstance(value_, float):
                            bucket[metric].append(value_)

    summary = []
    for method_name, by_layer_head in aggregates.items():
        for (layer_idx, head_idx), metrics in by_layer_head.items():
            row = {"method": method_name, "layer": layer_idx, "head": head_idx}
            for metric, values in metrics.items():
                row[metric] = mean(values)
            summary.append(row)
    summary.sort(key=lambda row: (row["method"], -float(row.get("score_rmse") or 0.0), row["layer"], row["head"]))

    output = {
        "config": {
            "dataset_key": args.dataset_key,
            "start_index": args.start_index,
            "max_examples": args.max_examples,
            "max_input_tokens": args.max_input_tokens,
            "methods": methods,
        },
        "num_records": len(records),
        "records": records,
        "summary": summary,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "num_records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
