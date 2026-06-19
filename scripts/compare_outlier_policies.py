#!/usr/bin/env python3
"""Compare TurboQuant outlier selection policies on calibration prompts."""

from __future__ import annotations

import argparse
import json
import sys
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

from experiments.longbench.run_full_cache_eval import apply_chat_template_if_needed, build_legacy_prompt, load_yaml, truncate_middle
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


def layer_inputs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(outputs.hidden_states[0], position_ids)
    return outputs.hidden_states, position_embeddings


def attention_metrics(query: torch.Tensor, key: torch.Tensor, key_hat: torch.Tensor, scaling: float) -> dict[str, float]:
    scores = torch.matmul(query.to(torch.float32), key.transpose(-1, -2).to(torch.float32)) * scaling
    scores_hat = torch.matmul(query.to(torch.float32), key_hat.transpose(-1, -2).to(torch.float32)) * scaling
    score_err = scores_hat - scores
    return {
        "score_mae": float(score_err.abs().mean().item()),
        "score_rmse": float(torch.sqrt(score_err.square().mean()).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--dataset-key", default="longbench_2wikimqa")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bits", type=float, default=2.5)
    parser.add_argument("--prompt-mode", choices=["longbench", "legacy"], default="longbench")
    parser.add_argument("--chat-template-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/incremental/outlier_policy_compare.json"))
    args = parser.parse_args()

    paths = load_yaml(Path(args.paths))
    cfg = load_yaml(Path(args.config))
    model_cfg = paths["models"][cfg["model_key"]]
    data_cfg = paths["datasets"][args.dataset_key]
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

    policies = ["dynamic_absmean", "error_gain"]
    results = {policy: {"score_mae": [], "score_rmse": [], "outlier_counts": []} for policy in policies}

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
            hidden_states, position_embeddings = layer_inputs(model, input_ids, attention_mask)

        for layer_idx, layer in enumerate(model.model.layers):
            hidden = hidden_states[layer_idx].detach()
            hidden_shape = (*hidden.shape[:-1], -1, layer.self_attn.head_dim)
            query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
            key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            query_tail = query[..., -1:, :].detach()
            key = key.detach()
            for policy in policies:
                kv_cfg = make_kv_config_from_effective_bits(
                    args.bits,
                    seed=idx,
                    codebook_grid_size=10_001,
                    outlier_policy=policy,
                )
                cache = TurboQuantDynamicCache(kv_cfg)
                key_hat, _ = cache.update(key, key, layer_idx=layer_idx)
                attn = attention_metrics(query_tail, repeat_kv(key, layer.self_attn.num_key_value_groups), repeat_kv(key_hat, layer.self_attn.num_key_value_groups), layer.self_attn.scaling)
                results[policy]["score_mae"].append(attn["score_mae"])
                results[policy]["score_rmse"].append(attn["score_rmse"])
                segment = cache.key_cache[layer_idx][0]
                if hasattr(segment, "outlier_indices"):
                    results[policy]["outlier_counts"].append(int(segment.outlier_indices.numel()))

    summary = {
        policy: {
            "avg_score_mae": sum(v["score_mae"]) / len(v["score_mae"]) if v["score_mae"] else None,
            "avg_score_rmse": sum(v["score_rmse"]) / len(v["score_rmse"]) if v["score_rmse"] else None,
            "avg_outlier_count": sum(v["outlier_counts"]) / len(v["outlier_counts"]) if v["outlier_counts"] else None,
        }
        for policy, v in results.items()
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
