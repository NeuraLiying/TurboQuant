#!/usr/bin/env python3
"""Build static per-layer channel scores for TurboQuant outlier selection."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--dataset-key", default="longbench_2wikimqa")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--prompt-mode", choices=["longbench", "legacy"], default="longbench")
    parser.add_argument("--chat-template-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/incremental/static_channel_scores.json"))
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
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    num_layers = len(model.model.layers)
    head_dim = model.model.layers[0].self_attn.head_dim
    key_scores = [torch.zeros(head_dim, device=model.device, dtype=torch.float64) for _ in range(num_layers)]
    value_scores = [torch.zeros(head_dim, device=model.device, dtype=torch.float64) for _ in range(num_layers)]
    counts = [0 for _ in range(num_layers)]

    for offset, row in enumerate(dataset):
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
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
            position_embeddings = model.model.rotary_emb(outputs.hidden_states[0], position_ids)

        for layer_idx, layer in enumerate(model.model.layers):
            hidden = outputs.hidden_states[layer_idx].detach()
            hidden_shape = (*hidden.shape[:-1], -1, layer.self_attn.head_dim)
            query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
            key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
            value = layer.self_attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            key_scores[layer_idx] += query.detach().to(torch.float64).square().mean(dim=(0, 1, 2))
            value_scores[layer_idx] += value.detach().to(torch.float64).square().mean(dim=(0, 1, 2))
            counts[layer_idx] += 1

    key_rows = []
    value_rows = []
    for layer_idx in range(num_layers):
        denom = max(counts[layer_idx], 1)
        key_rows.append((key_scores[layer_idx] / denom).detach().cpu().tolist())
        value_rows.append((value_scores[layer_idx] / denom).detach().cpu().tolist())

    output = {
        "config": {
            "dataset_key": args.dataset_key,
            "start_index": args.start_index,
            "max_examples": args.max_examples,
            "max_input_tokens": args.max_input_tokens,
            "selected_indices": selected_indices,
            "score_definition": {
                "key": "mean squared rotated query by layer/head-dimension",
                "value": "mean squared value activation by layer/head-dimension",
            },
        },
        "layer_key_channel_scores": key_rows,
        "layer_value_channel_scores": value_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "num_layers": num_layers, "head_dim": head_dim}, indent=2))


if __name__ == "__main__":
    main()
