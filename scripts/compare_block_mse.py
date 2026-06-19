#!/usr/bin/env python3
"""Compare scalar TurboQuant MSE and block-MSE on real LongBench K/V vectors."""

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
from turboquant.core import TurboQuantBlockMSE, TurboQuantLearnedBlockMSE, TurboQuantMSE
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


def mse(vectors: torch.Tensor, quantizer) -> float:
    q = quantizer.quantize(vectors)
    recon = quantizer.dequantize(q)
    return float((vectors - recon).square().sum(dim=-1).mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--dataset-key", default="longbench_2wikimqa")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-vectors-per-layer", type=int, default=512)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-mode", choices=["longbench", "legacy"], default="longbench")
    parser.add_argument("--chat-template-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/incremental/block_mse_compare.json"))
    args = parser.parse_args()

    paths = load_yaml(Path(args.paths))
    cfg = load_yaml(Path(args.config))
    model_cfg = paths["models"][cfg["model_key"]]
    data_cfg = paths["datasets"][args.dataset_key]
    dataset = load_dataset_from_config(data_cfg)
    selected_indices = list(range(args.start_index, min(len(dataset), args.start_index + args.max_examples)))
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

    rows = []
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
            _, key = apply_rotary_pos_emb(query, key, cos, sin)
            for name, tensor in [("key", key), ("value", value)]:
                vectors = tensor.reshape(-1, tensor.shape[-1]).detach().to(torch.float32).cpu()
                if vectors.shape[0] > args.max_vectors_per_layer:
                    stride = max(1, vectors.shape[0] // args.max_vectors_per_layer)
                    vectors = vectors[::stride][: args.max_vectors_per_layer]
                for bits in args.bits:
                    scalar = TurboQuantMSE(vectors.shape[-1], bits, seed=layer_idx, dtype=torch.float32, codebook_grid_size=10_001)
                    block = TurboQuantBlockMSE(
                        vectors.shape[-1],
                        bits,
                        block_size=2,
                        seed=layer_idx,
                        dtype=torch.float32,
                        codebook_grid_size=10_001,
                    )
                    learned_block = TurboQuantLearnedBlockMSE(
                        vectors.shape[-1],
                        bits,
                        block_size=2,
                        seed=layer_idx,
                        dtype=torch.float32,
                    )
                    scalar_mse = mse(vectors, scalar)
                    block_mse = mse(vectors, block)
                    learned_block_mse = mse(vectors, learned_block)
                    rows.append(
                        {
                            "layer": layer_idx,
                            "name": name,
                            "bits": bits,
                            "scalar_mse": scalar_mse,
                            "block2_mse": block_mse,
                            "learned_block2_mse": learned_block_mse,
                            "block2_relative_reduction": (scalar_mse - block_mse) / scalar_mse if scalar_mse else 0.0,
                            "learned_block2_relative_reduction": (
                                (scalar_mse - learned_block_mse) / scalar_mse if scalar_mse else 0.0
                            ),
                        }
                    )

    summary = {}
    for bits in args.bits:
        for name in ("key", "value"):
            values = [row for row in rows if row["bits"] == bits and row["name"] == name]
            scalar_avg = sum(row["scalar_mse"] for row in values) / len(values)
            block_avg = sum(row["block2_mse"] for row in values) / len(values)
            learned_block_avg = sum(row["learned_block2_mse"] for row in values) / len(values)
            summary[f"{name}_{bits}bit"] = {
                "scalar_mse": scalar_avg,
                "block2_mse": block_avg,
                "learned_block2_mse": learned_block_avg,
                "block2_relative_reduction": (scalar_avg - block_avg) / scalar_avg if scalar_avg else 0.0,
                "learned_block2_relative_reduction": (
                    (scalar_avg - learned_block_avg) / scalar_avg if scalar_avg else 0.0
                ),
            }

    output = {"config": vars(args), "summary": summary, "rows": rows}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
