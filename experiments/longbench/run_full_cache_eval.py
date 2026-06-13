#!/usr/bin/env python3
"""Run full-cache LongBench generation over a local Arrow shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
import yaml
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.kv_cache import TurboQuantDynamicCache, make_kv_config_from_effective_bits
from turboquant.longbench_metrics import category_for_dataset, score_prediction
from turboquant.longbench_prompts import (
    build_longbench_prompt,
    dataset_name_from_row,
    max_new_tokens_for,
    should_apply_chat_template,
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_from_config(data_cfg: dict) -> Dataset:
    arrow_files = data_cfg.get("arrow_files") or []
    parquet_files = data_cfg.get("parquet_files") or []
    if arrow_files:
        datasets = [Dataset.from_file(path) for path in arrow_files]
    elif parquet_files:
        datasets = [Dataset.from_parquet(path) for path in parquet_files]
    else:
        raise ValueError("dataset config must contain either arrow_files or parquet_files")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def build_legacy_prompt(row: dict) -> str:
    context = row["context"].strip()
    question = (row.get("question") or row.get("input") or "").strip()
    answer_prefix = (row.get("answer_prefix") or "Answer:").strip()
    if context.endswith(question):
        return f"{context}\n{answer_prefix}"
    return f"{context}\n\n{question}\n{answer_prefix}"


def truncate_middle(input_ids: torch.Tensor, max_input_tokens: int | None) -> torch.Tensor:
    if max_input_tokens is None or input_ids.shape[-1] <= max_input_tokens:
        return input_ids
    half = max_input_tokens // 2
    return torch.cat([input_ids[..., :half], input_ids[..., -half:]], dim=-1)


def apply_chat_template_if_needed(tokenizer, prompt: str, dataset_name: str, mode: str) -> str:
    if not should_apply_chat_template(dataset_name, mode):
        return prompt
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def contains_answer(prediction: str, answers: list[str]) -> bool:
    pred = normalize_text(prediction)
    return any(normalize_text(answer) in pred for answer in answers if answer)


def load_completed(output_path: Path) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add(str(record["index"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--dataset-key", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", choices=["single", "auto", "balanced"], default="single")
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--offload-folder", default=str(PROJECT_ROOT / "reproduce/offload"))
    parser.add_argument("--cache-mode", choices=["full", "turboquant"], default="full")
    parser.add_argument("--kv-bits", type=float, default=16.0)
    parser.add_argument("--key-bits", type=float, default=None)
    parser.add_argument("--value-bits", type=float, default=None)
    parser.add_argument("--key-quantizer", choices=["mse", "prod"], default="mse")
    parser.add_argument("--value-quantizer", choices=["mse", "prod"], default="mse")
    parser.add_argument("--effective-bit-allocation", choices=["blend", "quarter_high2"], default="blend")
    parser.add_argument("--no-quantize-decode", action="store_true")
    parser.add_argument("--codebook-grid-size", type=int, default=20_001)
    parser.add_argument("--turboquant-fast-materialized-eval", action="store_true")
    parser.add_argument("--prompt-mode", choices=["longbench", "legacy"], default="longbench")
    parser.add_argument("--chat-template-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/runs/longbench_full_cache_eval.jsonl"))
    args = parser.parse_args()

    paths = load_yaml(Path(args.paths))
    cfg = load_yaml(Path(args.config))
    model_cfg = paths["models"][cfg["model_key"]]
    dataset_key = args.dataset_key or cfg["longbench"]["primary_dataset_key"]
    data_cfg = paths["datasets"][dataset_key]

    dataset = load_dataset_from_config(data_cfg)
    full_dataset_len = len(dataset)
    start_index = max(args.start_index, 0)
    end_index = args.end_index if args.end_index is not None else full_dataset_len
    end_index = min(end_index, full_dataset_len)
    if start_index > end_index:
        raise ValueError(f"start-index {start_index} is greater than end-index {end_index}")
    if args.max_examples is not None:
        end_index = min(end_index, start_index + args.max_examples)
    selected_indices = list(range(start_index, end_index))
    dataset = dataset.select(selected_indices)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path) if args.resume else set()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["snapshot"], local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg["hardware"].get("dtype") == "bfloat16" else torch.float16
    max_memory = None
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory:
            key, value = item.split("=", 1)
            max_memory[int(key) if key.isdigit() else key] = value
    device_map = {"": args.device} if args.device_map == "single" else args.device_map
    offload_folder = Path(args.offload_folder)
    if args.device_map != "single":
        offload_folder.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["snapshot"],
        local_files_only=True,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=str(offload_folder),
        attn_implementation="sdpa",
    )
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    correct = 0
    total = 0
    skipped = 0
    started = perf_counter()
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for offset, row in enumerate(dataset):
            idx = selected_indices[offset]
            if str(idx) in completed:
                skipped += 1
                continue
            dataset_name = dataset_name_from_row(row, data_cfg.get("requested_config") or data_cfg["config"])
            if args.prompt_mode == "longbench":
                prompt = build_longbench_prompt(row, dataset_name)
                max_new_tokens = max_new_tokens_for(dataset_name)
            else:
                prompt = build_legacy_prompt(row)
                max_new_tokens = int(row.get("max_new_tokens") or 64)
            prompt = apply_chat_template_if_needed(tokenizer, prompt, dataset_name, args.chat_template_mode)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            inputs["input_ids"] = truncate_middle(inputs["input_ids"], args.max_input_tokens)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = truncate_middle(inputs["attention_mask"], args.max_input_tokens)
            input_device = model.get_input_embeddings().weight.device
            inputs = inputs.to(input_device)
            past_key_values = None
            if args.cache_mode == "turboquant":
                kv_cfg = make_kv_config_from_effective_bits(
                    args.kv_bits,
                    seed=idx,
                    codebook_grid_size=args.codebook_grid_size,
                    key_quantizer=args.key_quantizer,
                    value_quantizer=args.value_quantizer,
                    effective_bit_allocation=args.effective_bit_allocation,
                    fast_materialized_eval=args.turboquant_fast_materialized_eval,
                )
                kv_cfg = type(kv_cfg)(
                    key_bits=args.key_bits if args.key_bits is not None else kv_cfg.key_bits,
                    value_bits=args.value_bits if args.value_bits is not None else kv_cfg.value_bits,
                    seed=kv_cfg.seed,
                    codebook_grid_size=kv_cfg.codebook_grid_size,
                    quantize_prefill=kv_cfg.quantize_prefill,
                    quantize_decode=not args.no_quantize_decode,
                    key_quantizer=kv_cfg.key_quantizer,
                    value_quantizer=kv_cfg.value_quantizer,
                    effective_bit_allocation=kv_cfg.effective_bit_allocation,
                    fast_materialized_eval=kv_cfg.fast_materialized_eval,
                )
                past_key_values = TurboQuantDynamicCache(kv_cfg)
            example_started = perf_counter()
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": 1,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
                "past_key_values": past_key_values,
            }
            if dataset_name == "samsum":
                newline_ids = tokenizer.encode("\n", add_special_tokens=False)
                if newline_ids:
                    generate_kwargs["min_length"] = int(inputs["input_ids"].shape[-1]) + 1
                    generate_kwargs["eos_token_id"] = [tokenizer.eos_token_id, newline_ids[-1]]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    **generate_kwargs,
                )
            latency = perf_counter() - example_started
            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answers = row["answers"]
            is_correct = contains_answer(prediction, answers)
            longbench_score = score_prediction(dataset_name, prediction, answers, row.get("all_classes"))
            longbench_category = category_for_dataset(dataset_name)
            cache_stats = {}
            if past_key_values is not None and hasattr(past_key_values, "storage_nbytes"):
                storage_nbytes = int(past_key_values.storage_nbytes())
                materialized_nbytes = int(past_key_values.materialized_nbytes())
                cache_stats = {
                    "cache_storage_nbytes": storage_nbytes,
                    "cache_materialized_nbytes": materialized_nbytes,
                    "cache_storage_ratio": storage_nbytes / materialized_nbytes if materialized_nbytes else None,
                }
                if hasattr(past_key_values, "compression_summary"):
                    cache_stats["cache_compression_summary"] = past_key_values.compression_summary()
                if hasattr(past_key_values, "fast_materialized_nbytes"):
                    cache_stats["cache_fast_materialized_nbytes"] = int(past_key_values.fast_materialized_nbytes())
            correct += int(is_correct)
            total += 1
            record = {
                "index": idx,
                "id": row.get("_id"),
                "dataset": row.get("dataset"),
                "task": row.get("task"),
                "longbench_dataset": dataset_name,
                "longbench_category": longbench_category,
                "prompt_mode": args.prompt_mode,
                "chat_template_mode": args.chat_template_mode,
                "max_input_tokens": args.max_input_tokens,
                "cache_mode": args.cache_mode,
                "kv_bits": args.kv_bits,
                "key_bits": args.key_bits,
                "value_bits": args.value_bits,
                "key_quantizer": args.key_quantizer,
                "value_quantizer": args.value_quantizer,
                "effective_bit_allocation": args.effective_bit_allocation,
                "quantize_decode": not args.no_quantize_decode,
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "generated_tokens": int(new_tokens.shape[-1]),
                "max_new_tokens": max_new_tokens,
                "latency_seconds": latency,
                "prediction": prediction,
                "answers": answers,
                "all_classes": row.get("all_classes"),
                "contains_answer": is_correct,
                "longbench_score": longbench_score,
                **cache_stats,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if args.progress_every > 0 and total % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "completed_this_run": total,
                            "skipped": skipped,
                            "dataset_index": idx,
                            "requested": len(selected_indices),
                            "latency_seconds": round(latency, 4),
                            "contains_answer": is_correct,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    elapsed = perf_counter() - started
    # Summary for this invocation only. For resumed aggregate metrics, use the
    # JSONL records directly.
    summary = {
        "model": model_cfg["repo_id"],
        "dataset_key": dataset_key,
        "dataset": data_cfg["repo_id"],
        "config": data_cfg["config"],
        "split": data_cfg["split"],
        "cache_mode": args.cache_mode,
        "kv_bits": args.kv_bits,
        "key_bits": args.key_bits,
        "value_bits": args.value_bits,
        "key_quantizer": args.key_quantizer,
        "value_quantizer": args.value_quantizer,
        "effective_bit_allocation": args.effective_bit_allocation,
        "quantize_decode": not args.no_quantize_decode,
        "device_map": args.device_map,
        "max_memory": args.max_memory,
        "offload_folder": args.offload_folder if args.device_map != "single" else None,
        "prompt_mode": args.prompt_mode,
        "chat_template_mode": args.chat_template_mode,
        "max_input_tokens": args.max_input_tokens,
        "turboquant_fast_materialized_eval": args.turboquant_fast_materialized_eval,
        "full_dataset_len": full_dataset_len,
        "start_index": start_index,
        "end_index": end_index,
        "num_examples_requested": len(selected_indices),
        "num_examples_run": total,
        "num_examples_skipped": skipped,
        "contains_answer_accuracy_this_run": correct / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
