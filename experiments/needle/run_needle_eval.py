#!/usr/bin/env python3
"""Run Needle-In-A-Haystack generation over a local Arrow cache."""

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
from turboquant.longbench_metrics import qa_f1_score


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def contains_text(prediction: str, target: str) -> bool:
    return normalize_text(target) in normalize_text(prediction)


def load_completed(output_path: Path) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add(str(row["index"]))
    return completed


def load_dataset_from_config(data_cfg: dict) -> Dataset:
    arrow_files = data_cfg.get("arrow_files") or []
    parquet_files = data_cfg.get("parquet_files") or []
    if arrow_files:
        datasets = [Dataset.from_file(path) for path in arrow_files]
    elif parquet_files:
        datasets = [Dataset.from_parquet(path) for path in parquet_files]
    else:
        raise ValueError("dataset config must provide arrow_files or parquet_files")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--dataset-key", default="needle_16k_en")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--examples-per-position", type=int, default=None)
    parser.add_argument("--examples-per-position-lang", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--positions", nargs="*", default=None)
    parser.add_argument("--distractor-langs", nargs="*", default=None)
    parser.add_argument("--target-prompt-tokens", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-mode", choices=["full", "turboquant"], default="full")
    parser.add_argument("--kv-bits", type=float, default=16.0)
    parser.add_argument("--codebook-grid-size", type=int, default=10_001)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/runs/needle_eval.jsonl"))
    args = parser.parse_args()

    paths = load_yaml(Path(args.paths))
    cfg = load_yaml(Path(args.config))
    model_cfg = paths["models"][cfg["model_key"]]
    data_cfg = paths["datasets"][args.dataset_key]

    dataset = load_dataset_from_config(data_cfg)
    indices = list(range(args.start_index, len(dataset)))
    if args.target_prompt_tokens:
        allowed_targets = set(args.target_prompt_tokens)
        indices = [idx for idx in indices if dataset[idx].get("target_prompt_tokens") in allowed_targets]
    if args.distractor_langs:
        allowed = set(args.distractor_langs)
        indices = [idx for idx in indices if dataset[idx]["distractor_lang"] in allowed]
    if args.examples_per_position_lang is not None:
        positions = args.positions or sorted(set(dataset[idx]["needle_position"] for idx in indices))
        distractor_langs = args.distractor_langs or sorted(set(dataset[idx]["distractor_lang"] for idx in indices))
        selected = []
        for position in positions:
            for distractor_lang in distractor_langs:
                group_indices = [
                    idx
                    for idx in indices
                    if dataset[idx]["needle_position"] == position and dataset[idx]["distractor_lang"] == distractor_lang
                ]
                selected.extend(group_indices[: args.examples_per_position_lang])
        indices = selected
    elif args.examples_per_position is not None:
        positions = args.positions or sorted(set(dataset[idx]["needle_position"] for idx in indices))
        selected = []
        for position in positions:
            position_indices = [idx for idx in indices if dataset[idx]["needle_position"] == position]
            selected.extend(position_indices[: args.examples_per_position])
        indices = selected
    elif args.positions:
        allowed = set(args.positions)
        indices = [idx for idx in indices if dataset[idx]["needle_position"] in allowed]
    if args.max_examples is not None:
        indices = indices[: args.max_examples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path) if args.resume else set()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["snapshot"], local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg["hardware"].get("dtype") == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["snapshot"],
        local_files_only=True,
        torch_dtype=dtype,
        device_map={"": args.device},
        attn_implementation="sdpa",
    )
    model.eval()

    total = 0
    answer_hits = 0
    started = perf_counter()
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for idx in indices:
            if str(idx) in completed:
                continue
            row = dataset[idx]
            inputs = tokenizer(row["prompt"], return_tensors="pt", truncation=False).to(args.device)
            past_key_values = None
            if args.cache_mode == "turboquant":
                kv_cfg = make_kv_config_from_effective_bits(
                    args.kv_bits,
                    seed=idx,
                    codebook_grid_size=args.codebook_grid_size,
                )
                past_key_values = TurboQuantDynamicCache(kv_cfg)

            example_started = perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            latency = perf_counter() - example_started

            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer = row["answer_text_format"]
            answer_sentence = row["answer_sentence"]
            answer_contains = contains_text(prediction, answer)
            sentence_contains = contains_text(prediction, answer_sentence)
            answer_f1 = qa_f1_score(prediction, answer)
            sentence_f1 = qa_f1_score(prediction, answer_sentence)

            cache_stats = {}
            if past_key_values is not None and hasattr(past_key_values, "storage_nbytes"):
                storage_nbytes = int(past_key_values.storage_nbytes())
                materialized_nbytes = int(past_key_values.materialized_nbytes())
                cache_stats = {
                    "cache_storage_nbytes": storage_nbytes,
                    "cache_materialized_nbytes": materialized_nbytes,
                    "cache_storage_ratio": storage_nbytes / materialized_nbytes if materialized_nbytes else None,
                    "cache_compression_summary": past_key_values.compression_summary(),
                }

            total += 1
            answer_hits += int(answer_contains)
            record = {
                "index": idx,
                "id": row["id"],
                "dataset_key": args.dataset_key,
                "cache_mode": args.cache_mode,
                "kv_bits": args.kv_bits,
                "needle_position": row["needle_position"],
                "needle_lang": row["needle_lang"],
                "question_lang": row["question_lang"],
                "distractor_lang": row["distractor_lang"],
                "target_prompt_tokens": row.get("target_prompt_tokens"),
                "actual_prompt_tokens": row.get("actual_prompt_tokens"),
                "answer_start_index": row["answer_start_index"],
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "generated_tokens": int(new_tokens.shape[-1]),
                "max_new_tokens": args.max_new_tokens,
                "latency_seconds": latency,
                "prediction": prediction,
                "answer_text": answer,
                "answer_sentence": answer_sentence,
                "answer_contains": answer_contains,
                "answer_sentence_contains": sentence_contains,
                "answer_f1": answer_f1,
                "answer_sentence_f1": sentence_f1,
                **cache_stats,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    elapsed = perf_counter() - started
    summary = {
        "model": model_cfg["repo_id"],
        "dataset_key": args.dataset_key,
        "dataset": data_cfg["repo_id"],
        "config": data_cfg["config"],
        "split": data_cfg["split"],
        "cache_mode": args.cache_mode,
        "kv_bits": args.kv_bits,
        "num_examples_requested": len(indices),
        "num_examples_run": total,
        "answer_contains_accuracy_this_run": answer_hits / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
