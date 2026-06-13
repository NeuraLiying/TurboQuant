#!/usr/bin/env python3
"""Run a small full-cache LongBench generation smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_from_config(data_cfg: dict) -> Dataset:
    if data_cfg.get("arrow_files"):
        return Dataset.from_file(data_cfg["arrow_files"][0])
    if data_cfg.get("parquet_files"):
        datasets = [Dataset.from_parquet(path) for path in data_cfg["parquet_files"]]
        if len(datasets) == 1:
            return datasets[0]
        from datasets import concatenate_datasets

        return concatenate_datasets(datasets)
    raise ValueError("dataset config must contain either arrow_files or parquet_files")


def build_prompt(row: dict) -> str:
    context = row["context"].strip()
    question = (row.get("question") or row.get("input") or "").strip()
    answer_prefix = (row.get("answer_prefix") or "Answer:").strip()
    if context.endswith(question):
        return f"{context}\n{answer_prefix}"
    return f"{context}\n\n{question}\n{answer_prefix}"


def contains_answer(prediction: str, answers: list[str]) -> bool:
    pred = prediction.strip().lower()
    return any(answer.strip().lower() in pred for answer in answers if answer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/runs/longbench_full_cache_smoke.jsonl"))
    args = parser.parse_args()

    paths = load_yaml(Path(args.paths))
    cfg = load_yaml(Path(args.config))
    model_cfg = paths["models"][cfg["model_key"]]
    data_cfg = paths["datasets"][cfg["longbench"]["primary_dataset_key"]]

    dataset = load_dataset_from_config(data_cfg)
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    total = 0
    started = perf_counter()

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(dataset):
            prompt = build_prompt(row)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(args.device)
            max_new_tokens = int(row.get("max_new_tokens") or 64)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answers = row["answers"]
            is_correct = contains_answer(prediction, answers)
            correct += int(is_correct)
            total += 1
            record = {
                "index": idx,
                "id": row.get("_id"),
                "dataset": row.get("dataset"),
                "task": row.get("task"),
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "max_new_tokens": max_new_tokens,
                "prediction": prediction,
                "answers": answers,
                "contains_answer": is_correct,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    elapsed = perf_counter() - started
    summary = {
        "model": model_cfg["repo_id"],
        "dataset": data_cfg["repo_id"],
        "config": data_cfg["config"],
        "split": data_cfg["split"],
        "num_examples": total,
        "contains_answer_accuracy": correct / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
