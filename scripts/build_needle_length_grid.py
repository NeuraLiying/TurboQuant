#!/usr/bin/env python3
"""Build a local Needle length grid from cached 16k Needle examples."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_source_dataset(paths_cfg: dict[str, Any], dataset_key: str) -> Dataset:
    data_cfg = paths_cfg["datasets"][dataset_key]
    if data_cfg.get("arrow_files"):
        datasets = [Dataset.from_file(path) for path in data_cfg["arrow_files"]]
        if len(datasets) == 1:
            return datasets[0]
        return concatenate_datasets(datasets)
    if data_cfg.get("parquet_files"):
        datasets = [Dataset.from_parquet(path) for path in data_cfg["parquet_files"]]
        if len(datasets) == 1:
            return datasets[0]
        return concatenate_datasets(datasets)
    raise ValueError(f"{dataset_key} must provide arrow_files or parquet_files")


def select_indices(dataset: Dataset, positions: list[str], distractor_langs: list[str], examples_per_cell: int) -> list[int]:
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(dataset):
        key = (row["needle_position"], row["distractor_lang"])
        if key[0] in positions and key[1] in distractor_langs:
            by_cell[key].append(idx)

    selected = []
    missing = []
    for position in positions:
        for distractor_lang in distractor_langs:
            cell = by_cell.get((position, distractor_lang), [])
            if len(cell) < examples_per_cell:
                missing.append({"needle_position": position, "distractor_lang": distractor_lang, "available": len(cell)})
            selected.extend(cell[:examples_per_cell])
    if missing:
        raise ValueError(f"not enough examples for requested cells: {missing}")
    return selected


def token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def split_question_tail(prompt: str) -> tuple[str, str]:
    marker = "\n\nQuestion:"
    idx = prompt.rfind(marker)
    if idx == -1:
        return prompt, ""
    return prompt[:idx], prompt[idx:]


def make_prompt_for_target_tokens(tokenizer: AutoTokenizer, prompt: str, answer_sentence: str, target_tokens: int) -> str:
    body, tail = split_question_tail(prompt)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_tokens) <= target_tokens:
        return prompt

    tail_tokens = tokenizer(tail, add_special_tokens=False)["input_ids"] if tail else []
    body_budget = max(target_tokens - len(tail_tokens), 1)
    body_tokens = tokenizer(body, add_special_tokens=False)["input_ids"]
    answer_tokens = tokenizer(answer_sentence, add_special_tokens=False)["input_ids"]
    if not answer_tokens:
        cropped_body = tokenizer.decode(body_tokens[:body_budget], skip_special_tokens=False)
        return cropped_body + tail

    start = None
    for idx in range(0, len(body_tokens) - len(answer_tokens) + 1):
        if body_tokens[idx : idx + len(answer_tokens)] == answer_tokens:
            start = idx
            break
    if start is None:
        # Fall back to a centered crop if tokenization makes exact matching fail.
        midpoint = len(body_tokens) // 2
        begin = max(0, min(midpoint - body_budget // 2, len(body_tokens) - body_budget))
        cropped_body = tokenizer.decode(body_tokens[begin : begin + body_budget], skip_special_tokens=False)
        return cropped_body + tail

    answer_end = start + len(answer_tokens)
    if len(answer_tokens) >= body_budget:
        cropped_body = tokenizer.decode(body_tokens[start : start + body_budget], skip_special_tokens=False)
        return cropped_body + tail

    context_budget = body_budget - len(answer_tokens)
    left_budget = context_budget // 2
    right_budget = context_budget - left_budget
    begin = max(0, start - left_budget)
    end = min(len(body_tokens), answer_end + right_budget)
    if end - begin < body_budget:
        if begin == 0:
            end = min(len(body_tokens), body_budget)
        elif end == len(body_tokens):
            begin = max(0, len(body_tokens) - body_budget)
    cropped_body = tokenizer.decode(body_tokens[begin:end], skip_special_tokens=False)
    return cropped_body + tail


def build_rows(
    source: Dataset,
    tokenizer: AutoTokenizer,
    indices: list[int],
    target_tokens: list[int],
    source_dataset_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    stats = []
    for target in target_tokens:
        lengths = []
        for source_idx in indices:
            row = source[source_idx]
            prompt = make_prompt_for_target_tokens(tokenizer, row["prompt"], row["answer_sentence"], target)
            prompt_token_count = token_count(tokenizer, prompt)
            lengths.append(prompt_token_count)
            new_row = {
                "id": f"{row['id']}:target{target}",
                "source_index": int(source_idx),
                "source_dataset_key": source_dataset_key,
                "target_prompt_tokens": int(target),
                "actual_prompt_tokens": int(prompt_token_count),
                "needle_lang": row["needle_lang"],
                "question_lang": row["question_lang"],
                "distractor_lang": row["distractor_lang"],
                "needle_position": row["needle_position"],
                "answer_text_format": row["answer_text_format"],
                "answer_start_index": int(row["answer_start_index"]),
                "answer_sentence": row["answer_sentence"],
                "prompt": prompt,
            }
            rows.append(new_row)
        stats.append(
            {
                "target_prompt_tokens": target,
                "num_examples": len(lengths),
                "min_actual_prompt_tokens": min(lengths),
                "max_actual_prompt_tokens": max(lengths),
                "avg_actual_prompt_tokens": sum(lengths) / len(lengths),
            }
        )
    return rows, stats


def update_paths(paths_cfg: dict[str, Any], dataset_key: str, arrow_path: Path, num_examples: int) -> None:
    paths_cfg.setdefault("datasets", {})[dataset_key] = {
        "repo_id": "generated/local-needle-length-grid",
        "config": dataset_key,
        "split": "generated",
        "source": "generated_local",
        "cache_dir": str(arrow_path.parent),
        "num_examples": int(num_examples),
        "arrow_files": [str(arrow_path)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--source-dataset-key", default="needle_16k_en")
    parser.add_argument("--output-dataset-key", default="needle_generated_length_grid_en")
    parser.add_argument("--target-tokens", nargs="+", type=int, default=[4096, 8192, 16384, 32768, 65536, 104000])
    parser.add_argument("--positions", nargs="+", default=["start", "middle", "end"])
    parser.add_argument(
        "--distractor-langs",
        nargs="+",
        default=["ar", "de", "en", "es", "hi", "multilingual", "vi", "zh"],
    )
    parser.add_argument("--examples-per-cell", type=int, default=1)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reproduce/generated_data/needle_length_grid"))
    parser.add_argument("--update-paths", action="store_true")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "reproduce/logs/needle_length_grid_report.json"))
    args = parser.parse_args()

    paths_path = Path(args.paths)
    paths_cfg = load_yaml(paths_path)
    model_key = load_yaml(Path(args.config))["model_key"]
    tokenizer = AutoTokenizer.from_pretrained(paths_cfg["models"][model_key]["snapshot"], local_files_only=True, use_fast=True)

    source = load_source_dataset(paths_cfg, args.source_dataset_key)
    indices = select_indices(source, args.positions, args.distractor_langs, args.examples_per_cell)
    rows, length_stats = build_rows(source, tokenizer, indices, args.target_tokens, args.source_dataset_key)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = output_dir / f"{args.output_dataset_key}.arrow"
    dataset = Dataset.from_list(rows)
    hf_dataset_dir = output_dir / f"{args.output_dataset_key}.hf"
    if hf_dataset_dir.exists():
        shutil.rmtree(hf_dataset_dir)
    dataset.save_to_disk(str(hf_dataset_dir))
    saved_arrow_files = sorted(hf_dataset_dir.glob("*.arrow"))
    if not saved_arrow_files:
        raise RuntimeError(f"save_to_disk did not create an Arrow file under {hf_dataset_dir}")
    shutil.copyfile(saved_arrow_files[0], arrow_path)

    if args.update_paths:
        update_paths(paths_cfg, args.output_dataset_key, arrow_path, len(rows))
        write_yaml(paths_cfg, paths_path)

    report = {
        "source_dataset_key": args.source_dataset_key,
        "output_dataset_key": args.output_dataset_key,
        "target_tokens": args.target_tokens,
        "positions": args.positions,
        "distractor_langs": args.distractor_langs,
        "examples_per_cell": args.examples_per_cell,
        "source_indices": indices,
        "num_examples": len(rows),
        "arrow_path": str(arrow_path),
        "hf_dataset_dir": str(hf_dataset_dir),
        "paths_updated": args.update_paths,
        "length_stats": length_stats,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
