#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.longbench.run_full_cache_eval import apply_chat_template_if_needed, load_dataset_from_config, load_yaml
from turboquant.longbench_prompts import build_longbench_prompt, dataset_name_from_row
from transformers import AutoTokenizer


def features(prompt: str) -> dict[str, float]:
    lower = prompt.lower()
    chars = len(prompt)
    passage_count = len(re.findall(r"\bpassage\s+\d+\s*:", lower))
    question_marks = prompt.count("?")
    newline_count = prompt.count("\n")
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", prompt))
    caps_tokens = len(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", prompt))
    return {
        "chars": chars,
        "passage_count": passage_count,
        "question_marks": question_marks,
        "newline_count": newline_count,
        "sentence_count": sentence_count,
        "caps_token_ratio": caps_tokens / max(1, len(prompt.split())),
        "avg_passage_chars": chars / max(1, passage_count),
    }


def main() -> None:
    paths = load_yaml(PROJECT_ROOT / "configs/paths.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        paths["models"]["llama_3_1_8b_instruct"]["snapshot"],
        local_files_only=True,
        use_fast=True,
    )
    score_rows = json.loads((PROJECT_ROOT / "reproduce/incremental/per_example_official_scores_tq25_lowbit.json").read_text())
    score_by_key = {(row["dataset"], int(row["index"])): row for row in score_rows}
    table1_datasets = [
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "gov_report",
        "qmsum",
        "multi_news",
        "trec",
        "triviaqa",
        "samsum",
        "passage_retrieval_en",
        "passage_count",
        "lcc",
        "repobench-p",
    ]
    dataset_keys = [f"longbench_{dataset}" for dataset in table1_datasets]
    out = []
    for dataset_key in dataset_keys:
        data_cfg = paths["datasets"][dataset_key]
        dataset = load_dataset_from_config(data_cfg)
        for idx, row in enumerate(dataset):
            dataset_name = dataset_name_from_row(row, data_cfg.get("requested_config") or data_cfg["config"])
            prompt = build_longbench_prompt(row, dataset_name)
            prompt = apply_chat_template_if_needed(tokenizer, prompt, dataset_name, "auto")
            encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
            item = {
                "dataset": dataset_name,
                "index": idx,
                "prompt_tokens": int(encoded["input_ids"].shape[-1]),
                **features(prompt),
            }
            score = score_by_key.get((dataset_name, idx))
            if score is not None:
                item.update({"base": score["base"], "lowbit": score["lowbit"], "delta": score["delta"]})
            out.append(item)
    output = PROJECT_ROOT / "reproduce/incremental/prompt_gate_features_table1.json"
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    groups: dict[str, list[dict]] = {}
    for row in out:
        groups.setdefault(row["dataset"], []).append(row)
    for name, rows in groups.items():
        print(name, len(rows))
        for key in ["prompt_tokens", "passage_count", "question_marks", "avg_passage_chars", "delta"]:
            vals = [float(row[key]) for row in rows if key in row]
            print(" ", key, round(mean(vals), 4), min(vals), max(vals))


if __name__ == "__main__":
    main()
