#!/usr/bin/env python3
"""Build content-adaptive LowBit-Gain comparison from paired LongBench JSONL runs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.longbench.run_full_cache_eval import is_code_completion_prompt
from turboquant.longbench_metrics import TABLE1_CATEGORIES, category_for_dataset, normalize_dataset_name, score_prediction


DATASETS = [
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


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["index"])] = row
    return rows


def score_row(row: dict[str, Any]) -> float:
    dataset = normalize_dataset_name(row.get("longbench_dataset") or row.get("task") or row.get("dataset"))
    return 100.0 * score_prediction(dataset, row["prediction"], row.get("answers") or [], row.get("all_classes"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def prompt_structure_features(prompt: str) -> dict[str, int]:
    lower = prompt.lower()
    return {
        "passage_count": len(re.findall(r"\bpassage\s+\d+\s*:", lower)),
        "question_marks": prompt.count("?"),
    }


def load_feature_table(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(row["dataset"], int(row["index"])): row for row in rows}


def build(
    base_dir: Path,
    lowbit_dir: Path,
    threshold: int,
    *,
    max_passages: int | None,
    max_question_marks: int | None,
    feature_table: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    task_rows = []
    for dataset in DATASETS:
        base_path = base_dir / f"longbench_{dataset}_turboquant_2_5bit_all.jsonl"
        lowbit_path = lowbit_dir / f"lowbit_gain_mse_{dataset}_turboquant_2p5_full.jsonl"
        base_rows = load_jsonl(base_path)
        lowbit_rows = load_jsonl(lowbit_path)
        if set(base_rows) != set(lowbit_rows):
            raise ValueError(f"index mismatch for {dataset}: {base_path} vs {lowbit_path}")
        base_scores = []
        lowbit_scores = []
        candidate_scores = []
        active = 0
        for idx in sorted(base_rows):
            base = base_rows[idx]
            lowbit = lowbit_rows[idx]
            base_score = score_row(base)
            lowbit_score = score_row(lowbit)
            prompt = base.get("prompt") or ""
            code_prompt = bool(base.get("code_completion_prompt"))
            if "code_completion_prompt" not in base:
                dataset_name = normalize_dataset_name(base.get("longbench_dataset"))
                code_prompt = dataset_name in {"lcc", "repobench-p"} or is_code_completion_prompt(prompt, dataset_name)
            features = feature_table.get((dataset, idx))
            if features is None:
                features = prompt_structure_features(prompt)
            passage_count = int(features.get("passage_count") or 0)
            question_marks = int(features.get("question_marks") or 0)
            passage_gate_blocked = passage_count > 0 and (
                (max_passages is not None and passage_count > max_passages)
                or (max_question_marks is not None and question_marks > max_question_marks)
            )
            use_lowbit = int(base["prompt_tokens"]) > threshold and not code_prompt and not passage_gate_blocked
            active += int(use_lowbit)
            base_scores.append(base_score)
            lowbit_scores.append(lowbit_score)
            candidate_scores.append(lowbit_score if use_lowbit else base_score)
        category = category_for_dataset(dataset)
        task_rows.append(
            {
                "category": category,
                "dataset": dataset,
                "n": len(base_scores),
                "tq25": mean(base_scores),
                "lowbit25": mean(lowbit_scores),
                "content_adaptive25": mean(candidate_scores),
                "delta_vs_tq25": mean(candidate_scores) - mean(base_scores),
                "activation_rate": active / len(base_scores) if base_scores else 0.0,
            }
        )

    category_rows = []
    for category, datasets in TABLE1_CATEGORIES.items():
        rows = [row for row in task_rows if row["dataset"] in datasets]
        category_rows.append(
            {
                "category": category,
                "tq25": mean([row["tq25"] for row in rows]),
                "lowbit25": mean([row["lowbit25"] for row in rows]),
                "content_adaptive25": mean([row["content_adaptive25"] for row in rows]),
                "delta_vs_tq25": mean([row["delta_vs_tq25"] for row in rows]),
                "activation_rate": mean([row["activation_rate"] for row in rows]),
            }
        )

    return {
        "method": "Content-Adaptive LowBit-Gain",
        "threshold": threshold,
        "max_passages": max_passages,
        "max_question_marks": max_question_marks,
        "rule": (
            "Use lowbit_gain_mse when prompt_tokens > threshold, prompt is not code-completion-like, "
            "and Passage-style prompt structure does not exceed the configured passage/question thresholds; "
            "otherwise use TurboQuant MSE."
        ),
        "average": {
            "tq25": mean([row["tq25"] for row in category_rows]),
            "lowbit25": mean([row["lowbit25"] for row in category_rows]),
            "content_adaptive25": mean([row["content_adaptive25"] for row in category_rows]),
            "delta_vs_tq25": mean([row["delta_vs_tq25"] for row in category_rows]),
        },
        "category_rows": category_rows,
        "task_rows": task_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(PROJECT_ROOT / "reproduce/runs/table1_official_metric"))
    parser.add_argument("--lowbit-dir", default=str(PROJECT_ROOT / "reproduce/runs/incremental"))
    parser.add_argument("--threshold", type=int, default=6144)
    parser.add_argument("--max-passages", type=int, default=None)
    parser.add_argument("--max-question-marks", type=int, default=None)
    parser.add_argument("--feature-table", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = build(
        Path(args.base_dir),
        Path(args.lowbit_dir),
        args.threshold,
        max_passages=args.max_passages,
        max_question_marks=args.max_question_marks,
        feature_table=load_feature_table(Path(args.feature_table)) if args.feature_table else {},
    )
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["average"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
