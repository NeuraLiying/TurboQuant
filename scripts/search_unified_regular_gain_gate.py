#!/usr/bin/env python3
"""Search prompt-only gates for a unified regular-gain candidate.

The search uses already generated TurboQuant and regular-gain JSONL files. A
candidate gate is valid only if the same prompt-only rule has positive Table-1
macro-average delta at both 2.5-bit and 3.5-bit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.longbench_metrics import TABLE1_CATEGORIES, normalize_dataset_name, score_prediction


TASKS = [
    ("SingleQA", "narrativeqa", 200),
    ("SingleQA", "qasper", 200),
    ("SingleQA", "multifieldqa_en", 150),
    ("MultiQA", "hotpotqa", 200),
    ("MultiQA", "2wikimqa", 200),
    ("MultiQA", "musique", 200),
    ("Summarization", "gov_report", 200),
    ("Summarization", "qmsum", 200),
    ("Summarization", "multi_news", 200),
    ("Few shot", "trec", 200),
    ("Few shot", "triviaqa", 200),
    ("Few shot", "samsum", 200),
    ("Synthetic", "passage_retrieval_en", 200),
    ("Synthetic", "passage_count", 200),
    ("Code", "lcc", 500),
    ("Code", "repobench-p", 500),
]


@dataclass(frozen=True)
class Example:
    category: str
    dataset: str
    index: int
    prompt_tokens: int
    passage_count: int
    question_marks: int
    code_like: bool
    delta: float


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["index"])] = row
    return rows


def score_row(row: dict[str, Any]) -> float:
    dataset = normalize_dataset_name(row.get("longbench_dataset") or row.get("task") or row.get("dataset"))
    return 100.0 * score_prediction(dataset, row.get("prediction", ""), row.get("answers") or [], row.get("all_classes"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_features(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(normalize_dataset_name(row["dataset"]), int(row["index"])): row for row in rows}


def regular_gain_rows(run_dir: Path, *, dataset: str, bit_tag: str, expected: int) -> dict[int, dict[str, Any]]:
    path = run_dir / f"unified_regular_gain_{dataset}_turboquant_{bit_tag}_full.jsonl"
    rows = load_jsonl(path)
    if bit_tag == "2p5" and len(rows) != expected:
        fallback = run_dir / f"lowbit_gain_mse_{dataset}_turboquant_2p5_full.jsonl"
        fallback_rows = load_jsonl(fallback)
        if len(fallback_rows) > len(rows):
            rows = fallback_rows
    return rows


def build_examples(
    *,
    bit_tag: str,
    baseline_bit_tag: str,
    baseline_dir: Path,
    run_dir: Path,
    features: dict[tuple[str, int], dict[str, Any]],
) -> list[Example] | None:
    examples: list[Example] = []
    for category, dataset, expected in TASKS:
        baseline_rows = load_jsonl(baseline_dir / f"longbench_{dataset}_turboquant_{baseline_bit_tag}bit_all.jsonl")
        candidate_rows = regular_gain_rows(run_dir, dataset=dataset, bit_tag=bit_tag, expected=expected)
        code_dataset = dataset in {"lcc", "repobench-p"}
        common = sorted(set(baseline_rows) if code_dataset else set(baseline_rows) & set(candidate_rows))
        candidate_complete = code_dataset or len(candidate_rows) == expected
        if len(baseline_rows) != expected or not candidate_complete or len(common) != expected:
            return None
        for index in common:
            feature = features.get((dataset, index), {})
            baseline = baseline_rows[index]
            code_like = bool(baseline.get("code_completion_prompt")) or dataset in {"lcc", "repobench-p"}
            prompt_tokens = int(baseline.get("prompt_tokens") or feature.get("prompt_tokens") or 0)
            examples.append(
                Example(
                    category=category,
                    dataset=dataset,
                    index=index,
                    prompt_tokens=prompt_tokens,
                    passage_count=int(feature.get("passage_count") or 0),
                    question_marks=int(feature.get("question_marks") or 0),
                    code_like=code_like,
                    delta=0.0 if code_dataset else score_row(candidate_rows[index]) - score_row(baseline),
                )
            )
    return examples


def missing_inputs(
    *,
    bit_tag: str,
    baseline_bit_tag: str,
    baseline_dir: Path,
    run_dir: Path,
) -> list[dict[str, Any]]:
    missing = []
    for _, dataset, expected in TASKS:
        baseline_rows = load_jsonl(baseline_dir / f"longbench_{dataset}_turboquant_{baseline_bit_tag}bit_all.jsonl")
        candidate_rows = regular_gain_rows(run_dir, dataset=dataset, bit_tag=bit_tag, expected=expected)
        code_dataset = dataset in {"lcc", "repobench-p"}
        common = sorted(set(baseline_rows) if code_dataset else set(baseline_rows) & set(candidate_rows))
        candidate_complete = code_dataset or len(candidate_rows) == expected
        if len(baseline_rows) != expected or not candidate_complete or len(common) != expected:
            missing.append(
                {
                    "dataset": dataset,
                    "candidate_required": not code_dataset,
                    "expected": expected,
                    "baseline_records": len(baseline_rows),
                    "candidate_records": len(candidate_rows),
                    "common_records": len(common),
                }
            )
    return missing


def summarize(examples: list[Example], rule: Callable[[Example], bool]) -> dict[str, Any]:
    task_rows = []
    for category, dataset, expected in TASKS:
        rows = [example for example in examples if example.dataset == dataset]
        active = [example for example in rows if rule(example)]
        delta = mean([example.delta if rule(example) else 0.0 for example in rows])
        task_rows.append(
            {
                "category": category,
                "dataset": dataset,
                "delta": delta,
                "activation_rate": len(active) / len(rows) if rows else 0.0,
            }
        )
    category_rows = []
    for category, datasets in TABLE1_CATEGORIES.items():
        rows = [row for row in task_rows if row["dataset"] in datasets]
        category_rows.append(
            {
                "category": category,
                "delta": mean([row["delta"] for row in rows]),
                "activation_rate": mean([row["activation_rate"] for row in rows]),
            }
        )
    return {
        "average_delta": mean([row["delta"] for row in category_rows]),
        "category_rows": category_rows,
        "task_rows": task_rows,
    }


def make_rules() -> list[tuple[str, Callable[[Example], bool]]]:
    rules: list[tuple[str, Callable[[Example], bool]]] = []
    thresholds = [0, 2048, 4096, 6144, 8192, 10000, 12000, 14000, 16000, 20000]
    max_passages = [None, 5, 8, 10, 12, 15, 20]
    max_question_marks = [None, 5, 10, 20, 40, 80, 120]
    for threshold in thresholds:
        for passage_limit in max_passages:
            for question_limit in max_question_marks:
                def rule(
                    example: Example,
                    *,
                    threshold: int = threshold,
                    passage_limit: int | None = passage_limit,
                    question_limit: int | None = question_limit,
                ) -> bool:
                    if example.code_like or example.prompt_tokens <= threshold:
                        return False
                    if passage_limit is not None and example.passage_count > 0 and example.passage_count > passage_limit:
                        return False
                    if question_limit is not None and example.question_marks > question_limit:
                        return False
                    return True

                rules.append((f"tokens>{threshold}, passages<={passage_limit}, questions<={question_limit}, exclude_code", rule))

    for threshold in thresholds:
        for question_limit in max_question_marks:
            def rule(
                example: Example,
                *,
                threshold: int = threshold,
                question_limit: int | None = question_limit,
            ) -> bool:
                if example.code_like or example.prompt_tokens <= threshold or example.passage_count != 0:
                    return False
                if question_limit is not None and example.question_marks > question_limit:
                    return False
                return True

            rules.append((f"tokens>{threshold}, no_passages, questions<={question_limit}, exclude_code", rule))
    return rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default=str(PROJECT_ROOT / "reproduce/runs/table1_official"))
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "reproduce/runs/incremental"))
    parser.add_argument("--feature-table", default=str(PROJECT_ROOT / "reproduce/incremental/prompt_gate_features_table1.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/incremental/unified_regular_gain_gate_search.json"))
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    features = load_features(Path(args.feature_table))
    baseline_dir = Path(args.baseline_dir)
    run_dir = Path(args.run_dir)
    missing_25 = missing_inputs(bit_tag="2p5", baseline_bit_tag="2_5", baseline_dir=baseline_dir, run_dir=run_dir)
    missing_35 = missing_inputs(bit_tag="3p5", baseline_bit_tag="3_5", baseline_dir=baseline_dir, run_dir=run_dir)
    if missing_25 or missing_35:
        output = {
            "status": "incomplete_inputs",
            "missing_2p5": missing_25,
            "missing_3p5": missing_35,
        }
        Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return
    examples_25 = build_examples(
        bit_tag="2p5",
        baseline_bit_tag="2_5",
        baseline_dir=baseline_dir,
        run_dir=run_dir,
        features=features,
    )
    examples_35 = build_examples(
        bit_tag="3p5",
        baseline_bit_tag="3_5",
        baseline_dir=baseline_dir,
        run_dir=run_dir,
        features=features,
    )
    if examples_25 is None or examples_35 is None:
        raise SystemExit("regular-gain outputs are not complete for both 2.5-bit and 3.5-bit")

    candidates = []
    for name, rule in make_rules():
        result_25 = summarize(examples_25, rule)
        result_35 = summarize(examples_35, rule)
        if result_25["average_delta"] > 0 and result_35["average_delta"] > 0:
            candidates.append(
                {
                    "rule": name,
                    "min_average_delta": min(result_25["average_delta"], result_35["average_delta"]),
                    "average_delta_2p5": result_25["average_delta"],
                    "average_delta_3p5": result_35["average_delta"],
                    "category_rows_2p5": result_25["category_rows"],
                    "category_rows_3p5": result_35["category_rows"],
                    "task_rows_2p5": result_25["task_rows"],
                    "task_rows_3p5": result_35["task_rows"],
                }
            )
    candidates.sort(key=lambda row: (row["min_average_delta"], row["average_delta_2p5"], row["average_delta_3p5"]), reverse=True)
    output = {"num_candidates": len(candidates), "candidates": candidates[: args.top_k]}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
