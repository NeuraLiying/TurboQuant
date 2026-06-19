#!/usr/bin/env python3
"""Build Table-1 comparison for structure-adaptive regular-gain TurboQuant.

The candidate uses one prompt-only rule at both 2.5-bit and 3.5-bit:
select regular-gain reconstruction for long, non-code prompts whose
Passage-style structure is not blocked; otherwise use reproduced TurboQuant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

BITS = [("2p5", "2_5", 2.5), ("3p5", "3_5", 3.5)]


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


def load_feature_table(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(normalize_dataset_name(row["dataset"]), int(row["index"])): row for row in rows}


def score_row(row: dict[str, Any]) -> float:
    dataset = normalize_dataset_name(row.get("longbench_dataset") or row.get("task") or row.get("dataset"))
    return 100.0 * score_prediction(dataset, row.get("prediction", ""), row.get("answers") or [], row.get("all_classes"))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def regular_gain_paths(run_dir: Path, *, dataset: str, bit_tag: str) -> list[tuple[str, Path]]:
    paths = [
        (
            "regular_gain_mse",
            run_dir / f"unified_regular_gain_{dataset}_turboquant_{bit_tag}_full.jsonl",
        )
    ]
    if bit_tag == "2p5":
        paths.append(
            (
                "lowbit_gain_mse_equivalent",
                run_dir / f"lowbit_gain_mse_{dataset}_turboquant_2p5_full.jsonl",
            )
        )
    return paths


def load_regular_gain(run_dir: Path, *, dataset: str, bit_tag: str, expected: int) -> tuple[str, Path, dict[int, dict[str, Any]]]:
    loaded = []
    for source, path in regular_gain_paths(run_dir, dataset=dataset, bit_tag=bit_tag):
        rows = load_jsonl(path)
        loaded.append((source, path, rows))
        if len(rows) == expected:
            return source, path, rows
    return max(loaded, key=lambda item: len(item[2]))


def use_regular_gain(
    base: dict[str, Any],
    feature: dict[str, Any] | None,
    *,
    threshold: int,
    max_passages: int | None,
    max_question_marks: int | None,
) -> bool:
    prompt_tokens = int(base.get("prompt_tokens") or 0)
    dataset = normalize_dataset_name(base.get("longbench_dataset") or base.get("task") or base.get("dataset"))
    code_prompt = bool(base.get("code_completion_prompt")) or dataset in {"lcc", "repobench-p"}
    feature = feature or {}
    passage_count = int(base.get("prompt_passage_count") or feature.get("passage_count") or 0)
    question_marks = int(base.get("prompt_question_marks") or feature.get("question_marks") or 0)
    passage_blocked = passage_count > 0 and max_passages is not None and passage_count > max_passages
    question_blocked = max_question_marks is not None and question_marks > max_question_marks
    return prompt_tokens > threshold and not code_prompt and not passage_blocked and not question_blocked


def summarize_bit(
    *,
    bit_tag: str,
    baseline_bit_tag: str,
    kv_bits: float,
    baseline_dir: Path,
    run_dir: Path,
    feature_table: dict[tuple[str, int], dict[str, Any]],
    threshold: int,
    max_passages: int | None,
    max_question_marks: int | None,
) -> dict[str, Any]:
    task_rows = []
    for dataset in DATASETS:
        expected = 150 if dataset == "multifieldqa_en" else 500 if dataset in {"lcc", "repobench-p"} else 200
        base_path = baseline_dir / f"longbench_{dataset}_turboquant_{baseline_bit_tag}bit_all.jsonl"
        source, gain_path, gain_rows = load_regular_gain(run_dir, dataset=dataset, bit_tag=bit_tag, expected=expected)
        base_rows = load_jsonl(base_path)
        common = sorted(set(base_rows) & set(gain_rows))
        complete = len(base_rows) == expected and len(gain_rows) == expected and len(common) == expected
        base_scores = []
        gain_scores = []
        candidate_scores = []
        active = 0
        for index in common:
            base = base_rows[index]
            gain = gain_rows[index]
            base_score = score_row(base)
            gain_score = score_row(gain)
            feature = feature_table.get((dataset, index))
            selected = use_regular_gain(
                base,
                feature,
                threshold=threshold,
                max_passages=max_passages,
                max_question_marks=max_question_marks,
            )
            active += int(selected)
            base_scores.append(base_score)
            gain_scores.append(gain_score)
            candidate_scores.append(gain_score if selected else base_score)
        task_rows.append(
            {
                "category": category_for_dataset(dataset),
                "dataset": dataset,
                "expected": expected,
                "baseline_records": len(base_rows),
                "regular_gain_records": len(gain_rows),
                "common_records": len(common),
                "complete": complete,
                "regular_gain_source": source,
                "baseline_score": mean(base_scores) if complete else None,
                "regular_gain_score": mean(gain_scores) if complete else None,
                "candidate_score": mean(candidate_scores) if complete else None,
                "delta": (mean(candidate_scores) - mean(base_scores)) if complete else None,
                "activation_rate": active / len(common) if common else None,
                "baseline_jsonl": str(base_path),
                "regular_gain_jsonl": str(gain_path),
            }
        )

    category_rows = []
    for category, datasets in TABLE1_CATEGORIES.items():
        rows = [row for row in task_rows if row["dataset"] in datasets]
        complete = all(row["complete"] for row in rows)
        category_rows.append(
            {
                "category": category,
                "complete": complete,
                "baseline_score": mean([row["baseline_score"] for row in rows if row["baseline_score"] is not None])
                if complete
                else None,
                "regular_gain_score": mean(
                    [row["regular_gain_score"] for row in rows if row["regular_gain_score"] is not None]
                )
                if complete
                else None,
                "candidate_score": mean([row["candidate_score"] for row in rows if row["candidate_score"] is not None])
                if complete
                else None,
                "delta": mean([row["delta"] for row in rows if row["delta"] is not None]) if complete else None,
                "activation_rate": mean([row["activation_rate"] for row in rows if row["activation_rate"] is not None])
                if complete
                else None,
            }
        )

    complete_categories = [row for row in category_rows if row["complete"]]
    all_complete = len(complete_categories) == len(TABLE1_CATEGORIES)
    return {
        "kv_bits": kv_bits,
        "bit_tag": bit_tag,
        "complete": all_complete,
        "num_complete_tasks": sum(1 for row in task_rows if row["complete"]),
        "num_tasks": len(task_rows),
        "category_rows": category_rows,
        "task_rows": task_rows,
        "average": {
            "baseline_score": mean([row["baseline_score"] for row in complete_categories]) if all_complete else None,
            "regular_gain_score": mean([row["regular_gain_score"] for row in complete_categories]) if all_complete else None,
            "candidate_score": mean([row["candidate_score"] for row in complete_categories]) if all_complete else None,
            "delta": mean([row["delta"] for row in complete_categories]) if all_complete else None,
            "activation_rate": mean([row["activation_rate"] for row in complete_categories]) if all_complete else None,
        },
    }


def write_md(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Structure-Adaptive Regular-Gain Table 1 Comparison",
        "",
        "Candidate: one prompt-only gate at both 2.5-bit and 3.5-bit; use `regular_gain_mse` for long, non-code prompts whose Passage-style structure is not blocked, otherwise use TurboQuant MSE.",
        "",
    ]
    for result in results:
        avg = result["average"]
        lines.extend(
            [
                f"## KV {result['kv_bits']}",
                "",
                f"Complete tasks: {result['num_complete_tasks']} / {result['num_tasks']}",
                "",
                "| Category | TurboQuant | Regular-Gain | Structure-Adaptive | Delta | Activation | Complete |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in result["category_rows"]:
            lines.append(
                f"| {row['category']} | {fmt(row['baseline_score'])} | {fmt(row['regular_gain_score'])} | "
                f"{fmt(row['candidate_score'])} | {fmt(row['delta'])} | {fmt(row['activation_rate'])} | "
                f"{'yes' if row['complete'] else 'no'} |"
            )
        lines.append(
            f"| Average | {fmt(avg['baseline_score'])} | {fmt(avg['regular_gain_score'])} | "
            f"{fmt(avg['candidate_score'])} | {fmt(avg['delta'])} | {fmt(avg['activation_rate'])} | "
            f"{'yes' if result['complete'] else 'no'} |"
        )
        lines.extend(
            [
                "",
                "| Category | Task | Records | Source | TurboQuant | Regular-Gain | Structure-Adaptive | Delta | Activation | Complete |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in result["task_rows"]:
            records = f"{row['regular_gain_records']}/{row['expected']}"
            lines.append(
                f"| {row['category']} | `{row['dataset']}` | {records} | {row['regular_gain_source']} | "
                f"{fmt(row['baseline_score'])} | {fmt(row['regular_gain_score'])} | {fmt(row['candidate_score'])} | "
                f"{fmt(row['delta'])} | {fmt(row['activation_rate'])} | {'yes' if row['complete'] else 'no'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default=str(PROJECT_ROOT / "reproduce/runs/table1_official"))
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "reproduce/runs/incremental"))
    parser.add_argument("--feature-table", default=str(PROJECT_ROOT / "reproduce/incremental/prompt_gate_features_table1.json"))
    parser.add_argument("--threshold", type=int, default=6144)
    parser.add_argument("--max-passages", type=int, default=15)
    parser.add_argument("--max-question-marks", type=int, default=5)
    parser.add_argument(
        "--output-prefix",
        default=str(PROJECT_ROOT / "reproduce/incremental/structure_adaptive_regular_gain_table1"),
    )
    args = parser.parse_args()

    feature_table = load_feature_table(Path(args.feature_table) if args.feature_table else None)
    results = [
        summarize_bit(
            bit_tag=bit_tag,
            baseline_bit_tag=baseline_bit_tag,
            kv_bits=kv_bits,
            baseline_dir=Path(args.baseline_dir),
            run_dir=Path(args.run_dir),
            feature_table=feature_table,
            threshold=args.threshold,
            max_passages=args.max_passages,
            max_question_marks=args.max_question_marks,
        )
        for bit_tag, baseline_bit_tag, kv_bits in BITS
    ]
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(
            {
                "method": "Structure-Adaptive Regular-Gain TurboQuant",
                "threshold": args.threshold,
                "max_passages": args.max_passages,
                "max_question_marks": args.max_question_marks,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_md(results, md_path)
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
