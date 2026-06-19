#!/usr/bin/env python3
"""Prefill unified-gate inactive examples from TurboQuant baselines.

For examples where the unified regular-gain gate does not activate, the method
is exactly the TurboQuant MSE baseline. This script materializes those inactive
examples in the candidate JSONL while preserving already generated active
examples. The runner can then resume and spend GPU time only on missing active
examples.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ("narrativeqa", 200),
    ("qasper", 200),
    ("multifieldqa_en", 150),
    ("hotpotqa", 200),
    ("2wikimqa", 200),
    ("musique", 200),
    ("gov_report", 200),
    ("qmsum", 200),
    ("multi_news", 200),
    ("trec", 200),
    ("triviaqa", 200),
    ("samsum", 200),
    ("passage_retrieval_en", 200),
    ("passage_count", 200),
    ("lcc", 500),
    ("repobench-p", 500),
]
BITS = {
    "2p5": ("2.5", "2_5"),
    "3p5": ("3.5", "3_5"),
}
CODE_DATASETS = {"lcc", "repobench-p"}


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


def load_features(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(str(row["dataset"]), int(row["index"])): row for row in rows}


def gate_active(
    *,
    dataset: str,
    row: dict[str, Any],
    feature: dict[str, Any],
    threshold: int,
    max_passages: int | None,
    max_question_marks: int | None,
    exclude_code: bool,
) -> bool:
    if exclude_code and dataset in CODE_DATASETS:
        return False
    prompt_tokens = int(row.get("prompt_tokens") or feature.get("prompt_tokens") or 0)
    passage_count = int(feature.get("passage_count") or 0)
    question_marks = int(feature.get("question_marks") or 0)
    if prompt_tokens <= threshold:
        return False
    if max_passages is not None and passage_count > 0 and passage_count > max_passages:
        return False
    if max_question_marks is not None and question_marks > max_question_marks:
        return False
    return True


def inactive_row(
    row: dict[str, Any],
    *,
    dataset: str,
    kv_bits: str,
    feature: dict[str, Any],
    threshold: int,
    key_quantizer: str,
    value_quantizer: str,
    max_passages: int | None,
    max_question_marks: int | None,
    exclude_code: bool,
) -> dict[str, Any]:
    output = dict(row)
    passage_count = int(feature.get("passage_count") or 0)
    question_marks = int(feature.get("question_marks") or 0)
    output.update(
        {
            "kv_bits": float(kv_bits),
            "key_bits": None,
            "value_bits": None,
            "baseline_mode": "turboquant",
            "key_quantizer": "mse",
            "value_quantizer": "mse",
            "long_prompt_threshold": threshold,
            "long_prompt_max_tokens": None,
            "long_prompt_exclude_code": exclude_code,
            "long_prompt_max_passages": max_passages,
            "long_prompt_max_question_marks": max_question_marks,
            "code_completion_prompt": dataset in CODE_DATASETS,
            "prompt_passage_count": passage_count,
            "prompt_question_marks": question_marks,
            "passage_gate_blocked": max_passages is not None and passage_count > 0 and passage_count > max_passages,
            "question_gate_blocked": max_question_marks is not None and question_marks > max_question_marks,
            "long_prompt_key_quantizer": key_quantizer,
            "long_prompt_value_quantizer": value_quantizer,
            "effective_key_quantizer": "mse",
            "effective_value_quantizer": "mse",
        }
    )
    return output


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="unified_regular_gain_gate")
    parser.add_argument("--baseline-dir", default=str(PROJECT_ROOT / "reproduce/runs/table1_official"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reproduce/runs/incremental"))
    parser.add_argument(
        "--feature-table",
        default=str(PROJECT_ROOT / "reproduce/incremental/prompt_gate_features_table1.json"),
    )
    parser.add_argument("--bits", choices=sorted(BITS), nargs="+", default=sorted(BITS))
    parser.add_argument("--datasets", nargs="+", default=[dataset for dataset, _ in TASKS])
    parser.add_argument("--long-prompt-threshold", type=int, default=0)
    parser.add_argument("--long-prompt-key-quantizer", default="regular_gain_mse")
    parser.add_argument("--long-prompt-value-quantizer", default="regular_gain_mse")
    parser.add_argument("--long-prompt-max-passages", type=int, default=12)
    parser.add_argument("--long-prompt-max-question-marks", type=int, default=20)
    parser.add_argument("--long-prompt-exclude-code", action="store_true", default=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    features = load_features(Path(args.feature_table))
    expected_by_dataset = dict(TASKS)
    summaries = []
    for bit_tag in args.bits:
        kv_bits, baseline_bit_tag = BITS[bit_tag]
        for dataset in args.datasets:
            if dataset not in expected_by_dataset:
                raise ValueError(f"unknown dataset: {dataset}")
            baseline_path = baseline_dir / f"longbench_{dataset}_turboquant_{baseline_bit_tag}bit_all.jsonl"
            output_path = output_dir / f"{args.method_name}_{dataset}_turboquant_{bit_tag}_full.jsonl"
            baseline_rows = load_jsonl(baseline_path)
            if len(baseline_rows) != expected_by_dataset[dataset]:
                raise ValueError(f"{baseline_path} has {len(baseline_rows)} records, expected {expected_by_dataset[dataset]}")
            existing_rows = load_jsonl(output_path)
            merged = dict(existing_rows)
            prefilled = 0
            active_missing = 0
            inactive_total = 0
            for index, row in baseline_rows.items():
                feature = features.get((dataset, index), {})
                active = gate_active(
                    dataset=dataset,
                    row=row,
                    feature=feature,
                    threshold=args.long_prompt_threshold,
                    max_passages=args.long_prompt_max_passages,
                    max_question_marks=args.long_prompt_max_question_marks,
                    exclude_code=args.long_prompt_exclude_code,
                )
                if active:
                    if index not in merged:
                        active_missing += 1
                    continue
                inactive_total += 1
                if index not in merged:
                    merged[index] = inactive_row(
                        row,
                        dataset=dataset,
                        kv_bits=kv_bits,
                        feature=feature,
                        threshold=args.long_prompt_threshold,
                        key_quantizer=args.long_prompt_key_quantizer,
                        value_quantizer=args.long_prompt_value_quantizer,
                        max_passages=args.long_prompt_max_passages,
                        max_question_marks=args.long_prompt_max_question_marks,
                        exclude_code=args.long_prompt_exclude_code,
                    )
                    prefilled += 1
            write_jsonl_atomic(output_path, [merged[index] for index in sorted(merged)])
            summaries.append(
                {
                    "bit_tag": bit_tag,
                    "dataset": dataset,
                    "records": len(merged),
                    "existing_before": len(existing_rows),
                    "inactive_total": inactive_total,
                    "prefilled": prefilled,
                    "active_missing": active_missing,
                    "output": str(output_path),
                }
            )
    print(json.dumps({"summaries": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
