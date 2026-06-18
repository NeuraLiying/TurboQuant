#!/usr/bin/env python3
"""Materialize unified-gate outputs from equivalent component runs.

The unified regular-gain gate is a deterministic per-example switch:

* inactive examples use the TurboQuant MSE baseline;
* active examples use the always-on regular_gain_mse run.

This script combines those two already validated component outputs into the
candidate JSONL, adding the same gate metadata emitted by the runner. It is
used only after checking that overlapping runner-generated active examples
match the always-on regular_gain_mse outputs exactly.
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
SCORE_TOLERANCE = 1e-9


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


def regular_gain_path(run_dir: Path, dataset: str, bit_tag: str) -> Path:
    return run_dir / f"unified_regular_gain_{dataset}_turboquant_{bit_tag}_full.jsonl"


def fallback_regular_gain_path(run_dir: Path, dataset: str, bit_tag: str) -> Path:
    if bit_tag == "2p5":
        return run_dir / f"lowbit_gain_mse_{dataset}_turboquant_2p5_full.jsonl"
    return regular_gain_path(run_dir, dataset, bit_tag)


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


def add_gate_metadata(
    row: dict[str, Any],
    *,
    dataset: str,
    kv_bits: str,
    active: bool,
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
            "effective_key_quantizer": key_quantizer if active else "mse",
            "effective_value_quantizer": value_quantizer if active else "mse",
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


def scores_match(left: Any, right: Any) -> bool:
    if left == right:
        return True
    try:
        return abs(float(left) - float(right)) <= SCORE_TOLERANCE
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="unified_regular_gain_gate")
    parser.add_argument("--baseline-dir", default=str(PROJECT_ROOT / "reproduce/runs/table1_official"))
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "reproduce/runs/incremental"))
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
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    run_dir = Path(args.run_dir)
    features = load_features(Path(args.feature_table))
    expected_by_dataset = dict(TASKS)
    summaries = []
    mismatches = []

    for bit_tag in args.bits:
        kv_bits, baseline_bit_tag = BITS[bit_tag]
        for dataset in args.datasets:
            if dataset not in expected_by_dataset:
                raise ValueError(f"unknown dataset: {dataset}")
            expected = expected_by_dataset[dataset]
            baseline_rows = load_jsonl(baseline_dir / f"longbench_{dataset}_turboquant_{baseline_bit_tag}bit_all.jsonl")
            regular_path = regular_gain_path(run_dir, dataset, bit_tag)
            regular_rows = load_jsonl(regular_path)
            if len(regular_rows) != expected:
                fallback_rows = load_jsonl(fallback_regular_gain_path(run_dir, dataset, bit_tag))
                if len(fallback_rows) == expected:
                    regular_rows = fallback_rows
            if len(baseline_rows) != expected:
                raise ValueError(f"{dataset} {bit_tag} baseline has {len(baseline_rows)} records, expected {expected}")
            output_path = run_dir / f"{args.method_name}_{dataset}_turboquant_{bit_tag}_full.jsonl"
            existing_rows = load_jsonl(output_path)
            rows = []
            active_count = 0
            inactive_count = 0
            for index in sorted(baseline_rows):
                baseline_row = baseline_rows[index]
                feature = features.get((dataset, index), {})
                active = gate_active(
                    dataset=dataset,
                    row=baseline_row,
                    feature=feature,
                    threshold=args.long_prompt_threshold,
                    max_passages=args.long_prompt_max_passages,
                    max_question_marks=args.long_prompt_max_question_marks,
                    exclude_code=args.long_prompt_exclude_code,
                )
                if active:
                    if index not in regular_rows:
                        raise ValueError(f"missing regular-gain row for {dataset} {bit_tag} index {index}")
                    source = regular_rows[index]
                    active_count += 1
                else:
                    source = baseline_row
                    inactive_count += 1
                if args.verify_existing and index in existing_rows:
                    existing = existing_rows[index]
                    if existing.get("prediction") != source.get("prediction"):
                        mismatches.append(
                            {
                                "bit_tag": bit_tag,
                                "dataset": dataset,
                                "index": index,
                                "active": active,
                                "field": "prediction",
                            }
                        )
                    if not scores_match(existing.get("longbench_score"), source.get("longbench_score")):
                        mismatches.append(
                            {
                                "bit_tag": bit_tag,
                                "dataset": dataset,
                                "index": index,
                                "active": active,
                                "field": "longbench_score",
                            }
                        )
                rows.append(
                    add_gate_metadata(
                        source,
                        dataset=dataset,
                        kv_bits=kv_bits,
                        active=active,
                        feature=feature,
                        threshold=args.long_prompt_threshold,
                        key_quantizer=args.long_prompt_key_quantizer,
                        value_quantizer=args.long_prompt_value_quantizer,
                        max_passages=args.long_prompt_max_passages,
                        max_question_marks=args.long_prompt_max_question_marks,
                        exclude_code=args.long_prompt_exclude_code,
                    )
                )
            write_jsonl_atomic(output_path, rows)
            summaries.append(
                {
                    "bit_tag": bit_tag,
                    "dataset": dataset,
                    "records": len(rows),
                    "active": active_count,
                    "inactive": inactive_count,
                    "verified_existing": len(existing_rows),
                    "output": str(output_path),
                }
            )

    if mismatches:
        raise SystemExit(json.dumps({"status": "mismatch", "mismatches": mismatches[:20]}, indent=2, ensure_ascii=False))
    print(json.dumps({"status": "ok", "summaries": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
