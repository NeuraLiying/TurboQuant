#!/usr/bin/env python3
"""Materialize baseline-equivalent code-task outputs for the unified gate.

The unified regular-gain gate explicitly excludes code-completion prompts.
For LongBench Table 1, `lcc` and `repobench-p` therefore run exactly as the
TurboQuant MSE baseline. This script copies those baseline JSONL records into
the incremental run directory while adding the same gate metadata emitted by
the runner, so downstream tables can treat the tasks as complete without
rerunning an identical code path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DATASETS = ("lcc", "repobench-p")
BITS = {
    "2p5": ("2.5", "2_5"),
    "3p5": ("3.5", "3_5"),
}


def load_features(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(str(row["dataset"]), int(row["index"])): row for row in rows}


def count_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def transform_row(
    row: dict[str, Any],
    *,
    dataset: str,
    kv_bits: str,
    features: dict[tuple[str, int], dict[str, Any]],
    long_prompt_threshold: int,
    long_prompt_key_quantizer: str,
    long_prompt_value_quantizer: str,
    long_prompt_max_passages: int,
    long_prompt_max_question_marks: int,
) -> dict[str, Any]:
    index = int(row["index"])
    feature = features.get((dataset, index), {})
    output = dict(row)
    output.update(
        {
            "kv_bits": float(kv_bits),
            "key_bits": None,
            "value_bits": None,
            "baseline_mode": "turboquant",
            "key_quantizer": "mse",
            "value_quantizer": "mse",
            "long_prompt_threshold": long_prompt_threshold,
            "long_prompt_max_tokens": None,
            "long_prompt_exclude_code": True,
            "long_prompt_max_passages": long_prompt_max_passages,
            "long_prompt_max_question_marks": long_prompt_max_question_marks,
            "code_completion_prompt": True,
            "prompt_passage_count": int(feature.get("passage_count") or 0),
            "prompt_question_marks": int(feature.get("question_marks") or 0),
            "passage_gate_blocked": False,
            "question_gate_blocked": False,
            "long_prompt_key_quantizer": long_prompt_key_quantizer,
            "long_prompt_value_quantizer": long_prompt_value_quantizer,
            "effective_key_quantizer": "mse",
            "effective_value_quantizer": "mse",
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="unified_regular_gain_gate")
    parser.add_argument("--baseline-dir", default=str(PROJECT_ROOT / "reproduce/runs/table1_official"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reproduce/runs/incremental"))
    parser.add_argument(
        "--feature-table",
        default=str(PROJECT_ROOT / "reproduce/incremental/prompt_gate_features_table1.json"),
    )
    parser.add_argument("--datasets", choices=CODE_DATASETS, nargs="+", default=list(CODE_DATASETS))
    parser.add_argument("--bits", choices=sorted(BITS), nargs="+", default=sorted(BITS))
    parser.add_argument("--long-prompt-threshold", type=int, default=0)
    parser.add_argument("--long-prompt-key-quantizer", default="regular_gain_mse")
    parser.add_argument("--long-prompt-value-quantizer", default="regular_gain_mse")
    parser.add_argument("--long-prompt-max-passages", type=int, default=12)
    parser.add_argument("--long-prompt-max-question-marks", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = load_features(Path(args.feature_table) if args.feature_table else None)

    written = []
    for bit_tag in args.bits:
        kv_bits, baseline_bit_tag = BITS[bit_tag]
        for dataset in args.datasets:
            baseline_path = baseline_dir / f"longbench_{dataset}_turboquant_{baseline_bit_tag}bit_all.jsonl"
            output_path = output_dir / f"{args.method_name}_{dataset}_turboquant_{bit_tag}_full.jsonl"
            if not baseline_path.exists():
                raise FileNotFoundError(baseline_path)
            baseline_records = count_records(baseline_path)
            if output_path.exists() and not args.overwrite and count_records(output_path) == baseline_records:
                written.append({"output": str(output_path), "records": baseline_records, "status": "already_complete"})
                continue
            with baseline_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
                records = 0
                for line in source:
                    if not line.strip():
                        continue
                    row = transform_row(
                        json.loads(line),
                        dataset=dataset,
                        kv_bits=kv_bits,
                        features=features,
                        long_prompt_threshold=args.long_prompt_threshold,
                        long_prompt_key_quantizer=args.long_prompt_key_quantizer,
                        long_prompt_value_quantizer=args.long_prompt_value_quantizer,
                        long_prompt_max_passages=args.long_prompt_max_passages,
                        long_prompt_max_question_marks=args.long_prompt_max_question_marks,
                    )
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                    records += 1
            written.append({"output": str(output_path), "records": records, "status": "written"})
    print(json.dumps({"written": written}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
