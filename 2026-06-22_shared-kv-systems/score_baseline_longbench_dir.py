#!/usr/bin/env python3
"""Score a directory of LongBench-style jsonl predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

SKVQ_ROOT = Path("/home/liying/projects/SKVQ")
sys.path.insert(0, str(SKVQ_ROOT))
from score_longbench import dataset2metric  # noqa: E402


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction,
                    ground_truth,
                    all_classes=all_classes,
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2) if predictions else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    scores = {}
    counts = {}
    for path in sorted(pred_dir.glob("*.jsonl")):
        dataset = path.stem
        if dataset not in dataset2metric:
            continue
        predictions, answers = [], []
        all_classes = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                predictions.append(row["pred"])
                answers.append(row["answers"])
                all_classes = row["all_classes"]
        scores[dataset] = scorer(dataset, predictions, answers, all_classes)
        counts[dataset] = len(predictions)

    valid = [v for v in scores.values() if v is not None]
    result = {
        "pred_dir": str(pred_dir),
        "scores": scores,
        "counts": counts,
        "mean_score": round(float(np.mean(valid)), 4) if valid else None,
    }
    out_path = Path(args.output) if args.output else pred_dir / "result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
