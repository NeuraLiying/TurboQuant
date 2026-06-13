#!/usr/bin/env python3
"""Summarize Needle-In-A-Haystack JSONL outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_group(rows: list[dict]) -> dict:
    return {
        "num_examples": len(rows),
        "answer_contains_accuracy": mean([float(row["answer_contains"]) for row in rows]),
        "answer_sentence_contains_accuracy": mean([float(row["answer_sentence_contains"]) for row in rows]),
        "avg_answer_f1": mean([float(row["answer_f1"]) for row in rows]),
        "avg_answer_sentence_f1": mean([float(row["answer_sentence_f1"]) for row in rows]),
        "avg_prompt_tokens": mean([float(row["prompt_tokens"]) for row in rows]),
        "avg_generated_tokens": mean([float(row["generated_tokens"]) for row in rows]),
        "avg_latency_seconds": mean([float(row["latency_seconds"]) for row in rows]),
        "avg_cache_storage_ratio": mean(
            [float(row["cache_storage_ratio"]) for row in rows if row.get("cache_storage_ratio") is not None]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = Path(args.jsonl)
    rows = []
    by_position = defaultdict(list)
    by_distractor_lang = defaultdict(list)
    by_position_and_distractor_lang = defaultdict(list)
    by_target_prompt_tokens = defaultdict(list)
    by_target_and_position = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            position = row.get("needle_position", "unknown")
            distractor_lang = row.get("distractor_lang", "unknown")
            by_position[position].append(row)
            by_distractor_lang[distractor_lang].append(row)
            by_position_and_distractor_lang[f"{position}/{distractor_lang}"].append(row)
            target_prompt_tokens = row.get("target_prompt_tokens")
            if target_prompt_tokens is not None:
                target_key = str(target_prompt_tokens)
                by_target_prompt_tokens[target_key].append(row)
                by_target_and_position[f"{target_key}/{position}"].append(row)

    summary = {
        "path": str(path),
        "overall": summarize_group(rows),
        "by_position": {key: summarize_group(value) for key, value in sorted(by_position.items())},
        "by_distractor_lang": {key: summarize_group(value) for key, value in sorted(by_distractor_lang.items())},
        "by_position_and_distractor_lang": {
            key: summarize_group(value) for key, value in sorted(by_position_and_distractor_lang.items())
        },
        "by_target_prompt_tokens": {
            key: summarize_group(value) for key, value in sorted(by_target_prompt_tokens.items(), key=lambda item: int(item[0]))
        },
        "by_target_and_position": {
            key: summarize_group(value)
            for key, value in sorted(
                by_target_and_position.items(), key=lambda item: (int(item[0].split("/", 1)[0]), item[0])
            )
        },
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
