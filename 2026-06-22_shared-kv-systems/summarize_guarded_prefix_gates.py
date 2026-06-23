#!/usr/bin/env python3
"""Aggregate guarded quantized-prefix gate summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        data = load(path)
        task = "unknown"
        if "0000_0200" in path.name:
            task = "2wikimqa"
        elif "0800_1000" in path.name:
            task = "musique"
        elif "1600_1800" in path.name:
            task = "passage_retrieval_en"
        rows.append(
            {
                "task": task,
                "summary_path": str(path),
                "records": data["records"],
                "groups": data["groups"],
                "candidate_score": data["candidate_score"],
                "reference_score": data["reference_score"],
                "score_delta": data["candidate_score"] - data["reference_score"],
                "candidate_avg_latency_seconds": data["candidate_avg_latency_seconds"],
                "reference_avg_latency_seconds": data["reference_avg_latency_seconds"],
                "latency_ratio": data["candidate_avg_latency_seconds"] / data["reference_avg_latency_seconds"],
                "prefix_table_savings_ratio": data["prefix_table_savings_ratio"],
                "shared_total_savings_ratio": data["shared_total_savings_ratio"],
                "score_mismatch_count": data["score_mismatch_count"],
                "prediction_mismatch_count": data["prediction_mismatch_count"],
                "candidate_shared_total_storage_nbytes": data["candidate_shared_total_storage_nbytes"],
                "reference_shared_total_storage_nbytes": data["reference_shared_total_storage_nbytes"],
                "candidate_materialized_prefix_cache_nbytes": data.get(
                    "candidate_materialized_prefix_cache_nbytes",
                    0,
                ),
            }
        )

    records = sum(row["records"] for row in rows)
    candidate_score = sum(row["candidate_score"] * row["records"] for row in rows) / records
    reference_score = sum(row["reference_score"] * row["records"] for row in rows) / records
    candidate_storage = sum(row["candidate_shared_total_storage_nbytes"] for row in rows)
    reference_storage = sum(row["reference_shared_total_storage_nbytes"] for row in rows)
    report = {
        "records": records,
        "groups": sum(row["groups"] for row in rows),
        "candidate_score": candidate_score,
        "reference_score": reference_score,
        "score_delta": candidate_score - reference_score,
        "candidate_shared_total_storage_nbytes": candidate_storage,
        "reference_shared_total_storage_nbytes": reference_storage,
        "shared_total_savings_ratio": 1.0 - candidate_storage / reference_storage,
        "score_mismatch_count": sum(row["score_mismatch_count"] for row in rows),
        "prediction_mismatch_count": sum(row["prediction_mismatch_count"] for row in rows),
        "candidate_materialized_prefix_cache_nbytes": sum(
            row["candidate_materialized_prefix_cache_nbytes"] for row in rows
        ),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Guarded Quantized-Prefix Multi-Task Gate",
        "",
        "This aggregates three 200-row repeated-context gates. It is stronger than a smoke test but still not a full 2400-row decision result.",
        "",
        "## Aggregate",
        "",
        f"- Records: `{report['records']}`.",
        f"- Groups: `{report['groups']}`.",
        f"- Candidate score: `{report['candidate_score']:.6f}`.",
        f"- fp16 Shared-KV reference score: `{report['reference_score']:.6f}`.",
        f"- Score delta: `{report['score_delta']:.6f}`.",
        f"- Shared-total saving vs fp16 Shared-KV: `{fmt_pct(report['shared_total_savings_ratio'])}`.",
        f"- Transient materialized prefix-cache bytes: `{report['candidate_materialized_prefix_cache_nbytes']}`.",
        f"- Score mismatches: `{report['score_mismatch_count']}`.",
        f"- Prediction mismatches: `{report['prediction_mismatch_count']}`.",
        "",
        "## By Task",
        "",
        "| Task | Records | Candidate | Reference | Delta | Shared saving | Latency ratio | Transient prefix cache | Score mismatches | Prediction mismatches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['task']}` | {row['records']} | {row['candidate_score']:.6f} | "
            f"{row['reference_score']:.6f} | {row['score_delta']:.6f} | "
            f"{fmt_pct(row['shared_total_savings_ratio'])} | {row['latency_ratio']:.2f}x | "
            f"{row['candidate_materialized_prefix_cache_nbytes']} | "
            f"{row['score_mismatch_count']} | {row['prediction_mismatch_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this gate to decide whether a guarded policy is worth a full run. Passing evidence requires aggregate quality near the fp16 Shared-KV reference, no large task-level regression, and enough storage saving to justify the stronger full validation. Runtime remains a separate blocker because the current prototype decodes/materializes quantized prefix K/V in Python.",
            "",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
