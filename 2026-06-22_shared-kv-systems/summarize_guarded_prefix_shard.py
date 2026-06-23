#!/usr/bin/env python3
"""Summarize a guarded quantized-prefix shard against an fp16 Shared-KV reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def summarize(candidate_path: Path, reference_path: Path) -> dict[str, Any]:
    candidate = load_jsonl(candidate_path)
    reference = load_jsonl(reference_path)
    reference_by_index = {int(row["index"]): row for row in reference}
    rows = [row for row in candidate if int(row["index"]) in reference_by_index]
    if not rows:
        raise ValueError("no overlapping rows")

    score_mismatches = []
    prediction_mismatches = []
    for row in rows:
        ref = reference_by_index[int(row["index"])]
        if abs(float(row["longbench_score"]) - float(ref["longbench_score"])) > 1e-12:
            score_mismatches.append(
                {
                    "index": int(row["index"]),
                    "candidate_score": float(row["longbench_score"]),
                    "reference_score": float(ref["longbench_score"]),
                    "candidate_prediction": row["prediction"],
                    "reference_prediction": ref["prediction"],
                }
            )
        if row["prediction"] != ref["prediction"]:
            prediction_mismatches.append(
                {
                    "index": int(row["index"]),
                    "candidate_score": float(row["longbench_score"]),
                    "reference_score": float(ref["longbench_score"]),
                    "candidate_prediction": row["prediction"],
                    "reference_prediction": ref["prediction"],
                }
            )

    candidate_groups: dict[str, int] = {}
    reference_groups: dict[str, int] = {}
    candidate_group_materialized_cache: dict[str, int] = {}
    for row in rows:
        ref = reference_by_index[int(row["index"])]
        group_id = str(row["repeat_group_id"])
        candidate_groups.setdefault(group_id, int(row["shared_prefix_table_storage_nbytes"]))
        reference_groups.setdefault(group_id, int(ref["shared_prefix_table_storage_nbytes"]))
        candidate_group_materialized_cache.setdefault(
            group_id,
            int(row.get("shared_prefix_materialized_cache_nbytes") or 0),
        )

    candidate_prefix = sum(candidate_groups.values())
    reference_prefix = sum(reference_groups.values())
    candidate_materialized_cache = sum(candidate_group_materialized_cache.values())
    candidate_request = sum(int(row["request_storage_nbytes"]) for row in rows)
    reference_request = sum(int(reference_by_index[int(row["index"])]["request_storage_nbytes"]) for row in rows)
    candidate_total = candidate_prefix + candidate_request
    reference_total = reference_prefix + reference_request

    task_summaries = []
    task_names = sorted({str(row.get("task") or row.get("dataset") or "unknown") for row in rows})
    score_mismatch_indices = {item["index"] for item in score_mismatches}
    prediction_mismatch_indices = {item["index"] for item in prediction_mismatches}
    for task_name in task_names:
        task_rows = [row for row in rows if str(row.get("task") or row.get("dataset") or "unknown") == task_name]
        task_candidate_groups: dict[str, int] = {}
        task_reference_groups: dict[str, int] = {}
        task_candidate_materialized_cache_groups: dict[str, int] = {}
        for row in task_rows:
            ref = reference_by_index[int(row["index"])]
            group_id = str(row["repeat_group_id"])
            task_candidate_groups.setdefault(group_id, int(row["shared_prefix_table_storage_nbytes"]))
            task_reference_groups.setdefault(group_id, int(ref["shared_prefix_table_storage_nbytes"]))
            task_candidate_materialized_cache_groups.setdefault(
                group_id,
                int(row.get("shared_prefix_materialized_cache_nbytes") or 0),
            )
        task_candidate_prefix = sum(task_candidate_groups.values())
        task_reference_prefix = sum(task_reference_groups.values())
        task_candidate_request = sum(int(row["request_storage_nbytes"]) for row in task_rows)
        task_reference_request = sum(
            int(reference_by_index[int(row["index"])]["request_storage_nbytes"]) for row in task_rows
        )
        task_candidate_total = task_candidate_prefix + task_candidate_request
        task_reference_total = task_reference_prefix + task_reference_request
        task_candidate_score = mean(float(row["longbench_score"]) for row in task_rows)
        task_reference_score = mean(
            float(reference_by_index[int(row["index"])]["longbench_score"]) for row in task_rows
        )
        task_candidate_latency = mean(float(row["latency_seconds"]) for row in task_rows)
        task_reference_latency = mean(
            float(reference_by_index[int(row["index"])]["latency_seconds"]) for row in task_rows
        )
        task_summaries.append(
            {
                "task": task_name,
                "records": len(task_rows),
                "groups": len(task_candidate_groups),
                "candidate_score": task_candidate_score,
                "reference_score": task_reference_score,
                "score_delta": task_candidate_score - task_reference_score,
                "candidate_avg_latency_seconds": task_candidate_latency,
                "reference_avg_latency_seconds": task_reference_latency,
                "latency_ratio": task_candidate_latency / task_reference_latency,
                "score_mismatch_count": sum(int(row["index"]) in score_mismatch_indices for row in task_rows),
                "prediction_mismatch_count": sum(
                    int(row["index"]) in prediction_mismatch_indices for row in task_rows
                ),
                "candidate_prefix_table_storage_nbytes": task_candidate_prefix,
                "reference_prefix_table_storage_nbytes": task_reference_prefix,
                "prefix_table_savings_ratio": 1.0 - task_candidate_prefix / task_reference_prefix,
                "candidate_request_storage_nbytes": task_candidate_request,
                "reference_request_storage_nbytes": task_reference_request,
                "candidate_shared_total_storage_nbytes": task_candidate_total,
                "reference_shared_total_storage_nbytes": task_reference_total,
                "shared_total_savings_ratio": 1.0 - task_candidate_total / task_reference_total,
                "candidate_materialized_prefix_cache_nbytes": sum(task_candidate_materialized_cache_groups.values()),
            }
        )

    return {
        "candidate_jsonl": str(candidate_path),
        "reference_jsonl": str(reference_path),
        "records": len(rows),
        "candidate_records": len(candidate),
        "reference_records": len(reference),
        "groups": len(candidate_groups),
        "candidate_score": mean(float(row["longbench_score"]) for row in rows),
        "reference_score": mean(float(reference_by_index[int(row["index"])]["longbench_score"]) for row in rows),
        "candidate_avg_latency_seconds": mean(float(row["latency_seconds"]) for row in rows),
        "reference_avg_latency_seconds": mean(float(reference_by_index[int(row["index"])]["latency_seconds"]) for row in rows),
        "score_mismatch_count": len(score_mismatches),
        "prediction_mismatch_count": len(prediction_mismatches),
        "score_mismatch_preview": score_mismatches[:10],
        "prediction_mismatch_preview": prediction_mismatches[:10],
        "task_summaries": task_summaries,
        "candidate_prefix_table_storage_nbytes": candidate_prefix,
        "reference_prefix_table_storage_nbytes": reference_prefix,
        "prefix_table_savings_ratio": 1.0 - candidate_prefix / reference_prefix,
        "candidate_materialized_prefix_cache_nbytes": candidate_materialized_cache,
        "candidate_request_storage_nbytes": candidate_request,
        "reference_request_storage_nbytes": reference_request,
        "candidate_shared_total_storage_nbytes": candidate_total,
        "reference_shared_total_storage_nbytes": reference_total,
        "shared_total_savings_ratio": 1.0 - candidate_total / reference_total,
        "candidate_materialized_nbytes": sum(int(row["request_materialized_nbytes"]) for row in rows),
        "reference_materialized_nbytes": sum(
            int(reference_by_index[int(row["index"])]["request_materialized_nbytes"]) for row in rows
        ),
        "candidate_metadata": {
            "prefix_storage_mode": rows[0].get("prefix_storage_mode"),
            "prefix_kv_bits": rows[0].get("prefix_kv_bits"),
            "prefix_key_bits": rows[0].get("prefix_key_bits", rows[0].get("prefix_kv_bits")),
            "prefix_value_bits": rows[0].get("prefix_value_bits", rows[0].get("prefix_kv_bits")),
            "prefix_key_quantizer": rows[0].get("prefix_key_quantizer"),
            "prefix_value_quantizer": rows[0].get("prefix_value_quantizer"),
            "prefix_raw_start_tokens": rows[0].get("prefix_raw_start_tokens"),
            "prefix_raw_end_tokens": rows[0].get("prefix_raw_end_tokens"),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    delta = report["candidate_score"] - report["reference_score"]
    latency_ratio = report["candidate_avg_latency_seconds"] / report["reference_avg_latency_seconds"]
    lines = [
        "# Guarded Quantized-Prefix Summary",
        "",
        "This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.",
        "",
        "## Aggregate",
        "",
        f"- Records: `{report['records']}`.",
        f"- Groups: `{report['groups']}`.",
        f"- Candidate score: `{report['candidate_score']:.6f}`.",
        f"- fp16 Shared-KV reference score: `{report['reference_score']:.6f}`.",
        f"- Score delta: `{delta:.6f}`.",
        f"- Candidate avg latency: `{report['candidate_avg_latency_seconds']:.6f}s`.",
        f"- Reference avg latency: `{report['reference_avg_latency_seconds']:.6f}s`.",
        f"- Candidate/reference latency ratio: `{latency_ratio:.2f}x`.",
        f"- Score mismatches: `{report['score_mismatch_count']}`.",
        f"- Prediction mismatches: `{report['prediction_mismatch_count']}`.",
        "",
        "## Storage",
        "",
        f"- Prefix-table saving vs fp16 Shared-KV: `{fmt_pct(report['prefix_table_savings_ratio'])}`.",
        f"- Shared-total saving vs fp16 Shared-KV: `{fmt_pct(report['shared_total_savings_ratio'])}`.",
        f"- Candidate shared total bytes: `{report['candidate_shared_total_storage_nbytes']}`.",
        f"- Reference shared total bytes: `{report['reference_shared_total_storage_nbytes']}`.",
        f"- Candidate transient materialized prefix-cache bytes: `{report['candidate_materialized_prefix_cache_nbytes']}`.",
        "",
        "## Task Breakdown",
        "",
        "| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Transient prefix cache | Score mismatches | Prediction mismatches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in report["task_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(task["task"]),
                    str(task["records"]),
                    f"{task['candidate_score']:.6f}",
                    f"{task['reference_score']:.6f}",
                    f"{task['score_delta']:.6f}",
                    fmt_pct(task["shared_total_savings_ratio"]),
                    f"{task['latency_ratio']:.2f}x",
                    str(task["candidate_materialized_prefix_cache_nbytes"]),
                    str(task["score_mismatch_count"]),
                    str(task["prediction_mismatch_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Candidate Metadata",
            "",
            "```json",
            json.dumps(report["candidate_metadata"], indent=2),
            "```",
            "",
            "## Score Mismatch Preview",
            "",
        ]
    )
    if report["score_mismatch_preview"]:
        for item in report["score_mismatch_preview"]:
            lines.append(
                f"- index `{item['index']}`: candidate `{item['candidate_score']}` vs reference `{item['reference_score']}`"
            )
    else:
        lines.append("- None.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.candidate, args.reference)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
