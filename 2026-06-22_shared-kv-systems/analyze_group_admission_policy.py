#!/usr/bin/env python3
"""Analyze group-level admission for guarded shared-prefix quantization.

This is an offline policy study over already-generated candidate/reference JSONL files.
It simulates a shared-prefix admission policy: quantize a repeat group only if a cheap
probe says the guarded quantized-prefix path is safe; otherwise keep that group's prefix
entry in fp16 Shared-KV form.

The point is to move beyond a fixed "quantize every group with 512/2048 guards" policy
and quantify the quality/storage frontier available to a reuse-aware admission module.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


@dataclass(frozen=True)
class GroupDecision:
    group_id: str
    admit_quantized: bool
    reason: str
    probe_delta: float
    mean_delta: float
    min_delta: float
    candidate_prefix_bytes: int
    reference_prefix_bytes: int
    row_count: int


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_id = str(row["repeat_group_id"])
        groups.setdefault(group_id, []).append(row)
    for items in groups.values():
        items.sort(key=lambda row: int(row["index"]))
    return groups


def row_delta(row: dict[str, Any], reference_by_index: dict[int, dict[str, Any]]) -> float:
    ref = reference_by_index[int(row["index"])]
    return float(row["longbench_score"]) - float(ref["longbench_score"])


def decide_group(
    group_id: str,
    rows: list[dict[str, Any]],
    *,
    reference_by_index: dict[int, dict[str, Any]],
    policy: str,
    probe_variants: int,
    min_probe_delta: float,
    min_group_delta: float,
) -> GroupDecision:
    deltas = [row_delta(row, reference_by_index) for row in rows]
    probe = deltas[: max(1, min(probe_variants, len(deltas)))]
    probe_delta = mean(probe)
    mean_delta = mean(deltas)
    min_delta = min(deltas)
    first = rows[0]
    ref_first = reference_by_index[int(first["index"])]
    candidate_prefix = int(first["shared_prefix_table_storage_nbytes"])
    reference_prefix = int(ref_first["shared_prefix_table_storage_nbytes"])

    if policy == "all":
        return GroupDecision(group_id, True, "all", probe_delta, mean_delta, min_delta, candidate_prefix, reference_prefix, len(rows))
    if policy == "none":
        return GroupDecision(group_id, False, "none", probe_delta, mean_delta, min_delta, candidate_prefix, reference_prefix, len(rows))
    if policy == "oracle_mean":
        admit = mean_delta >= min_group_delta
        return GroupDecision(
            group_id,
            admit,
            "oracle_mean_pass" if admit else "oracle_mean_fail",
            probe_delta,
            mean_delta,
            min_delta,
            candidate_prefix,
            reference_prefix,
            len(rows),
        )
    if policy == "oracle_all":
        admit = min_delta >= min_group_delta
        return GroupDecision(
            group_id,
            admit,
            "oracle_all_pass" if admit else "oracle_all_fail",
            probe_delta,
            mean_delta,
            min_delta,
            candidate_prefix,
            reference_prefix,
            len(rows),
        )
    if policy == "probe_mean":
        admit = probe_delta >= min_probe_delta
        return GroupDecision(
            group_id,
            admit,
            "probe_mean_pass" if admit else "probe_mean_fail",
            probe_delta,
            mean_delta,
            min_delta,
            candidate_prefix,
            reference_prefix,
            len(rows),
        )
    if policy == "probe_all":
        admit = min(probe) >= min_probe_delta
        return GroupDecision(
            group_id,
            admit,
            "probe_all_pass" if admit else "probe_all_fail",
            probe_delta,
            mean_delta,
            min_delta,
            candidate_prefix,
            reference_prefix,
            len(rows),
        )
    raise ValueError(f"unknown policy: {policy}")


def simulate_policy(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    policy: str,
    probe_variants: int,
    min_probe_delta: float,
    min_group_delta: float,
) -> dict[str, Any]:
    reference_by_index = {int(row["index"]): row for row in reference_rows}
    candidate_by_group = group_rows(candidate_rows)
    decisions = [
        decide_group(
            group_id,
            rows,
            reference_by_index=reference_by_index,
            policy=policy,
            probe_variants=probe_variants,
            min_probe_delta=min_probe_delta,
            min_group_delta=min_group_delta,
        )
        for group_id, rows in candidate_by_group.items()
    ]
    decision_by_group = {decision.group_id: decision for decision in decisions}

    simulated_scores = []
    reference_scores = []
    simulated_latency = []
    reference_latency = []
    candidate_latency = []
    request_bytes = 0
    reference_request_bytes = 0
    score_mismatches = 0
    prediction_mismatches = 0
    accepted_rows = 0
    rejected_rows = 0
    task_stats: dict[str, dict[str, Any]] = {}

    for row in candidate_rows:
        idx = int(row["index"])
        ref = reference_by_index[idx]
        decision = decision_by_group[str(row["repeat_group_id"])]
        chosen = row if decision.admit_quantized else ref
        task = str(row.get("task") or row.get("dataset") or "unknown")
        stats = task_stats.setdefault(
            task,
            {
                "rows": 0,
                "accepted_rows": 0,
                "simulated_scores": [],
                "reference_scores": [],
                "candidate_scores": [],
            },
        )
        stats["rows"] += 1
        stats["simulated_scores"].append(float(chosen["longbench_score"]))
        stats["reference_scores"].append(float(ref["longbench_score"]))
        stats["candidate_scores"].append(float(row["longbench_score"]))
        if decision.admit_quantized:
            stats["accepted_rows"] += 1
            accepted_rows += 1
        else:
            rejected_rows += 1
        simulated_scores.append(float(chosen["longbench_score"]))
        reference_scores.append(float(ref["longbench_score"]))
        simulated_latency.append(float(chosen["latency_seconds"]))
        reference_latency.append(float(ref["latency_seconds"]))
        candidate_latency.append(float(row["latency_seconds"]))
        request_bytes += int(chosen["request_storage_nbytes"])
        reference_request_bytes += int(ref["request_storage_nbytes"])
        if abs(float(chosen["longbench_score"]) - float(ref["longbench_score"])) > 1e-12:
            score_mismatches += 1
        if chosen["prediction"] != ref["prediction"]:
            prediction_mismatches += 1

    accepted_groups = [decision for decision in decisions if decision.admit_quantized]
    rejected_groups = [decision for decision in decisions if not decision.admit_quantized]
    prefix_bytes = sum(
        decision.candidate_prefix_bytes if decision.admit_quantized else decision.reference_prefix_bytes
        for decision in decisions
    )
    reference_prefix_bytes = sum(decision.reference_prefix_bytes for decision in decisions)
    candidate_prefix_bytes = sum(decision.candidate_prefix_bytes for decision in decisions)
    shared_total = prefix_bytes + request_bytes
    reference_total = reference_prefix_bytes + reference_request_bytes
    candidate_total = candidate_prefix_bytes + sum(int(row["request_storage_nbytes"]) for row in candidate_rows)

    task_summaries = []
    for task, stats in sorted(task_stats.items()):
        task_summaries.append(
            {
                "task": task,
                "rows": stats["rows"],
                "accepted_rows": stats["accepted_rows"],
                "acceptance_ratio": stats["accepted_rows"] / stats["rows"],
                "simulated_score": mean(stats["simulated_scores"]),
                "reference_score": mean(stats["reference_scores"]),
                "candidate_score": mean(stats["candidate_scores"]),
                "score_delta_vs_reference": mean(stats["simulated_scores"]) - mean(stats["reference_scores"]),
            }
        )

    return {
        "policy": policy,
        "probe_variants": probe_variants,
        "min_probe_delta": min_probe_delta,
        "min_group_delta": min_group_delta,
        "rows": len(candidate_rows),
        "groups": len(decisions),
        "accepted_groups": len(accepted_groups),
        "rejected_groups": len(rejected_groups),
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "acceptance_ratio": len(accepted_groups) / len(decisions) if decisions else 0.0,
        "simulated_score": mean(simulated_scores),
        "reference_score": mean(reference_scores),
        "candidate_score": mean(float(row["longbench_score"]) for row in candidate_rows),
        "score_delta_vs_reference": mean(simulated_scores) - mean(reference_scores),
        "score_delta_vs_candidate_all": mean(simulated_scores) - mean(float(row["longbench_score"]) for row in candidate_rows),
        "score_mismatch_count": score_mismatches,
        "prediction_mismatch_count": prediction_mismatches,
        "simulated_avg_latency_seconds": mean(simulated_latency),
        "reference_avg_latency_seconds": mean(reference_latency),
        "candidate_avg_latency_seconds": mean(candidate_latency),
        "latency_ratio_vs_reference": mean(simulated_latency) / mean(reference_latency),
        "prefix_table_storage_nbytes": prefix_bytes,
        "reference_prefix_table_storage_nbytes": reference_prefix_bytes,
        "candidate_all_prefix_table_storage_nbytes": candidate_prefix_bytes,
        "prefix_table_savings_vs_reference": 1.0 - prefix_bytes / reference_prefix_bytes,
        "shared_total_storage_nbytes": shared_total,
        "reference_shared_total_storage_nbytes": reference_total,
        "candidate_all_shared_total_storage_nbytes": candidate_total,
        "shared_total_savings_vs_reference": 1.0 - shared_total / reference_total,
        "task_summaries": task_summaries,
        "decision_summary": {
            "accepted_mean_delta": mean([d.mean_delta for d in accepted_groups]) if accepted_groups else None,
            "rejected_mean_delta": mean([d.mean_delta for d in rejected_groups]) if rejected_groups else None,
            "accepted_probe_delta": mean([d.probe_delta for d in accepted_groups]) if accepted_groups else None,
            "rejected_probe_delta": mean([d.probe_delta for d in rejected_groups]) if rejected_groups else None,
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Group Admission Policy Analysis",
        "",
        "This offline analysis simulates a policy that quantizes a shared-prefix group only",
        "when an admission rule accepts it; rejected groups fall back to fp16 Shared-KV.",
        "",
        "## Aggregate",
        "",
        f"- Policy: `{report['policy']}`.",
        f"- Probe variants: `{report['probe_variants']}`.",
        f"- Rows/groups: `{report['rows']}` / `{report['groups']}`.",
        f"- Accepted groups: `{report['accepted_groups']}`; rejected groups: `{report['rejected_groups']}`.",
        f"- Acceptance ratio: `{fmt_pct(report['acceptance_ratio'])}`.",
        f"- Simulated score: `{report['simulated_score']:.6f}`.",
        f"- fp16 Shared-KV reference score: `{report['reference_score']:.6f}`.",
        f"- Score delta vs reference: `{report['score_delta_vs_reference']:.6f}`.",
        f"- Score mismatches vs reference: `{report['score_mismatch_count']}`.",
        f"- Prediction mismatches vs reference: `{report['prediction_mismatch_count']}`.",
        f"- Latency ratio vs reference: `{report['latency_ratio_vs_reference']:.2f}x`.",
        "",
        "## Storage",
        "",
        f"- Prefix-table saving vs fp16 Shared-KV: `{fmt_pct(report['prefix_table_savings_vs_reference'])}`.",
        f"- Shared-total saving vs fp16 Shared-KV: `{fmt_pct(report['shared_total_savings_vs_reference'])}`.",
        f"- Simulated shared-total bytes: `{report['shared_total_storage_nbytes']}`.",
        f"- Reference shared-total bytes: `{report['reference_shared_total_storage_nbytes']}`.",
        "",
        "## Task Breakdown",
        "",
        "| Task | Rows | Accepted rows | Acceptance | Sim score | Ref score | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["task_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task"],
                    str(row["rows"]),
                    str(row["accepted_rows"]),
                    fmt_pct(row["acceptance_ratio"]),
                    f"{row['simulated_score']:.6f}",
                    f"{row['reference_score']:.6f}",
                    f"{row['score_delta_vs_reference']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision Summary",
            "",
            "```json",
            json.dumps(report["decision_summary"], indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", choices=["all", "none", "probe_mean", "probe_all", "oracle_mean", "oracle_all"], required=True)
    parser.add_argument("--probe-variants", type=int, default=1)
    parser.add_argument("--min-probe-delta", type=float, default=0.0)
    parser.add_argument("--min-group-delta", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = simulate_policy(
        load_jsonl(args.candidate),
        load_jsonl(args.reference),
        policy=args.policy,
        probe_variants=args.probe_variants,
        min_probe_delta=args.min_probe_delta,
        min_group_delta=args.min_group_delta,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.output.with_suffix(".md"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
