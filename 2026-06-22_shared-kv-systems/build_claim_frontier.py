#!/usr/bin/env python3
"""Build the claim frontier for the Shared-KV systems direction.

This script consolidates existing complete repeated-context artifacts into a
new-direction report. It intentionally separates claim-ready evidence from
storage-only what-if estimates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
INCREMENTAL = PROJECT_ROOT / "reproduce/incremental"
GUARDED_PREFIX_FULL_SUMMARY = OUT_DIR / "guarded_prefix_tq35_s512_e2048_full_0000_2400_summary.json"
CACHED_PREFIX_GATE_SUMMARY = OUT_DIR / "guarded_prefix_tq35_s512_e2048_cached_materialized_multitask_gate_summary.json"
CACHED_PREFIX_FULL_SUMMARY = OUT_DIR / "guarded_prefix_tq35_s512_e2048_cached_materialized_full_0000_2400_summary.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_bytes(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    units = [("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]
    for suffix, scale in units:
        if abs(float(value)) >= scale:
            return f"{float(value) / scale:.2f} {suffix}"
    return f"{int(value)} B"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def find_run(quality: dict[str, Any], name: str) -> dict[str, Any]:
    for run in quality["runs"]:
        if run["name"] == name:
            return run
    raise KeyError(name)


def storage_estimate(run: dict[str, Any]) -> int | None:
    estimate = run.get("prefix_shared_storage_estimate")
    if not estimate:
        return None
    return int(estimate["estimated_prefix_shared_cache_storage_nbytes"])


def main() -> None:
    quality = load_json(INCREMENTAL / "repeated_context_quality_summary_current.json")
    plan = load_json(INCREMENTAL / "repeated_context_shared_kv_plan_current.json")
    reference = load_json(INCREMENTAL / "repeated_context_reference_runtime_full_summary.json")
    overhead = load_json(INCREMENTAL / "repeated_context_reference_materialization_overhead_full.json")
    split = load_json(INCREMENTAL / "repeated_context_split_full_scoreparity_summary.json")
    eager = load_json(INCREMENTAL / "repeated_context_materialize_eager_vs_clone_full_scoreparity_summary.json")
    guarded_prefix = load_json(GUARDED_PREFIX_FULL_SUMMARY) if GUARDED_PREFIX_FULL_SUMMARY.exists() else None
    cached_prefix_gate = load_json(CACHED_PREFIX_GATE_SUMMARY) if CACHED_PREFIX_GATE_SUMMARY.exists() else None
    cached_prefix_full = load_json(CACHED_PREFIX_FULL_SUMMARY) if CACHED_PREFIX_FULL_SUMMARY.exists() else None

    full = find_run(quality, "full_cache")
    tq25 = find_run(quality, "turboquant_2p5")
    tq35 = find_run(quality, "turboquant_3p5")
    plan_summary = plan["plan"]["summary"]
    ref_summary = reference["summary"]
    overhead_summary = overhead["summary"]
    shared_storage = int(overhead_summary["storage"]["shared_total_storage_nbytes"])
    materialized_storage = int(overhead_summary["storage"]["request_materialized_nbytes"])
    shared_score = float(ref_summary["score"]) * 100.0
    full_score = float(full["score"])
    shared_latency = float(ref_summary["latency"]["avg_latency_seconds"])
    full_latency = float(full["avg_latency_seconds"])

    tq25_raw_storage = int(tq25["prefix_shared_storage_estimate"]["raw_cache_storage_nbytes"])
    tq35_raw_storage = int(tq35["prefix_shared_storage_estimate"]["raw_cache_storage_nbytes"])
    tq25_shared_storage = storage_estimate(tq25)
    tq35_shared_storage = storage_estimate(tq35)

    # Storage-only what-if estimates. These are not quality claims.
    tq25_ratio = float(tq25["avg_cache_storage_ratio"])
    tq35_ratio = float(tq35["avg_cache_storage_ratio"])
    shared_prefix_bytes = int(plan_summary["shared_prefix_kv_bytes"])
    request_local_bytes = int(plan_summary["request_local_kv_bytes"])
    metadata_bytes = int(plan_summary["metadata_bytes"])
    prefix_tq25_local_fp16 = int(shared_prefix_bytes * tq25_ratio + request_local_bytes + metadata_bytes)
    prefix_tq35_local_fp16 = int(shared_prefix_bytes * tq35_ratio + request_local_bytes + metadata_bytes)
    prefix_fp16_local_tq25 = int(shared_prefix_bytes + request_local_bytes * tq25_ratio + metadata_bytes)
    prefix_fp16_local_tq35 = int(shared_prefix_bytes + request_local_bytes * tq35_ratio + metadata_bytes)

    rows = [
        {
            "name": "Full fp16 independent",
            "status": "claim-ready baseline",
            "quality_source": "measured full run",
            "score": full_score,
            "latency_seconds": full_latency,
            "persistent_kv_bytes": materialized_storage,
            "memory_delta_vs_full": 0.0,
            "quality_delta_vs_full": 0.0,
            "notes": "Each request materializes its own prompt and decode KV.",
        },
        {
            "name": "fp16 Shared-KV reference",
            "status": "claim-ready systems result",
            "quality_source": "measured full run, 2400/2400 clone parity",
            "score": shared_score,
            "latency_seconds": shared_latency,
            "persistent_kv_bytes": shared_storage,
            "memory_delta_vs_full": 1.0 - shared_storage / materialized_storage,
            "quality_delta_vs_full": shared_score - full_score,
            "notes": "Prefix table is stored once; stock attention still materializes prefix+local tensors.",
        },
        {
            "name": "TurboQuant 2.5 independent",
            "status": "measured negative codec baseline",
            "quality_source": "measured full run",
            "score": float(tq25["score"]),
            "latency_seconds": float(tq25["avg_latency_seconds"]),
            "persistent_kv_bytes": tq25_raw_storage,
            "memory_delta_vs_full": 1.0 - tq25_raw_storage / materialized_storage,
            "quality_delta_vs_full": float(tq25["score"]) - full_score,
            "notes": "Large storage reduction, but quality and latency fail the systems claim.",
        },
        {
            "name": "TurboQuant 3.5 independent",
            "status": "measured Pareto comparator",
            "quality_source": "measured full run",
            "score": float(tq35["score"]),
            "latency_seconds": float(tq35["avg_latency_seconds"]),
            "persistent_kv_bytes": tq35_raw_storage,
            "memory_delta_vs_full": 1.0 - tq35_raw_storage / materialized_storage,
            "quality_delta_vs_full": float(tq35["score"]) - full_score,
            "notes": "Nearly the same memory as fp16 Shared-KV, but lower quality and much higher latency.",
        },
    ]
    if guarded_prefix:
        guarded_score = float(guarded_prefix["candidate_score"]) * 100.0
        guarded_latency = float(guarded_prefix["candidate_avg_latency_seconds"])
        guarded_storage = int(guarded_prefix["candidate_shared_total_storage_nbytes"])
        rows.append(
            {
                "name": "Guarded TQ3.5 shared prefix, fp16 local",
                "status": "full storage-quality result; no speed claim",
                "quality_source": (
                    f"measured full run, {guarded_prefix['records']} rows; "
                    f"{guarded_prefix['score_mismatch_count']} score mismatches vs fp16 Shared-KV"
                ),
                "score": guarded_score,
                "latency_seconds": guarded_latency,
                "persistent_kv_bytes": guarded_storage,
                "memory_delta_vs_full": 1.0 - guarded_storage / materialized_storage,
                "quality_delta_vs_full": guarded_score - full_score,
                "notes": (
                    "Shared prefix table keeps first 512 and last 2048 prefix tokens fp16, "
                    "quantizes the middle prefix with TurboQuant 3.5, and keeps request-local KV fp16."
                ),
            }
        )
    rows.extend(
        [
        {
            "name": "TurboQuant 2.5 plus shared-prefix estimate",
            "status": "storage-only estimate",
            "quality_source": "not validated in shared-cache runtime",
            "score": float(tq25["score"]),
            "latency_seconds": None,
            "persistent_kv_bytes": tq25_shared_storage,
            "memory_delta_vs_full": 1.0 - tq25_shared_storage / materialized_storage if tq25_shared_storage else None,
            "quality_delta_vs_full": float(tq25["score"]) - full_score,
            "notes": "Potential multiplicative storage frontier, but current quality gap is too large.",
        },
        {
            "name": "TurboQuant 3.5 plus shared-prefix estimate",
            "status": "storage-only estimate",
            "quality_source": "not validated in shared-cache runtime",
            "score": float(tq35["score"]),
            "latency_seconds": None,
            "persistent_kv_bytes": tq35_shared_storage,
            "memory_delta_vs_full": 1.0 - tq35_shared_storage / materialized_storage if tq35_shared_storage else None,
            "quality_delta_vs_full": float(tq35["score"]) - full_score,
            "notes": "Strong storage frontier if quality can be recovered by prefix-aware quantization.",
        },
        {
            "name": "Quantized shared prefix, fp16 local estimate (2.5)",
            "status": "next-experiment storage target",
            "quality_source": "unknown; must run full protocol",
            "score": None,
            "latency_seconds": None,
            "persistent_kv_bytes": prefix_tq25_local_fp16,
            "memory_delta_vs_full": 1.0 - prefix_tq25_local_fp16 / materialized_storage,
            "quality_delta_vs_full": None,
            "notes": "Compresses the dominant shared prefix table while preserving local suffix/decode in fp16.",
        },
        {
            "name": "Quantized shared prefix, fp16 local estimate (3.5)",
            "status": "next-experiment storage target",
            "quality_source": "unknown; must run full protocol",
            "score": None,
            "latency_seconds": None,
            "persistent_kv_bytes": prefix_tq35_local_fp16,
            "memory_delta_vs_full": 1.0 - prefix_tq35_local_fp16 / materialized_storage,
            "quality_delta_vs_full": None,
            "notes": "Most plausible TurboQuant integration: large storage saving while leaving request-local states exact.",
        },
        {
            "name": "fp16 shared prefix, quantized local estimate (2.5)",
            "status": "low-value storage estimate",
            "quality_source": "unknown; likely small upside",
            "score": None,
            "latency_seconds": None,
            "persistent_kv_bytes": prefix_fp16_local_tq25,
            "memory_delta_vs_full": 1.0 - prefix_fp16_local_tq25 / materialized_storage,
            "quality_delta_vs_full": None,
            "notes": "Only compresses request-local KV; total gain is small because prefix dominates storage.",
        },
        {
            "name": "fp16 shared prefix, quantized local estimate (3.5)",
            "status": "low-value storage estimate",
            "quality_source": "unknown; likely small upside",
            "score": None,
            "latency_seconds": None,
            "persistent_kv_bytes": prefix_fp16_local_tq35,
            "memory_delta_vs_full": 1.0 - prefix_fp16_local_tq35 / materialized_storage,
            "quality_delta_vs_full": None,
            "notes": "Useful as a sanity check, not a strong contribution by itself.",
        },
        ]
    )

    derived = {
        "claim_ready_result": {
            "score_delta_fp16_shared_minus_full": shared_score - full_score,
            "latency_speedup_full_over_shared": full_latency / shared_latency,
            "storage_savings_shared_vs_full": 1.0 - shared_storage / materialized_storage,
            "clone_parity_rows": reference["clone_comparison"]["compared"],
            "clone_parity_mismatches": reference["clone_comparison"]["mismatch_count"],
        },
        "pareto_observation": {
            "tq35_memory_over_shared_fp16": tq35_raw_storage / shared_storage,
            "shared_fp16_memory_over_tq35": shared_storage / tq35_raw_storage,
            "shared_fp16_quality_advantage_over_tq35": shared_score - float(tq35["score"]),
            "shared_fp16_latency_speedup_over_tq35": float(tq35["avg_latency_seconds"]) / shared_latency,
            "shared_fp16_quality_advantage_over_tq25": shared_score - float(tq25["score"]),
            "shared_fp16_latency_speedup_over_tq25": float(tq25["avg_latency_seconds"]) / shared_latency,
        },
        "split_attention_limitation": {
            "split_score_mismatches": split["summary"]["score_mismatch_count"],
            "split_prediction_mismatches": split["summary"]["prediction_mismatch_count"],
            "materialized_eager_score_mismatches": eager["summary"]["score_mismatch_count"],
            "materialized_eager_prediction_mismatches": eager["summary"]["prediction_mismatch_count"],
        },
        "storage_decomposition": {
            "shared_prefix_table_bytes": shared_prefix_bytes,
            "request_local_bytes": request_local_bytes,
            "metadata_bytes": metadata_bytes,
            "shared_prefix_fraction_of_shared_total": shared_prefix_bytes / (shared_prefix_bytes + request_local_bytes),
            "request_local_fraction_of_shared_total": request_local_bytes / (shared_prefix_bytes + request_local_bytes),
        },
    }
    if guarded_prefix:
        derived["guarded_prefix_full_result"] = {
            "records": guarded_prefix["records"],
            "groups": guarded_prefix["groups"],
            "score_delta_vs_fp16_shared_points": (
                float(guarded_prefix["candidate_score"]) - float(guarded_prefix["reference_score"])
            )
            * 100.0,
            "score_delta_vs_full_fp16_points": float(guarded_prefix["candidate_score"]) * 100.0 - full_score,
            "score_mismatch_count": guarded_prefix["score_mismatch_count"],
            "prediction_mismatch_count": guarded_prefix["prediction_mismatch_count"],
            "latency_ratio_vs_fp16_shared": (
                float(guarded_prefix["candidate_avg_latency_seconds"])
                / float(guarded_prefix["reference_avg_latency_seconds"])
            ),
            "shared_total_savings_vs_fp16_shared": guarded_prefix["shared_total_savings_ratio"],
            "storage_savings_vs_full_fp16": (
                1.0 - int(guarded_prefix["candidate_shared_total_storage_nbytes"]) / materialized_storage
            ),
        }
    cached_source = cached_prefix_full or cached_prefix_gate
    if cached_source:
        source_label = "full" if cached_prefix_full else "balanced gate"
        cached_latency = cached_source.get("candidate_avg_latency_seconds")
        cached_storage = int(cached_source["candidate_shared_total_storage_nbytes"])
        cached_transient_bytes = int(
            cached_source.get("candidate_materialized_prefix_cache_nbytes", 0)
        )
        derived["cached_materialized_prefix_gate"] = {
            "source_label": source_label,
            "records": cached_source["records"],
            "groups": cached_source["groups"],
            "score_delta_vs_fp16_shared_points": (
                (float(cached_source["candidate_score"]) - float(cached_source["reference_score"])) * 100.0
                if "score_delta" not in cached_source
                else float(cached_source["score_delta"]) * 100.0
            ),
            "avg_latency_seconds": cached_latency,
            "reference_avg_latency_seconds": cached_source.get("reference_avg_latency_seconds"),
            "latency_ratio_vs_fp16_shared": (
                float(cached_latency)
                / float(cached_source["reference_avg_latency_seconds"])
                if cached_latency is not None
                else None
            ),
            "shared_total_savings_vs_fp16_shared": cached_source["shared_total_savings_ratio"],
            "transient_materialized_prefix_cache_bytes": cached_transient_bytes,
            "task_latency_ratios": {
                row["task"]: row["latency_ratio"]
                for row in cached_source.get("rows", cached_source.get("task_summaries", []))
            },
        }
        cached_score = float(cached_source["candidate_score"]) * 100.0
        rows.append(
            {
                "name": "Guarded TQ3.5 shared prefix + dense prefix cache",
                "status": "runtime diagnostic; no final speed claim",
                "quality_source": (
                    f"{source_label} run, {cached_source['records']} rows; "
                    f"extra transient prefix cache {fmt_bytes(cached_transient_bytes)}"
                ),
                "score": cached_score,
                "latency_seconds": float(cached_latency) if cached_latency is not None else None,
                "persistent_kv_bytes": cached_storage,
                "memory_delta_vs_full": 1.0 - cached_storage / materialized_storage,
                "quality_delta_vs_full": cached_score - full_score,
                "notes": (
                    "Same persistent storage as the guarded prefix result, but keeps one dense "
                    "materialization of each quantized shared prefix layer as transient runtime state."
                ),
            }
        )

    report = {
        "source_artifacts": {
            "quality_summary": str(INCREMENTAL / "repeated_context_quality_summary_current.json"),
            "shared_kv_plan": str(INCREMENTAL / "repeated_context_shared_kv_plan_current.json"),
            "reference_runtime": str(INCREMENTAL / "repeated_context_reference_runtime_full_summary.json"),
            "materialization_overhead": str(INCREMENTAL / "repeated_context_reference_materialization_overhead_full.json"),
            "split_parity": str(INCREMENTAL / "repeated_context_split_full_scoreparity_summary.json"),
        },
        "rows": rows,
        "derived": derived,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "claim_frontier.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Shared-KV Claim Frontier",
        "",
        "This report consolidates complete generated repeated-context artifacts for the new Shared-KV systems direction.",
        "Storage-only estimates are explicitly marked and are not quality claims.",
        "",
        "## Claim-Ready Result",
        "",
        f"- fp16 Shared-KV score delta vs full fp16: `{derived['claim_ready_result']['score_delta_fp16_shared_minus_full']:.4f}` points.",
        f"- fp16 Shared-KV latency speedup vs full fp16: `{derived['claim_ready_result']['latency_speedup_full_over_shared']:.2f}x`.",
        f"- fp16 Shared-KV persistent storage saving vs materialized full fp16: `{fmt_pct(derived['claim_ready_result']['storage_savings_shared_vs_full'])}`.",
        f"- Clone parity: `{derived['claim_ready_result']['clone_parity_rows']}` rows compared, `{derived['claim_ready_result']['clone_parity_mismatches']}` mismatches.",
        "",
        "## Pareto Observation",
        "",
        f"- TurboQuant 3.5 independent persistent KV / fp16 Shared-KV persistent KV: `{derived['pareto_observation']['tq35_memory_over_shared_fp16']:.3f}x`.",
        f"- fp16 Shared-KV is `{derived['pareto_observation']['shared_fp16_quality_advantage_over_tq35']:.2f}` points higher than TurboQuant 3.5 and `{derived['pareto_observation']['shared_fp16_latency_speedup_over_tq35']:.2f}x` faster.",
        f"- fp16 Shared-KV is `{derived['pareto_observation']['shared_fp16_quality_advantage_over_tq25']:.2f}` points higher than TurboQuant 2.5 and `{derived['pareto_observation']['shared_fp16_latency_speedup_over_tq25']:.2f}x` faster.",
        "",
        "Interpretation: in this repeated-context workload, exact cross-request sharing is a stronger first move than independent per-request low-bit compression. At roughly the same persistent KV size as TurboQuant 3.5, fp16 Shared-KV preserves quality and is much faster.",
        "",
    ]
    if guarded_prefix:
        guarded = derived["guarded_prefix_full_result"]
        lines.extend(
            [
                "## Guarded TurboQuant Shared-Prefix Result",
                "",
                f"- Full guarded-prefix run: `{guarded['records']}` rows, `{guarded['groups']}` groups.",
                f"- Score delta vs fp16 Shared-KV reference: `{guarded['score_delta_vs_fp16_shared_points']:.4f}` points.",
                f"- Score mismatches vs fp16 Shared-KV: `{guarded['score_mismatch_count']}`; prediction mismatches: `{guarded['prediction_mismatch_count']}`.",
                f"- Shared-total storage saving vs fp16 Shared-KV: `{fmt_pct(guarded['shared_total_savings_vs_fp16_shared'])}`.",
                f"- Persistent storage saving vs independent full fp16: `{fmt_pct(guarded['storage_savings_vs_full_fp16'])}`.",
                f"- Latency is `{guarded['latency_ratio_vs_fp16_shared']:.2f}x` slower than fp16 Shared-KV reference in the current Python materialized prototype.",
                "",
                "Interpretation: this is a full storage-quality result for a TurboQuant-composed Shared-KV table. It should not be reported as a speedup until the quantized prefix table has a non-materializing or fused decode path.",
                "",
            ]
        )
    if cached_source:
        cached = derived["cached_materialized_prefix_gate"]
        latency_items = ", ".join(
            f"`{task}` `{ratio:.2f}x`"
            for task, ratio in cached["task_latency_ratios"].items()
        )
        aggregate_latency = (
            f"`{cached['latency_ratio_vs_fp16_shared']:.2f}x` aggregate, "
            if cached["latency_ratio_vs_fp16_shared"] is not None
            else ""
        )
        lines.extend(
            [
                "## Cached-Materialized Prefix Runtime Diagnostic",
                "",
                f"- {cached['source_label'].capitalize()} run: `{cached['records']}` rows, `{cached['groups']}` groups.",
                f"- Score delta vs fp16 Shared-KV reference: `{cached['score_delta_vs_fp16_shared_points']:.4f}` points.",
                f"- Shared-total storage saving vs fp16 Shared-KV: `{fmt_pct(cached['shared_total_savings_vs_fp16_shared'])}`.",
                f"- Latency ratios vs fp16 Shared-KV: {aggregate_latency}by task {latency_items}.",
                f"- Additional transient dense prefix-cache bytes: `{fmt_bytes(cached['transient_materialized_prefix_cache_bytes'])}`.",
                "",
                "Interpretation: caching one dense materialization of the quantized shared prefix removes most of the Python dequantization/materialization latency, but it is a diagnostic tradeoff rather than a final kernel result because it uses extra transient dense memory.",
                "",
            ]
        )
    lines.extend(
        [
        "## Frontier Table",
        "",
        "| Method | Status | Score | Delta vs fp16 | Latency | Persistent KV | Saving vs full | Evidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row["status"],
                    fmt_num(row["score"]),
                    fmt_num(row["quality_delta_vs_full"]),
                    f"{fmt_num(row['latency_seconds'], 3)}s" if row["latency_seconds"] is not None else "n/a",
                    fmt_bytes(row["persistent_kv_bytes"]),
                    fmt_pct(row["memory_delta_vs_full"]),
                    row["quality_source"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Storage Decomposition",
            "",
            f"- Shared prefix table: `{fmt_bytes(shared_prefix_bytes)}`.",
            f"- Request-local suffix/decode KV: `{fmt_bytes(request_local_bytes)}`.",
            f"- Metadata: `{fmt_bytes(metadata_bytes)}`.",
            f"- Shared prefix fraction of non-metadata shared storage: `{fmt_pct(derived['storage_decomposition']['shared_prefix_fraction_of_shared_total'])}`.",
            "",
            "This makes suffix-only quantization a weak next contribution. The dominant experiment is prefix-table quantization with fp16 request-local KV.",
            "",
            "## Current Limitation",
            "",
            f"- Python split attention failed full row-level parity: `{derived['split_attention_limitation']['split_score_mismatches']}` score mismatches and `{derived['split_attention_limitation']['split_prediction_mismatches']}` prediction mismatches.",
            f"- Materialized eager vs clone also has backend drift: `{derived['split_attention_limitation']['materialized_eager_score_mismatches']}` score mismatches and `{derived['split_attention_limitation']['materialized_eager_prediction_mismatches']}` prediction mismatches.",
            "",
            "Do not claim non-materializing split attention yet. The next implementation target is SDPA-equivalent prefix-aware attention or a predeclared relaxed score-parity protocol.",
            "",
        ]
    )
    (OUT_DIR / "claim_frontier.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
