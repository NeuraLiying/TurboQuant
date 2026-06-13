#!/usr/bin/env python3
"""Build an automated first-stage reproduction report from current artifacts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


CANONICAL_PROJECT_ROOT = Path(os.environ.get("TURBOQUANT_PROJECT_ROOT", "/home/liying/projects/turboquant")).expanduser()
PROJECT_ROOT = CANONICAL_PROJECT_ROOT if CANONICAL_PROJECT_ROOT.exists() else Path(__file__).resolve().parents[1]
TABLE1_PAPER = [
    {
        "method": "Full Cache",
        "kv_size": 16.0,
        "SingleQA": 45.29,
        "MultiQA": 45.16,
        "Summarization": 26.55,
        "Few shot": 68.38,
        "Synthetic": 59.54,
        "Code": 46.28,
        "Average": 50.06,
    },
    {
        "method": "TurboQuant",
        "kv_size": 2.5,
        "SingleQA": 44.16,
        "MultiQA": 44.96,
        "Summarization": 24.80,
        "Few shot": 68.01,
        "Synthetic": 59.65,
        "Code": 45.76,
        "Average": 49.44,
    },
    {
        "method": "TurboQuant",
        "kv_size": 3.5,
        "SingleQA": 45.01,
        "MultiQA": 45.31,
        "Summarization": 26.00,
        "Few shot": 68.63,
        "Synthetic": 59.95,
        "Code": 46.17,
        "Average": 50.06,
    },
]


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def status(value: bool) -> str:
    return "complete" if value else "incomplete"


def table1_rows(table1_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_key = {(row["method"], float(row["kv_size"])): row for row in table1_summary}
    for paper in TABLE1_PAPER:
        local = by_key.get((paper["method"], paper["kv_size"]), {})
        rows.append({"paper": paper, "local": local})
    return rows


def figure5_top1_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row.get("topk", -1)) == 1 and int(row.get("bits", -1)) in {2, 4}
    ]


def count_keys(counts: dict[str, Any] | None) -> str:
    if not counts:
        return ""
    return ",".join(str(key) for key in sorted(counts, key=lambda value: float(value)))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    refresh = load_json(Path(args.refresh), {})
    table1 = load_json(Path(args.table1_summary), [])
    figure4 = load_json(Path(args.figure4_heatmap), [])
    figure5 = load_json(Path(args.figure5_summary), [])
    table2 = load_json(Path(args.table2_summary), [])
    kv_policy = load_json(Path(args.kv_policy), {})
    figure3 = load_json(Path(args.figure3_summary), {})

    refresh_summary = refresh.get("summaries", {})
    table1_manifest = refresh_summary.get("table1_manifest", {})
    table1_plan = refresh_summary.get("table1_plan", {})
    figure4_plan = refresh_summary.get("figure4_plan", {})
    asset = refresh_summary.get("asset_audit", {})

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sources": {
            "refresh": args.refresh,
            "table1_summary": args.table1_summary,
            "figure4_heatmap": args.figure4_heatmap,
            "figure5_summary": args.figure5_summary,
            "table2_summary": args.table2_summary,
            "kv_policy": args.kv_policy,
            "figure3_summary": args.figure3_summary,
        },
        "completion": {
            "table1_complete": table1_manifest.get("missing_dataset", 0) == 0 and table1_manifest.get("complete", 0) >= 96,
            "figure4_complete": figure4_plan.get("all_status_counts", {}).get("missing_dataset", 0) == 0,
            "figure5_dbpedia_complete": bool(figure5),
            "figure5_glove_complete": False,
            "core_validation_complete": bool(figure3),
        },
        "asset_audit": asset,
        "table1": {
            "paper_rows": TABLE1_PAPER,
            "local_rows": table1,
            "comparison_rows": table1_rows(table1),
            "manifest": table1_manifest,
            "plan": table1_plan,
        },
        "figure4": {
            "paper_scores": {"Full-Precision": 0.997, "TurboQuant": 0.997},
            "local_runs": figure4,
            "plan": figure4_plan,
        },
        "figure5": {
            "dbpedia_rows": figure5,
            "dbpedia_top1_rows": figure5_top1_rows(figure5),
            "glove_available": False,
        },
        "table2": {
            "rows": table2,
        },
        "kv_policy": kv_policy,
        "figure3": figure3,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    completion = report["completion"]
    table1 = report["table1"]
    figure4 = report["figure4"]
    figure5 = report["figure5"]
    lines = [
        "# TurboQuant First-Stage Reproduction Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Scope: `meta-llama/Llama-3.1-8B-Instruct`; compare Full Cache / Full-Precision vs TurboQuant only.",
        "",
        "## Completion",
        "",
        f"- Table 1 full reproduction: {status(completion['table1_complete'])}",
        f"- Figure 4 full reproduction: {status(completion['figure4_complete'])}",
        f"- Figure 5 DBpedia reproduction: {status(completion['figure5_dbpedia_complete'])}",
        f"- Figure 5 GloVe reproduction: {status(completion['figure5_glove_complete'])}",
        f"- Core algorithm validation: {status(completion['core_validation_complete'])}",
        "",
        "Current blocking data gaps remain explicit in the refresh summary below.",
        "",
        "## Refresh Summary",
        "",
        f"- Asset audit: `{report['asset_audit']}`",
        f"- Table 1 manifest: `{table1['manifest']}`",
        f"- Table 1 plan: `{table1['plan']}`",
        f"- Figure 4 plan: `{figure4['plan']}`",
        "",
        "## Table 1",
        "",
        "| Method | KV | Paper SingleQA | Paper MultiQA | Paper Summ. | Paper Few shot | Paper Synthetic | Paper Code | Paper Avg | Local SingleQA | Local MultiQA | Local Summ. | Local Few shot | Local Synthetic | Local Code | Local Available Avg | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in table1["comparison_rows"]:
        paper = row["paper"]
        local = row["local"]
        categories = local.get("categories", {})
        coverage = local.get("coverage", {})
        coverage_text = "; ".join(
            f"{name}: {','.join(value.get('available', [])) or 'none'}" for name, value in coverage.items()
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    paper["method"],
                    fmt(paper["kv_size"], 1),
                    fmt(paper["SingleQA"], 2),
                    fmt(paper["MultiQA"], 2),
                    fmt(paper["Summarization"], 2),
                    fmt(paper["Few shot"], 2),
                    fmt(paper["Synthetic"], 2),
                    fmt(paper["Code"], 2),
                    fmt(paper["Average"], 2),
                    fmt(categories.get("SingleQA"), 2),
                    fmt(categories.get("MultiQA"), 2),
                    fmt(categories.get("Summarization"), 2),
                    fmt(categories.get("Few shot"), 2),
                    fmt(categories.get("Synthetic"), 2),
                    fmt(categories.get("Code"), 2),
                    fmt(local.get("average_available_categories"), 2),
                    coverage_text,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Table 1 local values are partial until all LongBench / LongBench-E datasets are materialized.",
            "",
            "## Figure 4",
            "",
            "| Method | KV | Paper Score | Local Examples | Local Score | Token Limits | Depth Percents | Cache Ratio |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for run in figure4["local_runs"]:
        overall = run.get("overall", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    run.get("method", ""),
                    fmt(run.get("kv_size"), 1),
                    fmt(run.get("paper_score"), 3),
                    str(overall.get("num_examples", "")),
                    fmt(overall.get("score"), 4),
                    ",".join(str(value) for value in run.get("token_limits", [])),
                    ",".join(str(value) for value in run.get("depth_percents", [])),
                    fmt(overall.get("avg_cache_storage_ratio"), 4),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Figure 4 local values currently cover only materialized 16k EN data; missing token lengths remain absent.",
            "",
            "## Figure 5 DBpedia",
            "",
            "| Dataset | Dim | Bits | Top-k | Recall | Database | Queries |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in figure5["dbpedia_top1_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("dataset_key", ""),
                    str(row.get("dimension", "")),
                    str(row.get("bits", "")),
                    str(row.get("topk", "")),
                    fmt(row.get("recall_1_at_k"), 4),
                    str(row.get("num_database", "")),
                    str(row.get("num_queries", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "GloVe 200d is still unavailable, so Figure 5 is complete only for DBpedia.",
            "",
            "## Table 2 Timing",
            "",
            "| Dimension | Source | Local Seconds | Paper Seconds | Local/Paper |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["table2"]["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dimension", "")),
                    row.get("source", ""),
                    fmt(row.get("local_mean_seconds"), 6),
                    fmt(row.get("paper_turboquant_seconds"), 6),
                    fmt(row.get("local_over_paper"), 3),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## KV Compression Policy",
            "",
            "| Run | Regular bits | Outlier bits | Outlier channels | Effective KV bits | Cache ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["kv_policy"].get("runs", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("label", ""),
                    count_keys(row.get("regular_bit_values")),
                    count_keys(row.get("outlier_bit_values")),
                    count_keys(row.get("outlier_count_values")),
                    fmt(row.get("avg_effective_index_bits"), 2),
                    fmt(row.get("avg_cache_storage_ratio"), 4),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The current fractional-bit implementation stores packed regular channels plus higher-bit outlier channels, then dequantizes before attention. It is a faithful Python-level reproduction path, not a fused-kernel speed reproduction.",
        ]
    )
    lines.extend(
        [
            "",
            "## Core Validation",
            "",
            "| Bits | MSE mean | Inner-product error mean |",
            "| ---: | ---: | ---: |",
        ]
    )
    for bits, row in sorted(report["figure3"].items(), key=lambda item: int(item[0])):
        lines.append(f"| {bits} | {fmt(row.get('mse_mean'), 4)} | {fmt(row.get('mse_ip_error_mean'), 6)} |")
    lines.extend(
        [
            "",
            "## Sources",
            "",
        ]
    )
    for name, source in report["sources"].items():
        lines.append(f"- `{name}`: `{source}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", default=str(PROJECT_ROOT / "reproduce/logs/reproduction_state_refresh_2026-06-11_current.json"))
    parser.add_argument("--table1-summary", default=str(PROJECT_ROOT / "reproduce/runs/table1_llama_available_summary.json"))
    parser.add_argument("--figure4-heatmap", default=str(PROJECT_ROOT / "reproduce/runs/figure4_needle_available_heatmap.json"))
    parser.add_argument("--figure5-summary", default=str(PROJECT_ROOT / "reproduce/runs/figure5_turboquant_dbpedia_fullscale.json"))
    parser.add_argument("--table2-summary", default=str(PROJECT_ROOT / "reproduce/runs/table2_quantization_time_summary.json"))
    parser.add_argument("--kv-policy", default=str(PROJECT_ROOT / "reproduce/runs/kv_compression_policy_summary.json"))
    parser.add_argument("--figure3-summary", default=str(PROJECT_ROOT / "reproduce/runs/figure3_core_d256_summary.json"))
    parser.add_argument("--output-prefix", default=str(PROJECT_ROOT / "reproduce/FIRST_STAGE_REPORT_CURRENT"))
    args = parser.parse_args()

    report = build_report(args)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "md": str(md_path), "completion": report["completion"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
