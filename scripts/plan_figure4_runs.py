#!/usr/bin/env python3
"""Plan Figure 4 Needle-In-A-Haystack runs from local dataset keys."""

from __future__ import annotations

import argparse
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, concatenate_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN_CONFIGS = ["4k", "8k", "16k", "32k", "65k", "104k"]
POSITIONS = ["start", "middle", "end"]
METHODS = {
    "full_precision": {"label": "Full-Precision", "cache_mode": "full", "kv_bits": 16.0},
    "turboquant_2_5bit": {"label": "TurboQuant", "cache_mode": "turboquant", "kv_bits": 2.5},
}


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_from_config(data_cfg: dict[str, Any]) -> Dataset:
    arrow_files = data_cfg.get("arrow_files") or []
    parquet_files = data_cfg.get("parquet_files") or []
    if arrow_files:
        datasets = [Dataset.from_file(path) for path in arrow_files]
    elif parquet_files:
        datasets = [Dataset.from_parquet(path) for path in parquet_files]
    else:
        raise ValueError("dataset config must provide arrow_files or parquet_files")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def load_jsonl_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indices = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                indices.add(int(json.loads(line)["index"]))
    return indices


def inspect_grid(data_cfg: dict[str, Any], positions: list[str], examples_per_position_lang: int) -> dict[str, Any]:
    dataset = load_dataset_from_config(data_cfg)
    distractor_langs = sorted(set(dataset["distractor_lang"]))
    counts: dict[str, int] = {}
    missing_cells = []
    expected_examples = 0
    for position in positions:
        for distractor_lang in distractor_langs:
            group_size = sum(
                1
                for row in dataset
                if row["needle_position"] == position and row["distractor_lang"] == distractor_lang
            )
            selected = min(group_size, examples_per_position_lang)
            counts[f"{position}/{distractor_lang}"] = selected
            expected_examples += selected
            if selected < examples_per_position_lang:
                missing_cells.append(
                    {
                        "needle_position": position,
                        "distractor_lang": distractor_lang,
                        "available": group_size,
                        "requested": examples_per_position_lang,
                    }
                )
    return {
        "num_dataset_examples": len(dataset),
        "positions": positions,
        "distractor_langs": distractor_langs,
        "examples_per_position_lang": examples_per_position_lang,
        "expected_examples": expected_examples,
        "cell_counts": counts,
        "missing_cells": missing_cells,
    }


def output_path(
    token_config: str,
    lang: str,
    method_name: str,
    expected_examples: int | None,
    positions: list[str],
    examples_per_position_lang: int,
    output_tag: str | None,
) -> Path:
    tag_suffix = f"_{output_tag}" if output_tag else ""
    if (
        token_config == "16k"
        and lang == "en"
        and expected_examples == 24
        and positions == POSITIONS
        and examples_per_position_lang == 1
        and output_tag is None
    ):
        if method_name == "full_precision":
            return PROJECT_ROOT / "reproduce/runs/needle_16k_full_precision_grid_24.jsonl"
        if method_name == "turboquant_2_5bit":
            return PROJECT_ROOT / "reproduce/runs/needle_16k_turboquant_2_5bit_grid_24.jsonl"
    suffix = f"grid_{expected_examples}" if expected_examples is not None else "grid"
    return PROJECT_ROOT / f"reproduce/runs/needle_{token_config}_{lang}_{method_name}_{suffix}{tag_suffix}.jsonl"


def eval_command(
    entry: dict[str, Any],
    *,
    gpu: int,
    positions: list[str],
    distractor_langs: list[str],
    max_new_tokens: int,
    codebook_grid_size: int,
    resume: bool,
) -> str:
    method = METHODS[entry["method_name"]]
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "conda run -n turboquant python experiments/needle/run_needle_eval.py",
        f"--dataset-key {q(entry['dataset_key'])}",
        "--device cuda:0",
        f"--cache-mode {method['cache_mode']}",
        f"--max-new-tokens {max_new_tokens}",
        f"--examples-per-position-lang {entry['examples_per_position_lang']}",
        "--positions",
        *[q(position) for position in positions],
        "--distractor-langs",
        *[q(lang) for lang in distractor_langs],
    ]
    if method["cache_mode"] == "turboquant":
        parts.extend([f"--kv-bits {method['kv_bits']}", f"--codebook-grid-size {codebook_grid_size}"])
    if resume:
        parts.append("--resume")
    parts.append(f"--output {q(entry['output'])}")
    return " ".join(parts)


def summarize_command(jsonl_path: Path, aggregate_path: Path) -> str:
    return f"conda run -n turboquant python scripts/summarize_needle_results.py {q(jsonl_path)} --output {q(aggregate_path)}"


def build_heatmap_command(combined_inputs: dict[str, str], output_prefix: Path, *, plot: bool) -> str:
    parts = [
        "conda run -n turboquant python scripts/build_needle_heatmap.py",
        f"--output-prefix {q(output_prefix)}",
        "--title 'Figure 4 Needle Available Heatmap'",
        "--description 'This heatmap uses materialized local Needle runs only. Missing token limits remain absent until true local data is available.'",
    ]
    for method_name in ["full_precision", "turboquant_2_5bit"]:
        path = combined_inputs.get(method_name)
        if path is None:
            continue
        method = METHODS[method_name]
        parts.extend(["--run", q(method["label"]), str(method["kv_bits"]), q(path)])
    if plot:
        parts.append("--plot")
    return " ".join(parts)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    paths_cfg = load_yaml(Path(args.paths))
    datasets_cfg = paths_cfg.get("datasets", {})
    statuses = set(args.statuses)
    methods = args.methods
    gpus = [int(value) for value in args.gpus]

    planned_entries = []
    skipped_entries = []
    available_outputs_by_method: dict[str, list[Path]] = defaultdict(list)
    dataset_inspections: dict[str, dict[str, Any]] = {}
    gpu_cursor = 0

    for token_config in args.token_configs:
        dataset_key = f"needle_{token_config}_{args.lang}"
        data_cfg = datasets_cfg.get(dataset_key)
        inspection = None
        if data_cfg is not None:
            inspection = inspect_grid(data_cfg, args.positions, args.examples_per_position_lang)
            dataset_inspections[dataset_key] = inspection
        for method_name in methods:
            expected_examples = inspection["expected_examples"] if inspection is not None else None
            output = output_path(
                token_config,
                args.lang,
                method_name,
                expected_examples,
                args.positions,
                args.examples_per_position_lang,
                args.output_tag,
            )
            aggregate = output.with_suffix(".aggregate.json")
            records = load_jsonl_indices(output)
            if data_cfg is None:
                status = "missing_dataset"
            elif records and expected_examples is not None and len(records) == expected_examples:
                status = "complete"
            elif records:
                status = "partial"
            else:
                status = "not_started"

            entry = {
                "token_config": token_config,
                "lang": args.lang,
                "dataset_key": dataset_key,
                "method_name": method_name,
                "method": METHODS[method_name]["label"],
                "kv_bits": METHODS[method_name]["kv_bits"],
                "status": status,
                "expected_examples": expected_examples,
                "num_records": len(records),
                "examples_per_position_lang": args.examples_per_position_lang,
                "positions": args.positions,
                "distractor_langs": inspection["distractor_langs"] if inspection else [],
                "output": str(output),
                "aggregate": str(aggregate),
            }
            if status == "complete" or status in statuses:
                if data_cfg is not None:
                    available_outputs_by_method[method_name].append(output)
            if status not in statuses or data_cfg is None:
                skipped_entries.append(entry)
                continue
            gpu = gpus[gpu_cursor % len(gpus)]
            gpu_cursor += 1
            entry["gpu"] = gpu
            entry["command"] = eval_command(
                entry,
                gpu=gpu,
                positions=args.positions,
                distractor_langs=entry["distractor_langs"],
                max_new_tokens=args.max_new_tokens,
                codebook_grid_size=args.codebook_grid_size,
                resume=args.resume,
            )
            entry["post_commands"] = [summarize_command(output, aggregate)]
            planned_entries.append(entry)

    combined_inputs: dict[str, str] = {}
    combine_commands = []
    figure4_inputs_dir = Path(args.figure4_inputs_dir)
    if args.output_tag:
        figure4_inputs_dir = figure4_inputs_dir.with_name(f"{figure4_inputs_dir.name}_{args.output_tag}")
    for method_name in ["full_precision", "turboquant_2_5bit"]:
        paths = available_outputs_by_method.get(method_name, [])
        if not paths:
            continue
        combined_path = figure4_inputs_dir / f"{method_name}_available.jsonl"
        combined_inputs[method_name] = str(combined_path)
        combine_commands.append(f"cat {' '.join(q(path) for path in paths)} > {q(combined_path)}")

    final_heatmap_prefix = Path(args.final_heatmap_prefix)
    if args.output_tag:
        final_heatmap_prefix = final_heatmap_prefix.with_name(f"{final_heatmap_prefix.name}_{args.output_tag}")

    return {
        "paths": str(Path(args.paths)),
        "lang": args.lang,
        "token_configs": args.token_configs,
        "gpus": gpus,
        "selected_statuses": sorted(statuses),
        "methods": methods,
        "grid": {
            "positions": args.positions,
            "examples_per_position_lang": args.examples_per_position_lang,
            "max_new_tokens": args.max_new_tokens,
        },
        "dataset_inspections": dataset_inspections,
        "summary": {
            "num_planned_entries": len(planned_entries),
            "planned_status_counts": dict(Counter(entry["status"] for entry in planned_entries)),
            "skipped_status_counts": dict(Counter(entry["status"] for entry in skipped_entries)),
            "all_status_counts": dict(Counter(entry["status"] for entry in planned_entries + skipped_entries)),
        },
        "planned_entries": planned_entries,
        "skipped_entries": skipped_entries,
        "final_heatmap": {
            "figure4_inputs_dir": str(figure4_inputs_dir),
            "combined_inputs": combined_inputs,
            "combine_commands": combine_commands,
            "command": build_heatmap_command(
                combined_inputs,
                final_heatmap_prefix,
                plot=args.plot,
            ),
        },
    }


def build_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Figure 4 Needle Run Plan",
        "",
        f"Language: `{plan['lang']}`",
        f"Token configs: `{', '.join(plan['token_configs'])}`",
        f"GPUs: `{', '.join(str(gpu) for gpu in plan['gpus'])}`",
        "",
        "## Summary",
        "",
        f"- Planned entries: {plan['summary']['num_planned_entries']}",
        f"- All statuses: `{plan['summary']['all_status_counts']}`",
        f"- Planned statuses: `{plan['summary']['planned_status_counts']}`",
        f"- Skipped statuses: `{plan['summary']['skipped_status_counts']}`",
        "",
        "## Dataset Inspections",
        "",
        "| Dataset key | Dataset examples | Expected grid examples | Distractor languages | Missing cells |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for dataset_key, inspection in sorted(plan["dataset_inspections"].items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{dataset_key}`",
                    str(inspection["num_dataset_examples"]),
                    str(inspection["expected_examples"]),
                    ", ".join(inspection["distractor_langs"]),
                    str(len(inspection["missing_cells"])),
                ]
            )
            + " |"
        )
    if not plan["dataset_inspections"]:
        lines.append("| none | 0 | 0 | none | 0 |")

    lines.extend(
        [
            "",
            "## Planned Runs",
            "",
            "| Token config | Dataset key | Method | Status | Expected | Records | GPU |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for entry in plan["planned_entries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{entry['token_config']}`",
                    f"`{entry['dataset_key']}`",
                    f"`{entry['method_name']}`",
                    entry["status"],
                    str(entry["expected_examples"]),
                    str(entry["num_records"]),
                    str(entry["gpu"]),
                ]
            )
            + " |"
        )
    if not plan["planned_entries"]:
        lines.append("| none | none | none | none | 0 | 0 |  |")

    lines.extend(
        [
            "",
            "## Missing Or Complete",
            "",
            "| Token config | Dataset key | Method | Status | Expected | Records |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for entry in plan["skipped_entries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{entry['token_config']}`",
                    f"`{entry['dataset_key']}`",
                    f"`{entry['method_name']}`",
                    entry["status"],
                    str(entry["expected_examples"] or ""),
                    str(entry["num_records"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Final Heatmap Inputs", ""])
    for method_name, path in plan["final_heatmap"]["combined_inputs"].items():
        lines.append(f"- `{method_name}`: `{path}`")
    lines.append("")
    lines.append("The generated shell script runs planned Needle grids, summarizes them, concatenates available method outputs, and rebuilds a Figure-4-style heatmap.")
    lines.append("")
    return "\n".join(lines)


def build_shell(plan: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {q(PROJECT_ROOT)}",
        "",
        f"mkdir -p {q(plan['final_heatmap']['figure4_inputs_dir'])} reproduce/logs",
        "",
    ]
    batch: list[str] = []
    for entry in plan["planned_entries"]:
        batch.append(entry["command"])
        if len(batch) == len(plan["gpus"]):
            for command in batch:
                lines.append(f"{command} &")
            lines.append("wait")
            batch = []
    if batch:
        for command in batch:
            lines.append(f"{command} &")
        lines.append("wait")
    if plan["planned_entries"]:
        lines.append("")
        for entry in plan["planned_entries"]:
            for command in entry["post_commands"]:
                lines.append(command)
        lines.append("")
    if plan["final_heatmap"]["combine_commands"]:
        lines.append("# Build Figure-4-style heatmap from all complete/planned available outputs.")
        for command in plan["final_heatmap"]["combine_commands"]:
            lines.append(command)
        lines.append(plan["final_heatmap"]["command"])
    else:
        lines.append("# No available outputs to summarize.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--output-prefix", default=str(PROJECT_ROOT / "reproduce/logs/figure4_run_plan"))
    parser.add_argument("--lang", default="en")
    parser.add_argument("--token-configs", nargs="+", default=TOKEN_CONFIGS)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    parser.add_argument("--statuses", nargs="+", default=["not_started", "partial"])
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "4", "5"])
    parser.add_argument("--positions", nargs="+", default=POSITIONS)
    parser.add_argument("--examples-per-position-lang", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--codebook-grid-size", type=int, default=10001)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--figure4-inputs-dir", default=str(PROJECT_ROOT / "reproduce/runs/figure4_inputs"))
    parser.add_argument("--final-heatmap-prefix", default=str(PROJECT_ROOT / "reproduce/runs/figure4_needle_available_heatmap"))
    parser.add_argument("--output-tag", default=None, help="Optional suffix for newly planned run outputs.")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    sh_path = prefix.with_suffix(".sh")
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(plan), encoding="utf-8")
    sh_path.write_text(build_shell(plan), encoding="utf-8")
    sh_path.chmod(0o755)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "shell": str(sh_path),
                **plan["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
