#!/usr/bin/env python3
"""Plan runnable LongBench Table 1 jobs from a manifest."""

from __future__ import annotations

import argparse
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_LABEL = "Llama-3.1-8B-Instruct"
METHOD_ORDER = ["full", "turboquant_2_5bit", "turboquant_3_5bit"]
METHOD_LABELS = {
    "full": ("Full Cache", 16.0),
    "turboquant_2_5bit": ("TurboQuant", 2.5),
    "turboquant_3_5bit": ("TurboQuant", 3.5),
}


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def chunks_for(num_examples: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0 or num_examples <= chunk_size:
        return [(0, num_examples)]
    return [(start, min(start + chunk_size, num_examples)) for start in range(0, num_examples, chunk_size)]


def shard_path(final_output: Path, start: int, end: int) -> Path:
    return final_output.parent / "shards" / f"{final_output.stem}_{start:05d}_{end:05d}.jsonl"


def eval_command(
    entry: dict[str, Any],
    *,
    output_path: Path,
    gpu: int,
    start: int | None,
    end: int | None,
    codebook_grid_size: int,
    progress_every: int,
    resume: bool,
    prompt_mode: str,
    chat_template_mode: str,
    max_input_tokens: int | None,
) -> str:
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "conda run -n turboquant python experiments/longbench/run_full_cache_eval.py",
        f"--dataset-key {q(entry['dataset_key'])}",
        "--device cuda:0",
        f"--cache-mode {q('turboquant' if entry['method_name'].startswith('turboquant') else 'full')}",
        f"--prompt-mode {q(prompt_mode)}",
        f"--chat-template-mode {q(chat_template_mode)}",
    ]
    if max_input_tokens is not None:
        parts.append(f"--max-input-tokens {max_input_tokens}")
    if entry["method_name"].startswith("turboquant"):
        parts.extend([f"--kv-bits {entry['kv_bits']}", f"--codebook-grid-size {codebook_grid_size}"])
    if start is not None:
        parts.append(f"--start-index {start}")
    if end is not None:
        parts.append(f"--end-index {end}")
    if resume:
        parts.append("--resume")
    parts.extend([f"--output {q(output_path)}", f"--progress-every {progress_every}"])
    return " ".join(parts)


def merge_command(final_output: Path, expected_examples: int, shards: list[Path]) -> str:
    return (
        "conda run -n turboquant python scripts/merge_jsonl_by_index.py "
        f"--output {q(final_output)} --expected-start 0 --expected-end {expected_examples} "
        + " ".join(q(path) for path in shards)
    )


def summarize_command(jsonl_path: Path, aggregate_path: Path) -> str:
    return f"conda run -n turboquant python scripts/summarize_jsonl_accuracy.py {q(jsonl_path)} --output {q(aggregate_path)}"


def table1_summary_command(method_inputs: dict[str, Path], output_prefix: Path) -> str:
    parts = [
        "conda run -n turboquant python scripts/build_table1_summary.py",
        f"--output-prefix {q(output_prefix)}",
    ]
    for method_name in METHOD_ORDER:
        path = method_inputs.get(method_name)
        if path is None:
            continue
        label, kv_size = METHOD_LABELS[method_name]
        parts.extend(["--run", q(label), str(kv_size), q(MODEL_LABEL), q(path)])
    return " ".join(parts)


def should_plan(entry: dict[str, Any], statuses: set[str], methods: set[str] | None, datasets: set[str] | None) -> bool:
    if entry["status"] not in statuses:
        return False
    if methods is not None and entry["method_name"] not in methods:
        return False
    if datasets is not None and entry["dataset_key"] not in datasets:
        return False
    if entry.get("expected_examples") is None:
        return False
    return True


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    gpus = [int(value) for value in args.gpus]
    statuses = set(args.statuses)
    methods = set(args.methods) if args.methods else None
    datasets = set(args.datasets) if args.datasets else None

    planned_entries = []
    skipped_entries = []
    gpu_cursor = 0
    available_outputs_by_method: dict[str, list[Path]] = defaultdict(list)

    for entry in manifest["entries"]:
        output_path = Path(entry["output"])
        aggregate_path = Path(entry["aggregate"])
        if entry["status"] == "complete" or should_plan(entry, statuses, methods, datasets):
            available_outputs_by_method[entry["method_name"]].append(output_path)

        if not should_plan(entry, statuses, methods, datasets):
            skipped_entries.append(
                {
                    "dataset_key": entry["dataset_key"],
                    "method_name": entry["method_name"],
                    "status": entry["status"],
                    "expected_examples": entry.get("expected_examples"),
                    "reason": "not selected or not runnable",
                }
            )
            continue

        expected_examples = int(entry["expected_examples"])
        is_turboquant = entry["method_name"].startswith("turboquant")
        chunk_size = args.turboquant_chunk_size if is_turboquant else args.full_chunk_size
        if entry["status"] == "partial" and not args.rechunk_partial:
            chunk_ranges = [(None, None)]
        else:
            chunk_ranges = chunks_for(expected_examples, chunk_size)

        jobs = []
        shard_paths = []
        for start, end in chunk_ranges:
            gpu = gpus[gpu_cursor % len(gpus)]
            gpu_cursor += 1
            use_shard = start is not None and end is not None and len(chunk_ranges) > 1
            job_output = shard_path(output_path, start, end) if use_shard else output_path
            if use_shard:
                shard_paths.append(job_output)
            jobs.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "gpu": gpu,
                    "output": str(job_output),
                    "command": eval_command(
                        entry,
                        output_path=job_output,
                        gpu=gpu,
                        start=start,
                        end=end,
                        codebook_grid_size=args.codebook_grid_size,
                        progress_every=args.progress_every,
                        resume=args.resume,
                        prompt_mode=args.prompt_mode,
                        chat_template_mode=args.chat_template_mode,
                        max_input_tokens=args.max_input_tokens,
                    ),
                }
            )

        post_commands = []
        if shard_paths:
            post_commands.append(merge_command(output_path, expected_examples, shard_paths))
        post_commands.append(summarize_command(output_path, aggregate_path))
        planned_entries.append(
            {
                "dataset_key": entry["dataset_key"],
                "category": entry["category"],
                "dataset": entry["dataset"],
                "method": entry["method"],
                "method_name": entry["method_name"],
                "kv_bits": entry["kv_bits"],
                "status": entry["status"],
                "expected_examples": expected_examples,
                "final_output": str(output_path),
                "aggregate": str(aggregate_path),
                "num_jobs": len(jobs),
                "jobs": jobs,
                "post_commands": post_commands,
            }
        )

    combined_inputs: dict[str, str] = {}
    combine_commands = []
    table_inputs_dir = Path(args.table_inputs_dir)
    for method_name in METHOD_ORDER:
        paths = available_outputs_by_method.get(method_name, [])
        if not paths:
            continue
        combined_path = table_inputs_dir / f"{method_name}_available.jsonl"
        combined_inputs[method_name] = str(combined_path)
        combine_commands.append(f"cat {' '.join(q(path) for path in paths)} > {q(combined_path)}")

    final_summary_prefix = Path(args.final_summary_prefix)
    final_summary_command = table1_summary_command(
        {method: Path(path) for method, path in combined_inputs.items()},
        final_summary_prefix,
    )

    return {
        "manifest": str(manifest_path),
        "gpus": gpus,
        "selected_statuses": sorted(statuses),
        "chunking": {
            "full_chunk_size": args.full_chunk_size,
            "turboquant_chunk_size": args.turboquant_chunk_size,
            "rechunk_partial": args.rechunk_partial,
        },
        "generation": {
            "prompt_mode": args.prompt_mode,
            "chat_template_mode": args.chat_template_mode,
            "max_input_tokens": args.max_input_tokens,
        },
        "summary": {
            "manifest_status_counts": manifest.get("status_counts", {}),
            "num_planned_entries": len(planned_entries),
            "num_planned_jobs": sum(entry["num_jobs"] for entry in planned_entries),
            "planned_status_counts": dict(Counter(entry["status"] for entry in planned_entries)),
            "skipped_status_counts": dict(Counter(entry["status"] for entry in skipped_entries)),
        },
        "planned_entries": planned_entries,
        "skipped_entries": skipped_entries,
        "final_summary": {
            "table_inputs_dir": str(table_inputs_dir),
            "combined_inputs": combined_inputs,
            "combine_commands": combine_commands,
            "command": final_summary_command,
        },
    }


def build_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Table 1 Run Plan",
        "",
        f"Manifest: `{plan['manifest']}`",
        f"GPUs: `{', '.join(str(gpu) for gpu in plan['gpus'])}`",
        "",
        "## Summary",
        "",
        f"- Manifest statuses: `{plan['summary']['manifest_status_counts']}`",
        f"- Planned entries: {plan['summary']['num_planned_entries']}",
        f"- Planned jobs: {plan['summary']['num_planned_jobs']}",
        f"- Planned statuses: `{plan['summary']['planned_status_counts']}`",
        f"- Skipped statuses: `{plan['summary']['skipped_status_counts']}`",
        "",
        "## Planned Jobs",
        "",
        "| Dataset key | Method | Status | Expected examples | Jobs |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for entry in plan["planned_entries"]:
        lines.append(
            f"| `{entry['dataset_key']}` | `{entry['method_name']}` | {entry['status']} | {entry['expected_examples']} | {entry['num_jobs']} |"
        )
    if not plan["planned_entries"]:
        lines.append("| none | none | none | 0 | 0 |")
    lines.extend(
        [
            "",
            "## Missing Or Skipped",
            "",
            "| Status | Count |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted(plan["summary"]["skipped_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Final Summary Inputs",
            "",
        ]
    )
    for method_name, path in plan["final_summary"]["combined_inputs"].items():
        lines.append(f"- `{method_name}`: `{path}`")
    lines.append("")
    lines.append("The generated shell script runs planned chunks, merges shards, writes aggregate JSON, and rebuilds a Table-1-shaped summary from available method inputs.")
    lines.append("")
    return "\n".join(lines)


def build_shell(plan: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {q(PROJECT_ROOT)}",
        "",
        f"mkdir -p {q(Path(plan['final_summary']['table_inputs_dir']))} reproduce/logs",
        "",
    ]
    for entry in plan["planned_entries"]:
        lines.append(f"# {entry['dataset_key']} {entry['method_name']} ({entry['status']})")
        batch: list[str] = []
        for job in entry["jobs"]:
            batch.append(job["command"])
            if len(batch) == len(plan["gpus"]):
                for command in batch:
                    lines.append(f"{command} &")
                lines.append("wait")
                batch = []
        if batch:
            for command in batch:
                lines.append(f"{command} &")
            lines.append("wait")
        for command in entry["post_commands"]:
            lines.append(command)
        lines.append("")

    if plan["final_summary"]["combine_commands"]:
        lines.append("# Build Table-1-shaped summary from all complete/planned available outputs.")
        for command in plan["final_summary"]["combine_commands"]:
            lines.append(command)
        lines.append(plan["final_summary"]["command"])
    else:
        lines.append("# No available outputs to summarize.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "reproduce/logs/table1_manifest.json"))
    parser.add_argument("--output-prefix", default=str(PROJECT_ROOT / "reproduce/logs/table1_run_plan"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "4", "5"])
    parser.add_argument("--statuses", nargs="+", default=["not_started", "partial"])
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--full-chunk-size", type=int, default=0)
    parser.add_argument("--turboquant-chunk-size", type=int, default=50)
    parser.add_argument("--rechunk-partial", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--codebook-grid-size", type=int, default=10001)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--prompt-mode", choices=["longbench", "legacy"], default="longbench")
    parser.add_argument("--chat-template-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--table-inputs-dir", default=str(PROJECT_ROOT / "reproduce/runs/table1_inputs"))
    parser.add_argument("--final-summary-prefix", default=str(PROJECT_ROOT / "reproduce/runs/table1_llama_available_summary"))
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
