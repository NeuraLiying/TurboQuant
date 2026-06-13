#!/usr/bin/env python3
"""Refresh non-model reproduction state reports and run plans."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


CANONICAL_PROJECT_ROOT = Path(os.environ.get("TURBOQUANT_PROJECT_ROOT", "/home/liying/projects/turboquant")).expanduser()
PROJECT_ROOT = CANONICAL_PROJECT_ROOT if CANONICAL_PROJECT_ROOT.exists() else Path(__file__).resolve().parents[1]
DEFAULT_GPUS = ["0", "1", "4", "5"]


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def require_success(result: dict[str, Any]) -> None:
    if result["returncode"] != 0:
        command = " ".join(result["command"])
        raise RuntimeError(f"command failed with {result['returncode']}: {command}\n{result['stderr']}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    asset = report["summaries"].get("asset_audit", {})
    table1 = report["summaries"].get("table1_manifest", {})
    table1_plan = report["summaries"].get("table1_plan", {})
    figure4_plan = report["summaries"].get("figure4_plan", {})
    needle_prepare = report["summaries"].get("needle_prepare", {})
    longbench_prepare = report["summaries"].get("longbench_prepare", {})
    lines = [
        "# Reproduction State Refresh",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "This refresh does not run model generation. It updates cache registration, asset audits, manifests, and run plans.",
        "",
        "## Summary",
        "",
        f"- LongBench prepare: `{longbench_prepare}`",
        f"- Needle prepare: `{needle_prepare}`",
        f"- Asset audit LongBench: `{asset.get('longbench')}`",
        f"- Asset audit Needle: `{asset.get('needle')}`",
        f"- Asset audit DBpedia: `{asset.get('dbpedia')}`",
        f"- Table 1 manifest: `{table1}`",
        f"- Table 1 plan: `{table1_plan}`",
        f"- Figure 4 plan: `{figure4_plan}`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact_path in report["artifacts"].items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    lines.extend(["", "## Commands", ""])
    for result in report["commands"]:
        command = " ".join(result["command"])
        lines.append(f"- returncode={result['returncode']}: `{command}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="/home/liying/datasets/turboquant/hf_cache")
    parser.add_argument("--hub-cache-root", default="/home/liying/.cache/huggingface/hub")
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--tag", default=datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    parser.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    parser.add_argument("--skip-update-paths", action="store_true")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix) if args.output_prefix else PROJECT_ROOT / "reproduce/logs" / f"reproduction_state_refresh_{args.tag}"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "longbench_prepare": str(output_prefix.with_name(output_prefix.name + "_longbench_prepare.json")),
        "needle_prepare": str(output_prefix.with_name(output_prefix.name + "_needle_prepare.json")),
        "asset_audit": str(output_prefix.with_name(output_prefix.name + "_asset_audit.json")),
        "asset_audit_md": str(output_prefix.with_name(output_prefix.name + "_asset_audit.md")),
        "table1_manifest": str(output_prefix.with_name(output_prefix.name + "_table1_manifest.json")),
        "table1_plan": str(output_prefix.with_name(output_prefix.name + "_table1_plan.json")),
        "table1_plan_md": str(output_prefix.with_name(output_prefix.name + "_table1_plan.md")),
        "table1_plan_sh": str(output_prefix.with_name(output_prefix.name + "_table1_plan.sh")),
        "figure4_plan": str(output_prefix.with_name(output_prefix.name + "_figure4_plan.json")),
        "figure4_plan_md": str(output_prefix.with_name(output_prefix.name + "_figure4_plan.md")),
        "figure4_plan_sh": str(output_prefix.with_name(output_prefix.name + "_figure4_plan.sh")),
        "summary_md": str(output_prefix.with_suffix(".md")),
        "summary_json": str(output_prefix.with_suffix(".json")),
    }

    py = sys.executable
    commands = []
    update_flag = [] if args.skip_update_paths else ["--update-paths"]
    command_specs = [
        [
            py,
            "scripts/prepare_longbench_cache.py",
            "--paths",
            args.paths,
            "--cache-root",
            args.cache_root,
            "--hub-cache-root",
            args.hub_cache_root,
            *update_flag,
            "--output-report",
            artifacts["longbench_prepare"],
        ],
        [
            py,
            "scripts/prepare_needle_cache.py",
            "--paths",
            args.paths,
            "--configs",
            "4k",
            "8k",
            "16k",
            "32k",
            "65k",
            "104k",
            *update_flag,
            "--output-report",
            artifacts["needle_prepare"],
        ],
        [
            py,
            "scripts/audit_reproduction_assets.py",
            "--cache-root",
            args.cache_root,
            "--hub-cache-root",
            args.hub_cache_root,
            "--output",
            artifacts["asset_audit"],
            "--markdown-output",
            artifacts["asset_audit_md"],
        ],
        [
            py,
            "scripts/build_table1_manifest.py",
            "--paths",
            args.paths,
            "--output",
            artifacts["table1_manifest"],
        ],
        [
            py,
            "scripts/plan_table1_runs.py",
            "--manifest",
            artifacts["table1_manifest"],
            "--output-prefix",
            str(Path(artifacts["table1_plan"]).with_suffix("")),
            "--gpus",
            *args.gpus,
        ],
        [
            py,
            "scripts/plan_figure4_runs.py",
            "--paths",
            args.paths,
            "--output-prefix",
            str(Path(artifacts["figure4_plan"]).with_suffix("")),
            "--gpus",
            *args.gpus,
            "--plot",
        ],
    ]

    for command in command_specs:
        result = run_command(command, PROJECT_ROOT)
        commands.append(result)
        require_success(result)

    longbench_report = load_json(Path(artifacts["longbench_prepare"]))
    needle_report = load_json(Path(artifacts["needle_prepare"]))
    asset_report = load_json(Path(artifacts["asset_audit"]))
    table1_manifest = load_json(Path(artifacts["table1_manifest"]))
    table1_plan = load_json(Path(artifacts["table1_plan"]))
    figure4_plan = load_json(Path(artifacts["figure4_plan"]))

    summaries = {
        "longbench_prepare": {
            "num_entries": len(longbench_report.get("entries", [])),
            "num_cached": sum(1 for entry in longbench_report.get("entries", []) if entry.get("cached")),
            "num_missing": sum(1 for entry in longbench_report.get("entries", []) if not entry.get("cached")),
        },
        "needle_prepare": {
            "found_configs": needle_report.get("found_configs", []),
            "missing_configs": needle_report.get("missing_configs", []),
        },
        "asset_audit": {
            "longbench": asset_report["longbench"]["status_counts"],
            "needle": asset_report["needle"]["status_counts"],
            "dbpedia": asset_report["dbpedia"]["status_counts"],
        },
        "table1_manifest": table1_manifest.get("status_counts", {}),
        "table1_plan": table1_plan.get("summary", {}),
        "figure4_plan": figure4_plan.get("summary", {}),
    }

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "cache_root": args.cache_root,
        "hub_cache_root": args.hub_cache_root,
        "paths": args.paths,
        "update_paths": not args.skip_update_paths,
        "artifacts": artifacts,
        "summaries": summaries,
        "commands": commands,
    }
    Path(artifacts["summary_json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, Path(artifacts["summary_md"]))
    print(json.dumps({"summary_json": artifacts["summary_json"], "summary_md": artifacts["summary_md"], **summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
