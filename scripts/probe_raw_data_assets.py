#!/usr/bin/env python3
"""Probe raw/cache/hub files for first-stage reproduction datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

CANONICAL_PROJECT_ROOT = Path(os.environ.get("TURBOQUANT_PROJECT_ROOT", "/home/liying/projects/turboquant")).expanduser()
PROJECT_ROOT = CANONICAL_PROJECT_ROOT if CANONICAL_PROJECT_ROOT.exists() else Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.longbench_metrics import TABLE1_CATEGORIES


DATA_EXTENSIONS = {".arrow", ".parquet", ".jsonl", ".json", ".csv", ".zip"}
IGNORED_DATA_NAMES = {"dataset_info.json", "dataset_infos.json"}
LONG_BENCH_REPOS = [
    "datasets--THUDM--LongBench",
    "datasets--Xnhyacinth--LongBench",
    "datasets--Xnhyacinth--LongBench-e",
]
NEEDLE_REPO = "datasets--ameyhengle--Multilingual-Needle-in-a-Haystack"
TARGET_NEEDLE_CONFIGS = ["4k", "8k", "16k", "32k", "65k", "104k"]


def safe_relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def classify_path(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith(".lock"):
        return "lock"
    if name in IGNORED_DATA_NAMES:
        return "metadata_or_loader"
    if path.is_symlink() and suffix in DATA_EXTENSIONS:
        return "data_symlink"
    if suffix in DATA_EXTENSIONS:
        return "data_file"
    if suffix in {".py", ".md", ".yaml", ".yml"}:
        return "metadata_or_loader"
    return "blob_or_unknown"


def path_record(path: Path, base: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = None
    return {
        "path": safe_relative(path, base),
        "kind": classify_path(path),
        "suffix": path.suffix.lower(),
        "size": size,
        "is_symlink": path.is_symlink(),
        "target": os.readlink(path) if path.is_symlink() else None,
    }


def collect_matching_files(root: Path, patterns: list[str]) -> list[Path]:
    if not root.exists():
        return []
    lowered_patterns = [pattern.lower() for pattern in patterns]
    matches = []
    for path in root.rglob("*"):
        if not (path.is_file() or path.is_symlink()):
            continue
        lowered = str(path).lower()
        if any(pattern in lowered for pattern in lowered_patterns):
            matches.append(path)
    return sorted(matches)


def collect_hub_repo(hub_cache_root: Path, repo_dir: str) -> dict[str, Any]:
    root = hub_cache_root / repo_dir
    files = [path for path in root.rglob("*") if path.is_file() or path.is_symlink()] if root.exists() else []
    snapshots = []
    snapshots_root = root / "snapshots"
    if snapshots_root.exists():
        snapshots = [path for path in sorted(snapshots_root.iterdir()) if path.is_dir()]
    return {
        "repo_dir": repo_dir,
        "exists": root.exists(),
        "num_files": len(files),
        "kind_counts": dict(Counter(classify_path(path) for path in files)),
        "largest_files": [
            path_record(path, hub_cache_root)
            for path in sorted(files, key=lambda item: item.stat().st_size if item.exists() else 0, reverse=True)[:40]
        ],
        "snapshot_files": [
            path_record(path, hub_cache_root)
            for snapshot in snapshots
            for path in sorted(snapshot.rglob("*"))
            if path.is_file() or path.is_symlink()
        ],
    }


def summarize_longbench_files(files: list[Path]) -> dict[str, Any]:
    configs = [dataset for datasets in TABLE1_CATEGORIES.values() for dataset in datasets]
    summary = {}
    for config in configs:
        matched = [path for path in files if config.lower() in str(path).lower()]
        data_matched = [path for path in matched if classify_path(path) in {"data_file", "data_symlink"}]
        non_data_matched = [path for path in matched if path not in data_matched]
        summary[config] = {
            "num_matches": len(matched),
            "num_data_matches": len(data_matched),
            "num_non_data_matches": len(non_data_matched),
            "matches": [str(path) for path in matched[:40]],
            "data_matches": [str(path) for path in data_matched[:40]],
            "non_data_matches": [str(path) for path in non_data_matched[:40]],
        }
    return summary


def summarize_needle_files(files: list[Path]) -> dict[str, Any]:
    summary = {}
    for config in TARGET_NEEDLE_CONFIGS:
        marker = f"/{config}/"
        matched = [path for path in files if marker in str(path)]
        data_matched = [path for path in matched if classify_path(path) in {"data_file", "data_symlink"}]
        non_data_matched = [path for path in matched if path not in data_matched]
        summary[config] = {
            "num_matches": len(matched),
            "num_data_matches": len(data_matched),
            "num_non_data_matches": len(non_data_matched),
            "matches": [str(path) for path in matched[:40]],
            "data_matches": [str(path) for path in data_matched[:40]],
            "non_data_matches": [str(path) for path in non_data_matched[:40]],
        }
    return summary


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    hub_cache_root = Path(args.hub_cache_root)
    project_root = Path(args.project_root)

    longbench_patterns = [
        "longbench",
        "long_bench",
        "narrative",
        "qasper",
        "multifield",
        "hotpot",
        "musique",
        "gov_report",
        "qmsum",
        "multi_news",
        "trec",
        "trivia",
        "samsum",
        "passage",
        "lcc",
        "repobench",
    ]
    needle_patterns = ["needle", "haystack", "/4k/", "/8k/", "/16k/", "/32k/", "/65k/", "/104k/"]

    data_root_matches = collect_matching_files(data_root, longbench_patterns + needle_patterns)
    project_matches = collect_matching_files(project_root, longbench_patterns + needle_patterns)
    hub_matches = collect_matching_files(hub_cache_root, longbench_patterns + needle_patterns)

    hub_repos = [collect_hub_repo(hub_cache_root, repo) for repo in [*LONG_BENCH_REPOS, NEEDLE_REPO]]
    hub_snapshot_files = [
        Path(hub_cache_root / record["path"])
        for repo in hub_repos
        for record in repo["snapshot_files"]
    ]
    data_files = [path for path in data_root_matches if path.suffix.lower() in DATA_EXTENSIONS]

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "cache_root": str(cache_root),
        "hub_cache_root": str(hub_cache_root),
        "project_root": str(project_root),
        "data_root_match_counts": dict(Counter(classify_path(path) for path in data_root_matches)),
        "hub_match_counts": dict(Counter(classify_path(path) for path in hub_matches)),
        "project_match_counts": dict(Counter(classify_path(path) for path in project_matches)),
        "longbench_target_matches": summarize_longbench_files(data_root_matches + hub_snapshot_files),
        "needle_target_matches": summarize_needle_files(data_root_matches + hub_snapshot_files),
        "hub_repos": hub_repos,
        "largest_data_root_matches": [
            path_record(path, data_root)
            for path in sorted(data_root_matches, key=lambda item: item.stat().st_size if item.exists() else 0, reverse=True)[:80]
        ],
        "largest_hub_matches": [
            path_record(path, hub_cache_root)
            for path in sorted(hub_matches, key=lambda item: item.stat().st_size if item.exists() else 0, reverse=True)[:80]
        ],
        "data_files_under_data_root": [path_record(path, data_root) for path in data_files[:200]],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Raw Data Asset Probe",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "This is a read-only probe for unregistered raw/cache/hub files relevant to Table 1 and Figure 4.",
        "",
        "## Summary",
        "",
        f"- Data root matches by kind: `{report['data_root_match_counts']}`",
        f"- Hub matches by kind: `{report['hub_match_counts']}`",
        f"- Project matches by kind: `{report['project_match_counts']}`",
        "",
        "## LongBench Targets",
        "",
        "| Config | Data matches | Non-data matches |",
        "| --- | ---: | ---: |",
    ]
    for config, item in report["longbench_target_matches"].items():
        lines.append(f"| `{config}` | {item['num_data_matches']} | {item['num_non_data_matches']} |")
    lines.extend(["", "## Needle Targets", "", "| Config | Data matches | Non-data matches |", "| --- | ---: | ---: |"])
    for config, item in report["needle_target_matches"].items():
        lines.append(f"| `{config}` | {item['num_data_matches']} | {item['num_non_data_matches']} |")
    lines.extend(["", "## Hub Repositories", "", "| Repo | Exists | Files | Kind counts |", "| --- | --- | ---: | --- |"])
    for repo in report["hub_repos"]:
        lines.append(f"| `{repo['repo_dir']}` | {repo['exists']} | {repo['num_files']} | `{repo['kind_counts']}` |")
    lines.extend(["", "## Largest Relevant Hub Files", "", "| Kind | Size | Path |", "| --- | ---: | --- |"])
    for item in report["largest_hub_matches"][:40]:
        lines.append(f"| {item['kind']} | {item['size']} | `{item['path']}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/liying/datasets/turboquant")
    parser.add_argument("--cache-root", default="/home/liying/datasets/turboquant/hf_cache")
    parser.add_argument("--hub-cache-root", default="/home/liying/.cache/huggingface/hub")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/logs/raw_data_asset_probe.json"))
    parser.add_argument("--markdown-output", default=str(PROJECT_ROOT / "reproduce/logs/raw_data_asset_probe.md"))
    args = parser.parse_args()

    report = build_report(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, markdown_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "markdown_output": str(markdown_path),
                "data_root_match_counts": report["data_root_match_counts"],
                "hub_match_counts": report["hub_match_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
