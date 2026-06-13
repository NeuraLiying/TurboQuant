#!/usr/bin/env python3
"""Audit local assets needed for the first-stage TurboQuant reproduction."""

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


DATA_EXTENSIONS = {".arrow", ".parquet", ".jsonl", ".csv", ".json"}
IGNORED_JSON_NAMES = {"dataset_info.json", "dataset_infos.json"}
NEEDLE_TARGET_CONFIGS = ["4k", "8k", "16k", "32k", "65k", "104k"]
DBPEDIA_CACHE_DIRS = {
    "dbpedia_openai3_1536": "Qdrant___dbpedia-entities-openai3-text-embedding-3-large-1536-1_m",
    "dbpedia_openai3_3072": "Qdrant___dbpedia-entities-openai3-text-embedding-3-large-3072-1_m",
}

LONG_BENCH_REPOS = {
    "longbench": {
        "key_prefix": "longbench",
        "cache_dir_names": ["Xnhyacinth___long_bench", "THUDM___long_bench"],
        "hub_snapshot_dirs": ["datasets--Xnhyacinth--LongBench", "datasets--THUDM--LongBench"],
        "config_suffixes": [""],
    },
    "longbench_e": {
        "key_prefix": "longbench_e",
        "cache_dir_names": ["Xnhyacinth___long_bench-e", "THUDM___long_bench"],
        "hub_snapshot_dirs": ["datasets--Xnhyacinth--LongBench-e", "datasets--THUDM--LongBench"],
        "config_suffixes": ["", "_e"],
    },
}


def is_data_file(path: Path) -> bool:
    if path.suffix.lower() not in DATA_EXTENSIONS:
        return False
    if path.name in IGNORED_JSON_NAMES:
        return False
    if path.name.upper() == "README.md":
        return False
    return path.stat().st_size > 0


def count_symlinks(paths: list[Path]) -> int:
    return sum(1 for path in paths if path.is_symlink())


def safe_relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def aliases(config: str, suffixes: list[str]) -> list[str]:
    result = []
    for suffix in suffixes:
        alias = f"{config}{suffix}"
        if alias not in result:
            result.append(alias)
    return result


def collect_dataset_cache_hits(cache_root: Path, repo: dict[str, Any], config: str) -> dict[str, Any]:
    roots = []
    data_files = []
    info_files = []
    lock_files = []
    for cache_dir_name in repo["cache_dir_names"]:
        for config_alias in aliases(config, repo["config_suffixes"]):
            config_root = cache_root / cache_dir_name / config_alias
            if not config_root.exists():
                continue
            roots.append(config_root)
            info_files.extend(sorted(config_root.rglob("dataset_info.json")))
            lock_files.extend(sorted(config_root.rglob("*.lock")))
            for path in sorted(config_root.rglob("*")):
                if path.is_file() and is_data_file(path):
                    data_files.append(path)
    return {
        "roots": [safe_relative(path, cache_root) for path in roots],
        "data_files": [safe_relative(path, cache_root) for path in data_files],
        "dataset_info_files": [safe_relative(path, cache_root) for path in info_files],
        "lock_files": [safe_relative(path, cache_root) for path in lock_files],
        "num_data_files": len(data_files),
        "num_symlink_data_files": count_symlinks(data_files),
        "total_data_bytes": sum(path.stat().st_size for path in data_files),
    }


def collect_hub_snapshot_hits(hub_cache_root: Path, repo: dict[str, Any], config: str) -> dict[str, Any]:
    snapshot_roots = []
    data_files = []
    loader_files = []
    for snapshot_dir_name in repo["hub_snapshot_dirs"]:
        snapshots_root = hub_cache_root / snapshot_dir_name / "snapshots"
        if not snapshots_root.exists():
            continue
        for snapshot_root in sorted(path for path in snapshots_root.iterdir() if path.is_dir()):
            snapshot_roots.append(snapshot_root)
            for config_alias in aliases(config, repo["config_suffixes"]):
                config_root = snapshot_root / config_alias
                if not config_root.exists():
                    continue
                for path in sorted(config_root.rglob("*")):
                    if path.is_file() and is_data_file(path):
                        data_files.append(path)
            for path in sorted(snapshot_root.iterdir()):
                if path.is_file() and path.suffix in {".py", ".md"}:
                    loader_files.append(path)
    return {
        "snapshot_roots": [safe_relative(path, hub_cache_root) for path in snapshot_roots],
        "data_files": [safe_relative(path, hub_cache_root) for path in data_files],
        "loader_or_readme_files": [safe_relative(path, hub_cache_root) for path in loader_files],
        "num_data_files": len(data_files),
        "num_symlink_data_files": count_symlinks(data_files),
        "total_data_bytes": sum(path.stat().st_size for path in data_files),
    }


def find_lock_files_for_config(cache_root: Path, config: str) -> list[str]:
    lock_files = []
    config_lower = config.lower().replace("-", "_")
    for path in sorted(cache_root.glob("*.lock")) + sorted(cache_root.rglob("*_builder.lock")):
        normalized = str(path).lower().replace("-", "_")
        if config_lower in normalized:
            lock_files.append(safe_relative(path, cache_root))
    return lock_files


def build_longbench_audit(cache_root: Path, hub_cache_root: Path) -> dict[str, Any]:
    expected_datasets = [dataset for datasets in TABLE1_CATEGORIES.values() for dataset in datasets]
    entries = []
    for repo_name, repo in LONG_BENCH_REPOS.items():
        for category, datasets in TABLE1_CATEGORIES.items():
            for config in datasets:
                dataset_cache = collect_dataset_cache_hits(cache_root, repo, config)
                hub_snapshot = collect_hub_snapshot_hits(hub_cache_root, repo, config)
                extra_locks = find_lock_files_for_config(cache_root, config)
                materialized = dataset_cache["num_data_files"] > 0 or hub_snapshot["num_data_files"] > 0
                if materialized:
                    status = "materialized"
                elif dataset_cache["roots"] or extra_locks or hub_snapshot["loader_or_readme_files"]:
                    status = "metadata_or_lock_only"
                else:
                    status = "missing"
                entries.append(
                    {
                        "repo": repo_name,
                        "dataset_key": f"{repo['key_prefix']}_{config}",
                        "category": category,
                        "config": config,
                        "status": status,
                        "dataset_cache": dataset_cache,
                        "hub_snapshot": hub_snapshot,
                        "extra_lock_files": extra_locks,
                    }
                )
    return {
        "expected_dataset_configs": expected_datasets,
        "entries": entries,
        "status_counts": dict(Counter(entry["status"] for entry in entries)),
    }


def collect_needle_cache(cache_root: Path, hub_cache_root: Path) -> dict[str, Any]:
    cache_base = cache_root / "ameyhengle___multilingual-needle-in-a-haystack"
    hub_base = hub_cache_root / "datasets--ameyhengle--Multilingual-Needle-in-a-Haystack" / "snapshots"
    entries = []
    for config in NEEDLE_TARGET_CONFIGS:
        cache_files = []
        cache_info = []
        if (cache_base / config).exists():
            cache_files = [path for path in sorted((cache_base / config).rglob("*")) if path.is_file() and is_data_file(path)]
            cache_info = sorted((cache_base / config).rglob("dataset_info.json"))
        hub_files = []
        if hub_base.exists():
            for snapshot_root in sorted(path for path in hub_base.iterdir() if path.is_dir()):
                data_root = snapshot_root / "data" / config
                if data_root.exists():
                    hub_files.extend(path for path in sorted(data_root.rglob("*")) if path.is_file() and is_data_file(path))
        materialized = bool(cache_files or hub_files)
        entries.append(
            {
                "config": config,
                "status": "materialized" if materialized else "missing",
                "dataset_cache_files": [safe_relative(path, cache_root) for path in cache_files],
                "dataset_info_files": [safe_relative(path, cache_root) for path in cache_info],
                "hub_snapshot_files": [safe_relative(path, hub_cache_root) for path in hub_files],
                "num_dataset_cache_files": len(cache_files),
                "num_hub_snapshot_files": len(hub_files),
                "num_dataset_cache_symlink_files": count_symlinks(cache_files),
                "num_hub_snapshot_symlink_files": count_symlinks(hub_files),
                "total_bytes": sum(path.stat().st_size for path in cache_files + hub_files),
            }
        )
    return {
        "target_configs": NEEDLE_TARGET_CONFIGS,
        "entries": entries,
        "status_counts": dict(Counter(entry["status"] for entry in entries)),
    }


def collect_dbpedia_cache(cache_root: Path) -> dict[str, Any]:
    entries = []
    for key, dirname in DBPEDIA_CACHE_DIRS.items():
        root = cache_root / dirname
        data_files = [path for path in sorted(root.rglob("*")) if path.is_file() and is_data_file(path)] if root.exists() else []
        info_files = sorted(root.rglob("dataset_info.json")) if root.exists() else []
        entries.append(
            {
                "dataset_key": key,
                "status": "materialized" if data_files else "missing",
                "data_files": [safe_relative(path, cache_root) for path in data_files],
                "dataset_info_files": [safe_relative(path, cache_root) for path in info_files],
                "num_data_files": len(data_files),
                "total_data_bytes": sum(path.stat().st_size for path in data_files),
            }
        )
    return {
        "entries": entries,
        "status_counts": dict(Counter(entry["status"] for entry in entries)),
    }


def build_markdown(report: dict[str, Any]) -> str:
    longbench_counts = report["longbench"]["status_counts"]
    needle_counts = report["needle"]["status_counts"]
    dbpedia_counts = report["dbpedia"]["status_counts"]
    lines = [
        "# Reproduction Asset Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        "- Data-file detection follows symlinks, so Hugging Face snapshot parquet/csv symlinks count as materialized data when present.",
        f"- LongBench/LongBench-E Table 1 entries: {longbench_counts}",
        f"- Needle Figure 4 target configs: {needle_counts}",
        f"- DBpedia Figure 5 configs: {dbpedia_counts}",
        "- GloVe 200d is not audited as available because it is outside the provided non-GloVe cache scope.",
        "",
        "## LongBench Table 1",
        "",
        "| Repo | Config | Category | Status | Dataset cache data files | Hub snapshot data files |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for entry in report["longbench"]["entries"]:
        lines.append(
            "| {repo} | `{config}` | {category} | {status} | {cache_count} | {hub_count} |".format(
                repo=entry["repo"],
                config=entry["config"],
                category=entry["category"],
                status=entry["status"],
                cache_count=entry["dataset_cache"]["num_data_files"],
                hub_count=entry["hub_snapshot"]["num_data_files"],
            )
        )
    lines.extend(
        [
            "",
            "## Needle Figure 4",
            "",
            "| Config | Status | Dataset cache files | Hub snapshot files |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for entry in report["needle"]["entries"]:
        lines.append(
            f"| `{entry['config']}` | {entry['status']} | {entry['num_dataset_cache_files']} | {entry['num_hub_snapshot_files']} |"
        )
    lines.extend(
        [
            "",
            "## DBpedia",
            "",
            "| Dataset | Status | Data files |",
            "| --- | --- | ---: |",
        ]
    )
    for entry in report["dbpedia"]["entries"]:
        lines.append(f"| `{entry['dataset_key']}` | {entry['status']} | {entry['num_data_files']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="/home/liying/datasets/turboquant/hf_cache")
    parser.add_argument("--hub-cache-root", default="/home/liying/.cache/huggingface/hub")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/logs/reproduction_asset_audit.json"))
    parser.add_argument("--markdown-output", default=str(PROJECT_ROOT / "reproduce/logs/reproduction_asset_audit.md"))
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    hub_cache_root = Path(args.hub_cache_root)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cache_root": str(cache_root),
        "hub_cache_root": str(hub_cache_root),
        "longbench": build_longbench_audit(cache_root, hub_cache_root),
        "needle": collect_needle_cache(cache_root, hub_cache_root),
        "dbpedia": collect_dbpedia_cache(cache_root),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "markdown_output": str(markdown_path),
                "longbench": report["longbench"]["status_counts"],
                "needle": report["needle"]["status_counts"],
                "dbpedia": report["dbpedia"]["status_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
