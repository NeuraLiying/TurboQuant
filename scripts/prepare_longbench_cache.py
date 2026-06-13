#!/usr/bin/env python3
"""Prepare LongBench/LongBench-E cache entries and update configs/paths.yaml."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

CANONICAL_PROJECT_ROOT = Path(os.environ.get("TURBOQUANT_PROJECT_ROOT", "/home/liying/projects/turboquant")).expanduser()
PROJECT_ROOT = CANONICAL_PROJECT_ROOT if CANONICAL_PROJECT_ROOT.exists() else Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.longbench_metrics import TABLE1_CATEGORIES


REPOS = {
    "longbench": {
        "repo_id": "Xnhyacinth/LongBench",
        "dataset_names": ["long_bench"],
        "cache_dir_names": {
            "Xnhyacinth___long_bench": "Xnhyacinth/LongBench",
            "THUDM___long_bench": "THUDM/LongBench",
        },
        "hub_snapshot_dirs": {
            "datasets--Xnhyacinth--LongBench": "Xnhyacinth/LongBench",
            "datasets--THUDM--LongBench": "THUDM/LongBench",
        },
        "config_suffixes": [""],
        "key_prefix": "longbench",
    },
    "longbench_e": {
        "repo_id": "Xnhyacinth/LongBench-e",
        "dataset_names": ["long_bench-e", "long_bench"],
        "cache_dir_names": {
            "Xnhyacinth___long_bench-e": "Xnhyacinth/LongBench-e",
            "THUDM___long_bench": "THUDM/LongBench",
        },
        "hub_snapshot_dirs": {
            "datasets--Xnhyacinth--LongBench-e": "Xnhyacinth/LongBench-e",
            "datasets--THUDM--LongBench": "THUDM/LongBench",
        },
        "config_suffixes": ["", "_e"],
        "key_prefix": "longbench_e",
    },
}


TABLE1_DATASETS = [dataset for datasets in TABLE1_CATEGORIES.values() for dataset in datasets]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_aliases(repo: dict[str, Any], config: str) -> list[str]:
    aliases = []
    for suffix in repo["config_suffixes"]:
        alias = f"{config}{suffix}"
        if alias not in aliases:
            aliases.append(alias)
    return aliases


def parquet_num_rows(paths: list[Path]) -> int | None:
    try:
        import pyarrow.parquet as pq
    except Exception:  # pragma: no cover - report can still include parquet paths.
        return None
    total = 0
    for path in paths:
        total += pq.ParquetFile(path).metadata.num_rows
    return total


def find_cached_dataset(cache_root: Path, repo: dict[str, Any], config: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for cache_dir_name, repo_id in repo["cache_dir_names"].items():
        for config_alias in config_aliases(repo, config):
            config_root = cache_root / cache_dir_name / config_alias
            if not config_root.exists():
                continue
            for info_path in sorted(config_root.rglob("dataset_info.json")):
                info = load_json(info_path)
                if info.get("dataset_name") not in repo["dataset_names"] or info.get("config_name") != config_alias:
                    continue
                cache_dir = info_path.parent
                arrow_files = sorted(str(path) for path in cache_dir.glob("*.arrow"))
                if not arrow_files:
                    continue
                split_names = list((info.get("splits") or {}).keys())
                split = "test" if "test" in split_names else split_names[0] if split_names else "test"
                num_examples = (info.get("splits") or {}).get(split, {}).get("num_examples")
                candidates.append(
                    {
                        "source": "datasets_cache",
                        "repo_id": repo_id,
                        "requested_config": config,
                        "config": config_alias,
                        "split": split,
                        "cache_dir": str(cache_dir),
                        "arrow_files": arrow_files,
                        "num_examples": num_examples,
                        "dataset_info": str(info_path),
                    }
                )
    if not candidates:
        return None
    return candidates[-1]


def find_snapshot_dataset(hub_cache_root: Path, repo: dict[str, Any], config: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for snapshot_dir_name, repo_id in repo["hub_snapshot_dirs"].items():
        snapshots_root = hub_cache_root / snapshot_dir_name / "snapshots"
        if not snapshots_root.exists():
            continue
        for snapshot_root in sorted(path for path in snapshots_root.iterdir() if path.is_dir()):
            for config_alias in config_aliases(repo, config):
                config_root = snapshot_root / config_alias
                parquet_files = sorted(config_root.glob("*.parquet"))
                if not parquet_files:
                    continue
                candidates.append(
                    {
                        "source": "hub_snapshot",
                        "repo_id": repo_id,
                        "requested_config": config,
                        "config": config_alias,
                        "split": "test",
                        "cache_dir": str(config_root),
                        "parquet_files": [str(path) for path in parquet_files],
                        "num_examples": parquet_num_rows(parquet_files),
                        "dataset_info": None,
                        "snapshot_root": str(snapshot_root),
                    }
                )
    if not candidates:
        return None
    return candidates[-1]


def dataset_key(repo_name: str, config: str) -> str:
    return f"{REPOS[repo_name]['key_prefix']}_{config}"


def update_paths_config(paths: dict[str, Any], repo_name: str, config: str, cached: dict[str, Any]) -> None:
    entry = {
        "repo_id": cached["repo_id"],
        "config": cached["config"],
        "requested_config": config,
        "split": cached["split"],
        "cache_dir": cached["cache_dir"],
        "source": cached["source"],
    }
    if cached.get("num_examples") is not None:
        entry["num_examples"] = cached["num_examples"]
    if cached.get("arrow_files"):
        entry["arrow_files"] = cached["arrow_files"]
    if cached.get("parquet_files"):
        entry["parquet_files"] = cached["parquet_files"]
    paths.setdefault("datasets", {})[dataset_key(repo_name, config)] = entry


def download_dataset(repo_id: str, config: str, split: str, cache_root: Path) -> None:
    from datasets import load_dataset

    dataset = load_dataset(repo_id, config, split=split, cache_dir=str(cache_root))
    # Force materialization in case datasets returns a lazy object in future versions.
    _ = len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--cache-root", default="/home/liying/datasets/turboquant/hf_cache")
    parser.add_argument("--hub-cache-root", default="/home/liying/.cache/huggingface/hub")
    parser.add_argument("--repos", nargs="+", choices=sorted(REPOS), default=["longbench", "longbench_e"])
    parser.add_argument("--datasets", nargs="+", default=TABLE1_DATASETS)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--update-paths", action="store_true")
    parser.add_argument("--output-report", default=str(PROJECT_ROOT / "reproduce/logs/longbench_cache_prepare_report.json"))
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    hub_cache_root = Path(args.hub_cache_root)
    paths_path = Path(args.paths)
    paths_cfg = load_yaml(paths_path)

    report: dict[str, Any] = {
        "cache_root": str(cache_root),
        "hub_cache_root": str(hub_cache_root),
        "paths": str(paths_path),
        "download_missing": args.download_missing,
        "update_paths": args.update_paths,
        "entries": [],
    }

    for repo_name in args.repos:
        repo = REPOS[repo_name]
        for config in args.datasets:
            cached = find_cached_dataset(cache_root, repo, config)
            snapshot = find_snapshot_dataset(hub_cache_root, repo, config)
            if cached is None and snapshot is not None:
                cached = snapshot
            attempted_download = False
            download_error = None
            if cached is None and args.download_missing:
                attempted_download = True
                try:
                    download_dataset(repo["repo_id"], config, "test", cache_root)
                    cached = find_cached_dataset(cache_root, repo, config)
                    if cached is None:
                        cached = find_snapshot_dataset(hub_cache_root, repo, config)
                except Exception as exc:  # noqa: BLE001 - report exact cache/download failure.
                    download_error = f"{type(exc).__name__}: {exc}"
            if cached is not None and args.update_paths:
                update_paths_config(paths_cfg, repo_name, config, cached)
            report["entries"].append(
                {
                    "repo": repo_name,
                    "repo_id": repo["repo_id"],
                    "config": config,
                    "key": dataset_key(repo_name, config),
                    "cached": cached is not None,
                    "attempted_download": attempted_download,
                    "download_error": download_error,
                    "snapshot": snapshot,
                    "cache": cached,
                }
            )

    if args.update_paths:
        write_yaml(paths_cfg, paths_path)

    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cached_count = sum(1 for entry in report["entries"] if entry["cached"])
    print(
        json.dumps(
            {
                "output_report": str(output_path),
                "num_entries": len(report["entries"]),
                "num_cached": cached_count,
                "num_missing": len(report["entries"]) - cached_count,
                "paths_updated": args.update_paths,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
