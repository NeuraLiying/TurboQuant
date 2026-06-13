#!/usr/bin/env python3
"""Discover local Needle cache entries and update configs/paths.yaml."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

CANONICAL_PROJECT_ROOT = Path(os.environ.get("TURBOQUANT_PROJECT_ROOT", "/home/liying/projects/turboquant")).expanduser()
PROJECT_ROOT = CANONICAL_PROJECT_ROOT if CANONICAL_PROJECT_ROOT.exists() else Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


NEEDLE_CACHE_DIR_NAME = "ameyhengle___multilingual-needle-in-a-haystack"
NEEDLE_REPO_ID = "ameyhengle/Multilingual-Needle-in-a-Haystack"
DEFAULT_CONFIGS = ["4k", "8k", "16k", "32k", "65k", "104k"]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def split_from_arrow(path: Path) -> str:
    match = re.match(r"multilingual-needle-in-a-haystack-(.+)\.arrow$", path.name)
    if not match:
        raise ValueError(f"cannot infer Needle split from {path}")
    return match.group(1)


def count_rows(path: Path) -> int:
    return len(Dataset.from_file(str(path)))


def discover_needle(cache_root: Path, config: str) -> dict[str, Any]:
    config_root = cache_root / NEEDLE_CACHE_DIR_NAME / config
    entries = []
    for info_path in sorted(config_root.rglob("dataset_info.json")):
        cache_dir = info_path.parent
        arrow_files = sorted(cache_dir.glob("multilingual-needle-in-a-haystack-*.arrow"))
        if not arrow_files:
            continue
        split_entries = []
        for arrow_path in arrow_files:
            split_entries.append(
                {
                    "split": split_from_arrow(arrow_path),
                    "arrow_file": str(arrow_path),
                    "num_examples": count_rows(arrow_path),
                    "bytes": arrow_path.stat().st_size,
                }
            )
        entries.append(
            {
                "config": config,
                "cache_dir": str(cache_dir),
                "dataset_info": str(info_path),
                "splits": sorted(split_entries, key=lambda item: item["split"]),
            }
        )
    if not entries:
        return {"config": config, "found": False, "entries": []}
    return {"config": config, "found": True, "entries": entries}


def update_paths(paths_cfg: dict[str, Any], discovered: dict[str, Any]) -> None:
    datasets = paths_cfg.setdefault("datasets", {})
    for entry in discovered["entries"]:
        all_arrow_files = []
        total_examples = 0
        for split_entry in entry["splits"]:
            split = split_entry["split"]
            arrow_file = split_entry["arrow_file"]
            total_examples += int(split_entry["num_examples"])
            all_arrow_files.append(arrow_file)
            datasets[f"needle_{entry['config']}_{split}"] = {
                "repo_id": NEEDLE_REPO_ID,
                "config": entry["config"],
                "split": split,
                "source": "datasets_cache",
                "cache_dir": entry["cache_dir"],
                "num_examples": int(split_entry["num_examples"]),
                "arrow_files": [arrow_file],
            }
        datasets[f"needle_{entry['config']}_all"] = {
            "repo_id": NEEDLE_REPO_ID,
            "config": entry["config"],
            "split": "all",
            "source": "datasets_cache",
            "cache_dir": entry["cache_dir"],
            "num_examples": total_examples,
            "arrow_files": all_arrow_files,
        }


def build_report(cache_root: Path, paths_path: Path, update_paths_enabled: bool, discoveries: list[dict[str, Any]]) -> dict[str, Any]:
    found_configs = [item["config"] for item in discoveries if item["found"]]
    missing_configs = [item["config"] for item in discoveries if not item["found"]]
    report = {
        "cache_root": str(cache_root),
        "paths": str(paths_path),
        "update_paths": update_paths_enabled,
        "configs": [item["config"] for item in discoveries],
        "num_configs": len(discoveries),
        "num_found_configs": len(found_configs),
        "num_missing_configs": len(missing_configs),
        "found_configs": found_configs,
        "missing_configs": missing_configs,
        "results": discoveries,
    }
    if len(discoveries) == 1:
        report.update(discoveries[0])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--cache-root", default="/home/liying/datasets/turboquant/hf_cache")
    parser.add_argument("--config", default=None, help="Single Needle config to discover. Kept for backwards compatibility.")
    parser.add_argument("--configs", nargs="+", default=None, help="Needle configs to discover, e.g. 4k 8k 16k 32k 65k 104k.")
    parser.add_argument("--all-default-configs", action="store_true", help="Discover the paper-style default Needle configs.")
    parser.add_argument("--update-paths", action="store_true")
    parser.add_argument("--output-report", default=str(PROJECT_ROOT / "reproduce/logs/needle_cache_prepare_report.json"))
    args = parser.parse_args()

    paths_path = Path(args.paths)
    paths_cfg = load_yaml(paths_path)
    if args.configs:
        configs = args.configs
    elif args.all_default_configs:
        configs = DEFAULT_CONFIGS
    elif args.config:
        configs = [args.config]
    else:
        configs = ["16k"]

    cache_root = Path(args.cache_root)
    discoveries = [discover_needle(cache_root, config) for config in configs]
    if args.update_paths:
        for discovered in discoveries:
            if discovered["found"]:
                update_paths(paths_cfg, discovered)
        write_yaml(paths_cfg, paths_path)

    report = build_report(cache_root, paths_path, args.update_paths, discoveries)
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_report": str(output_path),
                "found_configs": report["found_configs"],
                "missing_configs": report["missing_configs"],
                "num_found_configs": report["num_found_configs"],
                "num_missing_configs": report["num_missing_configs"],
                "num_entries": sum(len(item["entries"]) for item in discoveries),
                "paths_updated": args.update_paths,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
