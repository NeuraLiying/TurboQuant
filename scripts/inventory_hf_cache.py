#!/usr/bin/env python3
"""Inventory local Hugging Face dataset cache contents."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_key_from_info(cache_root: Path, info_path: Path) -> str:
    rel = info_path.relative_to(cache_root)
    if len(rel.parts) < 4:
        return str(rel.parent)
    return "/".join(rel.parts[:2])


def arrow_group_from_path(cache_root: Path, arrow_path: Path) -> str:
    rel = arrow_path.relative_to(cache_root)
    if len(rel.parts) < 4:
        return str(rel.parent)
    return "/".join(rel.parts[:2])


def build_inventory(cache_root: Path) -> dict[str, Any]:
    dataset_info_paths = sorted(cache_root.rglob("dataset_info.json"))
    arrow_paths = sorted(cache_root.rglob("*.arrow"))
    lock_paths = sorted(cache_root.rglob("*.lock"))
    incomplete_paths = sorted(path for path in cache_root.rglob("*incomplete*") if path.is_file())
    arrows_by_group: dict[str, list[Path]] = defaultdict(list)
    for path in arrow_paths:
        arrows_by_group[arrow_group_from_path(cache_root, path)].append(path)

    datasets = []
    for info_path in dataset_info_paths:
        info = load_json(info_path)
        key = dataset_key_from_info(cache_root, info_path)
        arrow_files = arrows_by_group.get(key, [])
        datasets.append(
            {
                "key": key,
                "dataset_info": str(info_path),
                "builder_name": info.get("builder_name"),
                "config_name": info.get("config_name"),
                "dataset_name": info.get("dataset_name"),
                "splits": info.get("splits"),
                "num_arrow_files": len(arrow_files),
                "arrow_files": [str(path) for path in arrow_files],
                "total_arrow_bytes": sum(path.stat().st_size for path in arrow_files),
            }
        )

    return {
        "cache_root": str(cache_root),
        "num_dataset_info": len(dataset_info_paths),
        "num_arrow_files": len(arrow_paths),
        "num_lock_files": len(lock_paths),
        "num_incomplete_files": len(incomplete_paths),
        "datasets": datasets,
        "lock_files": [str(path) for path in lock_paths],
        "incomplete_files": [str(path) for path in incomplete_paths],
        "orphan_arrow_groups": {
            group: [str(path) for path in paths]
            for group, paths in sorted(arrows_by_group.items())
            if group not in {dataset["key"] for dataset in datasets}
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="/home/liying/datasets/turboquant/hf_cache")
    parser.add_argument("--output", default="reproduce/logs/hf_cache_inventory.json")
    args = parser.parse_args()

    inventory = build_inventory(Path(args.cache_root))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "num_dataset_info": inventory["num_dataset_info"]}, indent=2))


if __name__ == "__main__":
    main()
