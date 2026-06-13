#!/usr/bin/env python3
"""Build a runnable status manifest for Table 1 LongBench reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.longbench_metrics import TABLE1_CATEGORIES


REPO_PREFIXES = ("longbench", "longbench_e")
METHODS = (
    {"name": "full", "label": "Full Cache", "kv_bits": 16.0, "cache_mode": "full"},
    {"name": "turboquant_2_5bit", "label": "TurboQuant", "kv_bits": 2.5, "cache_mode": "turboquant"},
    {"name": "turboquant_3_5bit", "label": "TurboQuant", "kv_bits": 3.5, "cache_mode": "turboquant"},
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_jsonl_indices(path: Path) -> set[int]:
    indices: set[int] = set()
    if not path.exists():
        return indices
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                indices.add(int(json.loads(line)["index"]))
    return indices


def expected_examples_for(data_cfg: dict[str, Any]) -> int | None:
    if data_cfg.get("num_examples") is not None:
        return int(data_cfg["num_examples"])
    cache_dir = Path(data_cfg["cache_dir"])
    info_path = cache_dir / "dataset_info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    split = data_cfg.get("split", "test")
    split_info = (info.get("splits") or {}).get(split)
    if not split_info:
        return None
    return split_info.get("num_examples")


def output_stem(dataset_key: str, method_name: str) -> str:
    if method_name == "full":
        return f"{dataset_key}_full_cache_all"
    return f"{dataset_key}_{method_name}_chunked"


def command_for(dataset_key: str, method: dict[str, Any], output_path: Path, device: str) -> str:
    parts = [
        "CUDA_VISIBLE_DEVICES=0",
        "conda run -n turboquant python experiments/longbench/run_full_cache_eval.py",
        f"--dataset-key {dataset_key}",
        f"--device {device}",
        f"--cache-mode {method['cache_mode']}",
    ]
    if method["cache_mode"] == "turboquant":
        parts.extend([f"--kv-bits {method['kv_bits']}", "--codebook-grid-size 10001", "--resume"])
    parts.extend([f"--output {output_path}", "--progress-every 5"])
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/logs/table1_manifest.json"))
    parser.add_argument("--runs-root", default=str(PROJECT_ROOT / "reproduce/runs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repo-prefixes", nargs="+", choices=REPO_PREFIXES, default=list(REPO_PREFIXES))
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    datasets_cfg = paths_cfg.get("datasets", {})
    runs_root = Path(args.runs_root)

    entries = []
    for category, datasets in TABLE1_CATEGORIES.items():
        for dataset in datasets:
            for prefix in args.repo_prefixes:
                dataset_key = f"{prefix}_{dataset}"
                data_cfg = datasets_cfg.get(dataset_key)
                expected_examples = None
                if data_cfg:
                    expected_examples = expected_examples_for(data_cfg)
                for method in METHODS:
                    stem = output_stem(dataset_key, method["name"])
                    output_path = runs_root / f"{stem}.jsonl"
                    aggregate_path = runs_root / f"{stem}.aggregate.json"
                    indices = load_jsonl_indices(output_path)
                    status = "missing_dataset"
                    if data_cfg:
                        status = "not_started"
                        if indices:
                            status = "partial"
                        if aggregate_path.exists() and indices and len(indices) == expected_examples:
                            status = "complete"
                    entries.append(
                        {
                            "category": category,
                            "dataset": dataset,
                            "dataset_key": dataset_key,
                            "repo_id": data_cfg.get("repo_id") if data_cfg else None,
                            "config": data_cfg.get("config") if data_cfg else dataset,
                            "split": data_cfg.get("split") if data_cfg else "test",
                            "method": method["label"],
                            "method_name": method["name"],
                            "kv_bits": method["kv_bits"],
                            "status": status,
                            "num_records": len(indices),
                            "expected_examples": expected_examples,
                            "output": str(output_path),
                            "aggregate": str(aggregate_path),
                            "command": command_for(dataset_key, method, output_path, args.device) if data_cfg else None,
                        }
                    )

    summary = {
        "num_entries": len(entries),
        "status_counts": {
            status: sum(1 for entry in entries if entry["status"] == status)
            for status in sorted({entry["status"] for entry in entries})
        },
        "entries": entries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **summary["status_counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
