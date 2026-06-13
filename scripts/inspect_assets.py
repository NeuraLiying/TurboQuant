#!/usr/bin/env python3
"""Inspect local assets required for the first TurboQuant reproduction pass."""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Install the project environment first.") from exc


def file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() and path.is_file() else None


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return {"available": False, "error": f"{args[0]} not found"}
    return {
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def inspect_model(model_cfg: dict[str, Any]) -> dict[str, Any]:
    snapshot = Path(model_cfg["snapshot"])
    expected = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    shards = sorted(snapshot.glob("model-*-of-*.safetensors"))
    return {
        "repo_id": model_cfg.get("repo_id"),
        "snapshot": str(snapshot),
        "snapshot_exists": snapshot.exists(),
        "expected_files": {name: (snapshot / name).exists() for name in expected},
        "num_safetensor_shards": len(shards),
        "safetensor_shards": [{"path": str(path), "bytes": file_size(path)} for path in shards],
    }


def inspect_dataset(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    cache_dir = Path(dataset_cfg["cache_dir"])
    arrow_files = dataset_cfg.get("arrow_files")
    if arrow_files is None and dataset_cfg.get("arrow_glob"):
        arrow_files = sorted(glob.glob(dataset_cfg["arrow_glob"]))
    if arrow_files is None:
        arrow_files = []
    parquet_files = dataset_cfg.get("parquet_files") or []
    arrows = [Path(path) for path in arrow_files]
    parquets = [Path(path) for path in parquet_files]
    info_path = cache_dir / "dataset_info.json"
    info: dict[str, Any] | None = None
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    return {
        "repo_id": dataset_cfg.get("repo_id"),
        "config": dataset_cfg.get("config"),
        "split": dataset_cfg.get("split"),
        "cache_dir": str(cache_dir),
        "cache_dir_exists": cache_dir.exists(),
        "dataset_info_exists": info_path.exists(),
        "dataset_info_splits": None if info is None else info.get("splits"),
        "num_arrow_files": len(arrows),
        "total_arrow_bytes": sum(file_size(path) or 0 for path in arrows),
        "arrow_files_preview": [{"path": str(path), "bytes": file_size(path)} for path in arrows[:5]],
        "num_parquet_files": len(parquets),
        "total_parquet_bytes": sum(file_size(path) or 0 for path in parquets),
        "parquet_files_preview": [{"path": str(path), "bytes": file_size(path)} for path in parquets[:5]],
    }


def inspect_python() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for name in ["torch", "transformers", "datasets", "accelerate", "safetensors", "numpy", "scipy", "pyarrow"]:
        try:
            module = __import__(name)
        except Exception as exc:  # pragma: no cover
            packages[name] = {"available": False, "error": str(exc)}
            continue
        packages[name] = {"available": True, "version": getattr(module, "__version__", None)}

    torch_info: dict[str, Any] = {}
    if packages.get("torch", {}).get("available"):
        import torch

        torch_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": torch.cuda.get_device_capability(idx),
                }
                for idx in range(torch.cuda.device_count())
            ],
        }

    return {
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch": torch_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths_file = Path(args.paths)
    with paths_file.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    report = {
        "paths_file": str(paths_file.resolve()),
        "cwd": os.getcwd(),
        "python": inspect_python(),
        "nvidia_smi": run_command(["nvidia-smi"]),
        "models": {key: inspect_model(value) for key, value in cfg["models"].items()},
        "datasets": {key: inspect_dataset(value) for key, value in cfg["datasets"].items()},
    }

    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
