#!/usr/bin/env python3
"""Compare one full-cache and TurboQuant-cache LongBench generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kv-bits", type=float, default=3.0)
    args = parser.parse_args()

    env_python = sys.executable
    full_out = PROJECT_ROOT / "reproduce/runs/longbench_cache_compare_full.jsonl"
    tq_out = PROJECT_ROOT / f"reproduce/runs/longbench_cache_compare_tq_{str(args.kv_bits).replace('.', '_')}bit.jsonl"

    base = [
        env_python,
        str(PROJECT_ROOT / "experiments/longbench/run_full_cache_eval.py"),
        "--max-examples",
        str(args.max_examples),
        "--device",
        args.device,
    ]
    run(base + ["--cache-mode", "full", "--output", str(full_out)])
    run(
        base
        + [
            "--cache-mode",
            "turboquant",
            "--kv-bits",
            str(args.kv_bits),
            "--output",
            str(tq_out),
        ]
    )

    print(
        json.dumps(
            {
                "full_output": str(full_out),
                "turboquant_output": str(tq_out),
                "kv_bits": args.kv_bits,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
