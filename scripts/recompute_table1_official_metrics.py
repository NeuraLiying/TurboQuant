#!/usr/bin/env python3
"""Recompute LongBench scores for Table 1 JSONL outputs with current metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def summarize(path: Path) -> tuple[Path, int, str]:
    output = path.with_suffix(".aggregate.json")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/summarize_jsonl_accuracy.py"),
        str(path),
        "--recompute-longbench-score",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return output, result.returncode, result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(PROJECT_ROOT / "reproduce/runs/table1_official_metric"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    paths = sorted(run_root.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no JSONL files found under {run_root}")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(summarize, path) for path in paths]
        for future in as_completed(futures):
            output, returncode, stdout = future.result()
            if returncode != 0:
                failures.append((output, stdout))
            print(f"{'ok' if returncode == 0 else 'fail'} {output}", flush=True)

    if failures:
        for output, stdout in failures:
            print(f"\n--- {output} ---\n{stdout}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
