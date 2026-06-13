#!/usr/bin/env python3
"""Wait for Table 1 TurboQuant queue completion, then refresh and verify reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "reproduce/runs/table1_official"
LOG_ROOT = PROJECT_ROOT / "reproduce/logs/table1_turboquant_jobs"
STATUS_PATH = RUN_ROOT / "table1_turboquant_queue_status.json"
CONDA = Path("/home/liying/miniconda3/bin/conda")

TASKS = {
    "narrativeqa": 200,
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "passage_retrieval_en": 200,
    "passage_count": 200,
    "lcc": 500,
    "repobench-p": 500,
}

METHODS = ("full_cache", "turboquant_2_5bit", "turboquant_3_5bit")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def run_logged(command: list[str], log_path: Path) -> None:
    append_log(log_path, f"[finalize] run: {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        append_log(log_path, result.stdout.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"command failed with code {result.returncode}: {' '.join(command)}")


def queue_running() -> bool:
    result = subprocess.run(
        ["ps", "-eo", "cmd"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return "scripts/queue_table1_turboquant_jobs.py" in result.stdout


def read_queue_progress() -> tuple[int, int]:
    if not STATUS_PATH.exists():
        return 0, 32
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return int(status.get("complete", 0)), int(status.get("total", 32))


def count_unique_indexes(path: Path) -> tuple[int, int]:
    records = 0
    indexes: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records += 1
            indexes.add(int(json.loads(line)["index"]))
    return records, len(indexes)


def refresh_reports(log_path: Path) -> None:
    run_logged(
        [
            str(CONDA),
            "run",
            "-n",
            "turboquant",
            "python",
            "scripts/summarize_table1_method_progress.py",
            "--method-stem",
            "turboquant_2_5bit",
            "--title",
            "Table 1 TurboQuant 2.5-bit Progress",
        ],
        log_path,
    )
    run_logged(
        [
            str(CONDA),
            "run",
            "-n",
            "turboquant",
            "python",
            "scripts/summarize_table1_method_progress.py",
            "--method-stem",
            "turboquant_3_5bit",
            "--title",
            "Table 1 TurboQuant 3.5-bit Progress",
        ],
        log_path,
    )
    run_logged(
        [
            str(CONDA),
            "run",
            "-n",
            "turboquant",
            "python",
            "scripts/build_table1_official_comparison.py",
            "--run-root",
            "reproduce/runs/table1_official",
            "--output-prefix",
            "reproduce/TABLE1_OFFICIAL_COMPARISON",
        ],
        log_path,
    )


def verify_outputs() -> list[str]:
    errors: list[str] = []
    for method in METHODS:
        for dataset, expected in TASKS.items():
            output = RUN_ROOT / f"longbench_{dataset}_{method}_all.jsonl"
            aggregate = RUN_ROOT / f"longbench_{dataset}_{method}_all.aggregate.json"
            if not output.exists():
                errors.append(f"missing output: {output.relative_to(PROJECT_ROOT)}")
                continue
            if not aggregate.exists():
                errors.append(f"missing aggregate: {aggregate.relative_to(PROJECT_ROOT)}")
                continue
            records, unique = count_unique_indexes(output)
            if unique != expected:
                errors.append(
                    f"{method}/{dataset}: records={records}, unique={unique}, expected={expected}"
                )

    comparison = PROJECT_ROOT / "reproduce/TABLE1_OFFICIAL_COMPARISON.json"
    if not comparison.exists():
        errors.append("missing comparison: reproduce/TABLE1_OFFICIAL_COMPARISON.json")
    else:
        rows = json.loads(comparison.read_text(encoding="utf-8"))
        for row in rows:
            if not row.get("complete"):
                errors.append(f"comparison incomplete row: {row.get('method_stem')}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument(
        "--log",
        default="reproduce/logs/table1_turboquant_jobs/finalize_table1_when_done.log",
    )
    args = parser.parse_args()
    log_path = PROJECT_ROOT / args.log

    append_log(log_path, f"[finalize] start {now()}")
    while True:
        complete, total = read_queue_progress()
        append_log(log_path, f"[finalize] {now()} queue={complete}/{total}")
        if complete == total:
            break
        if not queue_running():
            append_log(log_path, "[finalize] queue process stopped before completion")
            return 2
        time.sleep(args.poll_seconds)

    # Give the queue's own report refresh a short head start, then write a final refresh.
    time.sleep(10)
    refresh_reports(log_path)
    errors = verify_outputs()
    if errors:
        append_log(log_path, "[finalize] verification failed")
        for error in errors:
            append_log(log_path, f"[finalize] {error}")
        return 3
    append_log(log_path, f"[finalize] verification passed {now()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
