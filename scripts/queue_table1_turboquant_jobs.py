#!/usr/bin/env python3
"""Background queue for Table 1 TurboQuant LongBench runs.

The queue is intentionally conservative:
- completed full task files are never rerun;
- partial files are resumed;
- active jobs are detected by their output path in the process table;
- tasks that have OOMed on one GPU are retried with two GPUs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "reproduce/runs/table1_official"
LOG_ROOT = PROJECT_ROOT / "reproduce/logs/table1_turboquant_jobs"
OFFLOAD_ROOT = PROJECT_ROOT / "reproduce/offload"
CONDA = Path("/home/liying/miniconda3/bin/conda")

TASKS = [
    ("narrativeqa", 200),
    ("qasper", 200),
    ("multifieldqa_en", 150),
    ("hotpotqa", 200),
    ("2wikimqa", 200),
    ("musique", 200),
    ("gov_report", 200),
    ("qmsum", 200),
    ("multi_news", 200),
    ("trec", 200),
    ("triviaqa", 200),
    ("samsum", 200),
    ("passage_retrieval_en", 200),
    ("passage_count", 200),
    ("lcc", 500),
    ("repobench-p", 500),
]

METHODS = [
    ("turboquant_2_5bit", "2.5"),
    ("turboquant_3_5bit", "3.5"),
]

DUAL_PREFERRED = {"narrativeqa", "qmsum", "passage_count", "repobench-p"}
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")


@dataclass(frozen=True)
class Job:
    dataset: str
    expected: int
    method_stem: str
    kv_bits: str

    @property
    def dataset_key(self) -> str:
        return f"longbench_{self.dataset}"

    @property
    def output(self) -> Path:
        return RUN_ROOT / f"longbench_{self.dataset}_{self.method_stem}_all.jsonl"

    @property
    def aggregate(self) -> Path:
        return RUN_ROOT / f"longbench_{self.dataset}_{self.method_stem}_all.aggregate.json"

    @property
    def stem(self) -> str:
        return f"{self.dataset}_{self.method_stem}"


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_unique_indexes(path: Path) -> int:
    if not path.exists():
        return 0
    indexes = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            indexes.add(int(json.loads(line)["index"]))
    return len(indexes)


def run_quiet(args: list[str], *, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def process_table() -> str:
    result = run_quiet(["ps", "-eo", "pid,ppid,pgid,sid,stat,etime,cmd"])
    return result.stdout


def is_running(job: Job, ps_text: str) -> bool:
    return str(job.output) in ps_text or str(job.output.relative_to(PROJECT_ROOT)) in ps_text


def gpu_memory() -> dict[int, int]:
    result = run_quiet(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    memory: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index_text, used_text = [part.strip() for part in line.split(",", 1)]
        memory[int(index_text)] = int(used_text)
    return memory


def log_paths(job: Job) -> list[Path]:
    return [
        LOG_ROOT / f"{job.stem}.log",
        LOG_ROOT / f"{job.stem}_dual.log",
    ]


def saw_oom(job: Job) -> bool:
    for path in log_paths(job):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in OOM_MARKERS):
            return True
    return False


def dual_failed(job: Job) -> bool:
    path = LOG_ROOT / f"{job.stem}_dual.log"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(marker in text for marker in OOM_MARKERS)


def needs_dual(job: Job) -> bool:
    return job.dataset in DUAL_PREFERRED or saw_oom(job)


def summarize(job: Job) -> bool:
    result = run_quiet(
        [
            str(CONDA),
            "run",
            "-n",
            "turboquant",
            "python",
            "scripts/summarize_jsonl_accuracy.py",
            str(job.output.relative_to(PROJECT_ROOT)),
            "--output",
            str(job.aggregate.relative_to(PROJECT_ROOT)),
        ]
    )
    if result.returncode != 0:
        (LOG_ROOT / "queue_summarize_errors.log").open("a", encoding="utf-8").write(
            f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} {job.stem} ---\n{result.stdout}\n"
        )
        return False
    return True


def refresh_reports() -> None:
    commands = [
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
    ]
    with (LOG_ROOT / "queue_report_refresh.log").open("a", encoding="utf-8") as log:
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} refresh ---\n")
        for command in commands:
            result = run_quiet(command)
            log.write(" ".join(command) + "\n")
            log.write(result.stdout + "\n")


def is_complete(job: Job) -> bool:
    unique_records = count_unique_indexes(job.output)
    if unique_records != job.expected:
        return False
    if not job.aggregate.exists():
        summarize(job)
    return job.aggregate.exists()


def progress_snapshot(jobs: list[Job]) -> dict:
    rows = []
    for job in jobs:
        records = count_jsonl(job.output)
        unique_records = count_unique_indexes(job.output)
        rows.append(
            {
                "dataset": job.dataset,
                "method_stem": job.method_stem,
                "records": records,
                "unique_records": unique_records,
                "expected": job.expected,
                "complete": unique_records == job.expected and job.aggregate.exists(),
                "aggregate": str(job.aggregate.relative_to(PROJECT_ROOT)),
            }
        )
    return {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "complete": sum(1 for row in rows if row["complete"]),
        "total": len(rows),
        "tasks": rows,
    }


def write_progress(jobs: list[Job]) -> None:
    snapshot = progress_snapshot(jobs)
    path = RUN_ROOT / "table1_turboquant_queue_status.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def launch(job: Job, gpus: list[int]) -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    OFFLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    dual = len(gpus) == 2
    mode_suffix = "_dual" if dual else ""
    log_path = LOG_ROOT / f"{job.stem}{mode_suffix}.log"
    script_path = LOG_ROOT / f"{job.stem}{mode_suffix}.sh"
    pid_path = LOG_ROOT / f"{job.stem}{mode_suffix}.pid"
    offload = OFFLOAD_ROOT / f"{job.method_stem}_{job.dataset}"

    command = [
        str(CONDA),
        "run",
        "-n",
        "turboquant",
        "python",
        "experiments/longbench/run_full_cache_eval.py",
        "--dataset-key",
        job.dataset_key,
        "--cache-mode",
        "turboquant",
        "--kv-bits",
        job.kv_bits,
        "--turboquant-fast-materialized-eval",
        "--prompt-mode",
        "longbench",
        "--chat-template-mode",
        "auto",
        "--start-index",
        "0",
        "--end-index",
        str(job.expected),
        "--resume",
        "--output",
        str(job.output.relative_to(PROJECT_ROOT)),
        "--progress-every",
        "20",
    ]
    if dual:
        offload.mkdir(parents=True, exist_ok=True)
        command.extend(
            [
                "--device-map",
                "auto",
                "--max-memory",
                "0=20GiB",
                "--max-memory",
                "1=20GiB",
                "--offload-folder",
                str(offload.relative_to(PROJECT_ROOT)),
            ]
        )
    else:
        command.extend(["--device", "cuda:0"])

    summarize_command = [
        str(CONDA),
        "run",
        "-n",
        "turboquant",
        "python",
        "scripts/summarize_jsonl_accuracy.py",
        str(job.output.relative_to(PROJECT_ROOT)),
        "--output",
        str(job.aggregate.relative_to(PROJECT_ROOT)),
    ]

    script = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {PROJECT_ROOT}",
        f"export CUDA_VISIBLE_DEVICES={','.join(str(gpu) for gpu in gpus)}",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"echo '[queue] start {time.strftime('%Y-%m-%d %H:%M:%S')} {job.stem} gpus={','.join(str(gpu) for gpu in gpus)}'",
        " ".join(f"'{part}'" for part in command),
        " ".join(f"'{part}'" for part in summarize_command),
        f"echo '[queue] done {time.strftime('%Y-%m-%d %H:%M:%S')} {job.stem}'",
    ]
    script_path.write_text("\n".join(script) + "\n", encoding="utf-8")
    script_path.chmod(0o755)

    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            ["bash", str(script_path)],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    return proc.pid


def ordered_jobs() -> list[Job]:
    jobs = []
    for method_stem, kv_bits in METHODS:
        for dataset, expected in TASKS:
            jobs.append(Job(dataset=dataset, expected=expected, method_stem=method_stem, kv_bits=kv_bits))
    return jobs


def active_or_incomplete_method(jobs: list[Job], ps_text: str) -> str | None:
    for method_stem, _ in METHODS:
        method_jobs = [job for job in jobs if job.method_stem == method_stem]
        if any((not is_complete(job)) or is_running(job, ps_text) for job in method_jobs):
            return method_stem
    return None


def schedulable_methods(start_method: str | None) -> list[str]:
    method_order = [method_stem for method_stem, _ in METHODS]
    if start_method is None:
        return []
    return method_order[method_order.index(start_method) :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--free-memory-threshold-mib", type=int, default=1000)
    parser.add_argument("--idle-exit-seconds", type=int, default=600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    jobs = ordered_jobs()
    last_launch_or_completion = time.time()
    while True:
        ps_text = process_table()
        memory = gpu_memory()
        free_gpus = [gpu for gpu, used in sorted(memory.items()) if used <= args.free_memory_threshold_mib]
        reserved: set[int] = set()
        launched = []
        active_jobs = [job.stem for job in jobs if is_running(job, ps_text)]
        complete_before = sum(1 for job in jobs if is_complete(job))
        schedulable_method = active_or_incomplete_method(jobs, ps_text)

        for method_stem in schedulable_methods(schedulable_method):
            for job in jobs:
                if job.method_stem != method_stem:
                    continue
                if is_complete(job):
                    continue
                if is_running(job, ps_text):
                    continue
                dual = needs_dual(job)
                if dual and dual_failed(job):
                    continue
                available = [gpu for gpu in free_gpus if gpu not in reserved]
                required = 2 if dual else 1
                if len(available) < required:
                    continue
                selected = available[:required]
                pid = launch(job, selected)
                reserved.update(selected)
                launched.append({"job": job.stem, "gpus": selected, "pid": pid})

        complete_after = sum(1 for job in jobs if is_complete(job))
        if launched or complete_after != complete_before:
            last_launch_or_completion = time.time()
        if complete_after != complete_before:
            refresh_reports()
        status = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "free_gpus": free_gpus,
            "launched": launched,
            "active_jobs": active_jobs,
            "schedulable_method": schedulable_method,
            "complete": complete_after,
            "total": len(jobs),
        }
        (LOG_ROOT / "queue_table1_turboquant_jobs.log").open("a", encoding="utf-8").write(
            json.dumps(status, ensure_ascii=False) + "\n"
        )
        write_progress(jobs)

        if complete_after == len(jobs):
            refresh_reports()
            break
        if args.once:
            break
        if not active_jobs and time.time() - last_launch_or_completion > args.idle_exit_seconds:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
