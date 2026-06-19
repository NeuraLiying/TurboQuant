#!/usr/bin/env python3
"""Queue unified regular-gain TurboQuant LongBench jobs.

This queue is for the incremental candidate that uses the same method at
2.5-bit and 3.5-bit: regular_gain_mse for both K and V under the original
TurboQuant fractional channel split.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "reproduce/runs/incremental"
LOG_ROOT = PROJECT_ROOT / "reproduce/logs/unified_regular_gain_jobs"
OFFLOAD_ROOT = PROJECT_ROOT / "reproduce/offload/unified_regular_gain"
PYTHON = Path("/home/liying/miniconda3/envs/turboquant/bin/python")

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
    ("2p5", "2.5", "2_5"),
    ("3p5", "3.5", "3_5"),
]

DUAL_PREFERRED = {"narrativeqa", "qmsum", "passage_count", "repobench-p"}
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")


@dataclass(frozen=True)
class Job:
    dataset: str
    expected: int
    bit_tag: str
    kv_bits: str
    baseline_bit_tag: str

    @property
    def dataset_key(self) -> str:
        return f"longbench_{self.dataset}"

    @property
    def output(self) -> Path:
        return RUN_ROOT / f"unified_regular_gain_{self.dataset}_turboquant_{self.bit_tag}_full.jsonl"

    @property
    def baseline(self) -> Path:
        return (
            PROJECT_ROOT
            / "reproduce/runs/table1_official"
            / f"longbench_{self.dataset}_turboquant_{self.baseline_bit_tag}bit_all.jsonl"
        )

    @property
    def stem(self) -> str:
        return f"unified_regular_gain_{self.dataset}_{self.bit_tag}"


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


def run_quiet(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def process_table() -> str:
    return run_quiet(["ps", "-eo", "pid,ppid,pgid,sid,stat,etime,cmd"]).stdout


def is_running(job: Job, ps_text: str) -> bool:
    return str(job.output) in ps_text or str(job.output.relative_to(PROJECT_ROOT)) in ps_text


def gpu_memory() -> dict[int, int]:
    result = run_quiet(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"])
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
    return [LOG_ROOT / f"{job.stem}.log", LOG_ROOT / f"{job.stem}_dual.log"]


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


def is_complete(job: Job) -> bool:
    return count_unique_indexes(job.output) == job.expected


def launch(job: Job, gpus: list[int], *, progress_every: int) -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    OFFLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    dual = len(gpus) == 2
    suffix = "_dual" if dual else ""
    log_path = LOG_ROOT / f"{job.stem}{suffix}.log"
    script_path = LOG_ROOT / f"{job.stem}{suffix}.sh"
    pid_path = LOG_ROOT / f"{job.stem}{suffix}.pid"
    offload = OFFLOAD_ROOT / job.stem

    command = [
        str(PYTHON),
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
        "--key-quantizer",
        "regular_gain_mse",
        "--value-quantizer",
        "regular_gain_mse",
        "--output",
        str(job.output.relative_to(PROJECT_ROOT)),
        "--progress-every",
        str(progress_every),
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

    script = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {PROJECT_ROOT}",
        f"export CUDA_VISIBLE_DEVICES={','.join(str(gpu) for gpu in gpus)}",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"echo '[queue] start {time.strftime('%Y-%m-%d %H:%M:%S')} {job.stem} gpus={','.join(str(gpu) for gpu in gpus)}'",
        " ".join("'" + part.replace("'", "'\\''") + "'" for part in command),
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


def ordered_jobs(bit_filter: set[str] | None = None) -> list[Job]:
    jobs: list[Job] = []
    for bit_tag, kv_bits, baseline_bit_tag in METHODS:
        if bit_filter is not None and bit_tag not in bit_filter:
            continue
        for dataset, expected in TASKS:
            jobs.append(
                Job(
                    dataset=dataset,
                    expected=expected,
                    bit_tag=bit_tag,
                    kv_bits=kv_bits,
                    baseline_bit_tag=baseline_bit_tag,
                )
            )
    return jobs


def write_progress(jobs: list[Job]) -> None:
    rows = []
    for job in jobs:
        rows.append(
            {
                "dataset": job.dataset,
                "bit_tag": job.bit_tag,
                "records": count_unique_indexes(job.output),
                "expected": job.expected,
                "complete": is_complete(job),
                "output": str(job.output.relative_to(PROJECT_ROOT)),
            }
        )
    status = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "complete": sum(1 for row in rows if row["complete"]),
        "total": len(rows),
        "tasks": rows,
    }
    (LOG_ROOT / "queue_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--free-memory-threshold-mib", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--bits", choices=["2p5", "3p5"], nargs="+", default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    jobs = ordered_jobs(set(args.bits) if args.bits is not None else None)
    while True:
        ps_text = process_table()
        memory = gpu_memory()
        free_gpus = [
            gpu
            for gpu, used in sorted(memory.items())
            if gpu in set(args.gpus) and used <= args.free_memory_threshold_mib
        ]
        reserved: set[int] = set()
        launched = []
        active_jobs = [job.stem for job in jobs if is_running(job, ps_text)]

        blocked_dual_job = any(
            (not is_complete(job))
            and (not is_running(job, ps_text))
            and job.baseline.exists()
            and needs_dual(job)
            and (not dual_failed(job))
            for job in jobs
        )
        dual_launched = False

        for job in jobs:
            if is_complete(job) or is_running(job, ps_text):
                continue
            if not job.baseline.exists():
                continue
            dual = needs_dual(job)
            if dual and dual_failed(job):
                continue
            available = [gpu for gpu in free_gpus if gpu not in reserved]
            required = 2 if dual else 1
            if len(available) < required:
                continue
            # Prefer pending dual-GPU jobs when two GPUs are immediately available,
            # but do not leave a single free GPU idle while waiting for another GPU.
            if blocked_dual_job and not dual and not dual_launched and len(available) >= 2:
                continue
            selected = available[:required]
            pid = launch(job, selected, progress_every=args.progress_every)
            reserved.update(selected)
            dual_launched = dual_launched or dual
            launched.append({"job": job.stem, "gpus": selected, "pid": pid})

        status = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "free_gpus": free_gpus,
            "launched": launched,
            "active_jobs": active_jobs,
            "complete": sum(1 for job in jobs if is_complete(job)),
            "total": len(jobs),
        }
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        (LOG_ROOT / "queue.log").open("a", encoding="utf-8").write(json.dumps(status, ensure_ascii=False) + "\n")
        write_progress(jobs)

        if status["complete"] == len(jobs):
            break
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
