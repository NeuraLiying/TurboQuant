#!/usr/bin/env python3
"""Queue Table-1 LongBench jobs for one unified TurboQuant method."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "reproduce/runs/incremental"
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
METHODS = [("2p5", "2.5", "2_5"), ("3p5", "3.5", "3_5")]
DUAL_PREFERRED = {"narrativeqa", "gov_report", "qmsum", "passage_count", "repobench-p"}
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")


@dataclass(frozen=True)
class Job:
    method_name: str
    key_quantizer: str
    value_quantizer: str
    dataset: str
    expected: int
    bit_tag: str
    kv_bits: str
    baseline_bit_tag: str
    extra_args: tuple[str, ...] = ()

    @property
    def dataset_key(self) -> str:
        return f"longbench_{self.dataset}"

    @property
    def output(self) -> Path:
        return RUN_ROOT / f"{self.method_name}_{self.dataset}_turboquant_{self.bit_tag}_full.jsonl"

    @property
    def baseline(self) -> Path:
        return (
            PROJECT_ROOT
            / "reproduce/runs/table1_official"
            / f"longbench_{self.dataset}_turboquant_{self.baseline_bit_tag}bit_all.jsonl"
        )

    @property
    def stem(self) -> str:
        return f"{self.method_name}_{self.dataset}_{self.bit_tag}"


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


def log_paths(job: Job, log_root: Path) -> list[Path]:
    return [log_root / f"{job.stem}.log", log_root / f"{job.stem}_dual.log"]


def saw_oom(job: Job, log_root: Path) -> bool:
    for path in log_paths(job, log_root):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in OOM_MARKERS):
            return True
    return False


def dual_failed(job: Job, log_root: Path) -> bool:
    path = log_root / f"{job.stem}_dual.log"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(marker in text for marker in OOM_MARKERS)


def needs_dual(job: Job, log_root: Path) -> bool:
    return job.dataset in DUAL_PREFERRED or saw_oom(job, log_root)


def is_complete(job: Job) -> bool:
    return count_unique_indexes(job.output) == job.expected


def launch(job: Job, gpus: list[int], *, log_root: Path, offload_root: Path, progress_every: int) -> int:
    log_root.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    offload_root.mkdir(parents=True, exist_ok=True)

    dual = len(gpus) == 2
    suffix = "_dual" if dual else ""
    log_path = log_root / f"{job.stem}{suffix}.log"
    script_path = log_root / f"{job.stem}{suffix}.sh"
    pid_path = log_root / f"{job.stem}{suffix}.pid"
    offload = offload_root / job.stem

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
        job.key_quantizer,
        "--value-quantizer",
        job.value_quantizer,
        "--output",
        str(job.output.relative_to(PROJECT_ROOT)),
        "--progress-every",
        str(progress_every),
    ]
    command.extend(job.extra_args)
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


def ordered_jobs(
    *,
    method_name: str,
    key_quantizer: str,
    value_quantizer: str,
    bit_filter: set[str] | None,
    dataset_filter: set[str] | None,
    extra_args: tuple[str, ...],
) -> list[Job]:
    jobs: list[Job] = []
    for bit_tag, kv_bits, baseline_bit_tag in METHODS:
        if bit_filter is not None and bit_tag not in bit_filter:
            continue
        for dataset, expected in TASKS:
            if dataset_filter is not None and dataset not in dataset_filter:
                continue
            jobs.append(
                Job(
                    method_name=method_name,
                    key_quantizer=key_quantizer,
                    value_quantizer=value_quantizer,
                    dataset=dataset,
                    expected=expected,
                    bit_tag=bit_tag,
                    kv_bits=kv_bits,
                    baseline_bit_tag=baseline_bit_tag,
                    extra_args=extra_args,
                )
            )
    return jobs


def build_extra_run_args(args: argparse.Namespace) -> tuple[str, ...]:
    extra: list[str] = []
    if args.long_prompt_threshold is not None:
        extra.extend(["--long-prompt-threshold", str(args.long_prompt_threshold)])
    if args.long_prompt_key_quantizer is not None:
        extra.extend(["--long-prompt-key-quantizer", args.long_prompt_key_quantizer])
    if args.long_prompt_value_quantizer is not None:
        extra.extend(["--long-prompt-value-quantizer", args.long_prompt_value_quantizer])
    if args.long_prompt_max_tokens is not None:
        extra.extend(["--long-prompt-max-tokens", str(args.long_prompt_max_tokens)])
    if args.long_prompt_exclude_code:
        extra.append("--long-prompt-exclude-code")
    if args.long_prompt_max_passages is not None:
        extra.extend(["--long-prompt-max-passages", str(args.long_prompt_max_passages)])
    if args.long_prompt_max_question_marks is not None:
        extra.extend(["--long-prompt-max-question-marks", str(args.long_prompt_max_question_marks)])
    extra.extend(args.extra_run_arg)
    return tuple(extra)


def write_progress(jobs: list[Job], log_root: Path) -> None:
    rows = [
        {
            "dataset": job.dataset,
            "bit_tag": job.bit_tag,
            "records": count_unique_indexes(job.output),
            "expected": job.expected,
            "complete": is_complete(job),
            "output": str(job.output.relative_to(PROJECT_ROOT)),
        }
        for job in jobs
    ]
    status = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "complete": sum(1 for row in rows if row["complete"]),
        "total": len(rows),
        "tasks": rows,
    }
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "queue_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--key-quantizer", required=True)
    parser.add_argument("--value-quantizer", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--free-memory-threshold-mib", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--bits", choices=["2p5", "3p5"], nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--long-prompt-threshold", type=int, default=None)
    parser.add_argument("--long-prompt-key-quantizer", default=None)
    parser.add_argument("--long-prompt-value-quantizer", default=None)
    parser.add_argument("--long-prompt-max-tokens", type=int, default=None)
    parser.add_argument("--long-prompt-exclude-code", action="store_true")
    parser.add_argument("--long-prompt-max-passages", type=int, default=None)
    parser.add_argument("--long-prompt-max-question-marks", type=int, default=None)
    parser.add_argument("--extra-run-arg", action="append", default=[])
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    log_root = PROJECT_ROOT / "reproduce/logs" / f"{args.method_name}_jobs"
    offload_root = PROJECT_ROOT / "reproduce/offload" / args.method_name
    jobs = ordered_jobs(
        method_name=args.method_name,
        key_quantizer=args.key_quantizer,
        value_quantizer=args.value_quantizer,
        bit_filter=set(args.bits) if args.bits is not None else None,
        dataset_filter=set(args.datasets) if args.datasets is not None else None,
        extra_args=build_extra_run_args(args),
    )
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
            and needs_dual(job, log_root)
            and (not dual_failed(job, log_root))
            for job in jobs
        )
        dual_launched = False

        for job in jobs:
            if is_complete(job) or is_running(job, ps_text):
                continue
            if not job.baseline.exists():
                continue
            dual = needs_dual(job, log_root)
            if dual and dual_failed(job, log_root):
                continue
            available = [gpu for gpu in free_gpus if gpu not in reserved]
            required = 2 if dual else 1
            if len(available) < required:
                continue
            if blocked_dual_job and not dual and not dual_launched and len(available) >= 2:
                continue
            selected = available[:required]
            pid = launch(job, selected, log_root=log_root, offload_root=offload_root, progress_every=args.progress_every)
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
        log_root.mkdir(parents=True, exist_ok=True)
        (log_root / "queue.log").open("a", encoding="utf-8").write(json.dumps(status, ensure_ascii=False) + "\n")
        write_progress(jobs, log_root)

        if status["complete"] == len(jobs):
            break
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
