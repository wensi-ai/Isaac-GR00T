# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark training throughput / GPU memory for scripts/b1k/train_b1k.py.

Launches short real training runs via torchrun, measures steady-state
throughput by timestamping the trainer's per-``logging_steps`` loss logs,
and polls nvidia-smi for peak GPU memory. Two modes:

``run``
    One measured run of a single configuration. Use ``--train-args`` to pass
    anything extra through to train_b1k.py (e.g. ``--decode-only-used-frames``).

``sweep``
    Coordinate-descent sweep to find the fastest ``OMP_NUM_THREADS`` →
    ``--dataloader-num-workers`` → ``--global-batch-size`` (in that order),
    starting from the ``--base-*`` values. The batch axis picks the highest
    throughput whose peak GPU memory stays under ``--mem-limit-frac``.

Examples::

    # Find optimal values on this machine (production-like data config):
    python scripts/b1k/benchmark_train_speed.py sweep \\
        --dataset-path $DATA_ROOT \\
        --train-args="--decode-only-used-frames" \\
        --results-json bench/sweep.json

    # Single measured run of one configuration:
    python scripts/b1k/benchmark_train_speed.py run \\
        --dataset-path $DATA_ROOT --label no-decode-flag-rate0.1 \\
        --global-batch-size 2048 --dataloader-num-workers 8 --omp-num-threads 4 \\
        --train-args="--episode-sampling-rate 0.1" \\
        --results-json bench/experiments.json

Notes:
  * Throughput is measured from step ``--measure-from-step`` (default:
    max-steps/2) to the last step, skipping model-load/startup and the
    "honeymoon" period during which batches are served from shards that the
    dataloader workers pre-decoded before step 1. Inspect ``window_sps`` in
    the results JSON to confirm the rate has plateaued; increase
    ``--max-steps`` if it has not.
  * The first run on a cold page cache is slower (video files not yet in RAM).
    Sweep mode runs an unmeasured warmup run first (disable with
    ``--no-warmup``); for ``run`` mode, do one throwaway run yourself if the
    cache is cold and comparability matters.
  * Normalization stats (meta/stats.json) must be precomputed for large
    multi-task datasets, exactly as for real training.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

LOSS_LINE = re.compile(r"\{'loss':")
OOM_PATTERNS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "CUDA error: out of memory",
    "cuda runtime error: out of memory",
)
# Kernel OOM-killer takes out dataloader workers when system RAM is exhausted.
RAM_OOM_PATTERNS = (
    "killed by signal: Killed",
    "exited unexpectedly",
)


@dataclasses.dataclass
class RunSpec:
    label: str
    global_batch_size: int
    dataloader_num_workers: int
    omp_num_threads: int
    train_args: list[str] = dataclasses.field(default_factory=list)

    def key(self):
        return (
            self.global_batch_size,
            self.dataloader_num_workers,
            self.omp_num_threads,
            tuple(self.train_args),
        )


def gpu_total_mem_mib() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
    )
    return [int(x) for x in out.split()]


class ResourcePoller(threading.Thread):
    """Polls per-GPU used memory, MemAvailable, and 1-min loadavg."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.max_gpu_used: list[int] = []
        self.min_mem_available_kb: int | None = None
        self.max_load1: float = 0.0

    def run(self):
        while not self.stop_event.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=15,
                )
                used = [int(x) for x in out.split()]
                if len(self.max_gpu_used) < len(used):
                    self.max_gpu_used += [0] * (len(used) - len(self.max_gpu_used))
                for i, u in enumerate(used):
                    self.max_gpu_used[i] = max(self.max_gpu_used[i], u)
            except Exception:
                pass
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            kb = int(line.split()[1])
                            if self.min_mem_available_kb is None or kb < self.min_mem_available_kb:
                                self.min_mem_available_kb = kb
                            break
                self.max_load1 = max(self.max_load1, os.getloadavg()[0])
            except Exception:
                pass
            self.stop_event.wait(self.interval)


def build_command(spec: RunSpec, args, run_dir: Path, port: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.num_gpus}",
        f"--master_port={port}",
        args.train_script,
        "--experiment-name",
        f"bench-{spec.label}",
        "--base-model-path",
        args.base_model_path,
        "--dataset-path",
        args.dataset_path,
        "--embodiment-tag",
        args.embodiment_tag,
        "--num-gpus",
        str(args.num_gpus),
        "--global-batch-size",
        str(spec.global_batch_size),
        "--dataloader-num-workers",
        str(spec.dataloader_num_workers),
        "--output-dir",
        str(run_dir),
        "--max-steps",
        str(args.max_steps),
        "--save-steps",
        str(10**9),  # never checkpoint mid-run
        "--save-total-limit",
        "1",
    ]
    if args.modality_config_path:
        cmd += ["--modality-config-path", args.modality_config_path]
    cmd += spec.train_args
    return cmd


def run_benchmark(spec: RunSpec, args, port: int) -> dict:
    """Run one configuration and return a result record."""
    work_dir = Path(args.work_dir)
    run_dir = work_dir / spec.label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / f"{spec.label}.log"

    cmd = build_command(spec, args, run_dir, port)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(spec.omp_num_threads)
    env.setdefault("WANDB_MODE", "offline")
    env["WANDB_DIR"] = str(run_dir)
    env["PYTHONUNBUFFERED"] = "1"
    for kv in args.extra_env:
        k, _, v = kv.partition("=")
        env[k] = v

    print(f"\n=== [{spec.label}] batch={spec.global_batch_size} workers="
          f"{spec.dataloader_num_workers} OMP={spec.omp_num_threads} "
          f"extra={' '.join(spec.train_args) or '(none)'}")
    print(f"    cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    if args.dry_run:
        return {"label": spec.label, "status": "dry-run"}

    loss_times: list[float] = []
    oom = False
    ram_oom = False
    shard_wait_total = 0.0
    shard_wait_count = 0

    poller = ResourcePoller()
    poller.start()
    t_start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        text=True,
        errors="replace",
    )

    def kill():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    shard_wait_re = re.compile(r"Wait for shard .* in ([0-9.]+) seconds")
    timed_out = False

    # Ensure the torchrun process group dies with the harness (SIGTERM/SIGINT).
    prev_handlers = {}

    def _forward_kill(signum, frame):
        kill()
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGINT):
        prev_handlers[sig] = signal.signal(sig, _forward_kill)
    try:
        with open(log_path, "w") as logf:
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                now = time.time()
                if LOSS_LINE.search(line):
                    loss_times.append(now)
                    n_steps = len(loss_times) * args.logging_steps
                    if n_steps % (10 * args.logging_steps) == 0 or len(loss_times) == 1:
                        print(f"    step {n_steps}/{args.max_steps} at t+{now - t_start:.0f}s")
                m = shard_wait_re.search(line)
                if m:
                    shard_wait_total += float(m.group(1))
                    shard_wait_count += 1
                if any(p in line for p in OOM_PATTERNS):
                    oom = True
                if any(p in line for p in RAM_OOM_PATTERNS):
                    ram_oom = True
                if now - t_start > args.run_timeout:
                    timed_out = True
                    print(f"    TIMEOUT after {args.run_timeout}s — killing")
                    kill()
                    break
        rc = proc.wait(timeout=300)
    except Exception as e:
        print(f"    harness error: {e}")
        kill()
        rc = -1
    finally:
        kill()
        for sig, h in prev_handlers.items():
            signal.signal(sig, h)
        poller.stop_event.set()
        poller.join(timeout=10)

    wall = time.time() - t_start
    expected_logs = args.max_steps // args.logging_steps
    record: dict = {
        "label": spec.label,
        "global_batch_size": spec.global_batch_size,
        "dataloader_num_workers": spec.dataloader_num_workers,
        "omp_num_threads": spec.omp_num_threads,
        "train_args": spec.train_args,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "measure_from_step": args.measure_from_step,
        "wall_time_s": round(wall, 1),
        "returncode": rc,
        "oom": oom,
        "timed_out": timed_out,
        "n_loss_logs": len(loss_times),
        "peak_gpu_mem_mib": max(poller.max_gpu_used) if poller.max_gpu_used else None,
        "peak_gpu_mem_per_gpu_mib": poller.max_gpu_used,
        "min_mem_available_gib": (
            round(poller.min_mem_available_kb / 2**20, 1)
            if poller.min_mem_available_kb is not None
            else None
        ),
        "max_load1": round(poller.max_load1, 1),
        "rank0_shard_wait_total_s": round(shard_wait_total, 1),
        "rank0_shard_wait_count": shard_wait_count,
        "log_file": str(log_path),
    }

    # One optimizer step consumes global_batch_size × gradient_accumulation_steps samples.
    accum = 1
    if "--gradient-accumulation-steps" in spec.train_args:
        accum = int(spec.train_args[spec.train_args.index("--gradient-accumulation-steps") + 1])
    samples_per_step = spec.global_batch_size * accum
    record["samples_per_step"] = samples_per_step

    if loss_times:
        record["time_to_first_log_s"] = round(loss_times[0] - t_start, 1)
        # Per-window rates (each window = logging_steps steps).
        window_sps = []
        for i in range(1, len(loss_times)):
            dt = loss_times[i] - loss_times[i - 1]
            window_sps.append(round(args.logging_steps * samples_per_step / dt, 1))
        record["window_sps"] = window_sps

    if oom:
        record["status"] = "oom"
    elif ram_oom:
        record["status"] = "ram-oom"
    elif timed_out:
        record["status"] = "timeout"
    elif len(loss_times) < expected_logs or rc != 0:
        record["status"] = "failed"
    else:
        start_idx = args.measure_from_step // args.logging_steps - 1
        steady_steps = args.max_steps - args.measure_from_step
        dt = loss_times[-1] - loss_times[start_idx]
        record["status"] = "ok"
        record["steady_steps_per_s"] = round(steady_steps / dt, 3)
        record["steady_samples_per_s"] = round(steady_steps * samples_per_step / dt, 1)

    status = record["status"]
    sps = record.get("steady_samples_per_s")
    print(
        f"    -> {status}"
        + (f", {sps} samples/s ({record['steady_steps_per_s']} steps/s)" if sps else "")
        + (f", peak GPU mem {record['peak_gpu_mem_mib']} MiB" if record["peak_gpu_mem_mib"] else "")
    )

    if not args.keep_run_outputs:
        shutil.rmtree(run_dir, ignore_errors=True)
    return record


def append_result(args, record: dict):
    path = Path(args.results_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if path.exists():
        results = json.loads(path.read_text())
    results.append(record)
    path.write_text(json.dumps(results, indent=2))


def markdown_table(records: list[dict]) -> str:
    header = (
        "| label | batch | workers | OMP | status | samples/s | steps/s | "
        "peak GPU MiB | startup s | rank0 shard-wait s |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in records:
        rows.append(
            f"| {r['label']} | {r.get('global_batch_size', '—')} | "
            f"{r.get('dataloader_num_workers', '—')} | "
            f"{r.get('omp_num_threads', '—')} | {r.get('status', '—')} | "
            f"{r.get('steady_samples_per_s', '—')} | "
            f"{r.get('steady_steps_per_s', '—')} | {r.get('peak_gpu_mem_mib', '—')} | "
            f"{r.get('time_to_first_log_s', '—')} | {r.get('rank0_shard_wait_total_s', '—')} |"
        )
    return header + "\n".join(rows)


def do_sweep(args):
    total_mem = min(gpu_total_mem_mib())
    mem_limit = total_mem * args.mem_limit_frac
    print(f"GPU memory limit for batch axis: {mem_limit:.0f} MiB ({args.mem_limit_frac:.0%} of {total_mem} MiB)")

    cache: dict = {}
    all_records: list[dict] = []
    port = [args.master_port]

    def measure(spec: RunSpec) -> dict:
        if spec.key() in cache:
            print(f"\n=== [{spec.label}] cached result reused")
            return cache[spec.key()]
        port[0] += 1
        rec = run_benchmark(spec, args, port[0])
        cache[spec.key()] = rec
        all_records.append(rec)
        append_result(args, rec)
        return rec

    best = {
        "omp": args.base_omp,
        "workers": args.base_workers,
        "batch": args.base_batch,
    }

    if args.warmup:
        print("\n### Warmup run (unmeasured; warms page cache / model cache)")
        port[0] += 1
        warm = run_benchmark(
            RunSpec(
                label="warmup",
                global_batch_size=best["batch"],
                dataloader_num_workers=best["workers"],
                omp_num_threads=best["omp"],
                train_args=args.train_args_list,
            ),
            args,
            port[0],
        )
        warm["label"] = "warmup (excluded from selection)"
        append_result(args, warm)

    axes = [
        ("omp", args.omp_candidates),
        ("workers", args.workers_candidates),
        ("batch", args.batch_candidates),
    ]

    for axis, candidates in axes:
        if not candidates:
            continue
        print(f"\n### Sweeping {axis}: {candidates} (base: {best})")
        axis_records = []
        for val in candidates:
            cfg = dict(best)
            cfg[axis] = val
            spec = RunSpec(
                label=f"omp{cfg['omp']}-w{cfg['workers']}-b{cfg['batch']}",
                global_batch_size=cfg["batch"],
                dataloader_num_workers=cfg["workers"],
                omp_num_threads=cfg["omp"],
                train_args=args.train_args_list,
            )
            rec = measure(spec)
            axis_records.append((val, rec))

        ok = [
            (val, rec)
            for val, rec in axis_records
            if rec.get("status") == "ok"
            and (axis != "batch" or (rec.get("peak_gpu_mem_mib") or 0) <= mem_limit)
        ]
        if not ok:
            print(f"!!! No successful runs on axis {axis}; keeping base value {best[axis]}")
            continue
        top_sps = max(rec["steady_samples_per_s"] for _, rec in ok)
        # Within tie tolerance of the top rate, prefer the smallest value
        # (fewer threads/workers = less contention; smaller batch = less memory).
        contenders = sorted(
            (val, rec)
            for val, rec in ok
            if rec["steady_samples_per_s"] >= top_sps * (1 - args.tie_tolerance)
        )
        best[axis] = contenders[0][0]
        print(f">>> best {axis} = {best[axis]} "
              f"({contenders[0][1]['steady_samples_per_s']} samples/s; top {top_sps})")

    print("\n### Sweep complete")
    print(f"Optimal: OMP_NUM_THREADS={best['omp']} "
          f"--dataloader-num-workers {best['workers']} "
          f"--global-batch-size {best['batch']}")
    print("\n" + markdown_table(all_records))
    summary = {
        "label": "SWEEP_SUMMARY",
        "optimal": {
            "OMP_NUM_THREADS": best["omp"],
            "dataloader_num_workers": best["workers"],
            "global_batch_size": best["batch"],
        },
    }
    append_result(args, summary)


def do_run(args):
    spec = RunSpec(
        label=args.label,
        global_batch_size=args.global_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        omp_num_threads=args.omp_num_threads,
        train_args=args.train_args_list,
    )
    rec = run_benchmark(spec, args, args.master_port)
    if not args.dry_run:
        append_result(args, rec)
        print("\n" + markdown_table([rec]))


def main():
    # Line-buffer stdout so progress is visible when piped/redirected.
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def add_common(sp):
        sp.add_argument("--dataset-path", required=True)
        sp.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
        sp.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
        sp.add_argument("--modality-config-path", default="examples/b1k/r1pro.py")
        sp.add_argument("--train-script", default="scripts/b1k/train_b1k.py")
        sp.add_argument("--num-gpus", type=int, default=len(gpu_total_mem_mib()))
        sp.add_argument("--max-steps", type=int, default=120)
        sp.add_argument("--measure-from-step", type=int, default=None,
                        help="Steady-state window start (default: max-steps/2, rounded to logging-steps)")
        sp.add_argument("--logging-steps", type=int, default=10,
                        help="Must match the trainer's logging_steps (10 in this repo)")
        sp.add_argument("--run-timeout", type=float, default=2700, help="Seconds before a run is killed")
        sp.add_argument("--train-args", default="",
                        help="Extra args passed through to the train script, e.g. \"--decode-only-used-frames --episode-sampling-rate 0.1\"")
        sp.add_argument("--work-dir", default="bench_runs")
        sp.add_argument("--results-json", default="bench_runs/results.json")
        sp.add_argument("--master-port", type=int, default=29510)
        sp.add_argument("--extra-env", action="append", default=[], metavar="KEY=VAL")
        sp.add_argument("--keep-run-outputs", action="store_true")
        sp.add_argument("--dry-run", action="store_true")

    pr = sub.add_parser("run", help="single measured run")
    add_common(pr)
    pr.add_argument("--label", required=True)
    pr.add_argument("--global-batch-size", type=int, default=2048)
    pr.add_argument("--dataloader-num-workers", type=int, default=8)
    pr.add_argument("--omp-num-threads", type=int, default=4)

    ps = sub.add_parser("sweep", help="coordinate-descent sweep over OMP → workers → batch")
    add_common(ps)
    ps.add_argument("--omp-candidates", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ps.add_argument("--workers-candidates", type=int, nargs="*", default=[2, 4, 8, 16])
    ps.add_argument("--batch-candidates", type=int, nargs="*", default=[512, 1024, 2048, 4096])
    ps.add_argument("--base-omp", type=int, default=4)
    ps.add_argument("--base-workers", type=int, default=8)
    ps.add_argument("--base-batch", type=int, default=2048)
    ps.add_argument("--mem-limit-frac", type=float, default=0.92,
                    help="Batch axis: reject configs whose peak GPU mem exceeds this fraction of total")
    ps.add_argument("--tie-tolerance", type=float, default=0.02,
                    help="Prefer smaller values within this relative throughput of the best")
    ps.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                    help="Unmeasured first run to warm the page cache")

    args = p.parse_args()
    args.train_args_list = shlex.split(args.train_args)
    if args.measure_from_step is None:
        args.measure_from_step = (args.max_steps // 2 // args.logging_steps) * args.logging_steps
    if args.measure_from_step % args.logging_steps != 0 or not (
        0 < args.measure_from_step < args.max_steps
    ):
        p.error("--measure-from-step must be a multiple of --logging-steps in (0, max-steps)")
    if args.max_steps % args.logging_steps != 0:
        p.error("--max-steps must be a multiple of --logging-steps")

    if args.mode == "run":
        do_run(args)
    else:
        do_sweep(args)


if __name__ == "__main__":
    main()
