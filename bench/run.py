#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///
"""Benchmark quintck (per thread count) and quint's TLC backend over the
bench corpus, appending one row per run to bench/results.parquet.

Usage:
    uv run bench/run.py [--reps N] [--threads 1,4,10] [--skip-tlc]
                        [--timeout SECS] [--specs A,B] [--out PATH]
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Spec:
    name: str
    quintck_args: list[str]  # after the fixture path
    exhaustive: bool  # True → TLC runs the same (unbounded) search
    invariants: str  # TLC --invariant argument


# Mirrors quintck-bench.sh (args) and fixtures/regen.sh (invariants).
SPECS = [
    Spec("TeachingConcurrency", ["--max-steps", "6"], False, "correctness"),
    Spec("ClockSync", ["--exhaustive"], True, "skewOK"),
    Spec("TwoPhaseCommit", ["--max-steps", "18"], False, "consistency"),
    Spec("ReadersWriters", ["--exhaustive"], True, "safety"),
    Spec("TwoLayeredCache", ["--exhaustive"], True, "cleanConsistency,dirtyInL1"),
    Spec("DiningPhilosophers", ["--max-steps", "16"], False, "consistent"),
    Spec("ReliableBroadcast", ["--exhaustive"], True, "validity,relayedBeforeDelivered"),
    Spec("LamportMutex", ["--max-steps", "15"], False, "mutex,requestConsistency"),
    Spec("Paxos", ["--max-steps", "11"], False, "agreement,oneValuePerBallot"),
    Spec("Raft", ["--max-steps", "15"], False, "electionSafety,logMatching,voteIntegrity"),
]

TIME_RE = re.compile(
    r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", re.M
)
RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")
QK_STATES_RE = re.compile(r"\[ok\] (\d+) states explored")
TLC_STATES_RE = re.compile(r"(\d+) states generated, (\d+) distinct states")
TLC_TIME_RE = re.compile(r"No violation found \((\d+)ms\)")


def run_timed(cmd: list[str], timeout: float) -> dict | None:
    """Run cmd under /usr/bin/time -l; None on timeout."""
    full = ["/usr/bin/time", "-l", *cmd]
    try:
        proc = subprocess.run(
            full, capture_output=True, text=True, timeout=timeout, cwd=ROOT
        )
    except subprocess.TimeoutExpired:
        return None
    out = proc.stdout + proc.stderr
    m = TIME_RE.search(out)
    rss = RSS_RE.search(out)
    return {
        "returncode": proc.returncode,
        "wall_s": float(m.group(1)) if m else None,
        "user_s": float(m.group(2)) if m else None,
        "sys_s": float(m.group(3)) if m else None,
        "max_rss_mb": int(rss.group(1)) / 1e6 if rss else None,
        "output": out,
    }


def base_row(spec: Spec, backend: str, threads: int, rep: int, commit: str) -> dict:
    return {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "commit": commit,
        "spec": spec.name,
        "backend": backend,
        "threads": threads,
        "rep": rep,
        "wall_s": None,
        "user_s": None,
        "sys_s": None,
        "max_rss_mb": None,
        "states": None,
        "states_distinct": None,
        "tlc_reported_s": None,
        "verdict": "error",
        "exhaustive": spec.exhaustive,
    }


def bench_quintck(spec: Spec, threads: int, rep: int, commit: str, timeout: float) -> dict:
    row = base_row(spec, "quintck", threads, rep, commit)
    cmd = [
        str(ROOT / "target/release/quintck"),
        str(ROOT / f"fixtures/bench_{spec.name}.json"),
        *spec.quintck_args,
        "--threads",
        str(threads),
    ]
    r = run_timed(cmd, timeout)
    if r is None:
        row["verdict"] = "timeout"
        return row
    row.update({k: r[k] for k in ("wall_s", "user_s", "sys_s", "max_rss_mb")})
    if m := QK_STATES_RE.search(r["output"]):
        row["states"] = int(m.group(1))
        row["states_distinct"] = int(m.group(1))
        row["verdict"] = "ok"
    return row


def bench_tlc(spec: Spec, rep: int, commit: str, timeout: float, workers: int) -> dict:
    # quint's TLC backend defaults to one worker per core.
    row = base_row(spec, "tlc", workers, rep, commit)
    cmd = [
        "quint",
        "verify",
        str(ROOT / f"specs/{spec.name}.qnt"),
        "--main=bench",
        f"--invariant={spec.invariants}",
        "--backend=tlc",
    ]
    r = run_timed(cmd, timeout)
    if r is None:
        row["verdict"] = "timeout"
        return row
    row.update({k: r[k] for k in ("wall_s", "user_s", "sys_s", "max_rss_mb")})
    if m := TLC_STATES_RE.search(r["output"]):
        row["states"] = int(m.group(1))
        row["states_distinct"] = int(m.group(2))
    if m := TLC_TIME_RE.search(r["output"]):
        row["tlc_reported_s"] = int(m.group(1)) / 1000
        row["verdict"] = "ok"
    return row


def main() -> int:
    ncpu = os.cpu_count() or 1
    default_threads = sorted({1, 2, 4, 8, ncpu})
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument(
        "--threads",
        default=",".join(map(str, default_threads)),
        help="comma-separated quintck thread counts",
    )
    ap.add_argument("--skip-tlc", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--specs", default=None, help="comma-separated subset")
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results.parquet")
    args = ap.parse_args()

    thread_counts = [int(t) for t in args.threads.split(",")]
    specs = SPECS
    if args.specs:
        wanted = set(args.specs.split(","))
        specs = [s for s in SPECS if s.name in wanted]
        if unknown := wanted - {s.name for s in specs}:
            sys.exit(f"unknown specs: {sorted(unknown)}")

    subprocess.run(
        ["cargo", "build", "--release"], cwd=ROOT, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()

    rows: list[dict] = []

    def record(row: dict) -> None:
        rows.append(row)
        wall = f"{row['wall_s']:8.2f}s" if row["wall_s"] is not None else "     n/a"
        states = f"{row['states']:>10}" if row["states"] else "         -"
        print(
            f"[{row['verdict']:>7}] {row['spec']:<20} {row['backend']:<7}"
            f" t={row['threads']:<3}{wall} {states} states",
            flush=True,
        )

    for rep in range(args.reps):
        for spec in specs:
            for t in thread_counts:
                record(bench_quintck(spec, t, rep, commit, args.timeout))
        if not args.skip_tlc:
            for spec in specs:
                if spec.exhaustive:
                    record(bench_tlc(spec, rep, commit, args.timeout, ncpu))

    df = pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Utf8,
            "commit": pl.Utf8,
            "spec": pl.Utf8,
            "backend": pl.Utf8,
            "threads": pl.Int32,
            "rep": pl.Int32,
            "wall_s": pl.Float64,
            "user_s": pl.Float64,
            "sys_s": pl.Float64,
            "max_rss_mb": pl.Float64,
            "states": pl.Int64,
            "states_distinct": pl.Int64,
            "tlc_reported_s": pl.Float64,
            "verdict": pl.Utf8,
            "exhaustive": pl.Boolean,
        },
    )
    if args.out.exists():
        df = pl.concat([pl.read_parquet(args.out), df], how="vertical")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"\n{len(rows)} runs appended -> {args.out} ({len(df)} rows total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
