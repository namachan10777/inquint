#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "matplotlib"]
# ///
"""Render charts from bench/results.parquet into bench/charts/.

Usage:
    uv run bench/plot.py [--commit ABC123 | --all] [--in PATH] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parent.parent

QK_1T = "#93b7e4"
QK_NT = "#1f5fbf"
TLC_C = "#c96a3f"


def load(path: Path, commit: str | None, use_all: bool) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if use_all:
        return df
    if commit is None:
        commit = df.sort("timestamp")["commit"].last()
    df = df.filter(pl.col("commit") == commit)
    if df.is_empty():
        sys.exit(f"no rows for commit {commit}")
    return df


def median_runs(df: pl.DataFrame) -> pl.DataFrame:
    """Median over reps per (spec, backend, threads); keeps counts/verdict."""
    return df.group_by(["spec", "backend", "threads"]).agg(
        pl.col("wall_s").median(),
        pl.col("user_s").median(),
        pl.col("max_rss_mb").median(),
        pl.col("states_distinct").max(),
        pl.col("exhaustive").first(),
        # a timeout in any rep marks the cell
        (pl.col("verdict") == "timeout").any().alias("timed_out"),
    )


def spec_order(df: pl.DataFrame) -> list[str]:
    """Specs sorted by parallel quintck wall time (fast → slow)."""
    max_t = df.filter(pl.col("backend") == "quintck")["threads"].max()
    base = (
        df.filter((pl.col("backend") == "quintck") & (pl.col("threads") == max_t))
        .sort("wall_s")
    )
    return base["spec"].to_list()


def get(df: pl.DataFrame, spec: str, backend: str, threads: int | None = None):
    q = (pl.col("spec") == spec) & (pl.col("backend") == backend)
    if threads is not None:
        q &= pl.col("threads") == threads
    rows = df.filter(q)
    return rows.row(0, named=True) if not rows.is_empty() else None


def bar_label(ax, y, x, text):
    ax.annotate(
        text, (x, y), xytext=(4, 0), textcoords="offset points",
        va="center", fontsize=8,
    )


def plot_wall(df: pl.DataFrame, out: Path):
    specs = spec_order(df)
    max_t = int(df.filter(pl.col("backend") == "quintck")["threads"].max())
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(specs) + 1.5))
    ys = range(len(specs))
    h = 0.27
    seen_labels: set[str] = set()
    for i, spec in enumerate(specs):
        for dy, backend, threads, color, label in [
            (h, "quintck", 1, QK_1T, "quintck (1 thread)"),
            (0, "quintck", max_t, QK_NT, f"quintck ({max_t} threads)"),
            (-h, "tlc", None, TLC_C, "TLC (all cores)"),
        ]:
            r = get(df, spec, backend, threads)
            if r is None:
                continue
            label = None if label in seen_labels else (seen_labels.add(label) or label)
            if r["timed_out"]:
                ax.barh(i + dy, 0.01, height=h, color=color, label=label)
                bar_label(ax, i + dy, 0.01, "T/O")
                continue
            ax.barh(i + dy, r["wall_s"], height=h, color=color, label=label)
            bar_label(ax, i + dy, r["wall_s"], f"{r['wall_s']:.1f}s")
    ax.set_yticks(list(ys), specs)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("wall time (s, log scale)")
    ax.set_title("Model checking wall time — quintck vs TLC (bench corpus)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "wall_time.png", dpi=150)
    plt.close(fig)


def plot_scaling(df: pl.DataFrame, out: Path):
    qk = df.filter((pl.col("backend") == "quintck") & ~pl.col("timed_out"))
    threads = sorted(qk["threads"].unique().to_list())
    fig, ax = plt.subplots(figsize=(7, 5))
    for spec in spec_order(df):
        rows = qk.filter(pl.col("spec") == spec).sort("threads")
        base = rows.filter(pl.col("threads") == 1)
        if base.is_empty():
            continue
        t1 = base["wall_s"][0]
        ax.plot(
            rows["threads"], [t1 / w for w in rows["wall_s"]],
            marker="o", markersize=3.5, linewidth=1.2, label=spec,
        )
    ax.plot(threads, threads, "--", color="gray", linewidth=1, label="ideal")
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup vs 1 thread")
    ax.set_title("quintck parallel scaling")
    ax.set_xticks(threads)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "scaling.png", dpi=150)
    plt.close(fig)


def plot_memory(df: pl.DataFrame, out: Path):
    specs = spec_order(df)
    max_t = int(df.filter(pl.col("backend") == "quintck")["threads"].max())
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(specs) + 1.5))
    h = 0.38
    seen_labels: set[str] = set()
    for i, spec in enumerate(specs):
        for dy, backend, threads, color, label in [
            (h / 2, "quintck", max_t, QK_NT, f"quintck ({max_t} threads)"),
            (-h / 2, "tlc", None, TLC_C, "TLC"),
        ]:
            r = get(df, spec, backend, threads)
            if r is None or r["max_rss_mb"] is None or r["timed_out"]:
                continue
            label = None if label in seen_labels else (seen_labels.add(label) or label)
            ax.barh(i + dy, r["max_rss_mb"], height=h, color=color, label=label)
            bar_label(ax, i + dy, r["max_rss_mb"], f"{r['max_rss_mb']:.0f} MB")
    ax.set_yticks(range(len(specs)), specs)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("max RSS (MB, log scale)")
    ax.set_title("Peak memory")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "memory.png", dpi=150)
    plt.close(fig)


def plot_throughput(df: pl.DataFrame, out: Path):
    specs = spec_order(df)
    max_t = int(df.filter(pl.col("backend") == "quintck")["threads"].max())
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(specs) + 1.5))
    h = 0.38
    seen_labels: set[str] = set()
    for i, spec in enumerate(specs):
        for dy, backend, threads, color, label in [
            (h / 2, "quintck", max_t, QK_NT, f"quintck ({max_t} threads)"),
            (-h / 2, "tlc", None, TLC_C, "TLC"),
        ]:
            r = get(df, spec, backend, threads)
            if (
                r is None
                or r["timed_out"]
                or not r["states_distinct"]
                or not r["wall_s"]
            ):
                continue
            rate = r["states_distinct"] / r["wall_s"] / 1e6
            label = None if label in seen_labels else (seen_labels.add(label) or label)
            ax.barh(i + dy, rate, height=h, color=color, label=label)
            bar_label(ax, i + dy, rate, f"{rate:.2f}M/s")
    ax.set_yticks(range(len(specs)), specs)
    ax.invert_yaxis()
    ax.set_xlabel("distinct states / second (millions)")
    ax.set_title("Exploration throughput")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=None, help="filter to this commit (default: latest)")
    ap.add_argument("--all", action="store_true", help="use every commit's rows")
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "bench/results.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "bench/charts")
    args = ap.parse_args()

    df = median_runs(load(args.inp, args.commit, args.all))
    args.out.mkdir(parents=True, exist_ok=True)
    plot_wall(df, args.out)
    plot_scaling(df, args.out)
    plot_memory(df, args.out)
    plot_throughput(df, args.out)
    for f in sorted(args.out.glob("*.png")):
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
