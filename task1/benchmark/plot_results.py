#!/usr/bin/env python3
"""task1/benchmark/plot_results.py

把 results/ 下的 CSV 画成报告需要的图（题目要求「至少用一张图解释 Shape 或 Tile Size
对性能的影响」）。

用法：
    cd task1 && python3 benchmark/plot_results.py

依赖：pandas、matplotlib（pip install pandas matplotlib）
数据还没跑出来时不会报错，只会提示缺少哪个文件。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_DIR = Path(__file__).resolve().parent.parent
RESULTS = TASK_DIR / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.is_file():
        print(f"[skip] 缺少 {path}（先在 NVIDIA 机器上跑 build_and_run.sh）")
        return None
    df = pd.read_csv(path, comment="#")
    if df.empty:
        print(f"[skip] {path} 还没有数据行")
        return None
    return df


def plot_vector_add() -> None:
    df = load("vector_add.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n, sub in df.groupby("N"):
        sub = sub.sort_values("block_size")
        ax.plot(sub["block_size"], sub["effective_GBps"], marker="o", label=f"N = 2^{int(n).bit_length()-1}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["block_size"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.set_title("Vector Add: Block Size vs Effective Bandwidth")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIGURES / "vector_add_block_size.png"
    fig.savefig(out, dpi=150)
    print(f"[ok] {out}")


def plot_gemm_shape() -> None:
    frames = []
    for name in ("gemm_naive.csv", "gemm_shared.csv", "gemm_register.csv"):
        df = load(name)
        if df is not None:
            frames.append(df)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    order = ["square", "tall-skinny", "small-M-1", "small-M-32", "small-M-256", "irregular"]
    present = [s for s in order if s in set(df["shape"])]
    pivot = df.pivot_table(index="shape", columns="variant", values="TFLOPS", aggfunc="max")
    pivot = pivot.reindex(present)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Matrix Shape")
    ax.set_ylabel("TFLOPS")
    ax.set_title("GEMM: Shape vs TFLOPS (custom kernels)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="variant", fontsize=8)
    fig.tight_layout()
    out = FIGURES / "gemm_shape_vs_tflops.png"
    fig.savefig(out, dpi=150)
    print(f"[ok] {out}")

    # 与 cuBLAS 的比例
    pct = df.pivot_table(index="shape", columns="variant", values="pct_of_cublas", aggfunc="max")
    pct = pct.reindex(present)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pct.plot(kind="bar", ax=ax)
    ax.set_xlabel("Matrix Shape")
    ax.set_ylabel("% of cuBLAS")
    ax.set_title("GEMM: Shape vs % of cuBLAS")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="variant", fontsize=8)
    fig.tight_layout()
    out = FIGURES / "gemm_shape_pct_cublas.png"
    fig.savefig(out, dpi=150)
    print(f"[ok] {out}")


def plot_tile_size() -> None:
    df = load("gemm_shared.csv")
    if df is None:
        return
    df = df[df["shape"] == "square"].copy()
    if df.empty:
        print("[skip] gemm_shared.csv 里没有 square shape")
        return
    df["tile"] = df["variant"].str.replace("shared-T", "", regex=False).astype(int)
    df = df.sort_values("tile")

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.plot(df["tile"], df["TFLOPS"], marker="o", color="#c0392b",
             label="TFLOPS (left)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(df["tile"])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("TILE size")
    ax1.set_ylabel("TFLOPS", color="#c0392b")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["tile"], df["pct_of_cublas"], marker="s", color="#2471a3",
             label="% of cuBLAS (right)")
    ax2.set_ylabel("% of cuBLAS", color="#2471a3")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
    ax1.set_title("Shared-Memory GEMM: TILE size vs Performance (square)")
    fig.tight_layout()
    out = FIGURES / "gemm_tile_size.png"
    fig.savefig(out, dpi=150)
    print(f"[ok] {out}")


def main() -> int:
    plot_vector_add()
    plot_gemm_shape()
    plot_tile_size()
    return 0


if __name__ == "__main__":
    sys.exit(main())
