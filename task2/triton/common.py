"""task2/triton/common.py

统一的 benchmark 口径，保证 Task 1 与 Task 2 的数字可比：

- warmup 10 次，正式测量 50 次；
- 用 CUDA Event 计时（只包住 GPU 工作，不含输入生成、不含首次 JIT 编译）；
- 报告 median 与 P95，并保留逐次原始数据；
- 输入统一 seed = 2026。

依赖：torch、triton。装环境（示例）：
    conda create -n mlsys python=3.11 -y && conda activate mlsys
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install triton numpy
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import torch

SEED = 2026
WARMUP = 10
ITERS = 50

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 测时
# ---------------------------------------------------------------------------
@torch.no_grad()
def bench_gpu(fn, warmup: int = WARMUP, iters: int = ITERS) -> list[float]:
    """返回 iters 次测量的逐次耗时（ms），只统计 fn 内部的 GPU 工作。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def pctl(samples: list[float], p: float) -> float:
    xs = sorted(samples)
    if not xs:
        return 0.0
    idx = p * (len(xs) - 1)
    lo, hi = int(idx // 1), int(-(-idx // 1))
    if lo == hi:
        return xs[lo]
    frac = idx - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def summarize(name: str, samples: list[float]) -> dict:
    med = statistics.median(samples)
    return {
        "name": name,
        "median_ms": med,
        "p95_ms": pctl(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "n": len(samples),
        "samples_ms": samples,
    }


def report(row: dict) -> None:
    """打印一行人类可读结果 + 一行机器可读 JSON（便于收集到 results/）。"""
    print(
        f"{row['name']:<38} median={row['median_ms']:.4f} ms  "
        f"p95={row['p95_ms']:.4f} ms  "
        + (f"TFLOPS={row['tflops']:.2f}  " if "tflops" in row else "")
        + (f"GB/s={row['gbps']:.1f}  " if "gbps" in row else "")
        + (f"err={row['max_abs_err']:.3e}  " if "max_abs_err" in row else "")
        + (f"pct_cublas={row['pct_of_cublas']:.1f}%" if "pct_of_cublas" in row else "")
    )


def dump_json(rows: list[dict], filename: str) -> Path:
    path = RESULTS_DIR / filename
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")
    return path


# ---------------------------------------------------------------------------
# 输入生成（题面约定：seed = 2026）
# ---------------------------------------------------------------------------
def make_vector(n: int, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(SEED)
    return torch.empty((n,), device="cuda", dtype=dtype).uniform_(-1.0, 1.0, generator=g)


def make_matrix(rows: int, cols: int, dtype=torch.float16) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(SEED)
    return torch.empty((rows, cols), device="cuda", dtype=dtype).uniform_(
        -1.0, 1.0, generator=g
    )


def assert_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "本任务需要 NVIDIA GPU + CUDA 版 PyTorch：torch.cuda.is_available() == False。"
            "请在具有 NVIDIA 显卡的机器/环境上运行。"
        )
    print(f"[env] GPU          : {torch.cuda.get_device_name(0)}")
    print(f"[env] torch        : {torch.__version__}")
    print(f"[env] cuda runtime : {torch.version.cuda}")
    try:
        import triton

        print(f"[env] triton       : {triton.__version__}")
    except Exception as exc:  # pragma: no cover
        print(f"[env] triton       : 未安装 ({exc})")
