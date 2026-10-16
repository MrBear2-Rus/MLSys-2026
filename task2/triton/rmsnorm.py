"""task2/triton/rmsnorm.py

Task 2 - 2.2 B：RMSNorm (Triton)

- FP16 输入输出，但**计算均值与归一化时用 FP32**；
- 与等价的 PyTorch 实现比较，正确性 max|err| < 1e-2；
- Shape：(tokens, hidden) ∈ {(1024, 4096), (8192, 4096)}；
- 报告有效带宽、median / P95 延迟。

运行：
    python3 triton/rmsnorm.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from common import assert_cuda, bench_gpu, dump_json, make_matrix, report, summarize

SHAPES = [(1024, 4096), (8192, 4096)]
EPS = 1e-5


@triton.jit
def rms_norm_kernel(x_ptr, w_ptr, y_ptr, stride, n_cols,
                    eps, BLOCK: tl.constexpr):
    """一行一个 program：整行的 x 一次性载入（hidden <= BLOCK），在寄存器里算 RMS。"""
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 均值与归一化都在 FP32 下完成
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    y = x * rstd * w

    tl.store(y_ptr + row * stride + cols, y.to(tl.float16), mask=mask)


def triton_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    tokens, hidden = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(hidden)
    rms_norm_kernel[(tokens,)](x, w, y, x.stride(0), hidden, eps, BLOCK=block)
    return y


def torch_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """等价参考实现：FP32 计算后转回 FP16。"""
    xf = x.float()
    mean_sq = xf.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    return (xf * rstd * w.float()).to(torch.float16)


def run_one(tokens: int, hidden: int) -> dict:
    x = make_matrix(tokens, hidden, torch.float16)
    w = make_matrix(1, hidden, torch.float16)[0]  # 权重按 hidden 维共享

    out = triton_rmsnorm(x, w)
    ref = torch_rmsnorm(x, w)
    max_err = (out.float() - ref.float()).abs().max().item()

    samples = bench_gpu(lambda: triton_rmsnorm(x, w))
    row = summarize(f"triton_rmsnorm {tokens}x{hidden}", samples)
    row["max_abs_err"] = max_err
    # 有效带宽口径：读 x(2B) + 写 y(2B) = 4 Bytes/element；权重每行读一次且可被 L2 缓存
    row["gbps"] = 4.0 * tokens * hidden / (row["median_ms"] * 1e-3) / 1e9
    row["config"] = {"tokens": tokens, "hidden": hidden}
    report(row)

    torch_rows = summarize("torch_rmsnorm", bench_gpu(lambda: torch_rmsnorm(x, w)))
    torch_rows["gbps"] = 4.0 * tokens * hidden / (torch_rows["median_ms"] * 1e-3) / 1e9
    report(torch_rows)
    row["torch_median_ms"] = torch_rows["median_ms"]
    return row


def main() -> None:
    assert_cuda()
    rows = [run_one(t, h) for (t, h) in SHAPES]
    dump_json(rows, "triton_rmsnorm.json")

    print("\n[提示] 报告里需要解释：RMSNorm 的计算量很小（每元素约 2 FLOP 量级），")
    print("       时间几乎全部花在把 x 读进来、把 y 写回去，属于典型 Memory Bound。")


if __name__ == "__main__":
    main()
