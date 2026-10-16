"""task2/triton/vector_add.py

Task 2 - 2.1 Triton Vector Add

对比对象：PyTorch 的逐元素加法（同 shape、同 dtype、同计时范围）。
配置：FP32，N ∈ {2^20, 2^24}，BLOCK_SIZE ∈ {128, 256, 512}，正确性阈值 max|err| < 1e-5。

运行：
    python3 triton/vector_add.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from common import assert_cuda, bench_gpu, dump_json, make_vector, report, summarize


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """每个 program 处理 BLOCK_SIZE 个元素：program_id -> offset -> mask 读写。"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def run_one(n: int, block_size: int) -> dict:
    x = make_vector(n, torch.float32)
    y = make_vector(n, torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, block_size),)

    # 首次 JIT 编译单独记录，不放进稳态延迟
    import time

    t0 = time.perf_counter()
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()
    jit_ms = (time.perf_counter() - t0) * 1000.0

    # 正确性
    ref = x + y
    max_err = (out - ref).abs().max().item()

    samples = bench_gpu(lambda: add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size))
    row = summarize(f"triton_add N={n} BLOCK={block_size}", samples)
    row["max_abs_err"] = max_err
    # 有效带宽：读 x(4B) + 读 y(4B) + 写 out(4B)
    row["gbps"] = 12.0 * n / (row["median_ms"] * 1e-3) / 1e9
    row["jit_compile_ms"] = jit_ms
    row["config"] = {"N": n, "BLOCK_SIZE": block_size}
    report(row)

    # 与 PyTorch 对照
    torch_rows = summarize("torch_add", bench_gpu(lambda: torch.add(x, y, out=out)))
    torch_rows["gbps"] = 12.0 * n / (torch_rows["median_ms"] * 1e-3) / 1e9
    report(torch_rows)
    row["torch_median_ms"] = torch_rows["median_ms"]
    return row


def main() -> None:
    assert_cuda()
    rows = []
    for n in (1 << 20, 1 << 24):
        for block_size in (128, 256, 512):
            rows.append(run_one(n, block_size))
    dump_json(rows, "triton_vector_add.json")

    # 回答题面问题所需的最小观察：program 数 = ceil(N / BLOCK_SIZE)
    print("\n[观察] program 数随 BLOCK_SIZE 反比变化：")
    for n in (1 << 20, 1 << 24):
        for block_size in (128, 256, 512):
            print(f"  N={n:<10} BLOCK={block_size:<5} programs={triton.cdiv(n, block_size)}")


if __name__ == "__main__":
    main()
