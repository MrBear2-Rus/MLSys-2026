"""task2/triton/gemm.py

Task 2 - 2.1 Triton GEMM

精度约定：**FP16 输入、FP32 accumulation、FP16 输出**
（输入输出省显存、走更快的计算路径，中间求和用 FP32 保住精度）。

Shape：(2048,2048,2048) / (32,4096,4096) / (1000,1536,769)
正确性：max|err| < 1e-1（与同精度、同计时范围的 torch.matmul 比较）。

注意：Task 1 测的是 FP32，这里是 FP16，两者可以讨论现象，但**不要直接写成"快了多少倍"**。

运行：
    python3 triton/gemm.py
"""
from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from common import assert_cuda, bench_gpu, dump_json, make_matrix, report, summarize

SHAPES = [(2048, 2048, 2048), (32, 4096, 4096), (1000, 1536, 769)]


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """经典 Blocked GEMM：每个 program 算一个 BLOCK_M x BLOCK_N 的 C 分块，
    K 方向按 BLOCK_K 步进，acc 用 fp32。GROUP_M 做 L2-aware 的 program 重排。"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)          # fp16 输入，fp32 累加
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b, block_m=128, block_n=128, block_k=64,
                  num_warps=8, num_stages=3, group_m=8):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_M=group_m, num_warps=num_warps, num_stages=num_stages,
    )
    return c


def run_one(m, n, k) -> dict:
    a = make_matrix(m, k, torch.float16)
    b = make_matrix(k, n, torch.float16)

    # 首次 JIT 编译时间单独记录
    t0 = time.perf_counter()
    out = triton_matmul(a, b)
    torch.cuda.synchronize()
    jit_ms = (time.perf_counter() - t0) * 1000.0

    # 严格 FP32 的参考（用于判断误差量级，不参与性能比较）
    ref = torch.matmul(a, b)
    max_err = (out.float() - ref.float()).abs().max().item()

    flops = 2.0 * m * n * k
    samples = bench_gpu(lambda: triton_matmul(a, b))
    row = summarize(f"triton_gemm {m}x{n}x{k}", samples)
    row["max_abs_err"] = max_err
    row["tflops"] = flops / (row["median_ms"] * 1e-3) / 1e12
    row["jit_compile_ms"] = jit_ms
    row["config"] = {"M": m, "N": n, "K": k}

    # 与同精度 torch.matmul 比较（同样 fp16 入 fp16 出）
    torch_samples = bench_gpu(lambda: torch.matmul(a, b))
    torch_rows = summarize("torch.matmul(fp16)", torch_samples)
    torch_rows["tflops"] = flops / (torch_rows["median_ms"] * 1e-3) / 1e12
    row["torch_median_ms"] = torch_rows["median_ms"]
    row["torch_tflops"] = torch_rows["tflops"]
    row["pct_of_cublas"] = 100.0 * row["tflops"] / torch_rows["tflops"]

    report(row)
    report(torch_rows)
    return row


def main() -> None:
    assert_cuda()
    rows = [run_one(m, n, k) for (m, n, k) in SHAPES]
    dump_json(rows, "triton_gemm.json")


if __name__ == "__main__":
    main()
