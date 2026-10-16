"""task2/triton/autotune_search.py

Task 2 - 2.2 C：Autotuning

搜索空间（共 72 个配置）：
    BLOCK_M, BLOCK_N ∈ {32, 64, 128}     -> 9 种组合
    BLOCK_K          ∈ {32, 64}          -> x2
    num_warps        ∈ {4, 8}            -> x2
    num_stages       ∈ {2, 3}            -> x2
                                       = 72

流程：
    1. 只在 (2048, 2048, 2048) 上选出最佳配置；
    2. 把这个配置原样拿到 (32, 4096, 4096) 与 (1000, 1536, 769) 上测试（held-out）；
    3. 算错或编译失败的配置记录下来并过滤；
    4. 搜索总耗时与编译后的稳态 Kernel 延迟**分开报告**。

运行：
    python3 triton/autotune_search.py
"""
from __future__ import annotations

import time
from itertools import product

import torch

from common import assert_cuda, bench_gpu, dump_json, make_matrix, pctl
from gemm import triton_matmul

SEARCH_SHAPE = (2048, 2048, 2048)
HELD_OUT_SHAPES = [(32, 4096, 4096), (1000, 1536, 769)]

BLOCK_MS = [32, 64, 128]
BLOCK_NS = [32, 64, 128]
BLOCK_KS = [32, 64]
NUM_WARPS = [4, 8]
NUM_STAGES = [2, 3]


def max_err_of(a, b, out, ref) -> float:
    return (out.float() - ref.float()).abs().max().item()


def measure(shape, cfg) -> dict:
    m, n, k = shape
    a = make_matrix(m, k, torch.float16)
    b = make_matrix(k, n, torch.float16)
    ref = torch.matmul(a, b)

    # ---- 编译期（含首次 JIT）+ 正确性 ----
    t0 = time.perf_counter()
    try:
        out = triton_matmul(a, b, **cfg)
        torch.cuda.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # 编译失败 / 资源超限
        return {"ok": False, "reason": f"compile/runtime error: {type(exc).__name__}: {exc}"}

    err = max_err_of(a, b, out, ref)
    if not torch.isfinite(out).all() or err >= 1e-1:
        return {"ok": False, "reason": f"correctness failed (max_abs_err={err:.3e})"}

    # ---- 稳态延迟（编译已完成）----
    samples = bench_gpu(lambda: triton_matmul(a, b, **cfg))
    med = sorted(samples)[len(samples) // 2]
    return {
        "ok": True,
        "cfg": cfg,
        "compile_ms": compile_ms,
        "median_ms": med,
        "p95_ms": pctl(samples, 0.95),
        "tflops": 2.0 * m * n * k / (med * 1e-3) / 1e12,
        "max_abs_err": err,
        "samples_ms": samples,
    }


def main() -> None:
    assert_cuda()

    configs = [
        dict(block_m=bm, block_n=bn, block_k=bk, num_warps=w, num_stages=s)
        for bm, bn, bk, w, s in product(BLOCK_MS, BLOCK_NS, BLOCK_KS, NUM_WARPS, NUM_STAGES)
    ]
    print(f"[search] {len(configs)} 个配置，搜索 Shape = {SEARCH_SHAPE}")

    search_t0 = time.perf_counter()
    results, failed = [], []
    for i, cfg in enumerate(configs, 1):
        r = measure(SEARCH_SHAPE, cfg)
        if r["ok"]:
            results.append(r)
        else:
            failed.append({"cfg": cfg, **r})
        print(f"  [{i:>2}/{len(configs)}] {cfg} -> "
              f"{'OK ' + format(r['tflops'], '.2f') + ' TFLOPS' if r['ok'] else 'FAIL: ' + r['reason'][:70]}")

    search_ms = (time.perf_counter() - search_t0) * 1000.0

    if not results:
        raise SystemExit("所有配置都失败，请检查环境。")

    best = max(results, key=lambda r: r["tflops"])
    print("\n[search] 搜索总耗时 = {:.1f} s（含 72 次编译与测量）".format(search_ms / 1000))
    print("[search] 最佳配置 = {}".format(best["cfg"]))
    print("[search] 该 Shape 最佳稳态延迟 = {:.4f} ms  ({:.2f} TFLOPS)".format(
        best["median_ms"], best["tflops"]))

    # ---- held-out shape：用"在方阵上选出"的配置原样测试 ----
    held_out = []
    print("\n[held-out] 用同一个配置测试未参与调参的 Shape：")
    for shape in HELD_OUT_SHAPES:
        r = measure(shape, best["cfg"])
        if r["ok"]:
            print(f"  {shape}: median={r['median_ms']:.4f} ms  {r['tflops']:.2f} TFLOPS")
        else:
            print(f"  {shape}: FAIL {r['reason'][:80]}")
        held_out.append({"shape": list(shape), **r})

    dump_json(
        {
            "search_shape": SEARCH_SHAPE,
            "search_total_ms": search_ms,
            "n_configs": len(configs),
            "best": best,
            "all_ok": results,
            "failed": failed,
            "held_out": held_out,
        },
        "triton_autotune.json",
    )


if __name__ == "__main__":
    main()
