# task2/results —— 原始数据（待 NVIDIA GPU 机器执行后填入）

本目录**没有**实测数据。原因同 Task 1：完成本仓库的机器是 Intel Arc Pro 140T，
没有 NVIDIA GPU，而 Triton 需要 NVIDIA 后端；题目要求「不得用预期结果冒充实测结果」。

## 生成方式

```bash
cd task2
python3 triton/bench_all.py        # 或分别跑 vector_add.py / gemm.py / rmsnorm.py / autotune_search.py
```

## 文件与字段

| 文件 | 内容 |
|---|---|
| `triton_vector_add.json` | 每个 (N, BLOCK_SIZE) 的逐次样本 `samples_ms`、`median_ms`、`p95_ms`、`max_abs_err`、`gbps`、`jit_compile_ms` |
| `triton_gemm.json` | 每个 Shape 的 median/P95、TFLOPS、`max_abs_err`、`jit_compile_ms`、`torch_*` 对照、`pct_of_cublas` |
| `triton_rmsnorm.json` | 每个 (tokens, hidden) 的 median/P95、`gbps`、`max_abs_err`、torch 对照 |
| `triton_autotune.json` | 72 个配置的编译/正确性结果、`search_total_ms`、`best`、`all_ok`、`failed`、`held_out` |

所有 JSON 都保留 `samples_ms`（逐次原始测量），保证结论可复查——题目明确要求
"保存逐次测量的原始数据，截图不能代替原始结果"。

## 报告里需要补的地方

`../report.md` 中标 **[待填]** 的表格，以及以下基于实测的结论：

- Vector Add：BLOCK_SIZE 与带宽的关系，是否接近带宽上限；
- GEMM：三个 Shape 上 TFLOPS 与 `% of cuBLAS`，以及首次 JIT 编译 vs 稳态延迟的对比；
- RMSNorm：有效带宽占峰值带宽的比例，验证 "AI ≈ 1 → Memory Bound" 的判断；
- Autotuning：最佳配置在 held-out Shape 上的迁移性。
