# task2/results —— 原始数据（RTX 5070 实测）

本目录的 JSON 均在朋友的 **NVIDIA RTX 5070（WSL2，sm_120）** 上跑出，由 `bench_all.py` 生成；
题目要求「不得用预期结果冒充实测结果」，以下全部为真实测量。

## 生成方式

```bash
cd task2
python3 triton/bench_all.py        # 或分别跑 vector_add.py / gemm.py / autotune_search.py
```

## 文件与字段

| 文件 | 内容 |
|---|---|
| `triton_vector_add.json` | 每个 (N, BLOCK_SIZE) 的逐次样本 `samples_ms`、`median_ms`、`p95_ms`、`max_abs_err`、`gbps`、`jit_compile_ms` |
| `triton_gemm.json` | 每个 Shape 的 median/P95、TFLOPS、`max_abs_err`、`jit_compile_ms`、`torch_*` 对照、`pct_of_cublas` |
| `triton_autotune.json` | 72 个配置的编译/正确性结果、`search_total_ms`、`best`、`all_ok`、`failed`、`held_out` |

所有 JSON 都保留 `samples_ms`（逐次原始测量），保证结论可复查——题目明确要求
"保存逐次测量的原始数据，截图不能代替原始结果"。

## 报告对应结论

`../report.md` 已填入以上实测数据：

- Vector Add：BLOCK_SIZE 与带宽的关系（2^20 时 BLOCK=512 带宽 421 GB/s，约为 BLOCK=128 的 3 倍）；
- GEMM：三个 Shape 的 TFLOPS 与 `% of cuBLAS`（85.9% / 49.6% / 136.1%），以及首次 JIT vs 稳态延迟的对比；
- Autotuning：最佳配置在 held-out Shape 上的迁移性（正确性达标、性能不保证最优）。
