# task1/results —— 原始数据（RTX 5070 实测，2026-10-16）

本目录已填入在朋友的 **NVIDIA RTX 5070（sm_120）** 上跑出的真实 CSV（Vector Add / Naive / Shared / Register），
以及 `environment.txt`。本仓库开发机为 Intel Arc（无 N 卡），故源码与脚本在此开发、实测在他机完成。
这里提供：

1. 数据文件格式说明（下面两张表）；
2. 空模板 `*.template.csv`；
3. 一键执行脚本 `../benchmark/build_and_run.sh`（如需图表可另跑 `../benchmark/plot_results.py`，作答默认纯文字）。

## 执行方式

```bash
cd task1
bash benchmark/build_and_run.sh        # 编译 + 跑满全部配置，写入本目录
# python3 benchmark/plot_results.py     # （可选）生成 results/figures/*.png，作答默认不需要
```

`build_and_run.sh` 会自动把 `nvidia-smi` / `nvcc --version` / `uname` / 编译参数写进
`environment.txt`。

## 数据文件与字段

### `vector_add.csv`

| 列 | 含义 |
|---|---|
| `N` | 向量长度 |
| `block_size` | 该次实验的 Block Size |
| `max_abs_err` | 与 CPU 参考实现的最大绝对误差（判据 < 1e-5） |
| `median_ms` | 50 次测量的中位数（CUDA Event，仅 Kernel） |
| `p95_ms` | 50 次测量的 P95 |
| `effective_GBps` | 有效带宽 `12N / t` |
| `peak_bw_util_pct` | 有效带宽 / 理论峰值带宽 |

### `gemm_naive.csv` / `gemm_shared.csv` / `gemm_register.csv`

| 列 | 含义 |
|---|---|
| `variant` | `naive` / `shared-T8` / `shared-T16` / `shared-T32` / `register-4x4` |
| `shape` | `square` / `tall-skinny` / `small-M-1` / `small-M-32` / `small-M-256` / `irregular` |
| `M,N,K` | 该次的矩阵形状 |
| `max_abs_err` | 与严格 FP32 cuBLAS 的最大绝对误差（判据 < 1e-2） |
| `median_ms` / `p95_ms` | 自写 Kernel 的延迟统计 |
| `TFLOPS` | `2MNK / t / 1e12` |
| `cublas_median_ms` / `cublas_TFLOPS` | 同一 Shape、同一计时口径下的 cuBLAS 基线 |
| `pct_of_cublas` | `TFLOPS / cuBLAS_TFLOPS * 100%` |

## 报告里必须补的项

跑完之后，把 `report.md` 中标注了 **[待填]** 的表格补充完整，并针对以下问题给出**基于实测**的解释：

- Block Size 增大到某个值后吞吐为何不再上升；1024 是否反而更慢；
- Tile 大小（8/16/32）与 Shape（方阵 vs 小 M）对性能的影响；
- Register Tiling 的收益与 Register Pressure 的代价。
