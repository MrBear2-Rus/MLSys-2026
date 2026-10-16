# MLSys-2026

焦糖工作室 2026 招新 —— **机器学习系统（MLSys）** 方向作答。

> 题目来源：`http://maomio.top/tasks.html?direction_id=5`，
> 其中 **6–10 题**为机器学习系统部分（1–5 题为编译器部分，二者择一）。
> 本目录只覆盖 MLSys 部分。

## 目录结构

```text
MLSys-2026/
│
├── task0/                 前置知识：环境、Benchmark 基础、Roofline
├── task1/                 CUDA：Vector Add / Naive GEMM / Shared Tiling / Register Tiling
├── task2/                 Kernel DSL：Triton / TileLang / RMSNorm / Autotuning
├── task3/                 一个 Token 的一生：Prefill·Decode、KV Cache、MLA、FlashAttention
└── task4/                 量化与分布式推理：PTQ、手算、DP / TP / PP
```

## ⚠️ 复现前必读：本机硬件限制

这套题里 **Task 1 / Task 2 必须在 NVIDIA GPU 上运行**，而完成本目录的机器配置为：

| 项目 | 本机实测 |
|---|---|
| GPU | Intel(R) Arc(TM) Pro 140T GPU (16GB) |
| `nvidia-smi` | 不存在（无 NVIDIA 显卡 / 驱动） |
| `nvcc` | 不存在（无 CUDA Toolkit） |

因此本仓库的处理方式是：

- **Task 3 / Task 4 / Task 0 的文字与手算部分**：已完成，结论可以直接看 `report.md`；
- **Task 1 / Task 2 的源码**：已完整实现并给出可复现的编译、运行与 benchmark 脚本；
- **Task 1 / Task 2 的 `results/`**：**留空并附上数据模板** —— 题目明确要求「不得用预期结果冒充实测结果」，所以这里不填任何编造的延迟 / TFLOPS 数字；
- 在任意一台有 NVIDIA GPU 的机器（原生 Ubuntu 或 WSL2）上执行：

```bash
# Task 1
cd task1 && bash benchmark/build_and_run.sh

# Task 2
cd task2 && python3 triton/bench_all.py
```

即可自动跑满全部配置并把原始逐次测量数据写入 `results/`，再对照 `report.md` 里已写好的「结果分析框架」补齐结论即可。

**没有 NVIDIA 机器？** 两条路都能拿到实测数据，任选其一。

### 路线 ①：租一台 GPU（推荐，最可控）

租好机器（AutoDL 等，RTX 4090 即可）后把本目录放上去，**一条命令跑完**：

```bash
cd MLSys-2026
bash run_all.sh
```

`run_all.sh` = 环境确认 → 按需补依赖 → Task 1 编译+跑分+画图 → Task 2 全部实验 → 打包 `mlsys_results.zip`。
选镜像、省钱、踩坑清单见 [`../租卡指南.md`](../租卡指南.md)。

也可以只跑其中一部分：

```bash
cd task1 && bash benchmark/build_and_run.sh && python3 benchmark/plot_results.py
cd task2 && python3 triton/bench_all.py
```

### 路线 ②：Google Colab 免费 T4

跑 [`colab_run_all.ipynb`](colab_run_all.ipynb)：上传到 [Colab](https://colab.research.google.com) →
**代码执行程序 → 更改运行时类型 → 硬件加速器选 T4 GPU** →
获取代码二选一（**A** 填 `REPO_URL` 后 `git clone`；**B** 直接上传 `../MLSys-2026-colab.zip`）→ 一路运行到底。

> 无论哪条路线，跑完把 `environment.txt` + CSV 发我，就能把下面各报告里的 **[待填]** 一次性补成带结论的完整版。
>
> 本目录已是一个初始化好的 git 仓库（含 `.gitignore`、`.gitattributes`）。
> `.gitattributes` 强制脚本类文件为 LF，避免 Windows 提交后在 Linux 上出现 `$'\r': command not found`；
> 推送命令见 [`../推送指南.md`](../推送指南.md)。

## 各 Task 完成度一览

| Task | 内容 | 状态 |
|---|---|---|
| 0 | 环境记录、Benchmark 基础、Roofline | ✅ 问答完成；`nvidia-smi` / `nvcc` 因无 NVIDIA 硬件记为 `null` 并说明原因 |
| 1 | CUDA Vector Add / GEMM（Naive → Shared → Register） | ✅ 4 个 `.cu` + benchmark 脚本 + 报告；⚠️ 实测数据待 GPU 机器补齐 |
| 2 | Triton Vector Add / GEMM、B: RMSNorm、C: Autotuning | ✅ 源码 + 报告；⚠️ 实测数据待 GPU 机器补齐 |
| 3 | Prefill/Decode、KV Cache 推导、MLA、FlashAttention 精读 | ✅ 完成（纯阅读 + 推导，无需 GPU） |
| 4 | 量化公式、INT8/INT4 手算、SmoothQuant/GPTQ/AWQ、DP/TP/PP | ✅ 完成（含两卡 MLP 切分图） |

## 关于 AI 辅助的说明

题目明确允许使用 AI/Agent 辅助，但要求能解释生成内容。本目录的作答原则：

- 所有**手算**（Task 4 的 INT8/INT4 量化）都用脚本独立复算过，脚本见 `task4/scripts/quant_check.py`；
- 所有**代码**都给出编译 / 运行命令与正确性判据，可以逐行解释；
- 所有**实测数据**都不虚构，缺硬件时留空 + 说明，而不是填一个"看起来合理"的数字。
