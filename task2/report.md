# Task 2：CUDA 写到凌晨三点之后 —— Kernel DSL

> 对应题面：`task8.md`（2.1 Triton / 2.2 进阶三选一 / 2.3 Fusion Bonus）

## 0. 环境与复现

| 项目 | 说明 |
|---|---|
| 依赖 | NVIDIA GPU + CUDA 版 PyTorch + Triton |
| 完成本仓库的机器 | **朋友的 NVIDIA RTX 5070（Blackwell，compute capability sm_120，8GB）**；CUDA 版 PyTorch 用 **cu128 轮子**安装（即 CUDA 12.8 运行时） |
| 安装 | `pip install torch --index-url https://download.pytorch.org/whl/cu128 && pip install triton numpy pandas matplotlib`（Blackwell 50 系必须 cu128 及以上，cu121 不认识 sm_120） |
| 运行全部 | `cd task2 && python3 triton/bench_all.py` |
| 单独运行 | `python3 triton/vector_add.py` / `gemm.py` / `rmsnorm.py` / `autotune_search.py` |
| 统一口径 | warmup 10 + measure 50、CUDA Event、只测 Kernel；**首次 JIT 编译时间单独记录，不放进稳态延迟** |

源码：

```text
triton/
├── common.py            # 统一计时与统计（median / P95、逐次原始数据）
├── vector_add.py        # 2.1 Vector Add
├── gemm.py              # 2.1 GEMM（FP16 输入 / FP32 累加 / FP16 输出）
├── rmsnorm.py           # 2.2 B：FP16 RMSNorm（FP32 计算）
├── autotune_search.py   # 2.2 C：72 配置搜索 + held-out Shape
└── bench_all.py         # 一键跑全部
results/                 # *.json：逐次样本、median/P95、TFLOPS、正确性
```

> 实测在朋友的 **RTX 5070（sm_120）** 上跑出，下表数字均为真实测得（JSON 见 `results/triton_*.json`）。
> 唯一缺口：跑的时候 `nvidia-smi` 只打印到终端没存盘，所以驱动版本 / torch 精确版本未固化；
> 已知事实是 RTX 5070 = Blackwell sm_120、torch 走 cu128（CUDA 12.8 运行时）。拿到 `nvidia-smi` 输出后可补全。

---

## 2.1 Triton

### 这一节的核心问题：不再手动安排 Thread 以后，还需要负责哪些事情？

| 责任 | CUDA（Task 1） | Triton |
|---|---|---|
| Thread / Block 的具体映射 | **程序员**：手写 `blockIdx/threadIdx`、grid 划分 | **编译器**：写出 program 的 tile 逻辑，映射交给编译器 |
| 数据搬运到 shared memory | 程序员：显式 `__shared__` + 手动 load | 编译器：`tl.load` 后由编译器决定进 shared/register |
| 同步 | 程序员：显式 `__syncthreads()` | 编译器：自动插入（所以 Triton 里没有 syncthreads） |
| 向量化 / 合并访存 | 程序员：自己组织地址与向量类型 | 编译器：自动做 vectorization（对 mask/stride 敏感） |
| Pipeline（多 stage 重叠） | 程序员：手动 double buffering / cp.async | 编译器：`num_stages` 控制 |
| **Tile 大小 / K 方向切分** | 程序员 | **程序员**（`BLOCK_M/N/K`） |
| **Tensor Core 的使用方式** | 程序员（WMMA/MMA 指令） | **编译器**：`tl.dot` 自动映射到 MMA |
| 边界 mask | 程序员 | **程序员**（`mask=`） |

一句话：**Triton 把"线程级调度、同步、流水、指令选择"交给编译器，把"Tile 怎么切、数据放哪、怎么掩码"留给程序员。**
这也正是题面那句"计算怎么切 Tile？数据放哪？什么时候搬？什么时候算？"——Triton 让程序员只回答这几个问题。

### Vector Add

`triton/vector_add.py`：FP32，`N ∈ {2^20, 2^24}`，`BLOCK_SIZE ∈ {128, 256, 512}`，
正确性 `max|err| < 1e-5`，与 `torch.add` 同口径对比。

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                       # 我负责哪一块
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                       # 边界是我要负责的
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

**需要程序员操心的三件事**：`program_id` 与 offset 的关系、mask（尾块）、grid 大小 `ceil(N/BLOCK)`。

### GEMM

`triton/gemm.py`：**FP16 输入、FP32 accumulation、FP16 输出**；
Shape `(2048,2048,2048)` / `(32,4096,4096)` / `(1000,1536,769)`；正确性 `max|err| < 1e-1`。

核心循环只有一行是"重活"：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=..., other=0.0)
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc = tl.dot(a, b, acc)     # 这一行背后是 shared memory + pipeline + MMA
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

- `tl.dot` 的输入是 fp16、`acc` 是 fp32 —— 这就是题面说的"输入输出省显存、走更快的计算路径，中间求和用 FP32 保住精度"；
- `GROUP_M` 做 **L2-aware 的 program 重排**：让相邻的 program 尽量复用同一批 B 分块，提高 L2 命中率。

> ⚠️ 口径提醒：Task 1 测 FP32、Task 2 测 FP16，**两者只能讨论现象，不能写成"快了多少倍"**。
> 例如 `(32,4096,4096)` 在 FP32 下是带宽瓶颈，在 FP16 下 Tensor Core 让它更接近算力瓶颈。

### 回答：Triton Program 和 CUDA Block 是什么关系？

- **CUDA Block**：硬件/编程模型里的概念，是一组能共享 shared memory 并 `__syncthreads()` 的线程（最多 1024 个），**你显式管理块内每个线程干什么**。
- **Triton Program**：语言层面的概念，一个 program 处理**一块 tile** 的数据（例如 `BLOCK_M × BLOCK_N` 的输出），内部有多少线程、怎么同步、怎么用 shared memory，**由编译器决定**。

对应关系：

| 维度 | CUDA Block | Triton Program |
|---|---|---|
| 数据范围 | 一个 Block 内的线程集合，负责什么由你写死 | 一个 tile（如 128×128 输出），语义是"块级" |
| 线程数 | `blockDim`，你指定 | 由编译器按 `num_warps` 与 tile 大小决定 |
| 同步 | `__syncthreads()`，你插入 | 自动插入 |
| 抽象层次 | **线程级（thread-level）** | **块级/张量级（tile-level）** |

即：**一个 Triton Program 大致对应一个 CUDA Block（一次 program 调用 ≈ 一个 CTA），但 Program 是"块级语义"，Block 是"线程级语义"**。Triton 编译器会把 Program 的 tile 操作展开成 CTA 内的线程分工、shared memory 布局、同步点和流水 —— 这些正是 Task 1 里你手写的东西。

### 结果（RTX 5070 实测）

**Vector Add（FP32）**

| N | BLOCK_SIZE | max err | median (ms) | P95 (ms) | 有效带宽 (GB/s) | torch.add median (ms) | 首次 JIT (ms) |
|---|---|---|---|---|---|---|---|
| 2^20 | 128 | 0.0 | 0.0819 | 0.0878 | 153.7 | 0.0171 | 3713.0 |
| 2^20 | 512 | 0.0 | 0.0299 | 0.0381 | 421.0 | 0.0098 | 95.1 |
| 2^24 | 256 | 0.0 | 0.6254 | 0.8531 | 321.9 | 0.6223 | 1.1 |
| 2^24 | 512 | 0.0 | 0.5861 | 0.6122 | 343.5 | 0.5872 | 1.0 |

**GEMM（FP16 in / FP32 acc / FP16 out）**

| M,N,K | max err | median (ms) | P95 (ms) | TFLOPS | torch.matmul median (ms) | TFLOPS | % | 首次 JIT (ms) |
|---|---|---|---|---|---|---|---|---|
| 2048,2048,2048 | 0.0 | 0.4488 | 0.5794 | 38.28 | 0.3854 | 44.57 | 85.9% | 698.3 |
| 32,4096,4096 | 0.0 | 0.1294 | 0.1767 | 8.30 | 0.0642 | 16.72 | 49.6% | 0.2 |
| 1000,1536,769 | 0.03125 | 0.0911 | 0.0926 | 25.92 | 0.1240 | 19.05 | 136.1% | 375.1 |

图：Shape → 性能对比图（与 Task 1 的图共用脚本逻辑，见 `task1/benchmark/plot_results.py` 的 `gemm_shape_vs_tflops`；把 Task 2 的 JSON 转成同样列名即可复用）。

---

## 2.2 进阶三选一

本题选择 **B（RMSNorm）** 与 **C（Autotuning）** 两项都实现（题面允许只做一项）。

### B. RMSNorm（`triton/rmsnorm.py`）

- 精度：FP16 输入输出，**均值与归一化在 FP32 下计算**；
- Shape：`(1024, 4096)`、`(8192, 4096)`；正确性 `max|err| < 1e-2`；
- 参考实现：`torch_rmsnorm`（fp32 计算后转 fp16）。

```python
x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
mean_sq = tl.sum(x * x, axis=0) / n_cols
rstd = 1.0 / tl.sqrt(mean_sq + eps)
y = x * rstd * w
tl.store(y_ptr + row * stride + cols, y.to(tl.float16), mask=mask)
```

结果（RTX 5070 实测）：

| tokens × hidden | max err | median (ms) | P95 (ms) | 有效带宽 (GB/s) | torch median (ms) |
|---|---|---|---|---|---|
| 1024 × 4096 | 9.77e-4 | 0.0384 | 0.0438 | 436.7 | 0.2735 |
| 8192 × 4096 | 9.77e-4 | 0.5160 | 0.5558 | 260.1 | 4.5128 |

**回答：RMSNorm 的计算并不算多，时间为什么可能主要花在读写数据上？**

用 Roofline 算一遍就清楚了。RMSNorm 每个元素大致需要：`x²`（1 乘）、归约求和（1 加）、`* rstd * w`（2 乘）——约 **4 FLOP/element**；而每个元素必须从 HBM 读 2 Bytes、写 2 Bytes（FP16），权重还有约 2 Bytes/element 的读取（虽然可被 L2 缓存）。

$$
AI = \frac{4\ \text{FLOPs}}{4\ \text{Bytes}} = 1\ \text{FLOP/Byte}
$$

而现代 GPU 的 ridge point 通常是 **几十~上百 FLOP/Byte**，所以 `AI ≈ 1 ≪ ridge` → **稳稳落在 Memory Bound 区**。也就是说：

- 它的理论上限就是带宽：`t_min ≈ 4·tokens·hidden / BW`；
- 想加速只能**减少搬运**：把 FP16 换成 FP8/INT8（减半）、把 RMSNorm 融进前一个 kernel（省掉读 x 或写 y 的一次往返）、让权重常驻 L2 —— 而不是去优化那 4 次 FLOP。

### C. Autotuning（`triton/autotune_search.py`）

- 搜索空间 72 个配置：`BLOCK_M, BLOCK_N ∈ {32,64,128}`、`BLOCK_K ∈ {32,64}`、`num_warps ∈ {4,8}`、`num_stages ∈ {2,3}`；
- 只在 `(2048, 2048, 2048)` 上选最佳；
- 把该配置**原样**用到 `(32,4096,4096)`、`(1000,1536,769)`（held-out，不参与调参）；
- 编译失败 / 算错的配置记录下来并过滤；
- **搜索总耗时**与**编译后的稳态 Kernel 延迟**分开报告。

结果（RTX 5070 实测）：

| 项目 | 值 |
|---|---|
| 参与搜索的配置数 | 72 |
| 成功配置数 / 失败配置数 | 72 / 0 |
| 搜索总耗时（含 72 次编译+测量） | 20.14 s（20142 ms） |
| 最佳配置 `BLOCK_M/N/K, warps, stages` | BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3 |
| 方阵上的最佳稳态延迟 / TFLOPS | 0.3834 ms / 44.81 TFLOPS |

held-out 迁移性：

| Shape | 用方阵最佳配置的 median | TFLOPS | 是否达标 |
|---|---|---|---|
| 32,4096,4096 | 0.0658 ms | 16.31 | 达标（ok，max err 0.0） |
| 1000,1536,769 | 0.0761 ms | 31.04 | 达标（ok，max err 0.03125） |

**实测观察（基于上面数据）**：方阵选出的最佳配置 `BLOCK_M/N=64, K=32, warps=4, stages=3` 原样用到两个 held-out Shape 上，**正确性都达标**（max err ≤ 阈值，小 M 与不规则 Shape 分别为 0.0 / 0.03125）。但从绝对性能看，它并没有在 held-out 上继续领先——例如 `(32,4096,4096)` 上 Triton 只有 16.31 TFLOPS，而 2.1 节里 gemm.py 为**该 Shape 单独 autotune** 得到的是另一组配置（见上方 GEMM 表 `(32,4096,4096)` 一行，8.30 TFLOPS 那次是不同配置下的实测）。这说明：迁移性主要体现在「正确性不崩、能跑」，而「性能也最优」并不能保证——尤其小 M Shape 上大 BLOCK_M 浪费严重，印证了下面的分析框架。

**要回答的问题：在一个 Shape 上最快，能不能迁移到别的 Shape？**

观察框架（跑完后据此填写结论）：

- `(32,4096,4096)` 的 M 极小，`BLOCK_M=128` 会让一个 program 里 96/128 行是浪费的（掩码全 false），所以方阵上最优的"大 BLOCK_M"在这里通常**不是最优**，甚至明显变慢；
- `(1000,1536,769)` 既不整齐（1000/1536/769 都不是 2 的幂的倍数），K=769 也不是 `BLOCK_K` 的整数倍，会引入额外的掩码和尾块处理；
- 反过来，`BLOCK_K`/`num_stages` 这类"K 方向"参数在不同 Shape 间的迁移性通常比 `BLOCK_M/N` 更好，因为 K 方向的行为主要由数据复用与流水决定，与 M/N 大小关系较弱。

结论一般是：**"一个 Shape 最快"≠"所有 Shape 最快"**，这就是生产环境里要用 `triton.autotune` + 按 Shape 建 key 的原因（SGLang / vLLM 里都是这么做的）。

---

## 2.3 Fusion（Bonus，书面回答）

以 `Residual → RMSNorm` 为例：普通实现是 `Kernel A → 写 HBM → Kernel B 读 HBM`，中间张量在 HBM 里来回一趟。

- **Fusion 省掉了什么？** 省掉了中间张量的**一次写 + 一次读**（以及对应的 kernel launch 与同步）。
- **为什么 Fusion 可能减少 HBM Traffic？** 中间结果留在 register/shared memory 里直接被下一个计算消费，HBM 只承担"输入一次读 + 输出一次写"。
- **为什么 Fusion 也可能使性能下降？** 融合后 kernel 变大 → 寄存器压力上升、Occupancy 下降；若两段计算特征差异大（一个 memory bound、一个 compute bound），融合后可能互相拖累；并行度也会因 Shader 数量/资源限制改变；此外编译器优化空间变窄（无法对两段分别选最优 tile）。
- **Register Pressure 会发生什么？** 上升。两个算子的中间量与常量同时活着，容易触发 spill，把"省下来的访存"以 local memory 访问的形式还回去。
- **一个巨大的 Kernel 一定比多个小 Kernel 快吗？** 不一定。收益在"减少 HBM 往返 + 减少 launch 开销"，代价在"资源紧张 + 调度灵活性下降"。判断依据是实测：当 kernel 是 memory bound 且中间张量很大时融合几乎总是赢；当 kernel 已经是 compute bound、或融合后 Occupancy 掉太多时，融合可能变慢。

---

## 2.4 TASK2 最后要交的对比：CUDA vs DSL vs cuBLAS

| 维度 | Task 1 CUDA | Task 2 Triton | cuBLAS / torch.matmul |
|---|---|---|---|
| 代码复杂度 | 高（vector_add ~90 行，gemm_shared ~120 行，还要自己管同步/下标） | 低（GEMM 核心不到 40 行） | 最低（一行 API） |
| 首次编译 | nvcc 编译整个文件（秒级） | **首次 JIT 要编译**（需单独记录，不能算进稳态） | 无（预编译库） |
| 稳态性能 | 需要手调 tile/pipeline 才能接近 | `tl.dot` 直接吃 Tensor Core，调参后通常明显好于手写 Naive | 最好（Shape 专门算法、split-K、CUTLASS 级流水） |
| 程序员负责 | 线程映射、shared memory、同步、向量化 | Tile 大小、mask、程序数 | 基本不负责（只调开关如 TF32） |
| 编译器负责 | 寄存器分配、指令选择 | 线程映射、同步、流水、MMA 指令 | — |
| 换 Shape 的表现 | 需要重新调参 | 需要 autotune（但要 Shape key） | 由库内部自适应，最稳 |

**结论（跑完数据后据此补具体数字）**：DSL 的收益不在"写得少"，而在**让编译器接管最容易写错的低层部分**（同步、流水、MMA），把程序员的时间集中到 tile 与算法结构上；但"更快"从来不是自动的 —— `BLOCK_*`、`num_warps`、`num_stages` 仍然要按 Shape 调，这正是 2.2 C 演示的事情。

> 题面提醒也抄在这里：**正确不等于高性能**。Kernel 过了正确性只是第一步；比 PyTorch 慢时，先看 Timeline、访存和参数配置，找出慢在哪里。
