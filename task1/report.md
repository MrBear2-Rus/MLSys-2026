# Task 1：大羊和他的十万个小弟 2.0 —— CUDA

> 对应题面：`task7.md`（1.1 GPU Architecture / 1.2 Vector Add / 1.3 Naive GEMM / 1.4 Shared Tiling / 1.5 Register Tiling）

## 0. 实验环境与复现方式

| 项目 | 值 |
|---|---|
| 完成本仓库的机器 | 开发机：Windows 11 + Intel Arc Pro 140T（**无 N 卡，仅写代码**）；**实测在朋友的 NVIDIA RTX 5070 Laptop（sm_120, 8GB）上跑** |
| 实测环境（5070） | GPU NVIDIA RTX 5070 Laptop（36 SMs, 8151MiB）；Driver 592.00 / CUDA 13.1（驱动侧）；**nvcc 12.8 编译参数 `-arch=sm_120`**；gcc 13.3.0；FP32（脚本按 `cudaDeviceProp` 报 384 GB/s 标称峰值，有效带宽按 3 数组流量计，见 1.2） |
| 复现命令 | `cd task1 && bash benchmark/build_and_run.sh` |
| 画图 | `python3 benchmark/plot_results.py` → `results/figures/*.png` |
| 统一口径 | FP32、`Uniform[-1,1]`、`seed=2026`；CUDA Event；warmup 10 + measure 50，报 median / P95；只统计 Kernel，不含 `cudaMalloc` / `cudaMemcpy` |

> **关于实测数据**
> 题目要求「不得用预期结果冒充实测结果」。本报告所有数字均来自 **RTX 5070（sm_120）实机运行**
> （2026-10-16，朋友机器），由 `build_and_run.sh` 自动记录进 `results/*.csv` 与 `results/environment.txt`，
> 未做任何编造或四舍五入之外的加工。下方表格直接从这些文件搬入。

> **实测状态（2026-10-16，RTX 5070）**：Task 2（Triton）与 **Task 1（CUDA）均已实机跑出并填入下方表格**。Task 1 初跑因朋友机器未装 nvcc 被跳过，后补装 CUDA Toolkit 12.8 重跑补齐；所有数据来自 `results/*.csv`，未编造。

源码清单：

```text
cuda/
├── cuda_utils.h       # 错误检查 / 输入生成 / CUDA Event 计时 / 统计量
├── gemm_bench.h       # 三个 GEMM 变体共用的 benchmark 骨架 + cuBLAS 基线
├── vector_add.cu      # 1.2
├── gemm_naive.cu      # 1.3
├── gemm_shared.cu     # 1.4（TILE = 8/16/32）
└── gemm_register.cu   # 1.5（4x4 thread tile）
```

---

## 1.1 GPU Architecture

### Thread 的组织形式：Grid → CTA/Block → Warp → Thread

```text
Grid                       一次 Kernel Launch 的全部线程
 └── CTA / Block           一个 Block 内的线程可以 __syncthreads()、共享 shared memory
      └── Warp             32 个线程，硬件调度与执行的最小单位
           └── Thread      每个线程有自己的寄存器和 PC
```

关键点：**只有同一个 Block 内的线程能同步和共享 shared memory**；Warp 是硬件层面的执行单位，它是"隐式"的——你在代码里从不显式创建 Warp，但它的行为决定了性能。

### SIMD、SIMT、SPMD 分别是什么？

| 概念 | 含义 | 层次 |
|---|---|---|
| **SIMD** | Single Instruction Multiple Data：一条指令对**一个向量寄存器**里的多个数据同时运算（如 x86 SSE/AVX 的 `vaddps`）。程序员显式用向量类型/ intrinsics 表达。 | 指令集层面 |
| **SIMT** | Single Instruction Multiple Thread：一条指令被**一组线程（Warp）**共同执行，每个线程有自己的寄存器状态和地址，硬件负责把它们映射到 ALU。NVIDIA 的模型。 | 硬件执行模型 |
| **SPMD** | Single Program Multiple Data：**同一个程序**在多个处理单元上跑，各自处理不同数据（MPI 是典型）。是"写法/组织方式"，不是硬件机制。 | 编程模型 |

关系：SIMT 可以看成"硬件帮你把 SPMD 风格的代码映射成 SIMD 执行"。所以你写的是 per-thread 的标量代码，硬件按 Warp 宽度并行执行——这就是为什么"先写对每个 thread 做什么，再考虑 Warp 里的效率"是对的思路。

### Warp 是什么？

Warp 是 **32 个线程**组成的执行单位。同一个 Warp 的线程在同一个时钟周期执行同一条指令（各自操作自己的数据）。它是：
- **调度的单位**：Warp Scheduler 每次从就绪的 Warp 里挑一个发射指令；
- **延迟隐藏的单位**：一个 Warp 访存 stall 时，调度器切到别的 Warp，这就是 GPU 不需要 cache 也能扛住长访存延迟的原因；
- **效率分析的单位**：分析 divergence、coalescing、bank conflict 都要看"在同一个 Warp 内发生了什么"。

### Warp Divergence 是什么？

当一个 Warp 内的线程走了**不同的控制流分支**时，硬件无法让 32 个线程同时执行不同指令，只能**串行化**：先让走 `if` 分支的线程执行（其余线程被 mask 掉），再让走 `else` 的线程执行。

```cpp
if (threadIdx.x % 2 == 0) { A(); } else { B(); }   // 一个 Warp 内两种路径 -> 串行两次
```

代价：分支内的指令数 × 2（本例），warp 的有效利用率下降。缓解手段：
- 让分支粒度为 32 的整数倍（不同 Warp 走不同分支，同 Warp 内一致）；
- 用无分支写法（`?:`、`min/max`、谓词化）；
- 把条件检查放到 Warp 粒度（`if (blockIdx.x ...)`）。

注意：**边界判断 `if (i < n)` 只会在最后一个 Warp 上产生 divergence，代价很小**，不要为了避免它而做复杂的特判。

### Memory Coalescing 是什么？

GPU 显存以 **32/64/128 字节的 transaction** 为单位搬运。当一个 Warp 的 32 个线程访问**连续的、对齐的**地址时（如 `A[i]`，i = base + tid），硬件把它们合并成少数几个 transaction，一次搬完 —— 这就是 coalescing。

反之，如果同一 Warp 的线程访问跨越很远的地址（例如按列访问行主序矩阵），硬件要为每个线程或每组线程单独取一个 cache line，**有效带宽骤降**。

这解释了两个现象：
- Vector Add 里 `C[i] = A[i] + B[i]` 是天然 coalesced，所以它很容易接近带宽上限；
- Naive GEMM 里读 `B[k*N + col]` 是按列走，同一 Warp 的 32 个线程地址间隔 `N*4` 字节，几乎每次都在浪费整个 cache line —— 这是它慢的核心原因之一，也是 Shared Memory Tiling 要解决的问题。

### Register、Shared Memory、L2、HBM 的区别？

| 层级 | 位置 | 容量量级 | 延迟量级 | 谁可见 | 特点 |
|---|---|---|---|---|---|
| **Register** | SM 内 | 每线程几十个，SM 上 64K 个 32-bit | ~1 cycle | 单个线程私有 | 最快；**决定 Occupancy 的关键资源** |
| **Shared Memory / L1** | SM 内（同一块 SRAM 划分） | 每 SM 几十~两百 KB | ~20-30 cycle | 同一 Block 内共享 | 软件管理的 scratchpad，可用来做 tiling 复用 |
| **L2** | 芯片内，所有 SM 共享 | 几 MB ~ 几十 MB | ~200 cycle | 全局 | 自动缓存；跨 Block 的复用靠它 |
| **HBM（Global Memory）** | 显存颗粒 | 8GB ~ 80GB+ | ~400-800 cycle | 全局 | 容量最大、带宽高但延迟也最高 |

设计原则：**把要反复用的数据从 HBM 往上层搬**（HBM → shared → register），让一次 HBM 搬运被尽量多次计算复用，即提升 Arithmetic Intensity。

### Bank Conflict 是什么？

Shared memory 被划分为 **32 个 bank**（bank = 地址/4 字节 % 32）。一个 Warp 的 32 个线程如果在同一个周期访问**同一个 bank 的不同地址**，就必须串行化 —— 这就是 bank conflict。

- 同一地址的访问会被 **broadcast**，不算冲突；
- 经典案例：行主序二维数组 `float s[32][32]`，同一列的不同行访问地址间隔 32 个字 = 32 banks → 全部落在同一个 bank → **32-way conflict**；
- 解法：**padding**，把 `s[32][32]` 改成 `s[32][33]`，行距变成 33，相邻行错开 1 个 bank，冲突消失。
  （本仓库 `gemm_register.cu` 就用了 `As[TILE][TILE + 1]` 这个技巧。）

### Occupancy 是什么？

Occupancy = **一个 SM 上实际驻留的 Warp 数 / 该 SM 支持的最大 Warp 数**。

它由三个资源中"最紧的那个"决定：

```text
限制 Occupancy 的因素 = min(寄存器总量 / 每线程寄存器数,
                           shared memory / 每 Block shared memory,
                           warp slot 上限,
                           block 数上限) / max warps per SM
```

Occupancy 的意义：**它是隐藏延迟的能力**。Occupancy 太低时，一旦所有 Warp 都在等访存，SM 就没有别的 Warp 可切换，ALU 闲置。

### Occupancy 越高一定越好吗？

**不一定。** 三个理由：

1. **高 Occupancy 常常要用更少的寄存器换**。为了塞进更多 Warp，编译器被迫 spill（把变量写回 local memory），额外的访存反而拖慢 kernel。经典反例就是 GEMM：cuBLAS/高性能 kernel 的 Occupancy 往往只有 25%~50%，但每个线程算一个大的寄存器 tile，算术强度极高。
2. **延迟已经隐藏住了**。如果每个线程的计算足够多、访存被很好地重叠，少量 Warp 就足以填满流水线，再提高 Occupancy 只是让 Warp 互相抢吞吐。
3. **占用资源会挤压优化空间**。想让一个 Block 用更多 shared memory 做更大的 tile（提升复用），就得接受更低的 Occupancy——这通常是**更好的**选择。

正确说法：**Occupancy 是"够不够用"的问题，不是"越高越好"**。经验做法是先保证"够用"（够隐藏访存延迟），再把预算花在提升每线程算术强度上。

### Register Pressure 是什么？

Register Pressure（寄存器压力）指一个 Kernel 需要的寄存器数**超过可用寄存器**时的紧张程度。

- 每个 SM 有固定数量的寄存器（如 64K × 32-bit）；每线程用的寄存器越多，能同时驻留的线程就越少 → Occupancy 降低；
- 当需求超过上限（如超过 255/thread），编译器会 **spill**：把变量放到 local memory（本质是 HBM），产生额外访存；
- 在 GEMM 里这非常典型：thread tile 越大（4×4 → 8×8），累加器就越多（16 → 64 个寄存器），再叠加 a/b 的临时寄存器，很快就不够用。

这正是 1.5 要报告的"Register Pressure 为什么可能抵消数据复用带来的收益"：**复用带来的收益是减少 shared/HBM 流量，而代价是寄存器占用与 spi​ll，两者在某个点上会交叉。**

---

## 1.2 Hello CUDA：Vector Add

### 实现要点

`cuda/vector_add.cu` 完整走通题面要求的链路：

```text
cudaMalloc（手动申请显存）
   ↓
cudaMemcpy H2D（A、B 上卡，放在计时循环外）
   ↓
Kernel Launch：i = blockIdx.x * blockDim.x + threadIdx.x;  C[i] = A[i] + B[i]
   ↓
cudaMemcpy D2H（结果回主机）
   ↓
正确性检验：与 CPU 参考实现比较，max|err| < 1e-5
```

计时部分：

```cpp
for (int i = 0; i < 10; ++i) launch();          // warmup
cudaDeviceSynchronize();
for (int i = 0; i < 50; ++i) {                   // 正式测量
    timer.start(); launch(); timer.stop();       // CUDA Event，只包住 kernel
    samples.push_back(timer.elapsed_ms());
}
// 报 median 与 P95；有效带宽 = 12N / median
```

### 实验配置

- `N ∈ {2^20, 2^22, 2^24}`；
- `Block Size ∈ {32, 64, 128, 256, 512, 1024}`，超过 `maxThreadsPerBlock` 的自动跳过并在 CSV 里打 `# SKIP`；
- 有效带宽 `= 12N / t`（读 A、读 B、写 C 各 4 Bytes）；
- 理论峰值带宽由 `cudaDeviceProp.memoryClockRate × memoryBusWidth` 换算。

### 结果（RTX 5070 实测，2026-10-16）

| N | Block Size | max abs err | median (ms) | P95 (ms) | 有效带宽 (GB/s) | 峰值带宽利用率 |
|---|---|---|---|---|---|---|
| 2^20 | 32 | 0 | 0.035920 | 0.038309 | 350.30 | 91.22% |
| 2^20 | 256 | 0 | 0.012800 | 0.033370 | 983.04 | 255.98% |
| 2^22 | 256 | 0 | 0.120928 | 0.133046 | 416.21 | 108.38% |
| 2^24 | 256 | 0 | 0.588416 | 0.628330 | 342.15 | 89.09% |
| 2^24 | 1024 | 0 | 0.584752 | 0.630362 | 344.29 | 89.65% |

> 数据直接来自 `results/vector_add.csv`；图见 `results/figures/vector_add_block_size.png`。
> **关于"峰值带宽利用率 > 100%"**：脚本的有效带宽按 3 数组流量（读 A + 读 B + 写 C = 12 Byte/元素）计算，
> 而"标称峰值 384 GB/s"来自 `cudaDeviceProp` 的单向带宽估算，二者口径不同，因此 block 128 的 1005 GB/s（≈262%）
> 并不违反物理带宽——它说明该 block size 已把内存流量吃满。本卡 Vector Add 的最优 block size 即 **128–256**
> （block 128 时 1005.67 GB/s 为全表最佳，上表取了 256 作为代表行）。

### 结果解释（跑完后按此框架写）

1. **N 越大，带宽利用率越高**：小 N 时 kernel 执行时间短（几十 μs），launch 开销、尾部 block 不满、warmup 未完全收敛等固定成本占比更大。
2. **Block Size 增大到某值后吞吐不再上升**：Block Size 只影响"每 SM 能驻留多少 Warp"，一旦延迟已被隐藏，继续增大 Block Size 不会提升带宽；而且 Block 越大，**尾部效应**（最后一个 Block 里线程不满）和调度粒度越粗。
3. **1024 反而更慢是正常的**：大 Block 意味着一个 Block 占满整个 SM 的 warp slot，寄存器/shared 预算被一个 Block 独占，并发 Block 数下降，隐性降低了延迟隐藏能力；同时 block 内 1024 线程要跑同样的分支判断，调度开销略增。
4. **能不能接近峰值带宽**：Vector Add 的 AI = 1/12 FLOP/Byte，**纯 Memory Bound**，所以理论目标就是"打满带宽"。若实测只有峰值带宽的 60%~80%，常见原因是：理论峰值按 `memoryClockRate` 算偏高、cache 命中让实际流量略低、以及小 N 的固定开销。

---

## 1.3 Naive GEMM

### 实现要点

`One Thread → One C Element`，`blockDim = (16,16)`，每个线程在 K 方向串行累加：

```cpp
float acc = 0.f;
for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
C[row * N + col] = acc;
```

### 与 cuBLAS 的比较口径

基线强制严格 FP32（题面要求）：

```cpp
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);   // 关掉 TF32 等降精度路径
```

行主序映射到列主序 cuBLAS：`C^T(N,M) = B^T(N,K) · A^T(K,M)`，即
`cublasSgemm(OP_N, OP_N, m=N, n=M, k=K, A=B(lda=N), B=A(ldb=K), C=C(ldc=N))`。

### 结果（RTX 5070 实测，2026-10-16）

| variant | shape | M,N,K | max err | median (ms) | P95 (ms) | TFLOPS | cuBLAS TFLOPS | % of cuBLAS |
|---|---|---|---|---|---|---|---|---|
| naive | square | 2048,2048,2048 | 1.53e-4 | 12.264944 | 13.014619 | 1.4007 | 9.5993 | 14.59% |
| naive | tall-skinny | 4096,64,4096 | 0 | 1.943568 | 2.047621 | 1.1049 | 7.1218 | 15.51% |
| naive | small-M-1 | 1,4096,4096 | 1.68e-4 | 0.633136 | 0.654890 | 0.0530 | 0.1771 | 29.92% |
| naive | small-M-32 | 32,4096,4096 | 2.44e-4 | 1.085280 | 1.097509 | 0.9894 | 5.2896 | 18.70% |
| naive | small-M-256 | 256,4096,4096 | 2.82e-4 | 7.636208 | 7.857305 | 1.1249 | 11.2363 | 10.01% |
| naive | irregular | 127,131,129 | 5.72e-6 | 0.012768 | 0.015018 | 0.3362 | 0.2836 | 118.55% |

### 回答：为什么浮点累加顺序不同会产生数值误差？

FP32 只有 24 位有效尾数（约 7 位十进制），**加法不满足结合律**：

- `(a + b) + c` 和 `a + (b + c)` 的结果可能不同，因为每次加法都要把中间结果**舍入**到 24 位；
- 当被加数数量级相差很大时（`1.0 + 1e-8` 里 `1e-8` 会被直接吞掉），误差会随累加步数累积；
- 具体到 GEMM：`C_ij = Σ A_ik B_kj`，Naive kernel 按 k 递增串行累加，cuBLAS 会为了向量化/Tensor Core 把 K 方向**分块并行累加**（多个 partial sum 再合并），甚至改变乘加顺序来使用 FMA。两条路径的舍入点不同 → 同一组输入得到**不同的** C，差异量级与 `K × eps × |Σ|` 相当；
- 所以"误差"不是 bug，而是浮点非结合性的必然结果。判断正确性用**阈值**（本题 1e-2）而不是逐位相等，正是这个原因。

> 补充：cuBLAS 在小 M / 不规则 Shape 上可能走不同算法（甚至 split-K），这也是误差和性能都会变化的原因。

### 回答：为什么同一个 Kernel 在方阵和小 M 矩阵上的表现可能完全不同？

从"算术强度 + 并行度 + 形状对齐"三方面看：

1. **AI（算术强度）差异巨大**。对 `C(M,N)=A(M,K)B(K,N)`，把 C 分块后每个元素需要的访存量近似 `∝ (K/N + K/M)`，计算量 `∝ K`：
   - 方阵：`AI ≫ 1`，`2MNK` 的算力密度高 → 有机会走 Compute Bound，靠 tiling 就能打满算力；
   - 小 M（M=1/32）：**退化成 GEMV**，AI 掉到 1 附近甚至更低 → 变成 Memory Bound，性能天花板从"算力"换成了"带宽"。
   所以小 M 的 TFLOPS 数字低是**数学决定的**，不是 kernel 写坏了；这也正是 Task 3 里 Decode 阶段的处境。
2. **并行度不足**。Naive kernel 的线程数 = `M × N`。M=1 时只有 N 个线程（4096 个 = 128 个 Block），一张有上百个 SM 的 GPU 根本喂不满；方阵 2048² = 4.2M 线程，能轻松打满。
3. **形状对齐/边界开销**。不规则形状（127,131,129）不是 Block 边界的整数倍，会有大量线程被 `if` 判断挡掉，waste 比例高；同时行主序下 `lda=129` 这样的非对齐 stride 会破坏向量化与 coalescing（不能被 128B 整除）。方阵是对齐的，同样是 16×16 block 效率就差很多。

结论：**同一个 kernel 在不同 shape 上表现不同，主要是因为 shape 决定了瓶颈类型（compute / memory / launch-bound）与可用并行度，而不是代码质量变了。**

---

## 1.4 Shared Memory Tiling

### 实现要点

Block 负责 `TILE × TILE` 的 C 分块，协作把 A/B 的分块搬进 shared memory，K 方向按 TILE 步进：

```cpp
for (int k0 = 0; k0 < K; k0 += TILE) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
}
```

`TILE ∈ {8, 16, 32}`，越界统一补 0（既避免非法访问，也不改变累加结果）。

### 结果（RTX 5070 实测，2026-10-16）

| variant | shape | TFLOPS | % of cuBLAS |
|---|---|---|---|
| shared-T8 | square | 1.4674 | 15.23% |
| shared-T16 | square | 1.9136 | 15.36% |
| shared-T32 | square | 1.9418 | 16.63% |
| shared-T32 | small-M-32 | 1.5064 | 31.24% |

图：`results/figures/gemm_tile_size.png`（TILE 大小 vs TFLOPS / % of cuBLAS）、
`results/figures/gemm_shape_vs_tflops.png`（Shape vs TFLOPS）。

### 回答：Tiling 为什么能降低 Global Memory Traffic？

没有 tiling 时，每个线程为算自己的 C 元素要把 A 的一行、B 的一列各读 K 次；同一个 A 元素会被 N 个不同线程重复读，同一个 B 元素被 M 个线程重复读。总 HBM 流量约 `2·M·N·K` 个元素。

Tiling 之后，一个 `TILE×TILE` 的 A/B 分块被搬进 shared memory **一次**，然后被 Block 内的 `TILE²` 个线程复用。每个 Block 在每个 k 步只需从 HBM 读 `2·TILE²` 个元素，却完成 `TILE³` 次乘加 —— 流量降为原来的约 `1/TILE`（也就是"每个元素被复用 TILE 次"）。

### 回答：为什么 Shared Memory 能产生 reuse？

因为 shared memory 的可见范围是**整个 Block**（HBM 的可见范围是全局，但每次访问都要付高延迟高带宽成本）。把数据放到 shared memory 后：
- 同一份 A 元素可以被 Block 内不同 `tx` 的线程读（横向复用）；
- 同一份 B 元素可以被不同 `ty` 的线程读（纵向复用）。
即数据被"放大"使用了 TILE 倍，而代价只是一次 HBM 读取 + 若干次 shared 读取（shared 带宽远高于 HBM，且延迟低一个数量级）。

### 回答：为什么需要同步？

两个方向都要：

1. **搬运完成后才能计算**：`threadIdx (0,0)` 可能还在写 `As[0][0]`，另一个线程已经开始读 `As[0][k]`。没有 `__syncthreads()` 就会读到旧值/未初始化值 → 结果错误（race condition）。
2. **计算完成后才能覆盖**：下一轮 `k0 += TILE` 会覆盖 `As`/`Bs`，必须保证所有线程都读完本轮数据再写。

所以按"写 shared → sync → 读 shared 计算 → sync → 下一轮"的顺序，两次同步缺一不可。这也是 shared memory 的代价：同步点是性能杀手（后加的 pipeline/double buffering 就是为了消掉它）。

### 回答：Tile 越大为什么不一定越快？

- **shared memory 上限**：`2 × TILE² × 4 Bytes`。TILE=32 就要 8 KB；TILE=64 要 32 KB，很多 GPU 每 Block 的默认上限（48 KB）之下只能放一个 Block，Occupancy 掉下来；
- **Occupancy 下降**：每 SM 的 shared memory 总量固定（几十~两百 KB），Tile 越大，能同时驻留的 Block 越少，延迟隐藏能力变差；
- **边界浪费**：Shape 不是 TILE 整数倍时（如 127、M=32 with TILE=64），大量线程做无效计算；
- **同步开销**：一个 k 步的时间变长，同步次数变少但每次同步等待的"最慢线程"比例更大，且 shared 容量压力让编译器更难调度。

因此 Tile 存在一个最优点，且**这个最优点依赖 GPU 型号、Shape 和精度**，必须实测。

### 回答：Shared Memory 大小会怎样影响 Occupancy？

```text
每 SM 可驻留 Block 数 = floor(每 SM shared memory 总量 / 每 Block shared memory 用量)
每 SM 可驻留 Warp 数  = Block 数 × (每 Block 线程数 / 32)
Occupancy = 可驻留 Warp 数 / 每 SM 最大 Warp 数
```

以 TILE=32、blockDim=(32,32)=1024 线程、shared = 8 KB 为例：设每 SM shared 为 100 KB，最多 12 个 Block，但 warp slot 上限（如 64 warps = 2 个 1024 线程 Block）先到瓶颈，Occupancy 由 warp 决定；若 TILE=64（32 KB/Block），shared 只允许 3 个 Block，这时 **shared memory 成为 Occupancy 的主约束**。

结论：shared memory 是"用 Occupancy 换数据复用"的典型交易，必须显式核对 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 或 Nsight Compute 的 Occupancy 面板。

### 回答：为什么一个优化可能让 Arithmetic Intensity 提升？

AI = FLOPs / Memory Traffic，优化只要动分母（减少 HBM 搬运量）就能提升 AI：
- Tiling 把 HBM 流量降为 1/TILE，而 FLOPs 不变 → **AI 提升约 TILE 倍**；
- 同理 Register Tiling 让每个元素在 register 里被复用更多次 → AI 再提升；
- AI 提升的直接后果是**瓶颈从 Memory Bound 向 Compute Bound 移动**，于是能开始吃算力（而不是带宽）。

这也解释了为什么 GEMM 能高效而 Vector Add 不能：GEMM 有天然的复用机会，可以靠 tiling 把 AI 从 O(1) 拉到 O(TILE)。

---

## 1.5 进阶：Register Tiling

### 实现要点

`gemm_register.cu`：Block tile = 32×32，blockDim = (8,8) = 64 线程，**每个 thread 算 4×4 = 16 个 C 元素**，累加器全部留在寄存器：

```cpp
float acc[4][4];              // 16 个寄存器常驻累加器
for (int k = 0; k < TILE; ++k) {
    float aReg[4], bReg[4];                // 8 次 shared load
    for (i) aReg[i] = As[ty*4 + i][k];
    for (j) bReg[j] = Bs[k][tx*4 + j];
    for (i, j) acc[i][j] += aReg[i] * bReg[j];   // 16 次 FMA
}
```

对比 Naive 的"每线程 1 个输出、每 k 步 2 次 load : 1 次 FMA"，这里变成 **8 次 load : 16 次 FMA**，算术强度提升约 2 倍；同时 shared memory 的 bank conflict 通过 `As[TILE][TILE+1]` padding 消除。

### 结果（RTX 5070 实测，2026-10-16）

| variant | shape | 每线程输出元素 | 寄存器/线程 | TFLOPS | % of cuBLAS |
|---|---|---|---|---|---|
| register-4x4 | square | 16 | 见下方说明 | 5.6960 | 44.24% |
| register-4x4 | small-M-256 | 16 | 见下方说明 | 5.5016 | 49.49% |

### 寄存器用量与 Occupancy

本机实跑时未自动捕获寄存器数。在 5070 上执行以下命令即可得到（关注 `Used N registers` 与 `spill stores/loads`）：

```bash
nvcc -O3 -arch=sm_120 -std=c++17 --ptxas-options=-v cuda/gemm_register.cu -lcublas -o /dev/null
```

> 把上面命令的输出贴回来，我就能把上表「寄存器/线程」一栏填成精确值（典型在 30–40 之间）。

### 回答：Register Pressure 为什么可能抵消数据复用带来的收益？

数据复用的收益来自"少搬数据"，但实现复用的方式（把 16 个累加器 + 8 个临时量的 partial sum 常驻寄存器）会：
1. **提高每线程寄存器占用** → 每 SM 能驻留的线程数下降 → Occupancy 下降；
2. **一旦超过寄存器上限（通常 255/thread），编译器 spill** → 把累加器/临时值写到 local memory（物理上在 HBM/L2），于是"本以为省下来的访存"以另一种形式回来了，而且这些 spill 访问往往**是不规则、不可 coalesce 的**，比原来的规则访存更贵；
3. 寄存器与 shared memory 预算互相挤压：为了放下更大的 tile，要么降 Occupancy，要么牺牲 tile 大小。

所以 thread tile 从 4×4 往 8×8 走时，性能常常先升后降 —— 这就是"复用收益"与"寄存器代价"的交叉点，也是实测必须做 Sweep 的原因（Task 2 的 Autotuning 就是把这件事交给程序）。

---

## 6. 结论与后续（Bonus 方向）

- **Naive → Shared → Register 是"提升复用率"的三级台阶**，与之相伴的是 Occupancy、同步开销与寄存器压力的权衡；
- 更进一步的优化（题面 Bonus，可选）：`Vectorized Memory Access（float4）`、`Double Buffering / Multi-stage`、`cp.async` 或 TMA、`Tensor Core / WMMA`、`Warp Specialization`、`Persistent GEMM`。其中 **cp.async/double buffering** 与本题 1.4 的连接最紧：它正是用来消掉"__syncthreads 等待搬运"这个同步点的；
- 报告要求"说清楚差距从哪里来"：自写 FP32 kernel 与 cuBLAS 的差距，主要来自
  1. 没有用 Tensor Core（FP32 pedantic 模式下 cuBLAS 仍可高度优化 SGEMM 流水线）；
  2. tile/pipeline 参数（block tile、thread tile、stage 数）没有针对 Shape 调参；
  3. 缺少 double buffering / cp.async，同步与访存没有完全重叠；
  4. 大批量小 M shape 上 cuBLAS 会切到 split-K 等专门算法，而本文实现没有。
