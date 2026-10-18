# Task 1：大羊和他的十万个小弟 2.0 —— CUDA
---

## 1.1 GPU Architecture


### SIMD、SIMT、SPMD 分别是什么？

SIMD:一条指令对一个向量寄存器里的多个数据同时计算

SIMT:一条指令被一组线程(Warp)共同执行，每个线程有其寄存器状态和地址，而硬件负责把它们映射到ALU。

SIMD:同一个程序在多个处理单元上跑，这种偏向于软件层的一种写法和组织方式，而不是硬件的机制

### Warp 是什么？

Warp 是 32 个线程组成的执行单位。同一个 Warp 的线程在同一个时钟周期执行同一条指令（各自操作自己的数据）。它同时是一个调度的单位和延迟隐藏、效率分析的单位

### Warp Divergence 是什么？

当一个 Warp 内的线程走了不同的控制流分支时，硬件无法让 32 个线程同时执行不同指令，只能串行化：先让走if分支的线程执行（其余线程被 mask 掉），再让走else的线程执行。但也有代价：分支内的指令数×2，warp的有效利用率下降

### Memory Coalescing 是什么？

GPU 显存以32/64/128 字节的 transaction为单位搬运。当一个 Warp 的 32 个线程访问连续的、对齐的地址时（如 A[i]，i = base + tid），硬件把它们合并成少数几个 transaction，一次搬完 —— 这就是 coalescing。(此事在计组的Cache中亦有记载)

反之，如果同一 Warp 的线程访问跨越很远的地址（例如按列访问行主序矩阵），硬件要为每个线程或每组线程单独取一个 cache line，有效带宽就会骤降。（一个篮子只放一个鸡蛋与一个篮子放一堆鸡蛋，运力当然不同）

### Register、Shared Memory、L2、HBM 的区别？

| 层级 | 位置 | 容量量级 | 延迟量级 | 谁可见 | 特点 |
|---|---|---|---|---|---|
| **Register** | SM 内 | 每线程几十个，SM 上 64K 个 32-bit | ~1 cycle | 单个线程私有 | 最快；**决定 Occupancy 的关键资源** |
| **Shared Memory / L1** | SM 内（同一块 SRAM 划分） | 每 SM 几十~两百 KB | ~20-30 cycle | 同一 Block 内共享 | 软件管理的 scratchpad，可用来做 tiling 复用 |
| **L2** | 芯片内，所有 SM 共享 | 几 MB ~ 几十 MB | ~200 cycle | 全局 | 自动缓存；跨 Block 的复用靠它 |
| **HBM（Global Memory）** | 显存颗粒 | 8GB ~ 80GB+ | ~400-800 cycle | 全局 | 容量最大、带宽高但延迟也最高 |

（该表来自AI。我认为的是这四个相当于是类似CPU的层级存储和Cache机制，越往高层越快量越小，越往底层量越大但是越慢。Register是最快的决定资源，然后L1就可以在Block内共享拿来做复用，L2更慢但是跨Block传播,然后HBM就是显存中的最底层的东西）

### Bank Conflict 是什么？

Shared memory 被划分为 32 个 bank 。一个 Warp 的 32 个线程如果在同一个周期访问同一个 bank 的不同地址，就必须串行化。

### Occupancy 是什么？

Occupancy = 一个 SM 上实际驻留的 Warp 数 / 该 SM 支持的最大 Warp 数。

它由三个资源中"最紧的那个"决定：

```text
限制 Occupancy 的因素 = min(寄存器总量 / 每线程寄存器数,
                           shared memory / 每 Block shared memory,
                           warp slot 上限,
                           block 数上限) / max warps per SM
```

故在我看来Occupancy的意义类似于隐藏延迟。Occupancy 太低时，一旦所有 Warp 都在等访存，SM 就没有别的 Warp 可切换，计算的ALU资源就会闲置。

### Occupancy 越高一定越好吗？

不一定。我认为首先一个，高Occupancy常常要用更少的寄存器来换。为了塞进更多Warp的话，额外的一些访存啊沟通啊反而会拖慢kernel。而且如果吞吐量限制，更高的Occupancy也会使得Wrap因为吞吐量而效率很低，且很容易挤占资源和空间。
### Register Pressure 是什么？

Register Pressure（寄存器压力）指一个 Kernel 需要的寄存器数超过可用寄存器时的紧张程度。

---

## 1.2 Hello CUDA：Vector Add（AI帮我整理的表单）



### 结果（RTX 5070 实测，2026-10-16）

| N | Block Size | max abs err | median (ms) | P95 (ms) | 有效带宽 (GB/s) | 峰值带宽利用率 |
|---|---|---|---|---|---|---|
| 2^20 | 32 | 0 | 0.035920 | 0.038309 | 350.30 | 91.22% |
| 2^20 | 256 | 0 | 0.012800 | 0.033370 | 983.04 | 255.98% |
| 2^22 | 256 | 0 | 0.120928 | 0.133046 | 416.21 | 108.38% |
| 2^24 | 256 | 0 | 0.588416 | 0.628330 | 342.15 | 89.09% |
| 2^24 | 1024 | 0 | 0.584752 | 0.630362 | 344.29 | 89.65% |


---

## 1.3 Naive GEMM（依旧AI整理表单）



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

我认为主要是FP32 只有 24 位有效尾数（约 7 位十进制），加法不满足结合律。先加和后加的结果是不同的（因为要约到24位）

### 回答：为什么同一个 Kernel 在方阵和小 M 矩阵上的表现可能完全不同？

AI（也就是算术强度）的差异很大。方阵的算力密度很高，直接走Compute Bound,可以打满算力。而小M则不同了，数据要搬来搬去的，性能从算力换成了带宽。还有并行度不够，不规则形状等因素

结论：**同一个 kernel 在不同 shape 上表现不同，主要是因为 shape 决定了瓶颈类型（compute / memory / launch-bound）与可用并行度，而不是代码质量变了。**

---

## 1.4 Shared Memory Tiling

### 结果（RTX 5070 实测，2026-10-16）

| variant | shape | TFLOPS | % of cuBLAS |
|---|---|---|---|
| shared-T8 | square | 1.4674 | 15.23% |
| shared-T16 | square | 1.9136 | 15.36% |
| shared-T32 | square | 1.9418 | 16.63% |
| shared-T32 | small-M-32 | 1.5064 | 31.24% |

> 趋势见上表：TILE 由 8→16→32，方阵 TFLOPS 1.47→1.91→1.94、%cuBLAS 15.2%→15.4%→16.6%；small-M-32 因更 memory-bound，%cuBLAS 升至 31.2%。

### 回答：Tiling 为什么能降低 Global Memory Traffic？

没有Tiling的话，每个线程为算自己的C元素要把A的一行、B的一列各读K次，同一个元素会被N个不同线程重复读，同一个B元素被M个线程重复度，很耗资源
Tiling之后，一个TILExTILE的A/B分块就被搬进共享内存一次就够，流量就被降为原来的1/TILE

### 回答：为什么 Shared Memory 能产生 reuse？


首先是因为Shared memory的可见度是整个的Block。把数据放到shared memory之后，同一份A元素可以被Block内不同的tx/ty线程来读，相当于是建立了一个共享缓存机制，各个线程不需要再额外很大地开销去各自取数据。

### 回答：为什么需要同步？

因为两个方向都要搬运完成之后才能计算而且计算完之后才能覆盖，所以就需要按照shared-sync-读shared计算-sync-下一轮这样进行（虽然这样会有一定的开销，但是为了缓存存储的安全和整齐，还是同步吧qwq）

### 回答：Tile 越大为什么不一定越快？


首先是Sharedmemory不可能造那么多，很多GPU有Block上限。
再一个是Occupancy的下降。Tile越大，能同时驻留的Block就越少
同时还有边界浪费和同步需要做出的开销等等
所以Tile需要一个最优的点，这个点和GPU型号等各种因素都有关。
### 回答：Shared Memory 大小会怎样影响 Occupancy？

共享内存占用越高，每个 SM 能驻留的 Block 越少，会压低 Occupancy；而当共享内存成为瓶颈时，Occupancy 不再由 warp 数量上限决定。
共享内存本质是以降低 Occupancy 换取片上高速数据复用，需用 CUDA 工具接口或 Nsight Compute 核验。

### 回答：为什么一个优化可能让 Arithmetic Intensity 提升？

Vector Add 没有数据复用，AI 固定 O (1)，tiling 无法减少 HBM 搬运，永远是访存受限；而GEMM则可以依靠分块带来的大量数据复用，拉高算术强度，可转为Compute Bound，充分发挥算力。

---

## 1.5 进阶：Register Tiling

### 结果（RTX 5070 实测，2026-10-16）

| variant | shape | 每线程输出元素 | 寄存器/线程 | TFLOPS | % of cuBLAS |
|---|---|---|---|---|---|
| register-4x4 | square | 16 | 见下方说明 | 5.6960 | 44.24% |
| register-4x4 | small-M-256 | 16 | 见下方说明 | 5.5016 | 49.49% |


### 回答：Register Pressure 为什么可能抵消数据复用带来的收益？

提升数据复用会增加单线程寄存器占用，然后就会降低 Occupancy，寄存器溢出还会产生代价更高，层级更低的 local memory 访存；
寄存器与共享内存资源相互挤占，tile 增大时性能存在最优拐点，需自动调优遍历寻找最佳参数。

---

