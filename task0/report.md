# Task 0：前置知识


## 1. 环境记录

本任务的环境分两台机器记录，原始输出都在 `env/env_raw.txt`：
**开发/作答机**（Windows + Intel Arc，在它上面执行了题面要求的四条命令）和
**GPU 实测机**（朋友的 RTX 5070，Task 1 / Task 2 的实测数据来自这里）。



### 在朋友机子上跑的，脚本输出在同目录 `env/` 文件夹

这台机器上的四条命令同样记录如下：

| 命令 | 实测输出 |
|---|---|
| `nvidia-smi` | `NVIDIA-SMI 590.62   Driver Version: 592.00   CUDA Version: 13.1` |
| `nvcc --version` | `Cuda compilation tools, release 12.8`（CUDA Toolkit 12.8，编译统一用 `-arch=sm_120`） |
| `gcc --version` | `gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0` |
| `python --version` | 由 `run_all.sh` 在仓库内创建的 `.venv` 提供（配套 torch cu128） |

| 项目 | 值 |
|---|---|
| 操作系统 | Ubuntu 24.04 x86_64，内核 `6.6.87.2-microsoft-standard-WSL2` |
| GPU 型号 | NVIDIA GeForce RTX 5070 Laptop GPU，8151 MiB，算力 **sm_120（Blackwell）** |
| Driver | 592.00 |
| CUDA | 驱动侧最高 13.1（`nvidia-smi` 那一栏）；Toolkit / 编译用 12.8 |
| PyTorch | cu128 轮子（运行时 CUDA 12.8），`torch.cuda.is_available() = True` |

> 注：`nvidia-smi` 里的 `CUDA Version: 13.1` 是**驱动能支持到的最高版本**，不等于编译用的 Toolkit 版本，两者的区别见下面 2.1。
>
> 开发机没有 N 卡，所以它的 `nvidia-smi` / `nvcc` / NVIDIA GPU / PyTorch 按题目要求记为 `null` 并注明原因。

---

## 2. 环境相关问题

### 2.1 GPU Driver、CUDA Toolkit、PyTorch CUDA Runtime 分别是什么？它们之间有什么关系？

三者实际上构成了一个金字塔形的GPU软件技术栈，越往上层越接近软件，越往下层越接近硬件，每一层都只能通过下一层来访问更底层的东西

GPUDriver实际上是内核模块和用户态库，提供了硬件初始化，现存管理，上下文，CUDA API这些底层的接口，也就是GPU这个硬件的“驱动Driver"

CUDA Toolkit则作为中间套件，由nvcc,头文件，libcudart等等组成，它提供了一套CUDA Runtime API加编译器，性能库，调试/性能工具的工具

而PyTorch CUDA Runtime则是PyTorch自带和依赖的libcudart和预编译kernel,给Python层提供tensor抽象，同时把算子派发到GPU

关系（总结自AI）：

1. 驱动与 Toolkit 是两个独立的安装包。驱动由 NVIDIA 显卡驱动包提供，Toolkit 单独下载安装。所以存在"驱动有、nvcc 没有"的状态。（补：在跑题目7和8的时候就曾有没装nvcc的问题）
2. 驱动是唯一真正和硬件对话的一层。CUDA Runtime 内部最终还是调用 Driver API。驱动版本通常要求 ≥ Runtime/Toolkit 的编译要求（不过新版驱动向下兼容旧的 runtime，不赖）。
3. PyTorch 的 wheel 已经把 CUDA Runtime 和算子库打包进去了，所以 `pip install torch` 之后不需要单独装完整 Toolkit 也能跑前向推理/训练；只有要自己写 `.cu`、用 `nvcc` 编译时，才必须有完整 Toolkit。

### 2.2 nvcc 是什么，为什么 `nvidia-smi` 可以正常执行，并不意味着 `nvcc` 一定存在？

-nvcc是 CUDA 的编译器驱动（compiler driver）：它把 .cu 文件拆成 host 代码和 device 代码进行编译，再链接成 fatbin 塞进可执行文件。没有 nvcc 就不能编译一个 .cu。
-nvidia-smi属于驱动包，不属于 Toolkit。它通过驱动暴露的接口检查 GPU 的状态（温度、显存占用、进程），只要有驱动和英伟达的显卡就能运行。
- 因此两者的依赖完全不同。后者可用只能说明N卡存在，但是前者可用就说明CUDA Toolkit是已被安装的

### 2.3 什么是动态链接库？

动态链接库（Linux `.so` / Windows `.dll` / macOS `.dylib`）是**在运行时才被加载并解析符号**的可执行代码集合。（来自Deepseek）

- **机制**：编译链接阶段只在可执行文件里留下”我需要什么“的记录，也就是include，而不复制实现；程序启动时由动态链接器找到对应的 `.so`，把符号地址填进去。
- **对比静态**静态链接会把实现直接拷贝进可执行文件；而动态链接多份程序直接共享同一份库，体积大幅缩小的同时，还可以单独升级，非常的好用。但是有的时候会碰上版本不匹配的情况
- **在 CUDA 场景里的意义**：libcudart.so、libcublas.so 这些都是动态库。


### 2.4 `PATH` 和 `LD_LIBRARY_PATH` 分别有什么作用？

| 变量 | 谁在用 | 作用 | CUDA 场景 |
|---|---|---|---|
| `PATH` | shell / `execvp` | 决定「输入一个命令名时去哪里找可执行文件」 | 要把 `$CUDA_HOME/bin` 放进 `PATH`，否则 `nvcc: command not found` |
| `LD_LIBRARY_PATH` | 动态链接器 `ld.so` | 决定「加载 `.so` 时去哪些额外目录找」 | 要把 `$CUDA_HOME/lib64` 放进 `LD_LIBRARY_PATH`，否则运行时报找不到 `libcudart.so` |

（更改自AI给的表格）

### 2.5 为什么我们通常会使用 Python 虚拟环境？

1. 依赖隔离：不同项目要torch 2.1 + CUDA 12.1和torch 2.6 + CUDA 12.4，全局安装只能留一份，虚拟环境可以各留各的。
2. 可复现：requirements.txt/environment.yml锁版本后，换机器能重建出同样的环境；MLSys 的 benchmark 尤其在意"别人的数字能不能复现"。
3. 避免污染系统解释器：Windows/Linux 自带的 Python 常被系统工具依赖，装坏了会牵连系统组件。
4. 用完直接丢：环境坏了删掉目录重建，而不是重装系统 Python。
---

## 3. 0.1 Benchmark 基础

### Latency 是什么？
**延迟**：一个任务从开始到结束所经历的时间，面向「单次请求有多快」。
### Throughput 是什么？
**吞吐**：单位时间内完成的任务量，面向「系统整体能处理多少」。（也就是吞吐量吧）
### Bandwidth 是什么？
**带宽**：单位时间内能搬运的数据量（GB/s）。GPU上会存在有理论最大带宽和实际上能跑的带宽，两者由于物理限制是不同的。
### FLOPs 和 FLOPS 有什么区别？

- **FLOPs**：浮点运算的数量。
- **FLOPS**：每秒浮点运算次数
- 二者关系：`TFLOPS = FLOPs / (t × 10^12)`

### P50、P95、P99 是什么？

时延的**分位数**：把 N 次测量排序后，P50 是中位数（典型表现），P95 表示 95% 的请求都快于该值，P99 是尾部延迟。

### Warmup 是什么？

在正式计时前先跑若干次（题目里约定是10次），目的是排除冷启动效应：
- GPU 时钟从空闲的低频升到高频需要时间；
- 首次调用要付页表建立 / 显存 slab 分配 / CUDA context 初始化成本；
- Triton/cuDNN 这类框架的 **JIT 编译与 autotune 缓存**发生在首次调用；
- 首次访问的 L2/HBM 是冷的，cache 里还没有数据。

不warmup的话前几轮跑的数据不稳，就会导致鲁棒性的假性下降

### 为什么一个实验通常不能只运行一次？

1. **测量噪声**：GPU 时钟、功耗墙、其他进程（桌面/浏览器）都会扰动结果；
2. **异步执行**：单次测到的可能是"提交耗时"而不是"执行耗时"；
3. **需要判断稳定性**：只有多次测量才能看出离散程度（标准差大说明结论不可信）。

### GPU 为什么是异步执行的？

GPU是独立协处理器，CPU只负责把kernel,内存拷贝等等命令提交到队列和流之后就不管了，然后GPU按流调度执行

主要目的：隐藏延迟，提高吞吐，并发算的快

### `torch.cuda.synchronize()` 有什么作用？

阻塞当前线程，直到当前设备上所有已提交的工作（kernel、拷贝）全部完成。（欸，这不就是操作系统的阻塞态吗）

这样可以保证计算的正确性和时间足够

### CUDA Event 和 Python Timer 有什么区别？

| 维度 | Python Timer (`time.time()`) | CUDA Event (`cudaEventRecord`) |
|---|---|---|
| 计时位置 | CPU 侧挂钟 | GPU 流内的时间戳 |
| 是否异步感知 | 否，需要 `synchronize()` 兜底 | 是，天然记录 GPU 执行区间 |
| 精度 | 受 Python 调度抖动影响，μs~ms 级 | GPU 时钟级，通常更稳更准 |
| 是否打乱流水 | 强同步会破坏 GPU 流水 | 事件插入流中，不强制 CPU 等待 |
| 适用场景 | 端到端 wall clock（含 CPU 开销） | **纯 kernel / GPU 区间计时**（本题要求） |


本表修改自AI。我的体会是，Python Timer更像是在GPU出现前为CPU定制的用于OS式的ARM式的定时器结构，有很浓厚的CPU味道，而CUDA Event则是一种为GPU专门定制的事件中断系统

---

## 4. 0.2 Roofline Model


### Arithmetic Intensity 的单位是什么？它高或低分别意味着算子更偏向计算，还是更偏向搬运数据？

- 定义：`AI = FLOPs / Memory Traffic in Bytes`，单位是 **FLOP/Byte**（FLOP per byte）。（FLOP每单位比特）
- **AI 高** ，每搬运一个字节能换来很多次计算 ，更偏向 **Compute Bound**（算力是瓶颈）；
- **AI 低**，搬一个字节只换来很少的计算，更偏向 **Memory Bound**（搬运和带宽是瓶颈）。

### 已知一个算子的 AI 后，怎样判断它更可能是 Compute Bound 还是 Memory Bound？这个判断会不会随着 GPU 型号变化？

判断需要一个"分水岭"，即 Roofline 的 **ridge point（拐点）**：
```text
ridge point = P_peak / BW_peak     (单位 FLOP/Byte)
```
- `AI > ridge` → 落在斜线以上的平顶区 → **Compute Bound**；
- `AI < ridge` → 落在斜线以下的斜坡区 → **Memory Bound**。
（资料来自AI）


改参数同时也会跟着GPU的不同来变化。比如卡的算力超强，那带宽可能就会成为瓶颈，所以这样对算子的判断不能脱离硬件来谈

### 对 FP32 Vector Add `C = A + B`，计算 FLOPs、最少 Memory Traffic 和 AI，并判断瓶颈

按题面假设"每个元素读取一次 A、一次 B，写回一次 C"：

| 量 | 计算 | 值 |
|---|---|---|
| FLOPs | 每个元素 1 次加法 → `N` | `N` FLOPs |
| Memory Traffic | `4N + 4N + 4N` | `12N` Bytes |
| AI | `N / 12N` | **`1/12 ≈ 0.083` FLOP/Byte** |

（如果把 `a+b` 记为 2 个 FLOP 的常见口径，则 AI = `1/6 ≈ 0.167`，量级不变。）

结论：**必然属于 Memory Bound**。以题干默认的带宽公式 `12N / t` 反推，
例如 N = 2^24、FP32：读 A/B + 写 C = 12 × 16777216 B ≈ 201 MB，
即使跑满 1000 GB/s，也只要约 0.2 ms，而这期间只做了 1.7e7 次加法（约 0.0002% 的算力）。
它离任何现代 GPU 的 ridge point（通常 10~100 FLOP/Byte）都差两个数量级，
所以**优化方向是省带宽，不是省计算**——这也是后面"算子融合能提速、而减少算术强度无关紧要"的根本原因。

（本题的严谨计算来自AI给出。但是我认为结论其实挺明显的，一次计算的算力只要N，运力却要12N，所以几乎可以肯定是Memory Bound）
### Roofline 给出的是性能上限而不是实测性能。哪些因素会让实测偏离预测？

按影响机理分四类：
Cache复用：真实HBM流量少,更近上限。
非合并/重复访存：搬运放大,实测远低。
Launch开销：小shape/小kernel端到端不达理论。
利用率不足：occupancy/指令/延迟/降频→实测低于上限。


