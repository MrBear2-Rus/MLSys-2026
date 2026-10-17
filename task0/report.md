# Task 0：前置知识

> 对应题面：`task6.md` —— 环境准备 + Benchmark 基础 + Roofline Model

## 1. 环境记录

### 1.1 四条命令的实测输出

本任务的开发（知识作答 + 写代码）在本机完成，GPU 实测在朋友的 RTX 5070（经 WSL2）完成，故环境由两台机器共同组成：

| 命令 | 开发 / 作答机（本机，Intel Arc 无 N 卡） | GPU 实测机（朋友 RTX 5070，经 WSL2） |
|---|---|---|
| `nvidia-smi` | `null` | `RTX 5070 Laptop GPU`，Driver 592.00，CUDA 13.1（SMI 报告） |
| `nvcc --version` | `null` | `12.8`（CUDA Toolkit 12.8，`-arch=sm_120`） |
| `gcc --version` | `gcc.exe (x86_64-posix-seh-rev0, MinGW-W64) 12.2.0` | `gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0` |
| `python --version` | `Python 3.13.14` | `Python 3.x`（`run_all.sh` 所建 `.venv`，torch+cu128 可用） |

### 1.2 系统与硬件信息

| 项目 | 开发 / 作答机（本机） | GPU 实测机（朋友，经 WSL2） |
|---|---|---|
| 操作系统 | Microsoft Windows 11 家庭版 中文版（10.0.26200 / Build 26200） | Linux 6.6.87.2-microsoft-standard-WSL2（Ubuntu 24.04） |
| GPU 型号 | Intel(R) Arc(TM) Pro 140T GPU (16GB)，驱动 `32.0.101.8801` | NVIDIA GeForce RTX 5070 Laptop GPU (8151MiB)，计算能力 `sm_120`（Blackwell） |
| CUDA Toolkit | `null`（无 NVIDIA 显卡） | `12.8`（`nvcc` 可编译 `.cu`，`-arch=sm_120`） |
| PyTorch | `null`（系统 Python 3.13 未装 torch） | torch（cu128 运行时 = CUDA 12.8 运行时，`torch.cuda.is_available()=True`） |

原始输出见 [`env/env_raw.txt`](env/env_raw.txt)（含两台机器的完整快照）。

### 1.3 环境与"为何有两台机器"的说明

- **本机（开发 / 作答机）**：只有 Intel Arc 核显、无 NVIDIA 驱动与 CUDA Toolkit，所以 `nvidia-smi` / `nvcc` / NVIDIA GPU / PyTorch 在本机上记为 `null`。本机的 `gcc` 12.2.0 与 `Python 3.13.14` 足以完成 Task 0 的知识作答、Task 4 的手算校验脚本，以及把 Task 1 / Task 2 的源码与 benchmark 脚本完整写好。
- **朋友的 RTX 5070（经 WSL2，GPU 实测机）**：为了拿到 Task 1（CUDA）/ Task 2（Triton）的真实跑分，借用了朋友的 NVIDIA RTX 5070（sm_120，8GB）机器，在 WSL2 + CUDA Toolkit 12.8 下执行 `task1/benchmark/build_and_run.sh` 与 `task2/triton/bench_all.py`，真实延迟 / TFLOPS 已填入 `task1/report.md` 与 `task2/report.md`。
- 因此：本任务的"环境"由两台机器共同组成——**知识作答与开发在本机，GPU 实测在朋友的 5070**。这种"无卡开发 + 借卡实测"的分工在招新题里是允许的（题目要求「缺失项写 `null` 并说明」，我们正是如实标注了本机的 `null`）。
- 原始输出见 [`env/env_raw.txt`](env/env_raw.txt)（含两台机器的完整快照）。

---

## 2. 环境相关问题

### 2.1 GPU Driver、CUDA Toolkit、PyTorch CUDA Runtime 分别是什么？它们之间有什么关系？

三者是一条自底向上的栈，越靠上越"软件化"，但**每一层都只能通过下一层访问硬件**：

| 层 | 实体 | 提供什么 |
|---|---|---|
| GPU Driver | 内核模块 + 用户态库 `libcuda.so`（Windows: `nvcuda.dll`） | 硬件初始化、显存管理、上下文、**CUDA Driver API**（`cuLaunchKernel` 等底层接口） |
| CUDA Toolkit | `nvcc`、头文件、`libcudart`、cuBLAS/cuDNN/Nsight 等 | **CUDA Runtime API**（`cudaMalloc` 等友好封装）+ 编译器 + 性能库 + 调试/性能工具 |
| PyTorch CUDA Runtime | PyTorch 自带/依赖的 `libcudart` 与预编译 kernel（`sm_XX` 机器码 + `libcublas` 等） | 给 Python 层提供 tensor 抽象，把算子派发到 GPU |

关系要点：

1. **驱动与 Toolkit 是两个独立的安装包**。驱动由 NVIDIA 显卡驱动包提供，Toolkit 单独下载安装。所以存在"驱动有、nvcc 没有"的状态，本机就是"驱动也没有"的状态。
2. **驱动是唯一真正和硬件对话的一层**。CUDA Runtime 内部最终还是调用 Driver API。驱动版本通常要求 **≥** Runtime/Toolkit 的编译要求（新版驱动向下兼容旧的 runtime，见 2.2 的解释）。
3. **PyTorch 的 wheel 已经把 CUDA Runtime 和算子库打包进去了**（例如 `torch 2.x+cu121`），所以 `pip install torch` 之后**不需要**单独装完整 Toolkit 也能跑前向推理/训练；只有要自己写 `.cu`、用 `nvcc` 编译时，才必须有完整 Toolkit。
4. 版本关系上：`PyTorch(cu121)` 要求驱动 ≥ 525.x 之类的最低版本；驱动太旧时 `torch.cuda.is_available()` 会返回 `False` 并给出 "driver version is insufficient" 的提示。

### 2.2 nvcc 是什么，为什么 `nvidia-smi` 可以正常执行，并不意味着 `nvcc` 一定存在？

- **`nvcc` 是 CUDA 的编译器驱动（compiler driver）**：它把 `.cu` 文件拆成 host 代码（交给 `gcc`/`cl.exe`）和 device 代码（编译成 PTX / `sm_XX` 的 SASS），再链接成 fatbin 塞进可执行文件。没有 `nvcc` 就不能编译一个 `.cu`。
- **`nvidia-smi` 属于驱动包，不属于 Toolkit**。它通过驱动暴露的 NVML 接口读取 GPU 的状态（温度、显存占用、进程），只要有**驱动 + NVIDIA 显卡**就能运行。
- 因此两者的依赖完全不同：
  - `nvidia-smi` 可用 ⇔ **驱动存在**；
  - `nvcc` 可用 ⇔ **CUDA Toolkit 被安装了**。
- 想写 `C = A + B` 的 CUDA kernel，仅有 `nvidia-smi` 是不够的：必须装 Toolkit（或用 conda 的 `cuda-toolkit`、或走 Docker/官方镜像）。

### 2.3 什么是动态链接库？

动态链接库（Linux `.so` / Windows `.dll` / macOS `.dylib`）是**在运行时才被加载并解析符号**的可执行代码集合。

- **机制**：编译链接阶段只在可执行文件里留下"我需要 `cudaMalloc` 这个符号"的记录（PLT/GOT 表项），不复制实现；程序启动（或 `dlopen`）时由动态链接器 `ld.so` 找到对应的 `.so`，把符号地址填进去。
- **与静态库对比**：静态链接会把实现直接拷贝进可执行文件（体积大、升级要重编译）；动态链接多份程序共享同一份库（体积小、可单独升级、可被多个版本共存）。
- **在 CUDA 场景里的意义**：`libcudart.so`、`libcublas.so` 都是动态库。驱动升级后，只要 ABI 兼容，就不需要重新编译 PyTorch；反过来，如果运行时找不到 `libcublas.so.12` 这样的库，就会报 `libcublas.so.12: cannot open shared object file`。
- **代价**：多了运行时查找开销和"版本地狱"（DLL Hell）——库版本不匹配是 CUDA 环境里最常见的一类报错。

### 2.4 `PATH` 和 `LD_LIBRARY_PATH` 分别有什么作用？

| 变量 | 谁在用 | 作用 | CUDA 场景 |
|---|---|---|---|
| `PATH` | shell / `execvp` | 决定「输入一个命令名时去哪里找可执行文件」 | 要把 `$CUDA_HOME/bin` 放进 `PATH`，否则 `nvcc: command not found` |
| `LD_LIBRARY_PATH` | 动态链接器 `ld.so` | 决定「加载 `.so` 时去哪些额外目录找」 | 要把 `$CUDA_HOME/lib64` 放进 `LD_LIBRARY_PATH`，否则运行时报找不到 `libcudart.so` |

补充两点：

- 二者**互不替代**：能跑 `nvcc` 不代表能加载 `libcudart`，反之亦然。所以实践里通常两个都配。
- `LD_LIBRARY_PATH` 是"临时覆盖"，优先级高于系统默认路径，容易和其他版本的库冲突。更干净的做法是用 `rpath`（链接时 `-Wl,-rpath`）或 `ldconfig` 写进系统配置；`LD_LIBRARY_PATH` 更适合临时排障。

### 2.5 为什么我们通常会使用 Python 虚拟环境？

1. **依赖隔离**：不同项目要 `torch 2.1 + CUDA 12.1` 和 `torch 2.6 + CUDA 12.4`，全局安装只能留一份，虚拟环境可以各留各的。
2. **可复现**：`requirements.txt` / `environment.yml` 锁版本后，换机器能重建出同样的环境；MLSys 的 benchmark 尤其在意"别人的数字能不能复现"。
3. **避免污染系统解释器**：Windows/Linux 自带的 Python 常被系统工具依赖，装坏了会牵连系统组件。
4. **可回滚/可丢弃**：环境坏了删掉目录重建，而不是重装系统 Python。
5. **Python 没有原生的依赖解析隔离机制**（不像 C++ 的 header-only 或无 Ruby 的 bundler），`pip` 装进哪个环境由解释器的 `sys.prefix` 决定，所以必须靠 venv/conda/uv 显式隔离。

---

## 3. 0.1 Benchmark 基础

> 一句话总结这一节：**"跑得快"不是一个数，而是一套语言**。同一段程序可以说"延迟 10 ms"，也可以说"吞吐 1.2 GB/s"，它们回答的是不同问题。

### Latency 是什么？

**延迟**：一个任务从开始到结束所经历的时间，面向「单次请求有多快」。
例：一次 kernel 执行 1.8 ms，就是 latency = 1.8 ms。它是分布，不是常数，所以通常报 median / P95。

### Throughput 是什么？

**吞吐**：单位时间内完成的任务量，面向「系统整体能处理多少」。
例：1200 samples/s、8000 tokens/s、14 GB/s。它和 latency 在系统里常常是权衡关系（增大 batch 提升吞吐、同时抬高单请求延迟）。

### Bandwidth 是什么？

**带宽**：单位时间内能搬运的数据量（GB/s）。
在 GPU 上有两个口径：
- **理论峰值带宽（peak BW）**：显存规格决定，如某卡 1008 GB/s；
- **实测有效带宽（effective BW）**：`实际搬运字节数 / 实际耗时`。
两者之比就是题目要的"理论显存带宽利用率"。做 Roofline 判断时用峰值，做性能归因时看有效值。

### FLOPs 和 FLOPS 有什么区别？

- **FLOPs**：浮点运算的**数量**（复数名词，s 是复数）。GEMM 需要 `2MNK` 次浮点运算，这是计算量，也可以指算法复杂度。
- **FLOPS**：每秒浮点运算**次数**（速率，是 FLOPS = floating-point operations per second）。说某卡 82 TFLOPS，是它的峰值算力。
- 二者关系：`TFLOPS = FLOPs / (t × 10^12)`，这正是题面给的 GEMM 公式。

### P50、P95、P99 是什么？

时延的**分位数**：把 N 次测量排序后，P50 是中位数（典型表现），P95 表示 95% 的请求都快于该值，P99 是尾部延迟。
- 只看 P50 会掩盖卡顿：在线推理更关心 P99，因为它对应 SLA 违约和用户感知的卡顿；
- 离线批处理更关心吞吐（总时间），分位数的重要性下降。

### Warmup 是什么？

在正式计时前先跑若干次（题面约定 10 次），目的是排除冷启动效应：
- GPU 时钟从空闲的低频升到高频需要时间；
- 首次调用要付页表建立 / 显存 slab 分配 / CUDA context 初始化成本；
- Triton/cuDNN 这类框架的 **JIT 编译与 autotune 缓存**发生在首次调用；
- 首次访问的 L2/HBM 是冷的，cache 里还没有数据。

不 warmup 会把"启动成本"算进"稳态性能"，得出偏低且不可复现的数字。

### 为什么一个实验通常不能只运行一次？

1. **测量噪声**：GPU 时钟、功耗墙、其他进程（桌面/浏览器）都会扰动结果；
2. **异步执行**：单次测到的可能是"提交耗时"而不是"执行耗时"（见下一条）；
3. **需要统计量**：题目要求 median 和 P95，样本量为 1 时无从谈起；
4. **需要判断稳定性**：只有多次测量才能看出离散程度（标准差大说明结论不可信）。
题目约定 warmup 10 次 + 正式测量 50 次，并保存**逐次原始数据**，就是为了让结论可复查。

### GPU 为什么是异步执行的？

- CPU 把 kernel launch 写进流的命令队列后**立刻返回**，实际执行由 GPU 的硬件调度器异步完成；
- 这样 CPU 才能继续准备下一步（launch 下一个 kernel、做 H2D 拷贝、跑 Python），实现 **CPU 与 GPU 重叠**；
- 如果每次 launch 都同步等待，几微秒的 kernel launch 会引入毫秒级的往返开销，流水线彻底断掉；
- 代价是：**"函数返回"不等于"计算完成"**，任何读取结果的代码（`.item()`、`.cpu()`、拷贝回主机）都会隐式同步。

### `torch.cuda.synchronize()` 有什么作用？

阻塞当前线程，直到**当前设备上所有已提交的工作（kernel、拷贝）全部完成**。

它的两个用途：
1. **保证正确性**：只有同步后才能安全读取结果；
2. **保证计时正确**：`t0 = time.time(); kernel(); t1 = time.time()` 测到的是"CPU 提交耗时"（可能只有几十微秒），必须 `kernel(); torch.cuda.synchronize(); t1 = time.time()` 才包含真实执行时间。

### CUDA Event 和 Python Timer 有什么区别？

| 维度 | Python Timer (`time.time()`) | CUDA Event (`cudaEventRecord`) |
|---|---|---|
| 计时位置 | CPU 侧挂钟 | GPU 流内的时间戳 |
| 是否异步感知 | 否，需要 `synchronize()` 兜底 | 是，天然记录 GPU 执行区间 |
| 精度 | 受 Python 调度抖动影响，μs~ms 级 | GPU 时钟级，通常更稳更准 |
| 是否打乱流水 | 强同步会破坏 GPU 流水 | 事件插入流中，不强制 CPU 等待 |
| 适用场景 | 端到端 wall clock（含 CPU 开销） | **纯 kernel / GPU 区间计时**（本题要求） |

本题约定用 **CUDA Event** 统计 GPU 时间，正是因为要"默认只统计 Kernel 执行，不包含内存申请、初始化和 H2D/D2H"；如果还想测 End-to-End，必须**另起一列**，两种时间不能混在一起。

---

## 4. 0.2 Roofline Model

> Roofline 的价值：**在动手测之前，先猜这个算子是 Compute Bound 还是 Memory Bound**，再用实验去验证自己的判断。

### Arithmetic Intensity 的单位是什么？它高或低分别意味着算子更偏向计算，还是更偏向搬运数据？

- 定义：`AI = FLOPs / Memory Traffic in Bytes`，单位是 **FLOP/Byte**（FLOP per byte）。
- **AI 高** → 每搬运一个字节能换来很多次计算 → 更偏向 **Compute Bound**（瓶颈在算力）；
- **AI 低** → 搬一个字节只换来很少的计算 → 更偏向 **Memory Bound**（瓶颈在显存带宽/搬运）。
典型的：GEMM（大 K）AI 高，Vector Add / GEMV / RMSNorm / Softmax 的 AI 都很低。

### 已知一个算子的 AI 后，怎样判断它更可能是 Compute Bound 还是 Memory Bound？这个判断会不会随着 GPU 型号变化？

判断需要一个"分水岭"，即 Roofline 的 **ridge point（拐点）**：

```text
ridge point = P_peak / BW_peak     (单位 FLOP/Byte)
```

- `AI > ridge` → 落在斜线以上的平顶区 → **Compute Bound**；
- `AI < ridge` → 落在斜线以下的斜坡区 → **Memory Bound**。

**会随 GPU 型号变化**，而且是双向变化：
- 一张卡算力更高但带宽没跟上，`ridge point` 上移，原来 Compute Bound 的算子可能变成 Memory Bound；
- Tensor Core 让 `P_peak`（FP16）远高于 FP32，因此同一个算子在"开 Tensor Core"和"关 Tensor Core"两种模式下可能落在不同区域。
所以结论必须写成"**在某张卡、某种精度下**是 Memory Bound"，不能脱离硬件谈。

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

### Roofline 给出的是性能上限而不是实测性能。哪些因素会让实测偏离预测？

按影响机理分四类：

1. **Cache / 复用让实测"变好"**：Roofline 的 traffic 按"必须从 HBM 搬"估算，但 L2/L1 命中、shared memory 复用会减少真实 HBM 流量。若 kernel 有复用一个 kernel 内反复访问的数据（如 GEMM 的 A/B 分块），有效 AI 会高于朴素估算，实测点会跑到 Roofline 线**上方**（不同口径）或更靠近上限。
2. **重复访存 / 非合并访问让实测"变差"**：未 coalesced 的访问会把 128 B 的 cache line 只用 4 B，实际搬运量翻几十倍；strided、broadcast 冲突同理。此时实测远低于上限。
3. **Kernel Launch Overhead**：小 shape / 小 kernel 时，launch 的固定开销（几微秒）和 CPU 提交时间占比显著，端到端根本达不到 kernel 的理论值。Task 1 里的"小 M 矩阵"和 Task 3 的 Decode 就属于这一类。
4. **硬件利用率达不到 100%**：occupancy 受限（寄存器压力、shared memory 占用）、指令发射瓶颈、非 matmul 指令占比、访存延迟未被隐藏（流水不足）、时钟/功耗墙降频，都会让实测落在上限下方。

**Roofline 是"上界，不是承诺"**：它假设完美重叠、无延迟、无额外指令，因此实测低于上限是常态，高于上限通常意味着你的 traffic 估算口径（含 cache 复用）与实际不同。

---

## 5. 小结：Task 0 想建立的直觉

在写任何 CUDA 之前，先能把一句话拆成可验证的问题：

```text
"这段程序跑了 10 ms。"
  ↓ 快在哪？
10 ms 花在：CPU 提交？Kernel 执行？H2D/D2H 拷贝？还是同步等待？
  ↓ 它的天花板是多少？
按 Roofline 算：这个算子的 AI 是多少，ridge point 在哪里，理论上限是多少 ms？
  ↓ 实测离上限多远？差在 cache、访存、launch 还是 occupancy？
```

后面的 Task 1（CUDA）和 Task 2（DSL）就是把这套"先测再优化"的流程真正跑一遍。
