# Task 4：模型装不下怎么办 —— 量化与分布式推理

> 对应题面 `task10.md`。不跑模型、不写框架。回答三件事：INT8/INT4 理论省多少显存？PTQ 不重训为何还要校准数据？DP/TP/PP 各拆什么、通信代价？

## 4.1 浮点数到低比特
- 权重显存下界 `M = P·b/8`：FP32→FP16/BF16 省 1/2，→INT8 省 1/4，→INT4 省 1/8。
- "理论"二字重要：真实还需存 Scale/Zero-point、KV Cache、Activation、Workspace，所以"权重省 4×"≠"显存省 4×"≠"延迟 1/4"。
- 量化：`q = clip(round(x/s)+z, qmin, qmax)`，`x̂ = s(q−z)`。对称量化 z=0（权重），非对称 z≠0（ReLU 后激活）。
- **PTQ vs QAT**：PTQ 训练后直接量化、只需几百条校准数据、8bit 几乎无损、分钟级；QAT 训练中插伪量化、需重训。**PTQ 不重训 ≠ 不看数据**：scale 由数据动态范围决定，激活每 token 都变，必须用校准集估典型范围、定 per-tensor/per-channel scale。
- **Weight-only(W4A16) vs W8A8**：
  | 方案 | 权重 | 激活 | 省什么 |
  |---|---|---|---|
  | Weight-only | INT4 | FP16 | 权重存储+带宽（计算仍 FP16，Decode 主流） |
  | W8A8 | INT8 | INT8 | 存储+带宽+整数 TC 提速 |
- 粒度 per-tensor/per-channel/per-group：越细越保精度，但元数据越多、Kernel 越复杂（GPTQ/AWQ 用 group=128 取平衡）。
- **Activation Outlier**：少数固定 channel 幅值大 20~100×，per-tensor 量化会被 outlier 拉大 s、普通值挤成 0——这是"激活比权重难量化"的根因，也是 SmoothQuant 要解的。

**手算一**（`x=[-1.0,-0.4,0.2,0.9]`，`s=max|x|/qmax`，z=0）：
| | INT8 (s=1/127) | INT4 (s=1/7) |
|---|---|---|
| x=-1.0 | q=-127, x̂=-1.000 (0%) | q=-7, x̂=-1.000 (0%) |
| x=-0.4 | q=-51, x̂=-0.402 (0.39%) | q=-3, x̂=-0.429 (7.14%) |
| x=0.2 | q=25, x̂=0.197 (1.57%) | q=1, x̂=0.143 (**28.57%**) |
| x=0.9 | q=114, x̂=0.898 (0.26%) | q=6, x̂=0.857 (4.76%) |
> 由 `scripts/quant_check.py` 复算：`python3 task4/scripts/quant_check.py`

**哪些数最先失精度？绝对值最小的非零元素。** 同 tensor 内绝对误差上界 s/2 相同，相对误差≈(s/2)/|x| 与 |x| 成反比；|x|=max|x| 恰好映射到 qmax 被精确表示。s 放大 18 倍(127/7)，相对误差也放大 18 倍，首当其冲是小值。工程含义：用 per-channel/per-group 缩小局部 s；outlier 拉开 range 正是 SmoothQuant"把难度搬给权重"的动机。

**模型变小为何不一定变快**：①硬件有无对应低比特 TC；②Kernel 是否真走低比特路径；③dequant 开销；④瓶颈在哪（权重带宽→近线性，激活/KV 带宽→无用，通信→只减传输量，算力→仅 W8A8 有用）；⑤小 batch 被 launch 开销主导。

## 4.2 三种 PTQ 思路
- **SmoothQuant (W8A8)**：激活比权重难量化（权重静态平滑、激活有 channel-wise outlier，且 Y=XW 一边 range 大整型 GEMM 就用不了）。做法：per-channel 缩放 `Y=(X·diag(s)⁻¹)(diag(s)·W)` 数学等价，把激活 outlier 难度迁到权重；α≈0.5 让两边同好量化，从而用上整数 TC 的 W8A8 GEMM。校准数据观测 per-channel 激活幅值、定 s_j。
- **GPTQ (Weight-only)**：只量化权重(3~4bit)，激活 FP16，收益来自权重存储/带宽（Decode 最痛）。关键是按"对输出的敏感度"分配误差：`argmin‖WX−ŴX‖²`，Hessian≈2XXᵀ 决定哪些方向重要、该量更准。逐层处理（避免误差累积）、二阶信息（决定误差往哪放）、one-shot 批量更新（Cholesky，免重训免反向）让它可行。
- **AWQ (Weight-only)**：Salient Weight = 对输出影响最大的 ~0.1~1% 通道（对应激活幅值大）。保护少量通道即可拿回大部分精度。用校准集 per-channel 激活幅值排序重要性，对重要通道 per-channel 缩放(s>1)使 INT4 更易表示、对应激活除以 s 保持等价；s 网格搜索最小化层输出误差。**保持统一 INT4 布局**，Kernel 无需特殊分支 → 硬件友好。

| | SmoothQuant | GPTQ | AWQ |
|---|---|---|---|
| 类型 | W8A8 | Weight-only | Weight-only |
| 机制 | 等价缩放搬 outlier 到权重 | 二阶信息按输出敏感度分配误差 | 激活统计挑 salient 通道缩放保护 |
| 省什么 | 存储+带宽+整数 GEMM 提速 | 权重存储/带宽 | 权重存储/带宽（实现最简单） |
| 场景 | 提吞吐/降算力 | 单卡显存受限要高质量 4bit | 端侧/边缘，泛化好 |

## 4.3 DP、TP 与 PP
| 并行 | 切什么 | 每卡持有 | 通信 | 解决装不下 |
|---|---|---|---|---|
| DP | 数据/请求 | 完整模型 | 推理几乎无（训练梯度 All-Reduce） | ❌ |
| TP | 层内大矩阵 | 权重分片 | 每层 1~2 次 All-Reduce | ✅ |
| PP | 层（按深度） | 部分层 | Stage 间传 Activation | ✅ |

- **DP**：每卡完整模型，提吞吐；但单份模型仍要装进一张卡，KV Cache 还要各存一份。
- **TP**：Column Parallel（沿输出维切，GeLU 前无通信）+ Row Parallel（沿输入维切，部分和需 All-Reduce）。Transformer 里 QKV 用 Column（按 head）、输出投影用 Row。通信收敛到每层 1~2 次 All-Reduce，需 NVLink 机内。两卡 MLP 切分图见 `figures/mlp_tp_2gpu.svg`。
- **Collective**：All-Reduce（各卡同形张量→每卡得总和）、All-Gather（各卡分片→每卡得完整）、Reduce-Scatter（→求和的分片）。All-Reduce = Reduce-Scatter + All-Gather；NCCL 是标准实现。
- **PP**：按层切，Stage 间传激活，通信量比 TP 小、适合跨机；缺点 Pipeline Bubble（micro-batch/1F1B 缓解，但影响延迟）。
- 三者不同维度切分互不干扰，常组合（DP+TP+PP，如 Megatron：机内 TP、跨机 PP、最外层 DP）。

## 总结
- **量化**：32bit→4~8bit 理论省 4~8×，实际取决于瓶颈是权重/激活带宽还是算力，以及硬件有无低比特路径（Roofline 仍适用：量化本质降 Memory Traffic 推高 AI）。
- **PTQ 不重训但必须看数据**：scale 由分布定，激活每 token 在变。
- **分布式**：DP 不动模型、TP 切层内矩阵（机内高带宽）、PP 切层（跨机少通信）；三者不同维度可自由组合。
