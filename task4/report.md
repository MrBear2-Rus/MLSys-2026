# Task 4：模型装不下怎么办 —— 量化与分布式推理

> 对应题面 `task10.md`。不跑模型、不写框架。回答三件事：INT8/INT4 理论省多少显存？PTQ 不重训为何还要校准数据？DP/TP/PP 各拆什么、通信代价？

## 4.1 浮点数到低比特

- PTQ vs QAT：PTQ是训练完了，再统一把权重什么的换成低精度，只要几百条校准数据，8bit几乎不掉点就可搞定
QAT是在训练的时候就假装“已经量化了”，让模型自己适应刻度，通常还要重训。


- **Weight-only(W4A16) vs W8A8**：
 Weight-only是权重存INT4,激活还是FP16,省的是权重存储和带宽，计算也仍然走FP16.（Decode阶段卡在般权重，所以这是主流）
 W8A8是权重、激活都是INT8,存储带宽什么的都占。
- 粒度 per-tensor/per-channel/per-group：越细越保精度，但元数据越多、Kernel 越复杂（GPTQ/AWQ 用 group=128 取平衡）。
- **Activation Outlier**：少数固定 channel 幅值大 20~100×，per-tensor 量化会被 outlier 拉大 s、普通值挤成 0——这是"激活比权重难量化"的根因，也是 SmoothQuant 要解的。

**手算一**（`x=[-1.0,-0.4,0.2,0.9]`，`s=max|x|/qmax`，z=0）：
| | INT8 (s=1/127) | INT4 (s=1/7) |
|---|---|---|
| x=-1.0 | q=-127, x̂=-1.000 (0%) | q=-7, x̂=-1.000 (0%) |
| x=-0.4 | q=-51, x̂=-0.402 (0.39%) | q=-3, x̂=-0.429 (7.14%) |
| x=0.2 | q=25, x̂=0.197 (1.57%) | q=1, x̂=0.143 (**28.57%**) |
| x=0.9 | q=114, x̂=0.898 (0.26%) | q=6, x̂=0.857 (4.76%) |
> 由 `scripts/quant_check.py` 复算：`python3 task4/scripts/quant_check.py`（AI）

（AI搞得表格，这个表格AI做的好像还挺好的，我都是给数据让它帮我做md的表）

**哪些数最先失精度？**量化最先坏的是小数字。半格优化（四舍五入）的相对误差来说，数字越小就越拉。

## 4.2 三种 PTQ 思路
- **SmoothQuant (W8A8)**：激活比权重难量化（权重静态平滑、激活有 channel-wise outlier，且 Y=XW 一边 range 大整型 GEMM 就用不了）。做法：per-channel 缩放 `Y=(X·diag(s)⁻¹)(diag(s)·W)` 数学等价，把激活 outlier 难度迁到权重；α≈0.5 让两边同好量化，从而用上整数 TC 的 W8A8 GEMM。校准数据观测 per-channel 激活幅值、定 s_j。
- **GPTQ (Weight-only)**：只量化权重(3~4bit)，激活 FP16，收益来自权重存储/带宽（Decode 最痛）。关键是按"对输出的敏感度"分配误差：`argmin‖WX−ŴX‖²`，Hessian≈2XXᵀ 决定哪些方向重要、该量更准。逐层处理（避免误差累积）、二阶信息（决定误差往哪放）、one-shot 批量更新（Cholesky，免重训免反向）让它可行。
- **AWQ (Weight-only)**：Salient Weight = 对输出影响最大的 ~0.1~1% 通道（对应激活幅值大）。保护少量通道即可拿回大部分精度。用校准集 per-channel 激活幅值排序重要性，对重要通道 per-channel 缩放(s>1)使 INT4 更易表示、对应激活除以 s 保持等价；s 网格搜索最小化层输出误差。**保持统一 INT4 布局**，Kernel 无需特殊分支 → 硬件友好。

1.SmoothQuant的思想是，权重是静态的，比较平滑，而激活则每个token都变。所以数学等价缩放，把激活的ontlier难度迁移一部分到权重上，让两边都好量化
2.GPTQ则是权重只留个3-4bit,但是误差不要乱丢，要按“对输出多敏感”来分配
它只量化权重，激活还是FP16,这样权重存储和带宽就会变快了
3.AWQ则是抓重点思想，Salient Weight是对输出影响最大的少数通道通常对应激活幅值大的那些。
## 4.3 DP、TP 与 PP
| 并行 | 切什么 | 每卡持有 | 通信 | 解决装不下 |
|---|---|---|---|---|
| DP | 数据/请求 | 完整模型 | 推理几乎无（训练梯度 All-Reduce） | ❌ |
| TP | 层内大矩阵 | 权重分片 | 每层 1~2 次 All-Reduce | ✅ |
| PP | 层（按深度） | 部分层 | Stage 间传 Activation | ✅ |

- **DP**：每卡完整模型，提吞吐；但单份模型仍要装进一张卡，KV Cache 还要各存一份。
- **TP**：Column Parallel（沿输出维切，GeLU 前无通信）+ Row Parallel（沿输入维切，部分和需 All-Reduce）。Transformer 里 QKV 用 Column（按 head）、输出投影用 Row。通信收敛到每层 1~2 次 All-Reduce，需 NVLink 机内。例（2 卡 MLP 4H→H→H→4H）：第一层 Column 沿输出维切（各卡算 H/2，无通信），第二层 Row 沿输入维切（各卡算 H/2 后 All-Reduce 求和），每层净 1 次 All-Reduce。

（AI的，还在看）