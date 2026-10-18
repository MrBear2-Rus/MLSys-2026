# Task 3：一个 Token 的一生

> 对应题面 `task9.md`。不跑模型、不写 Kernel，是一次"带着系统问题读论文"的练习：先推公式 → 再读论文（看它减少了公式里哪一项）→ 最后把注意力从 FLOPs 转到数据搬运。

## 阅读清单
| 论文 | 我关注的点 |
|---|---|
| DeepSeek-V2 (2405.04434) | MLA 用模型结构换 KV Cache |
| KV Cache Survey sKis (2607.08057) | 系统视角三维 taxonomy |
| FlashAttention V1/V2 (2205.14135 / 2307.08691) | 重排数据搬运 |
| FlashAttention-3/4 (2407.08608 / 2603.05451) | 硬件协同设计 |
| NSA (2502.11089, Bonus) | 训得起来的稀疏注意力 |

## 3.1 Prefill & Decode
- 自回归：每步只产 1 token，拼回输入再前向，T 个 token 要 T 次前向（严格串行）。
- 阶段对比：
  | 阶段 | 输入 | 核心运算 | 特征 |
  |---|---|---|---|
  | Prefill | 整段 prompt (T) | 大 GEMM `T×d · d×4d` + `T×T` Attention | Compute Bound，MFU 40~60% |
  | Decode | 1 token + KV | GEMV `1×d · d×n` + `1×T` Attention | Memory Bound，SM 利用率 1~5% |
- 差别在 M 维：Prefill M=T（权重被 T 个 token 复用，AI 高）；Decode M=1（权重读一遍只用一次，AI≈1）。
- 一句话：**Prefill 是算力问题，Decode 是带宽问题。**
- 与 Task1 呼应：`M∈{1,32,256}` 的小 M GEMM 就是 Decode 的缩影——M=1 退化成 GEMV，AI≈1，瓶颈从算力切到带宽。

## 3.2 KV Cache
- 单 token 单层 K/V 为 `2·H_kv·D_h·S` 字节；全层 `2·L·H_kv·D_h·S`；再乘 B、T：
  `M_KV = 2 B L T H_kv D_h S` Bytes
- 验证 Llama-2-7B (L=32, H_kv=32, D_h=128, S=2)：单 token 层=16KB，100请求×10000token≈512GB（单卡 24GB 装不下）。
- 为何引入：无 Cache 每步重算全部历史 K/V（O(T²)）；有 Cache 只算新 token 的 QKV 再做 `1×T` 注意力，降到 O(T) 增量——**空间换计算**。
- 缺点：显存随 B×T 线性增长（并发硬上限）；Decode 每步读全量 Cache（AI≈1）；碎片/动态管理（PagedAttention 解决）；首 token 延迟与抢占代价；压缩/驱逐带来质量损失。
- MHA / GQA / MQA：KV Cache 正比于 `H_kv`，让多个 Q head 共享更少 KV head 即可等比例缩小。
  | 方案 | H_kv | Cache（相对 MHA） |
  |---|---|---|
  | MHA | H | 1× |
  | GQA | H/G | 1/G |
  | MQA | 1 | 1/H |
  | MLA | 压成 latent | 见下 |
  GQA 是质量/显存折中（Llama-3/Mistral 默认），还顺带缓解 Decode 带宽。

## 3.3 花式玩法
> 读法：先问"它减少了 `M_KV=2BLTH_kv D_h S` 里的哪一项"，再问代价。

**MLA (DeepSeek-V2)**：低秩联合压缩，Cache 只存低维 latent `c^KV`(d_c=512) + 解耦 RoPE 键(d_h^R=64)；Matrix Absorption 把上投影吸收进 W^Q/W^O，Decode 表现得像 MQA。每 token 层 Cache：等价 MHA 32768 元素 vs MLA 576 → 约 1.8%（论文称综合降 93.3%）。代价：需端到端训练、Kernel 更复杂。

**Survey sKis 三维**：
- 时间（Execution&Scheduling）：KV 调度 / 流水线重叠 / 硬件感知执行；
- 空间（Placement&Migration）：内存层级编排（offload/CXL）/ 计算设备编排（跨机传输）；
- 结构（Representation&Retention）：KVCC 压缩（量化/低秩/剪枝）/ KVRM 保留（分配复用/驱逐）。
- 保留管理重点：PagedAttention（分页减碎片+前缀共享）、Eviction（H2O/StreamingLLM/SnapKV，保留 sink+滑窗+按 attention 分数）。
- "哪些 KV 值得留"四类信号：①attention 累积分 ②近期性/位置 ③query 侧观察 ④层/head 差异化预算。

**NSA (Bonus)**：动态分层稀疏，三分支——Compressed（粗粒度扫全局）/ Selected（细粒度选 top-n block 全精度）/ Sliding（局部窗口）。硬件对齐：算术强度均衡+block 对齐 Tensor Core；GQA 组内共享选择；Triton 实现可流水；可端到端训练（无剪枝掉点）。稀疏的是 token 维度的 KV 参与集合，省 O(n²) 计算与 O(n) KV 读带宽。

## 3.4 Attention IO：FlashAttention
普通 Attention 把 `S=n×n`、`P=n×n` 中间矩阵落 HBM，长序列下 O(n²) 往返是最大瓶颈。
- **V1 省的是 HBM IO 不是 FLOPs**：Tiling + Online Softmax，把 S/P 留在 SRAM 增量处理，避免 O(n²) 的 HBM 写回/读回；FLOPs 不变（精确），显存降到 O(n)，实测 2~4×。
- **V2 解决"IO 少了但算力仍用不满"**：①并行度/work partitioning 不足（Q 切 64 行、warp 少、跨 warp 需 shared mem 同步）；②non-matmul FLOPs（rescale/mask/boundary）占比高，拖低 Tensor Core 利用率；③Occupancy 不足。V2 改 K/V 串行+Q 并行、减少跨 warp 通信、重排 online softmax 顺序 → 更接近纯 GEMM（约 50~70% 峰值）。
- **V3/V4**：V3(Hopper)=异步 + warp specialization + TMA + WGMMA + FP8；V4(Blackwell)=应对非对称 scaling（TC 翻倍但 SM 带宽/exp 单元没翻倍），用全异步 MMA、软件模拟 exp、tensor memory、2-CTA MMA、CuTe-DSL，B200 BF16 达 1613 TFLOPS(71%)。印证 Task0：ridge point 随硬件变。

**回答题面**：V1 省掉 `n×n` 中间矩阵的 HBM 往返（FLOPs 不变）；V2 解决并行度不足与 non-matmul 占比过高，让矩阵乘占比上升、更接近 GEMM 执行效率。

## 交付：连接 Task 1
Task1 的小 M GEMM（M=1→GEMV, AI≈1）就是 Decode 每步的缩影；Task3 所有优化都能解释成"减少 Decode 每步必须搬运的字节数"：GQA/MQA 砍 KV head 维、MLA 压成 latent、量化降 S、稀疏/驱逐只读重要 token、PagedAttention 减碎片、FlashAttention 在 Attention 内部重演同一件事。**一句话：Task1 教会"AI 低就必吃带宽"，Task3 展示了整个推理栈为对抗这条规律所做的全部努力。**
