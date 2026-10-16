# Task 3：一个 Token 的一生

> 对应题面：`task9.md`（3.1 Prefill & Decode / 3.2 KV Cache / 3.3 花式玩法 / 3.4 Attention IO）

本题**不运行模型、不实现 Kernel**，是一次"带着系统问题读论文"的练习。本报告的结构：
先自己推公式 → 再读论文（看它们减少了公式里的哪一项）→ 最后把注意力从 FLOPs 转到数据搬运。

阅读的论文：

| 论文 | 定位 | 我关注的部分 |
|---|---|---|
| DeepSeek-V2 (arXiv 2405.04434) | 用**模型结构**换 KV Cache | MLA（Multi-head Latent Attention）、decoupled RoPE、matrix absorption |
| System-Aware KV Cache Optimization Survey (arXiv 2607.08057, ACL 2026 Findings) | **系统视角**的 KV Cache 优化地图 | 三维 taxonomy；Pruning / Eviction / 量化 / 低秩 |
| FlashAttention V1 / V2 (2205.14135 / 2307.08691) | 重新安排**数据搬运** | HBM IO、online softmax、并行度与 work partitioning |
| FlashAttention-3 / 4 (2407.08608 / 2603.05451) | 硬件协同设计 | asynchrony、warp specialization、Blackwell 的非对称 scaling |
| NSA (2502.11089) 【Bonus】 | 把稀疏做成**训得起来**的 | compressed / selected / sliding 三分支 + 硬件对齐 |

---

## 3.1 Prefill & Decode

### 什么是 Autoregressive Generation？

自回归生成：模型每次只产生一个 token，然后把**自己刚产生的 token** 拼回输入序列，继续产生下一个，直到结束。

```text
prompt: [t1 t2 ... tn] ──> forward ──> t(n+1)
[t1 ... tn t(n+1)]      ──> forward ──> t(n+2)
[t1 ... tn t(n+1) t(n+2)] ──> forward ──> t(n+3)
...                                            (串行，T 个 token 就要 T 次 forward)
```

关键性质：**计算是严格串行的、每步只推进一个 token**。这是所有推理系统优化都要面对的约束——LLM 推理不是"一次大 GEMM"，而是"一次大 GEMM + 很多次小运算"。

### Prefill 做什么？Decode 做什么？

| 阶段 | 输入 | 一次 forward 产生 | 核心运算 | 特征 |
|---|---|---|---|---|
| **Prefill** | 整段 prompt（长度 T） | 全部 T 个位置的 hidden state + **第一个**新 token | 大 GEMM（矩阵乘矩阵，`T × d` 乘 `d × 4d` 等），Attention 是 `T × T` | 大 batch 维度、**算力密集**（Compute Bound） |
| **Decode** | 上一步的 1 个 token + KV Cache | 1 个 token | GEMM 退化成 **GEMV**（`1 × d` 乘 `d × n`），Attention 是 `1 × T` | 每步只算 1 行，**访存密集**（Memory Bound） |

两者的量化差别（以 Llama-2-7B、FP16、T=4096 为例，便于建立直觉）：

- Prefill：一次要算 `4096 × (7B) × 2` FLOPs 量级，算力能被喂满，**秒级**完成数千 token；
- Decode：每次只算 1 个 token，却要把**整个模型的权重（13 GB）+ 全部 KV Cache** 从 HBM 读一遍，所以**每步的耗时基本上等于"读一遍权重和 Cache"的耗时**。这就是为什么 `nvidia-smi` 里 Decode 阶段显存带宽打满、而 SM 利用率很低。

### 为什么 Prefill 和 Decode 的矩阵 Shape 差别很大？

用 GEMM 的维度看最清楚。设 hidden = d，中间维度 = 4d：

| | 输入矩阵 | 权重 | 输出 | 计算量占比 |
|---|---|---|---|---|
| Prefill | `T × d`（T 很大） | `d × 4d` | `T × 4d` | 一次处理 T 个 token，M = T |
| Decode | `1 × d`（M = 1） | `d × 4d` | `1 × 4d` | 一次只处理 1 个 token，M = 1 |

**差别在于 M 维度：Prefill 的 M = T（几千），Decode 的 M = 1。**

- M 大 → 权重分块后能被多个 token 复用 → 算术强度高 → 打满算力；
- M = 1 → 权重读进来只用一次 → 算术强度 ≈ 1 → 纯粹被带宽限制。

Attention 侧的差别同理：Prefill 是 `Q(T×d) · K^T(d×T)` → `T×T`，是"矩阵乘矩阵"；Decode 是 `Q(1×d) · K^T(d×T)` → `1×T`，是"向量乘矩阵"，并且还要**把整条 KV Cache 读一遍**。

### 为什么 Prefill 往往更容易利用 GPU Compute？

1. **有大量可并行的 token**：T 个位置之间互相独立（causal mask 之外），可以铺满几千个 block，Occupancy 高；
2. **算术强度高**：M = T 让同一个权重分块被 T 个 token 复用，`AI = FLOPs / Bytes` 远高于 ridge point → Compute Bound，能靠 Tensor Core 打满算力；
3. **shape 大且整齐**：大矩阵乘法正是 GPU 最擅长、cuBLAS/CUTLASS 优化最充分的形态。
   实际测量中 Prefill 的 MFU（Model FLOPs Utilization）可以到 40%~60%，而 Decode 常常只有 1%~5%。

### 为什么 Decode 往往更加 Memory Sensitive？

1. **M = 1 导致权重无法复用**：每算 1 个 token 就要把全部权重从 HBM 搬到 SM，`AI ≈ 1`（每读 2 Bytes 权重只换来约 2 FLOPs），远低于 ridge point → **Memory Bound**；
2. **KV Cache 每步都要读全量**：第 t 步要读 `2 × L × H_kv × D_h × t` 的数据，随上下文线性增长；
3. **不能靠"多算一点"掩盖**：算力有大量空闲，但带宽已经饱和，所以再多的 SM 也没用；
4. **因此优化方向完全不同**：Decode 的优化目标是**减少搬运量**（GQA/MQA、MLA、量化、KV Cache 稀疏化/驱逐、算子融合、连续批处理），而不是"提高算力利用率"。

一句话：**Prefill 是算力问题，Decode 是带宽问题。**

### 回头看看 Task 1：小 M GEMM 和 Decode 有没有关系？

**有，而且是同一件事。**

Task 1 的 `M ∈ {1, 32, 256}, N = K = 4096` 就是 Decode 的缩影：

- `M = 1` 时，`C(1,N) = A(1,K) · B(K,N)` 退化成 GEMV：读 B 的 `K×N` 个元素，只做 `K×N` 次乘加 → `AI ≈ 1`；
- 在 Task 1 的报告里我们已经预判：小 M shape 的 TFLOPS 会**远低于**方阵，因为瓶颈从算力变成了带宽——这正是 Decode 每步的处境；
- 区别只是 Decode 里"每一层每一次 decode"都要重复这个过程，并且还要额外读 KV Cache，所以 Decode 的带宽压力比单个 GEMV 更大。

---

## 3.2 KV Cache

### 推导：`M_KV = 2 B L T H_kv D_h S`

对每一层的 Attention：`Q = XW_Q, K = XW_K, V = XW_V`。

- 一个 token 在该层产生的 K：`H_kv` 个 head × 每个 head `D_h` 个元素 × 每元素 `S` 字节 → `H_kv · D_h · S` 字节；
- V 同理，再乘 2（K 与 V 各存一份）→ `2 H_kv D_h S` 字节/token/层；
- 乘层数 `L` → `2 L H_kv D_h S` 字节/token；
- 乘序列长度 `T`、批大小 `B` → 

$$
\boxed{M_{KV} = 2\,B\,L\,T\,H_{kv}\,D_h\,S\ \text{Bytes}}
$$

每一项的物理含义：

| 符号 | 含义 | 谁在决定它 |
|---|---|---|
| 2 | K 和 V 各一份 | 算法 |
| B | batch / 并发请求数 | 系统（连续批处理决定） |
| L | 层数 | 模型 |
| T | 已生成的上下文长度 | 请求（上下文窗口决定上界） |
| H_kv | **KV Head 数** | **模型结构（MHA / GQA / MQA / MLA 的分水岭）** |
| D_h | 每个 head 的维度 | 模型结构 |
| S | 每元素字节数 | 精度（FP16 = 2，INT8 = 1，FP8 = 1，INT4 = 0.5） |

**验证题面给出的 Llama-2-7B 例子**（L = 32, H_kv = 32, D_h = 128, S = 2）：

- 单 token 单层：`2 × 32 × 128 × 2 = 16384 B = 16 KB` ✔
- 单 token 全部层：`32 × 16 KB = 512 KB` ✔
- 100 请求 × 10000 token：`100 × 10000 × 512 KB = 512,000,000 KB ≈ 512 GB` ✔（单卡 RTX 4090 只有 24 GB，装不下）

这也说明：**KV Cache 的规模是"B × T"的双重放大**——长上下文 + 高并发会把它从"空间换计算的优化"变成"限制并发的主要瓶颈"。

### 为什么引入 KV Cache？

因为没有 Cache 时，每生成一个 token 都要**为全部历史 token 重新计算 K 和 V**：

- 无 Cache：第 t 步要算 t 个 token 的 K/V → 总计算量 `O(T²)`（且大部分是重复劳动）；
- 有 Cache：K/V 只算一次并保存，第 t 步只算新 token 的 Q/K/V（1 个 token），再做 `1 × T` 的注意力 → 计算量从 `O(T²)` 降到 `O(T)` 每步，总共 `O(T²)` 变 `O(T)` 的增量计算。

即：**KV Cache 是"空间换计算"。**

### KV Cache 有什么缺点？

1. **显存占用随 B×T 线性增长**，成为并发的硬上限（上例 512 GB）；
2. **Decode 变成带宽瓶颈**：每步都要读全量 Cache，`AI ≈ 1`；
3. **碎片化与动态管理成本**：请求长度不一、不断增长 → 显存碎片、分配/回收开销（这正是 **PagedAttention** 要解决的问题）；
4. **首 token 延迟与抢占代价**：Cache 不足时需要换出/重算（recompute）或抢占，影响 SLO（尤其是尾延迟）；
5. **精度损失**：压缩/量化/驱逐 Cache 都会带来质量下降（Survey 里的 O5）。

### 比较 MHA / GQA / MQA：为什么 GQA / MQA 可以降低 KV Cache？

设 Query Head 数 `H`，KV Head 数 `H_kv`：

| 方案 | H_kv | KV Cache（相对 MHA） | 说明 |
|---|---|---|---|
| **MHA**（Multi-Head Attention） | `H_kv = H` | 1×（基准） | 每个 Q head 有自己的 K/V head |
| **GQA**（Grouped-Query Attention） | `H_kv = H / G` | **1/G** | G 个 Q head 共享一组 K/V；G 可以取 4、8 等 |
| **MQA**（Multi-Query Attention） | `H_kv = 1` | **1/H** | 所有 Q head 共用同一组 K/V |
| **MLA**（DeepSeek-V2） | 压缩成潜变量 | 见 3.3 | 从模型结构上把 K/V 压到低维 latent |

原理一句话：**KV Cache 的大小正比于 `H_kv`，所以让多个 Query Head 共享更少的 KV Head，就能等比例减小 Cache。**

- MQA 把 `H_kv` 压到 1，节省最多，但因为所有 head 共享同一份 K/V，**模型表达能力下降明显**（质量损失、训练不稳定）；
- GQA 是折中：牺牲少量质量，换取 4~8 倍的 Cache 缩减，已成为主流 LLM 的默认设计（Llama-2-70B / Llama-3 / Mistral 等）；
- 工程上 GQA 还有个额外好处：**Decode 时读的 KV 更少**，直接缓解带宽瓶颈；同时 Attention Kernel 里 K/V 的读取天然被 G 个 head 复用，算术强度更高。

---

## 3.3 KV Cache 的花式玩法

> 读法：先问"它在减少公式 `2BLTH_kv D_h S` 里的哪一项？"，再问"代价是什么？"

### 3.3.1 DeepSeek-V2 / MLA：把 K/V 压成低维潜变量

**它解决什么问题？**
MHA 的 KV Cache 太大（每 token 每层 `2 H D_h` 个元素），而简单地砍 head（MQA）会掉点。要在**不损失模型能力**的前提下大幅降低 Cache 与 Decode 访存。

**核心办法（MLA）**

1. **低秩联合压缩**：不再独立缓存 K 和 V，而是把输入投影成一个**低维潜向量** `c_t^{KV}`（维度 `d_c`，远小于 `2 n_h d_h`），K 和 V 都在需要时由 `c^{KV}` 上投影得到：
   $$
   c_t^{KV} = W^{DKV} h_t,\qquad k_t = W^{UK} c_t^{KV},\qquad v_t = W^{UV} c_t^{KV}
   $$
   **Cache 里只存 `c^{KV}`**，它的维度是 `d_c` 而不是 `2 n_h d_h`。
2. **Decoupled RoPE**：RoPE 是位置相关的、不能和"压缩再上投影"随意结合（否则位置信息会被破坏）。MLA 额外维护一份小的**解耦 RoPE 键** `k_t^R`，维度 `d_h^R`，只对这部分加位置编码。
   → 于是每 token 每层的 Cache = **`d_c + d_h^R` 个元素**。
3. **Matrix Absorption（矩阵吸收）**：推理时把 `W^{UK}` 吸收进 `W^Q`、把 `W^{UV}` 吸收进 `W^O`，这样 Decode 时**不必显式物化完整的 K/V**，而是直接对低维 latent 做注意力 —— 从访存形态上看，MLA 在 Decode 时**表现得像 MQA**（所有 head 共享一份 latent）。

**数一下省了多少**（DeepSeek-V2 的配置：`d_c = 512`、`d_h^R = 64`、`n_h = 128`、`d_h = 128`）：

| | 每 token 每层 Cache 元素数 |
|---|---|
| 等价的 MHA | `2 × 128 × 128 = 32768` |
| MLA | `512 + 64 = 576` |
| 比值 | **约 1.8%** |

论文给出的综合结论是 **KV Cache 降低 93.3%**（与其 67B 的 MHA 基线相比），同时保持甚至超过同规模 MHA 模型的质量。

**省下了什么资源？** 显存（Cache 大小）与 Decode 的访存带宽。
**代价是什么？**
- 训练要配合（低秩结构需要端到端训练才能学到好用的压缩），不是纯推理期的技巧；
- 推理时多了"上投影"的计算（把 latent 还原成 K/V 或做吸收），Kernel 更复杂；
- 实现上要小心 RoPE 的解耦与吸收顺序（这也是 MLA Kernel 比普通 MHA 难写的原因）。

### 3.3.2 KV Cache 优化综述（sKis, arXiv 2607.08057）：一张系统地图

这篇 Survey 的贡献不在提出某个方法，而在于**把所有方法按系统行为组织成三个维度**（缩写 sKis = system-aware KV infrastructure for serving）：

```text
                       时间 (temporal)
              Execution & Scheduling
                    │            │
   KVS 调度      OVLP 流水线    HAE 硬件感知执行
                    │
   ─────────────────┼──────────────────────────
   空间 (spatial)   │   结构 (structural)
Placement & Migration      Representation & Retention
     MHO 内存层级编排            KVCC 压缩（量化/低秩/结构剪枝）
     CDO 计算设备编排            KVRM 保留管理（分配复用/驱逐）
```

**维度一：Execution and Scheduling（时间）**——KV 何时被访问/计算/调度。
- KV-centric Scheduling：请求级（TetriInfer 用 KV 预测缓解 PD 干扰）、token 级、kernel 级（FlashInfer 按 Q/KV 长度在 block 间调度）；
- Pipelining & Overlapping：用重叠掩盖 KV 相关延迟（Comp–Comp 如 FastDecode/Neo；Comp–I/O 如 CachedAttention、AsyncKV；I/O–Comm 如 PRESERVE）；
- Hardware-aware Execution：PD 解聚（DistServe、Mooncake、MuxServe）、计算卸载到 CPU / CSD / PIM。

**维度二：Placement and Migration（空间）**——KV 放在哪、怎么搬。
- 内存层级编排：重要性感知的 offload（ArkVale、OmniKV、ClusterKV、PQCache、SpeCache）、成本模型（FlexGen）、CXL 内存池（CXL-SpecKV、TraCT）、GPU 内 L2 预取；
- 计算设备编排：集群内（DistServe 的 pull-based KV 拉取）、远程传输（CacheGen 把 KV 编码成 bitstream 按带宽自适应流式传）、异构加速器。

**维度三：Representation and Retention（结构）**——本次题目重点关注的 prunning/eviction 就在这一维。

1. **KV Cache 量化**：把浮点压成低比特。要点：**K 与 V 的离群模式不同**（常见做法是 K 按 channel、V 按 token 量化）；离群值需要专门处理；向量量化（VQ-LLM、CQ、CommVQ）用码本捕捉元素相关性。代表方法：KIVI、KVQuant、ZipCache、GEAR、CacheGen、SmoothQuant 等。
   **关键提醒**：更低 bit **不一定**带来端到端收益，还要看 (de)quantization 的运行时成本和 kernel 边界能否融合。
2. **低秩近似**：把 KV 张量/投影矩阵/注意力子空间约束到低维。代表：xKV（layer-group SVD）、ShadowKV（只压 K）、Palu（对 `W^K`/`W^V` 因子化）、EigenAttention。
3. **结构压缩（Structural Compression）**——**Pruning 与 Merging**：
   - **Pruning**：按结构单元丢弃。`CHAI`（head 级）、`ThinK`（channel 级）、`ClusterAttn`（token 级）；
   - **Merging**：把单元融合成共享形式。`KVMerger`/`CaM`（token）、`KVSharer`/`MiniCache`（layer）、`D2O`（token）。
4. **保留管理**：
   - **分配与复用**：**PagedAttention**（固定大小页 + 逻辑→物理映射，减少碎片、支持共享）、Prompt Cache、radix tree（前缀共享）、LazyLLM（只对关键 token 算 KV）；
   - **驱逐（Eviction）**：在预算下丢掉不关键的 token KV。Survey 总结出三个关键观察：
     - **时机**：静态（prefill 后一次性定，如 StreamingLLM 风格）vs 动态（decode 中在线更新，跟踪重要性漂移）；
     - **策略**：通常保留**最近窗口** + **attention sink（开头 token）**，再按 attention 分数补充；
     - **预算**：从"全层统一预算"演进到"跨层/跨 head 自适应分配"。
     代表方法：**H2O**（累积 attention 分数驱逐）、**StreamingLLM**（保留 sink + 滑窗）、**SnapKV**（prefill 末尾 query 观察窗口 + 压缩）、**PyramidKV**（层间预算不均）、**Ada-KV**、Scissorhands、TOVA、CAKE、L2KV 等。

**Cross-behavior co-design 与开放挑战（O1–O7 / C1–C6，摘要）**
- O1 结构类研究最多（主导显存节省）；O2 时间类直接决定延迟/吞吐，但**尾延迟报告稀缺**；O4 **能耗探索不足**；O5 **质量损失普遍存在**；O6 **HAE–CDO 是最强的协同设计模式**；O7 **KVCC 相对孤立**，存在错失的协同机会。
- C1 SLO 驱动的尾部控制、C2 energy-aware、C3 可信与安全（dual-use）、C4 可泛化的 HAE–CDO、C5 统一预算下的协同优化、C6 **统一基准**（指标定义不一致，导致跨论文无法可靠比较）。

**对"哪些 KV 状态值得保留"的回答（Pruning/Eviction 的判断依据）**，可以归纳成四类信号：
1. **attention 权重累积**（H2O、Scissorhands）——历史 attention 得分高的 token 更重要；
2. **近期性/位置**（StreamingLLM、滑窗）——最近 token 与被反复 attend 的开头 sink token；
3. **query 侧的观察**（SnapKV、PyramidKV）——用 prefill 末端的 query 看"哪些位置被看中"；
4. **层/head 间的差异化预算**（PyramidKV、Ada-KV）——不同层对上下文的依赖不同，不该平均分配预算。

> 我的补充判断：这四类信号里，1 与 3 都是"用过去预测未来"，在长尾/分布外请求上天然脆弱；而 Survey 的 O5 也提示"质量损失普遍存在"。所以工程上真正的问题是**如何让劣化可控且可观测**（这正是 C1/C6 的主张）。

### 3.3.3 【Bonus】Sparse Attention：NSA（Native Sparse Attention）

**背景**：Causal Attention 里每个 Query 要看全部历史 Key，复杂度 `O(n²)`；长上下文下既算不动，KV Cache 也读不动。稀疏注意力的直觉是"不是所有历史 token 都同等重要"，所以只让一部分参与计算。

NSA 的做法是一个**动态分层稀疏**：三个分支并行，各司其职，最后按学习到的门控加权合并。

| 分支 | 粒度 | 干什么 | 类比 |
|---|---|---|---|
| **Compressed Attention** | 粗粒度（block 级） | 把连续的 token block 通过轻量 MLP 压成**压缩 KV**，先扫一遍全局 | "先看一眼全局目录" |
| **Selected Attention** | 细粒度（block 级） | 用压缩分支得到的注意力分数，**选出 top-n 重要的 block**，对这些 block 用**全精度** K/V 做注意力 | "再精读最相关的几页" |
| **Sliding Attention** | 局部（窗口） | 只对最近的窗口做注意力，保证局部精度 | "顺带看相邻几句话" |

**"硬件对齐"具体体现在哪？**
1. **算术强度平衡**：稀疏注意力最容易踩的坑是"省了 FLOPs 却更慢"——因为不规则 gather 会让内存访问效率暴跌。NSA 让每个分支的**算术强度尽量接近**、并让 block 大小对齐 Tensor Core 与内存事务，使 kernel 落在接近 Compute Bound 的区间；
2. **GQA 组内共享选择**：同一个 GQA group 内的多个 Query Head 共享同一份 KV Cache 与同一套被选中的 block，于是内存加载被多个 head 复用，访存效率高，且天然负载均衡；
3. **Triton 实现 + kernel 级优化**：把选择、压缩、滑窗三种访存模式在 kernel 里组织成可流水执行的形态，而不是三条独立的稀疏路径；
4. **可训练**：稀疏模式是模型的一部分（端到端训练，梯度能穿过被选中的 block），不需要"先训练稠密再事后剪枝"，因此也就没有"剪枝后掉点"的问题。论文报告在 64k 序列上 decode / forward / backward 全部相对 Full Attention 有明显加速，且质量不降。

**所以 NSA 到底"稀疏"了什么？**
稀疏的是 **Attention 在 token 维度上实际参与的 Key/Value 集合**（以及对应的 KV Cache 访问/存储），而**不是** head 维度或 feature 维度。它省下的是 `O(n²)` 的注意力计算量与 `O(n)` 的 KV 读带宽，并用"压缩 + 选择 + 滑窗"三级结构保证精度不塌。

---

## 3.4 Attention IO：FlashAttention

普通 Attention：

$$
S = QK^\top,\qquad P = \mathrm{Softmax}(S),\qquad O = PV
$$

当序列长度 `n` 变大时，`S` 和 `P` 都是 `n × n` 的中间矩阵。如果按"算完 S → 写回 HBM → 读出来做 softmax → 写出 P → 读回来做 PV"来做，HBM 上要多出 **`O(n²)` 级别的写入 + 读取** —— 而 HBM 带宽是 Decode 场景里最紧张的资源。

### FlashAttention V1：真正省的是 HBM IO，不是 FLOPs

- **做法**：Tiling + Online Softmax。
  把 Q 按行分块载入 SRAM，K/V 按列分块循环载入；对每个 K/V 分块增量更新 softmax 的统计量（running max `m`、running sum `ℓ`），并用它们**按需 rescale** 已经累加的输出。
  因为 softmax 是"需要看到整行才能归一化"的运算，online softmax 的技巧就是：**不落盘，边算边修正**（`m` 变化时把之前的部分和乘上 `exp(m_old − m_new)`）。
- **省掉了什么**：**省掉了把 `n × n` 的 `S`/`P` 写入并读回 HBM 的那些往返**。FLOPs 完全没变（仍是精确 attention，不是近似），但 HBM 访问次数大幅下降，中间矩阵根本不存在于 HBM 上。
- **副产品**：显存占用从 `O(n²)`（中间矩阵）降到 `O(n)`（只需存 Q/K/V/O，KV Cache 本身仍需 `O(n)`）；而且因为 IO 少了，实测反而更快（2~4×）。

一句话：**FlashAttention 是"IO-aware"的换搬运而不是换计算**。这也解释了为什么它对长序列收益特别大。

### FlashAttention V2 相比 V1 解决了哪些性能瓶颈？

V1 虽然 IO 少了，但**没有充分发挥 GPU 算力**（GEMM 利用率上不去，尤其在长序列 / 小 head dim 时），V2 主要解决三件事：

1. **并行度与 work partitioning 不足**：V1 把 Q 分成 64 行一块、每个 thread block 的 warp 数少，且在 Q 方向并行、K/V 方向串行 → 对长序列来说 block 数不够、SM 吃不满，warp 之间还要通过 shared memory 交换中间结果。
   **V2 改为"K/V 方向串行、Q 方向并行"的更细粒度划分**：把 Q 的行进一步切给不同的 warp，使每个 warp 的工作量更均衡，同时**避免 warp 之间来回同步**（少了一次 shared memory 通信 + 同步）。
2. **non-matmul FLOPs 太多**：softmax 的 rescale、mask、边界的比较与选择这些"非矩阵乘"指令在 V1 里占比不低。GPU 的 Tensor Core 峰值远高于普通 FMA/INT 单元，这些指令会直接把利用率拖下来。
   **V2 通过重新安排 online softmax 的更新顺序（例如把 rescale 尽量推迟/合并）减少 non-matmul FLOPs 占比**。
3. **结果**：矩阵乘的占比上升 → 整体表现**更接近一个纯 GEMM**，所以能更接近硬件的算力上限（论文报告约 2× 于 V1，达到约 50%~70% 的理论峰值）。

### V3 / V4（顺着看，理解"硬件在变，瓶颈也在变"）

- **FlashAttention-3（Hopper）**：核心是 **asynchrony + warp specialization** —— 用 TMA 做异步搬运、用 WGMMA 指令把"计算"和"softmax 的非矩阵乘部分"在不同 warp group 间流水起来，再叠加 FP8，让 H100 在低精度下也能接近峰值。
- **FlashAttention-4（Blackwell, arXiv 2603.05451）**：正面回应**非对称硬件 scaling**——Blackwell 上 Tensor Core 吞吐翻倍，但 shared memory 带宽、exponential 单元等**没有同步翻倍**，于是原来的流水线会卡在"非矩阵乘"环节。它用四类手段应对：(1) 重新设计 pipeline 以利用**全异步 MMA** 和更大 tile；(2) **软件模拟 exp 与条件 rescale**，减少 non-matmul 操作；(3) 利用 **tensor memory** 与 **2-CTA MMA** 模式减少 shared memory 流量与 backward 的 atomic add；(4) 用 **CuTe-DSL（Python 内嵌）** 实现，编译时间比传统 C++ 模板快 20~30×。B200 BF16 上最高 1613 TFLOPS/s（71% 利用率），比 cuDNN 9.13 快最多 1.3×、比 Triton 快最多 2.7×。
  → 这条演进线正好验证了 Task 0 的判断：**Ridge point 会随硬件变化，瓶颈会从算力搬到 shared memory 带宽 / 特殊函数单元，所以"照着上一代的经验调参"不一定有效。**

### 回答题面问题

> **FlashAttention V1 到底省掉了什么？V2 相比 V1 又解决了哪些性能瓶颈，为什么这些修改能够让 Attention 更接近 GEMM 的执行效率？**

**V1 省掉的是 HBM IO，不是 FLOPs。** 它用 tiling + online softmax 把 `S`、`P` 这两个 `n×n` 中间矩阵留在 SRAM 里增量处理，从而避免了 `O(n²)` 的 HBM 写入/读取往返；显存占用也降到 `O(n)`。因为 attention 在长序列下本来就是 IO-bound（算力远未用满），所以"少搬数据"直接换来了 2~4× 加速，且结果精确。

**V2 解决的是"IO 少了但算力还是用不满"的三个瓶颈**：(1) 并行度与 work partitioning 不合理（Q 方向切 64 行、warp 数少、warp 之间需要 shared memory 通信与同步）；(2) non-matmul FLOPs（softmax rescale、mask、boundary check）占比过高，拖低 Tensor Core 利用率；(3) 由此导致的 Occupancy 与流水效率不足。V2 通过"K/V 串行 + Q 并行、把工作切给更多 warp 且减少跨 warp 通信"提高并行度，通过重排 online softmax 的更新顺序减少非矩阵乘指令占比。

**为什么这样就更接近 GEMM？** 因为 GPU 的峰值算力是"Tensor Core 做矩阵乘"的峰值，而 attention 里 softmax/masking 这些非矩阵乘操作走的是慢得多的普通单元。**当 non-matmul 指令在总指令流里的占比下降、矩阵乘占比上升时，kernel 的"有效算力"就会向纯 GEMM 靠拢**；同时并行度提升让更多的 MMA 能同时在飞，掩盖了访存与特殊函数延迟 —— 于是实测利用率从 V1 的水平提升到接近 GEMM 的量级（约 50%~70% 峰值）。

---

## TASK3 交付：连接 Task 1 的 Small-M GEMM、Decode 与 Memory Bandwidth

> **一段话**

Task 1 里 `M ∈ {1, 32, 256}, N = K = 4096` 的那组 GEMM，就是 Decode 阶段每一层、每一步都在重复的运算：当 M 退化成 1 时，`C(1,N) = A(1,K)·B(K,N)` 变成 GEMV，权重矩阵 B 被完整读一遍却只参与 `K×N` 次乘加，算术强度掉到 1 FLOP/Byte 附近，远低于 GPU 的 ridge point，于是瓶颈从算力整体切换到 HBM 带宽 —— 这就是为什么我在 Task 1 里预判小 M 的 TFLOPS 会远低于方阵，也是为什么 Decode 阶段的 SM 利用率常年只有个位数百分比却把显存带宽跑满。沿着这个视角看，Task 3 读到的所有优化都能被解释成"减少 Decode 每步必须搬运的字节数"：GQA/MQA 砍掉 KV Cache 的 head 维度、MLA 把 K/V 压成低维 latent（再叠加 matrix absorption 让 Decode 表现得像 MQA）、KV Cache 量化把 `S` 从 2 Bytes 降到 1 或 0.5、稀疏化/驱逐只读真正重要的 token、PagedAttention 减少碎片与重复分配，而 FlashAttention 做的则是同一件事在 Attention 内部的重演 —— 它不改变 FLOPs，只把 `n×n` 中间矩阵的 HBM 往返搬到片上，用 online softmax 把 softmax 也变成"可流式"的。换句话说，**Task 1 教会我"AI 低就必然吃带宽"，Task 3 则展示了整个 LLM 推理栈为了对抗这一条规律所做的全部努力。**

---

## 附：四问框架下的论文总结表

| 论文 | 解决什么问题 | 核心办法 | 省下哪种资源 | 代价 |
|---|---|---|---|---|
| DeepSeek-V2 / MLA | MHA 的 KV Cache 太大，MQA 掉点 | 低维 latent 压缩 K/V + decoupled RoPE + matrix absorption | 显存 + Decode 访存带宽（Cache 降约 93.3%） | 需要端到端训练、投影开销、Kernel 复杂 |
| KV Cache Survey (sKis) | 方法碎片化、难以比较 | 时间/空间/结构三维 taxonomy + 行为-目标矩阵 + 协同设计网络 | 提供选型与评估框架（本身不省资源） | 未给统一基准（C6），跨论文数字仍不可比 |
| FlashAttention V1 | `n×n` 中间矩阵的 HBM 往返 | Tiling + Online Softmax | **HBM IO**（FLOPs 不变） | 实现复杂、backward 需重算 |
| FlashAttention V2 | V1 算力用不满 | 更好的并行度与 work partitioning、减少 non-matmul FLOPs | 算力利用率（更接近 GEMM） | 仍受 head dim / 精度限制 |
| FlashAttention-3/4 | Hopper/Blackwell 的硬件非对称 | 异步执行、warp specialization、软件模拟 exp、tensor memory、2-CTA MMA、CuTe-DSL | 新一轮的 shared memory 带宽与特殊函数瓶颈 | 强硬件依赖、需要新 DSL 工具链 |
| NSA（Bonus） | 长上下文 `O(n²)` 与 KV 读带宽 | 压缩 + 选择 + 滑窗三分支，硬件对齐且可端到端训练 | 计算量 + KV Cache 访问/存储 | 结构更复杂、稀疏模式的泛化性仍需验证 |
