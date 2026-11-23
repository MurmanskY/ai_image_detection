# 自有方法设计探索：跨 Tokenizer 亲和力差异检测（CTAD）

> 历史记录：本文保留 2026-11-17 的探索记录和 pilot 数值；后续全量、
> 配对重评估已修正其性能结论。当前方法与最新结果以
> [`docs/method_dca_fuse.md`](./method_dca_fuse.md) §8 为准。

## 1. 实验证据回顾（motivation 的事实基础）

从当前已完成实验中，有四条直接指导方法设计的证据：

### 1.1 跨生成器泛化是所有 black-box 方法的共同短板

| 方法 | overall AUC | LlamaGen | 最弱生成器 |
|---|---|---|---|
| CNNDetection | 0.819 | 0.986 acc | RAR 0.031 / Open_MAGVIT2 0.056 acc |
| UnivFD（最强） | 0.913 | >0.9 | Infinity/RAR 接近随机 |
| DIRE（迁移） | 0.859 | 0.97 | RAR 0.17 / Open_MAGVIT2 0.31 |
| RIGID / ZED | 0.74 / 0.58 | — | 整体失效 |

语义级（CLIP）与低层级（CNN/NPR）特征学到的主要是
**训练所见生成器（LlamaGen）的 artifact**，对未见架构
（RAR 随机序、Open-MAGVIT2、Infinity bitwise）失效。

### 1.2 单 tokenizer 统计方向跨生成器翻转（token_stats v2）

在 LlamaGen VQ tokenizer 下，fake 相对 real 的量化误差方向不一致：

| 方向 | 生成器 |
|---|---|
| qerr 低于 real | LlamaGen, Open_MAGVIT2, RAR, VAR |
| qerr 高于 real | Infinity, Janus_Pro, Switti |

**没有任何单一阈值/线性分类器能在所有生成器上同时成立**——
这正是 D³QE-lite（AUC 0.689）失败、以及"通用 AR 指纹"假说被证伪的原因。
但这同时暴露了一个更强的结构信号：**每个 AR 生成器与其自身
codebook/tokenizer 家族存在亲和力（affinity）**，而方向（偏高/偏低）
取决于该生成器与所用参考 tokenizer 的关系。

### 1.3 PRADA 证明了 token 条件概率是强信号，但代价是白盒

PRADA T2I AUROC 0.99+，但它需要对**每个候选生成模型本身**做
逐 token 似然计算——需要模型权重、架构适配、算力巨大，
且只能检测"已知集合内"的生成器。

### 1.4 格式混杂是真实但次要的因素

JPEG95 重编码对照：UnivFD AUC 0.953→0.889。约 6 个点来自
JPEG/PNG 格式捷径。任何新方法都必须做**统一重编码对照**，
否则结论不可信。

## 2. 方法设计：CTAD（Cross-Tokenizer Affinity Discrepancy）

### 2.1 核心假设

> 真实摄影图像对所有 VQ tokenizer 而言**同等程度地"在流形之外"**；
> 而 AR 生成图像在其**源生成器的 tokenizer 家族下异常贴合**
> （量化误差异常低 / token 分布异常吻合），在其他 tokenizer 下表现正常。
> 因此，"跨 tokenizer 统计量的离散程度"本身即是与方向无关的
> 生成证据——无论该生成器使我们的统计量偏高还是偏低。

这把 D³QE（单 tokenizer 统计）、Data Provenance（codebook 几何归因）、
PRADA（似然比）统一到同一个解释框架下，同时规避了 1.2 的方向翻转问题。

### 2.2 方法流程

```
输入图像 x
  │  K 个冻结 VQ tokenizer（覆盖主要 AR 家族）:
  │    T1 = LlamaGen VQ      (next-token, 已有 pretrained/vq_ds16_c2i.pt)
  │    T2 = Open-MAGVIT2 VQ  (next-token, 需下载)
  │    T3 = VAR VQ           (next-scale, PRADA model_weights 中有)
  │    T4 = RAR tokenizer    (random-order, PRADA model_weights 中有)
  ▼
对每个 Tk 提取每图像特征 φ_k(x):
  qerr_mean/std, token histogram JS 散度(vs real 参考),
  2D transition 条件熵, bigram 互信息
  ▼
用真实图像验证集拟合每个特征的参考分布 → z 分数 z_k(x)
  ▼
检测分数（免训练版）:
  S(x) = -min_k z_k(qerr_k)        # 异常贴合某 tokenizer = fake
       + λ · spread_k(z_k)         # 跨 tokenizer 离散度
归因输出（副产物）:
  argmin_k z_k → 预测生成器家族
可选轻量头（LOGO 协议训练）:
  logistic/MLP([z_1..z_K, min, spread]) — 验证可学习组合的上界
```

### 2.3 与现有工作的差异化（创新点）

| 方法 | 需要的资源 | 检测粒度 | 局限 | CTAD 的区别 |
|---|---|---|---|---|
| D³QE | 单 LlamaGen tokenizer + 训练 | 检测 | 方向翻转失效(1.2) | 多 tokenizer 相对亲和力，方向无关 |
| PRADA | 每个候选 AR 模型权重+逐 token 似然 | 检测+归因 | 白盒、闭集、算力大 | 只需公开 VQ tokenizer（轻量几个数量级），开集 |
| Data Provenance | 每模型 encoder 微调 | 归因 | 需逐模型训练 | 免微调，检测为先，归因是副产物 |
| RIGID/ZED | 基础模型/压缩模型 | 检测 | 在我们的测试上弱 | token 空间 + real-only 校准 |
| UnivFD | CLIP 特征+线性头 | 检测 | 未见生成器失效 | 信号来自生成机制本身而非语义/纹理 artifact |

**三个明确的创新点：**

1. **跨 tokenizer 亲和力离散度作为方向无关的通用 AR 证据**——
   直接回应并解释了单 tokenizer 统计方向翻转的现象（1.2），
   把"缺陷"转化为"信号"。
2. **Real-only / one-class 校准**：只用真实图像拟合参考分布，
   不需要任何 fake 训练数据，天然开集，适配未知新生成器。
3. **检测-归因一体化**：min-z 的 argmin 给出弱归因（哪个 tokenizer 家族），
   无需 PRADA 级别的逐模型似然。

### 2.4 预期实验表现（假设，待验证）

- **主要预期**：未见生成器（RAR/Open_MAGVIT2/Infinity）上 min-gen AUC
  显著高于 UnivFD（其 min-gen 接近随机），因为信号来自生成机制而非
  学习到的 artifact；
- overall AUC 目标 ≥0.90，macro-min-gen AUC 目标 ≥0.80；
- 如果假设不成立（fake 在自身 tokenizer 下 qerr 并不显著低），
  实验同样有价值：将把问题准确定位为"亲和力信号不足"，
  退化为 multi-tokenizer ensemble，与 D³QE/DP 形成清晰对话。

## 3. 实验计划（全部基于现有框架与小批量）

### 实验 A：可行性验证（半天，GPU 分钟级）

1. 准备 T1–T4 tokenizer（T1 已有；T3/T4 从 PRADA model_weights 提取 VQ；
   T2 下载 Open-MAGVIT2 VQ，需代理）；
2. 在 ARForensics test 500/类 子集上计算每 tokenizer 的 φ_k 与 z 分数；
3. 验证核心假设：fake 的 min-z(qerr) 是否显著低于 real
   （per-generator AUC、效应量、方向一致性检查）。

**通过标准**：min-z 在 ≥5/7 生成器上 AUC>0.8 且方向一致。

### 实验 B：免训练检测器 + LOGO

- S(x) 阈值只用 real val 校准（target FPR=1%）；
- LOGO：轻量头在 6 个生成器上训练，测试第 7 个，轮换；
- 与 UnivFD / D³QE-lite / RIGID 在同一 predictions 级别对比
  （统一 evaluator，分层 bootstrap CI）。

### 实验 C：鲁棒性与混杂控制（必须）

- JPEG95 统一重编码对照（沿用 §1.4 协议）；
- JPEG Q50、resize 0.5、Gaussian blur 下 AUC 保持率；
- 归因混淆矩阵 vs PRADA cfm。

### 实验 D：消融

- K ∈ {1, 2, 4} tokenizer 数量；
- 特征组合：仅 qerr / qerr+transition / 全部；
- min-z vs spread vs 学习组合。

## 4. 备选方向（v0.2，若 A 失败）

**Surrogate-PRADA**：只用真实图像在第三方 tokenizer 上训练一个小型
AR prior（GPT-style，~10M 参数），score = token NLL surprise。
本质是"real-only 版 PRADA"，不需要目标生成器。
成本高于 CTAD（需训练 prior），作为第二梯队。

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| fake 在自身 tokenizer 下 qerr 不显著（亲和力弱） | 实验 A 先验证；不行转 ensemble/备选 |
| T2 下载困难 | 先用 T1/T3/T4 三个 tokenizer 起步 |
| 格式捷径混入 | 所有主结果必须附 JPEG95 对照 |
| 与 DP 撞车 | 强调检测优先、免微调、real-only 开集协议 |

---

# 实验结果与方法定型 v0.2（2026-11-17）

实验 A/B/C 已完成，设计据此从"CTAD 独立检测器"修正为**双通道融合检测器**。

## A. 亲和力假设：强成立（实验 A, 4 tokenizer, 500/类）

每个生成器在**自己的 tokenizer 通道**下都可检测（单通道 AUC 经方向翻转后）：

| 通道 | 检测对象 | 翻转 AUC | 机制 |
|---|---|---|---|
| RAR tokenizer | RAR fake | 0.843 | qerr 1.16 vs real 1.99（异常贴合） |
| Open-MAGVIT2 tokenizer | Open_MAGVIT2 fake | **0.924** | qerr 0.220 vs real 0.403 |
| VAR tokenizer | LlamaGen fake | 0.705 | qerr 0.190 vs real 0.278 |
| LlamaGen tokenizer | — | ~0.5 | 中性通道（所有组无差异） |

**方向一致性**：S_min 的 fake 均值在全部 7 个生成器上均高于 real——
亲和力以"异常低 qerr"单向出现，与 §2.1 假设一致。

## B. 学习组合不可迁移，免训练融合有效（实验 B, LOGO）

1. **LOGO 否定性结果**：LR/MLP 在留出生成器上 mean AUC 0.654/0.656，
   不超过免训练 S_min（0.681）；Open_MAGVIT2 上 MLP 崩至 0.478。
   结论：亲和力信号逐生成器特异，**监督组合无法泛化到未见生成器**，
   CTAD 必须保持 real-only 免训练形态。
2. **融合帕累托改进**（UnivFD 分数 + CTAD S_min，real 参考 z 归一，
   α=UnivFD 权重）：

| 方法 | macro mean AUC | 最差生成器 |
|---|---|---|
| UnivFD | 0.953 | 0.911 (Open_MAGVIT2) |
| CTAD S_min | 0.681 | 0.565 |
| **融合 α=0.7** | **0.953** | **0.922 (Open_MAGVIT2 0.928)** |

融合在几乎不损失强项的情况下把最弱生成器抬高 ~1.1 个点。
注意：α=0.7 为分析扫描值，正式协议须在 val 上固定 α 后报 test。

## C. 格式稳健性：CTAD 显著优于语义通道（实验 C, JPEG95 统一重编码）

| 方法 | 原始 AUC | JPEG95 AUC | Δ |
|---|---|---|---|
| UnivFD | 0.953 | 0.889 | **-0.064** |
| CTAD S_min | 0.681 | 0.658 | **-0.023** |

CTAD 受格式混杂影响仅为语义通道的约 1/3，进一步证明其信号来自
生成机制而非压缩痕迹。两通道在稳健性上也互补。

## 定型方法：DCA-Fuse（Dual-Channel Affinity Fusion）

```
通道 1 (语义 artifact): 冻结 CLIP ViT-L/14 + 线性头 (现成 UnivFD checkpoint)
通道 2 (机制 affinity): K 个冻结 VQ tokenizer (LlamaGen/VAR/RAR/Open-MAGVIT2)
        → 每图 qerr → val real 参考 z 分数 → S_min = -min_k z_k
融合: score = α · z(real-ref, 通道1) + (1-α) · z(real-ref, S_min)
        (z 均以 real 参考分布归一; α 在 val 上固定, 候选 0.7)
校准: 阈值仅在 val 上确定 (target FPR=1%), test 只评一次
```

**相对现有工作的最终定位**：
- vs UnivFD：min-gen +1.1 点且更抗格式捷径（通道 2 补足）；
- vs D³QE：解释并解决了单 tokenizer 方向翻转（多通道 + 方向无关 min 聚合）；
- vs PRADA：不需要目标生成器权重，开集、代价低几个数量级；
- vs Data Provenance：免逐模型微调，检测优先、归因（argmin 通道）为副产品。

## 下一步（按优先级）

1. 在 val 上固定 α（候选 0.7）与阈值，test 一次性评估并归档；
2. 补齐通道 2 的其余家族 tokenizer（Janus/Infinity/Switti 若有公开 VQ），
   预期进一步提升 T2I 家族覆盖；
3. LOGO 完整协议重跑 + 分层 bootstrap CI；
4. 消融：K 数量、仅 qerr vs +transition、min vs top-2 聚合。

## DCA-Fuse 定型结果（500/类协议, α=0.7, val Youden 校准阈值 0.805）

归档: `reports/runs/dca_fuse_arforensics_test_sub500/`
(predictions.csv.gz + metrics.json + manifest.json)

| 生成器 | AUC | TPR@1%FPR |
|---|---|---|
| LlamaGen | 0.978 | 0.606 |
| RAR | 0.980 | 0.634 |
| VAR | 0.963 | 0.514 |
| Janus_Pro | 0.958 | 0.338 |
| Switti | 0.943 | 0.228 |
| Open_MAGVIT2 | **0.927** | 0.304 |
| Infinity | **0.921** | 0.052 |
| **overall** | **0.953** (CI95 [0.950, 0.956]) | 0.382 |

与 UnivFD 单通道 (macro 0.953 / min-gen 0.911) 相比: macro 持平,
最差生成器 +1.0 点 (Open_MAGVIT2 0.911→0.927, Infinity 0.921),
且 CTAD 通道格式稳健性更好 (JPEG95 掉 2.3 点 vs 6.4 点)。

五通道版 (加入 Switti VAE, reso=512) 的 CTAD S_min macro 0.695
(四通道 0.681), Switti 通道同时强化了 LlamaGen (0.653→0.715)
与 Janus (0.659→0.680) 的亲和力覆盖。

## 消融与通道裁剪（六通道试验）

**Infinity VAE 通道试验（否定）**：BSD d32reg VAE 在 256px/1M schedule 下
qerr 对所有组几乎无差异（real 0.00855 vs Infinity fake 0.00837），
单通道 AUC 0.534 近随机；加入后 min 聚合被稀释（6 通道 0.661 < 5 通道
0.695），故剔除。**启示：通道必须携带亲和力信号，中性通道有害。**

聚合方式消融（min/top2_mean/mean_low）：min 0.661 ≈ top2 0.657 >
mean_low 0.635，保持 min 聚合。
最佳成对通道：rar+switti 0.698、rar+open_magvit2 0.694。

最终 K = {llamagen, var, rar, open_magvit2, switti}，
DCA-Fuse 最终版（α=0.7, 阈值 0.805）：
overall AUC 0.953 / min-gen 0.921 (Infinity) / Open_MAGVIT2 0.927，
已归档 `reports/runs/dca_fuse_arforensics_test_sub500/`。

## 后续全量结果的更正（2026-11-23）

以上 0.953 overall AUC 是 500/类 pilot，不应作为 full-scale claim。
路径匹配的 84k full test 得到 DCA-Fuse AUROC 0.91275；相对 UnivFD 的
paired ΔAUROC 为 +0.000001，95% CI [-0.000670, +0.000687]，不能支持
整体提升。Open_MAGVIT2 有显著改善，但多个生成器出现显著回退。
非参数 empirical-tail/conformal CTAD 的 500/类 pilot macro AUROC 为
0.7003，对旧 S_min 的 0.6952 仅有轻微提升，最差生成器没有改善。
完整 protocol、逐生成器 paired CI 和复现命令见
[`docs/method_dca_fuse.md`](./method_dca_fuse.md)。
