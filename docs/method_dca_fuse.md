# DCA-Fuse：双通道亲和力融合的自回归生成图像检测方法

> 版本：v1.0（2026-11-18）
> 状态：已完成假设验证、消融与定型评估，结果已归档
> （`reports/runs/dca_fuse_arforensics_test_sub500/`）。
> 本文档完整记录方法探索的 motivation、设计、创新点、实验设置、
> 实验结果与分析。代码位于 `analysis/ctad_*.py` 与
> `analysis/dca_fuse_eval.py`。

## 1. Motivation：从 baseline 实验结果出发

本方法不是凭空设计的，而是直接由本仓库已完成的 baseline
复现实验中的四组观察驱动：

### 1.1 跨生成器泛化是所有 black-box 检测器的共同短板

在 ARForensics test（7 个 AR 生成器 × 6000 real + 6000 fake）上，
即使是表现最好的 UnivFD（冻结 CLIP ViT-L/14 + 线性头，
overall AUC 0.913），对训练中未见过的生成器（RAR、Open_MAGVIT2、
Infinity）也接近随机；CNNDetection/NPR/DIRE 等同样只对
训练同源的 LlamaGen 有效。这说明语义/纹理 artifact 特征无法构成
跨架构的通用检测信号。

### 1.2 单 tokenizer 统计方向跨生成器翻转

token 统计分析（`analysis/token_stats.py` v2，LlamaGen VQ tokenizer）
发现：fake 相对 real 的量化误差（qerr）方向随生成器家族不同而**翻转**——
LlamaGen/RAR/VAR/Open_MAGVIT2 偏低，Infinity/Janus/Switti 偏高。
任何"单 tokenizer + 单阈值"路线（如 D³QE-lite，AUC 仅 0.689）
都注定失败。但翻转本身揭示了更深的结构：**每个 AR 生成器与其自身的
tokenizer/codebook 家族存在亲和力（affinity）**，偏离方向取决于
参考 tokenizer 与该生成器的关系。

### 1.3 PRADA 证明了 token 级信号的强度与代价

PRADA（CVPR Findings 2026）在 DARG T2I 上达到 AUROC 0.99+，
证明 AR 模型的条件概率是比值极强的检测信号。但它需要**每个候选
生成模型的完整权重**，并逐 token 计算似然——白盒、闭集、
算力开销巨大。

### 1.4 格式捷径是真实混杂因素

JPEG95 统一重编码对照实验（`reports/RESULTS.md` §8.1）显示
UnivFD AUC 0.953→0.889：约 6 个 AUC 点来自"real 全 JPEG /
fake 全 PNG"的格式捷径。新方法必须内嵌格式稳健性控制。

**结论**：理想的检测器应当——信号来自生成机制本身（而非学到的
artifact）、方向无关（应对 1.2 的翻转）、轻量开集（避免 1.3 的
白盒代价）、对格式混杂稳健（应对 1.4）。

## 2. 方法设计

### 2.1 核心假设

> 真实摄影图像对所有 VQ tokenizer 而言**同等程度地"在流形之外"**；
> 而 AR 生成图像在其**源生成器的 tokenizer 家族下异常贴合**
> （量化误差异常低）。因此"某图像在某 tokenizer 下的 z 分数异常低"
> 即是与方向无关的生成证据。

### 2.2 DCA-Fuse 架构

```
输入图像 x
  │  通道 1 (语义 artifact):
  │    冻结 CLIP ViT-L/14 + 线性头 (UnivFD checkpoint)
  │    → s1(x) (BCE logit 经 sigmoid)
  │
  │  通道 2 (机制 affinity, CTAD):
  │    K=5 个冻结 VQ tokenizer:
  │      T1 LlamaGen VQ      (pretrained/vq_ds16_c2i.pt)
  │      T2 VAR VAE          (vae_ch160v4096z32.pth)
  │      T3 RAR MaskGIT-VQ   (maskgit-vqgan-imagenet-f16-256.bin)
  │      T4 Open-MAGVIT2     (imagenet_256_L.ckpt, TencentARC)
  │      T5 Switti VAE       (PRADA model_weights/switti/vae, reso=512)
  │    → 每 tokenizer 每图 qerr(量化误差)
  │    → val real 参考分布归一 → z_k(x)
  │    → s2(x) = -min_k z_k(x)   (异常贴合 = fake)
  ▼
融合: score = α·z(s1) + (1-α)·z(s2)   (α=0.7, real 参考归一)
校准: 阈值仅在 val 上以 Youden's J 确定 (test 只评估一次)
归因: argmin_k z_k(x) → 预测的生成器家族 (副产品)
```

### 2.3 与现有工作的关系（创新点）

| 维度 | D³QE | PRADA | Data Provenance | UnivFD | **DCA-Fuse** |
|---|---|---|---|---|---|
| 信号 | 单 tokenizer 统计 | 逐 token 似然比 | codebook 距离 | CLIP 语义 | 多 tokenizer 亲和力 + 语义 |
| 需要的资源 | 1 个 VQ | 每个生成模型权重 | 逐模型 encoder 微调 | CLIP 权重 | 仅公开 VQ tokenizer |
| 训练 | 需要 | 评分函数预训练 | 需要 | 线性头 | 通道 2 免训练 |
| 方向问题 | 翻转失效(§1.2) | — | — | — | **min-z 方向无关** |
| 开集 | 闭集倾向 | 闭集 | 闭集 | 闭集 artifact | **real-only 校准** |
| 归因 | 否 | 是 | 是 | 否 | 弱归因(副产品) |

**三个明确的创新点：**

1. **把"方向翻转"从缺陷转化为信号**：用 K 个 tokenizer 的
   min-z 聚合，使检测不依赖任何单一方向假设——直接回应并解释了
   D³QE-lite 失败的根因。
2. **Real-only 免训练机制通道**：亲和力通道只用真实图像拟合
   参考分布，无需任何 fake 训练数据，天然开集。LOGO 实验证明
   这是唯一可行形态（见 §4.2）。
3. **real 参考归一的双通道融合**：机制通道与语义通道按 real
   参考 z 分数加权融合，同时改善 worst-case 生成器与格式稳健性，
   且代价比 PRADA 低几个数量级。

## 3. 实验设置

- **数据**：ARForensics test 500/类 示例子集（7 生成器 × 500 real
  + 500 fake，与既有 subsample-500 协议一致）；参考分布用 val real
  （1000 张 ImageNet）。
- **通道 2 特征**：每个 tokenizer 输出每图 qerr（latent 与 quantized
  的 MSE）。预处理与各自官方 wrapper 对齐（LlamaGen/VAR/Open_MAGVIT2
  256px [-1,1]，RAR 256px [0,1]，Switti 512px [-1,1]）。
- **评估指标**：AUC/AP/TPR@FPR/FPR@95%TPR（离散 ROC，不插值）；
  分层 bootstrap 95% CI。阈值仅用 val Youden's J。
- **对照协议**：LOGO（leave-one-generator-out）、JPEG95 统一
  重编码稳健性、通道子集与聚合方式消融。
- **实现**：`analysis/ctad_feasibility.py`（特征管线与可行性）、
  `analysis/ctad_logo.py`（LOGO）、`analysis/ctad_reencode95.py`
  （格式对照）、`analysis/ctad_ablation.py`（消融）、
  `analysis/dca_fuse_eval.py`（定型评估）。
- **环境**：torch 2.9.1+cu128，单张 RTX 5090（评估全程 < 30 分钟）。

## 4. 实验结果与分析

### 4.1 实验 A：亲和力假设验证（4→5→6 tokenizer）

每个生成器在**自己的 tokenizer 通道**下都可检测
（单通道 AUC，经方向翻转）：

| tokenizer 通道 | 检测对象 | 翻转 AUC | 机制证据 |
|---|---|---|---|
| RAR | RAR fake | 0.843 | qerr 1.16 vs real 1.99 |
| Open-MAGVIT2 | Open_MAGVIT2 fake | **0.924** | qerr 0.220 vs real 0.403 |
| VAR | LlamaGen fake | 0.705 | qerr 0.190 vs real 0.278 |
| Switti | LlamaGen fake | 0.800+ | qerr 0.154 vs real 0.287 |
| LlamaGen | — | ~0.5 | 中性通道（所有组无差异） |
| Infinity BSD | — | 0.534 | **中性通道（已剔除，见 §4.4）** |

**方向一致性**：S_min 的 fake 均值在全部 7 个生成器上均高于
real——亲和力以"异常低 qerr"单向出现，假设成立。

### 4.2 实验 B：LOGO 否定性发现 → 免训练定位

以 6 个生成器训练 LR/MLP 组合器，测试留出的第 7 个：

| 组合器 | macro mean AUC | 关键失败 |
|---|---|---|
| LR | 0.654 | Open_MAGVIT2 仅 0.583 |
| MLP | 0.656 | Open_MAGVIT2 崩至 **0.478** |
| 免训练 S_min | **0.681–0.695** | 无 |

**分析**：亲和力信号是逐生成器特异的——留出生成器的亲和通道
在它自己的 tokenizer 上，训练集中根本不存在该模式，因此
**监督组合在原理上就无法泛化到未见生成器**。这决定了 CTAD
必须保持 real-only 免训练形态，而不是缺陷。

### 4.3 实验 C：格式稳健性（JPEG95 统一重编码）

| 方法 | 原始 AUC | JPEG95 AUC | Δ |
|---|---|---|---|
| UnivFD（语义通道） | 0.953 | 0.889 | **-0.064** |
| CTAD S_min（机制通道） | 0.681 | 0.658 | **-0.023** |

机制通道受格式混杂影响仅为语义通道的约 1/3，证明其信号确实
来自生成机制而非压缩痕迹——两通道在稳健性上互补。

### 4.4 消融（`outputs/ctad/ablation.json`）

- **聚合方式**：min（0.661）≈ top2_mean（0.657）> mean_low（0.635）
  → 保持 min。
- **Infinity VAE 通道否定**：BSD d32reg 的 qerr 对所有组几乎无差异
  （real 0.00855 vs fake 0.00837），单通道 0.534 近随机；
  加入后稀释 min 聚合（6 通道 0.661 < 5 通道 0.695）。
  **教训：通道必须携带亲和力信号，中性通道有害，已剔除。**
- **最佳成对通道**：rar+switti（0.698）、rar+open_magvit2（0.694）。
- **最终 K** = {llamagen, var, rar, open_magvit2, switti}。

### 4.5 定型结果（α=0.7，val 校准阈值 0.805，test 一次评估）

归档：`reports/runs/dca_fuse_arforensics_test_sub500/`
（含逐样本 predictions.csv.gz、metrics、环境、git 元数据）。

| 生成器 | DCA-Fuse AUC | UnivFD 单通道 AUC | TPR@1%FPR |
|---|---|---|---|
| LlamaGen | 0.978 | 0.979 | 0.606 |
| RAR | 0.980 | 0.979 | 0.634 |
| VAR | 0.963 | 0.970 | 0.514 |
| Janus_Pro | 0.958 | 0.961 | 0.338 |
| Switti | 0.943 | 0.950 | 0.228 |
| Open_MAGVIT2 | **0.927** | 0.911 | 0.304 |
| Infinity | **0.921** | 0.921 | 0.052 |
| **overall** | **0.953** (CI95 [0.950, 0.956]) | 0.953 | 0.382 |

**仅限 pilot 的观察**：在 500/类子集上，Open-MAGVIT2 有提升；该结果
没有证明总体或跨生成器优势。全量配对结果见 §8，不能将这组子集结果
表述为稳定的 Pareto improvement。

## 5. 局限与后续方向

1. **通道 2 在 T2I 家族（Switti/Infinity/Janus）上仍偏弱**
   （S_min 0.54–0.66）：这些家族自身的 VAE 亲和力较弱
   （Infinity）或需要更大输入分辨率（Switti 原协议 1024px）。
   后续可尝试 Infinity 的原生多尺度 schedule 或更高分辨率。
2. **α 与阈值协议**：α=0.7 为分析扫描值，正式论文应在 val 上
   以目标 FPR 固定；融合阈值目前以 LlamaGen-only 的 val 校准，
   对 T2I 家族可能次优。
3. **通道 1 依赖 UnivFD checkpoint**（在 val-derived 协议上训练）：
   若更换训练协议，融合参考统计需同步更新。
4. **Janus 通道**：Janus 与 LlamaGen 共享同一 VQ-VAE，
   已由 llamagen 通道覆盖（对 Janus fake 呈中性），
   无需单独通道。
5. 尚待完成：注册进统一框架（`baselines/dca_fuse.py`）参与主表，以及
   leave-one-tokenizer-family-out 和外部数据集验证。84k 全量结果已按图像
   路径匹配并做 paired bootstrap（§8）；全量评估脚本会按当前
   \(\alpha\) 在 validation 上重算融合阈值与 real-reference 标准化，
   改变参数后应重新运行，不能复用旧阈值。

## 6. 复现指南

```bash
# 1. 可行性 (亲和力假设, ~5 min GPU)
python analysis/ctad_feasibility.py --n 500 \
    --out outputs/ctad/feasibility.json

# 2. LOGO + 特征缓存 (~8 min GPU)
python analysis/ctad_logo.py --n 500 --out outputs/ctad/logo.json

# 3. JPEG95 稳健性 (~4 min GPU)
python analysis/ctad_reencode95.py

# 4. 消融 (纯 CPU, 读缓存)
python analysis/ctad_ablation.py --cache outputs/ctad/zfeat.npz

# 5. 定型评估 (~3 min GPU, 归档 predictions)
python analysis/dca_fuse_eval.py --alpha 0.7 --n_boot 1000 \
    --out outputs/dca_fuse/results_test.json

# 6. 归档
python scripts/archive_run.py --run_id dca_fuse_arforensics_test_sub500 \
    --config configs/univfd_arforensics.yaml \
    --results outputs/dca_fuse/results_test.json \
    --predictions outputs/dca_fuse/results_test_predictions.csv \
    --method dca_fuse --split test \
    --checkpoint outputs/univfd_arforensics/best.pth
```

依赖的 tokenizer 权重：
`pretrained/vq_ds16_c2i.pt`（LlamaGen）、
`third_party/PRADA/model_weights/{var,rar,switti}/`、
`pretrained/open_magvit2/imagenet_256_L.ckpt`（Open-MAGVIT2，
代码见 `third_party/SEED-Voken`）。

## 7. 非参数亲和力校准（pilot）

当前历史结果使用 `S_min = -min(z_k)`，隐含每个 tokenizer 的 qerr
可由均值/标准差充分标准化的假设，并且 tokenizer 数量增加时没有显式
校正极值搜索。`analysis/ctad_affinity.py` 提供一个可比较的非参数版本：

1. 将仅含真实图像的 validation qerr 按固定随机种子拆成互斥的 reference-fit
   与 calibration 子集；经验分布只在 fit 子集上估计。
2. 每个 tokenizer 的低尾概率用带 add-one 修正的经验 CDF
   \(\hat F_k(q)=(1+\#\{q_i\le q\})/(n_k+1)\) 估计，并定义
   \(A(x)=\max_k[-\log \hat F_k(q_k(x))]\)。
3. 用 calibration 子集上的 \(A\) 分数构造上尾 conformal p-value。
   检测阈值 \(p\le\alpha\) 在真实 calibration 与未来真实样本可交换、
   且 fit/calibration 独立的条件下具有有限样本边际错误率控制；这不保证
   在数据集偏移、不同真实分布或有依赖的重复图像下仍然成立。

`analysis/ctad_feasibility.py` 会并列输出此分数的 AUROC、校准集大小以及
测试真实图像/生成图像的观测 FPR/TPR。连续 \(A(x)\) 用于排序指标，conformal
p-value 用于 operating-point 判定；离散 p-value 不应用作主要 AUROC
分数。它是新实验而非对已有 DCA-Fuse 数值的重新解释，必须重新运行并
独立报告，不能将新旧结果拼接成单一结果表。

`engine.metrics.paired_bootstrap_delta` 可对两组逐样本对齐的分数计算
配对分层 bootstrap 的 metric 差值区间。只有在样本 ID、标签和顺序完全
一致时才能使用；若同一真实图被多个生成器重复复用，应按图像/来源做
group bootstrap，而不能把重复行当作独立观测。

在固定 seed=42、500 张 fit real + 500 张 calibration real 的 500/类
pilot 上，经验尾部连续分数的 macro AUROC 为 0.7003，旧 `S_min` 为
0.6952；最差生成器 AUROC 分别为 0.5585 和 0.5594。以 nominal
5% conformal p-value 为 operating point，独立 test real 上观测 FPR 为
3.89%，但各生成器 TPR 仅 2.8%–37.0%。这是小样本机制检验，不构成
open-set 或检测性能提升的证据；仍需 full-scale、外部数据与多 seed 验证。

## 8. 全量路径配对重评估（84k）

在 \(\alpha=0.7\) 设置下，`analysis/dca_fuse_eval_full.py` 现按归一化图像
路径连接 UnivFD 与 CTAD 分数；validation 阈值和通道标准化均由当前
\(\alpha\) 重新计算。Overall
指标使用 42,000 个 real 一次、与 42,000 个 fake 合并，避免在逐生成器
循环中重复计入 real。输出的 84,000 行预测路径唯一。

| 生成器 | DCA-Fuse AUROC | paired ΔAUROC vs. UnivFD (95% CI) |
|---|---:|---:|
| Infinity | 0.8243 | -0.00617 [-0.00801, -0.00445] |
| Janus_Pro | 0.9360 | -0.00016 [-0.00117, 0.00084] |
| LlamaGen | 0.9612 | +0.00126 [+0.00055, +0.00199] |
| Open_MAGVIT2 | 0.9054 | +0.01914 [+0.01778, +0.02057] |
| RAR | 0.9706 | +0.00083 [+0.00013, +0.00155] |
| Switti | 0.8429 | -0.00780 [-0.00927, -0.00636] |
| VAR | 0.9489 | -0.00710 [-0.00798, -0.00629] |
| **Overall** | **0.91275** | **+0.000001 [-0.000670, +0.000687]** |

Overall paired CI 跨 0，故当前结果不支持整体 AUROC 优于 UnivFD。Open-MAGVIT2
上的提升显著，但 Switti、VAR、Infinity 也有显著回退；最合适的结论仍是
generator-dependent 的互补性，而不是总体 SOTA。paired bootstrap 以路径为
配对键，按标签分层重采样；若不同路径实际指向重复内容，仍需改用内容/来源
group bootstrap。

归档：`reports/runs/dca_fuse_arforensics_test_full_recalibrated/`（metrics、
84k predictions、manifest、environment 和 config）；原始输出位于
`outputs/dca_fuse/results_test_full_recalibrated.json` 与
`outputs/dca_fuse/results_test_full_recalibrated_predictions.csv`。

```bash
python analysis/dca_fuse_eval_full.py --alpha 0.7 --n_boot 1000 \
    --out outputs/dca_fuse/results_test_full_recalibrated.json
```

---

# v5 指南验证：CTRF 关系结构 MVP（负结果，已按预注册标准停止）

按 `deep-research-report-new-method-v5.pdf` 的主线 CTRF（跨 tokenizer 关系结构）
做了 P0-A MVP：`analysis/ctrf_qerr_mvp.py`，经验 CDF 秩归一化 → 成对差关系特征
（5 tokenizer → 5 边际 + 10 成对差）→ real-only shrinkage Mahalanobis / kNN。

结果（500/类协议，`outputs/ctrf/qerr_relational_mvp.json`）：

| 生成器 | Mahalanobis | kNN | S_min (CTAD) |
|---|---|---|---|
| Open_MAGVIT2 | **0.972** | 0.903 | 0.857 |
| Switti | **0.695** | 0.646 | 0.615 |
| RAR | 0.734 | 0.724 | 0.767 |
| Infinity | 0.621 | 0.579 | 0.672 |
| Janus_Pro | 0.666 | 0.643 | 0.680 |
| LlamaGen | 0.652 | 0.644 | 0.715 |
| VAR | 0.515 | 0.491 | 0.559 |
| **macro** | **0.694** | 0.661 | **0.695** |

预注册 stop 标准（macro +0.03 或 worst-gen +0.02）：**未通过，按指南停止**。

分析：关系结构呈生成器特异互补（Open_MAGVIT2 +0.11、Switti +0.08），
但 VAR/RAR 反噬，整体与标量 min 持平——**multivariate dependency 并不
比 scalar minimum 更稳定**。tokenizer-affinity 主线到此为止，不再投入
structured feature 大工程。v5 建议的备选（Forensic Self-Description +
mechanism response）记录在案，不展开。

## PARD（v6 指南主线，P0 MVP）：负结果

按 `deep-research-report-v6.pdf` 的主线 PARD（扰动感知响应动力学）做了
P0 MVP：`analysis/pard_mvp.py`。DINOv2 ViT-S/14 冻结（与 RIGID 同骨干），
5 类扰动 × 2 强度，4 层 patch 一阶/二阶响应 + 空间统计 → 370 维描述子，
val real-only Mahalanobis / kNN；subsample-500 协议，RIGID 同子集对照。

结果（`outputs/pard/pard_mvp.json`）：PARD-kNN macro 0.620 vs RIGID 0.628，
预注册 GO 三条均未通过（macro 增益 -0.008，worst +0.032，3/7 不差于
RIGID），**按 v6 停止条件不扩大规模**。诊断消融（`pard_ablation.py`）
显示无特征子块超过 RIGID +0.03；多扰动全局 CLS 动态（0.476）反而不如
RIGID 单噪声，低层特征略优于深层。详见 `reports/RESULTS.md` §12。

至此两条免训练新机制主线（CTRF、PARD）均按预注册标准终止；
监督式 UnivFD 仍为最强基线。

## MPID（v7 指南主线，P0）：负结果

按 `deep-research-report-v7.pdf` 的主线 MPID（机制特权不变蒸馏）做了 P0：
`analysis/mpid_p0.py`。PRADA 完整分数作训练特权教师（离线复算与官方一致），
CLIP ViT-L/14 冻结特征 + MLP 学生，LOGO-4，BCE-only vs BCE+MSE(λ=1)。

结果（`outputs/mpid/mpid_p0.json`，mpid-p0-1000 示例协议）：
macro AUROC A=0.9980 vs B=0.9978（增益 -0.0002，要求 ≥+0.02），0/4 折提升，
JPEG95 增益 -0.0001——**GO 三条均未通过，按 v7 停止**。

关键诊断：(1) Ridge 探针显示 PRADA 机制分数从冻结 CLIP 特征**不可线性解码**
（held-out 生成器 Spearman ρ 中位 0.026），蒸馏前提不成立；(2) BCE-only 已
近饱和（0.9965–0.9989），协议无增益余量；(3) 蒸馏反而提高生成器身份探针
准确率（0.94→0.97），与 invariance 目标相反。详见 `reports/RESULTS.md` §13。

至此三条免训练/特权监督新机制主线（CTRF、PARD、MPID）均按预注册标准终止。

## MPID 严格版修正与 STEM-AR（v8 复核）：均负结果

**MPID 严格 teacher holdout**（`analysis/mpid_p0_strict.py`，v8 §2.2 修正）：
教师集合逐折排除 held-out 模型、genprobe 改独立测试评估后，结论不变
（macro 增益 -0.0002，0.869→0.885 leakage 略升，GO=false）。

**DARG T2I 难度审计**（`analysis/darg_audit.py`，v8 §2.3）：无近重复泄漏，
但元数据/格式可完美分类（AUC 1.0000：real JPEG 多分辨率 vs fake PNG 固定
方形分辨率）；codec-matched 后仍 0.997——简单性源自内容域差，
DARG T2I LOGO 0.998 仅为"当前划分下的表现"。

**STEM-AR 诊断**（`analysis/stem_ar_p0.py`，v8 §9 P0-C）：token-to-grid
映射验证通过（四模型 scale_lengths 均为完全平方或精确池化）；但 per-cell
ridge 探针在 LOGO 全部 4 折、orig 与 jpeg95 两种 codec 下
ρ(real) ≡ ρ(shuffled)（margin=0.0）——token 级机制证据图从 DINOv2 局部
特征完全不可预测，**按 v8 停止**。

联合结论：PRADA 机制证据对冻结视觉表征的不可解码性在标量（MPID）与
空间图（STEM-AR）两种粒度上均成立，机制蒸馏路线整体终止。

## CARF（v9 指南主线，P0）：首个 GO=true

按 `deep-research-report-v9.pdf` 的主线 CARF（内容与采集条件条件化的
取证残差建模）完成 P0：`analysis/carf_p0.py`。语义仅用于检索内容相近的
真实参考；判别证据为局部取证残差对条件化真实分布 p(r|c,a) 的偏离
（核加权局部高斯）；cal real conformal 校准。

结果（`outputs/carf/carf_p0.json`，协议 A 全 JPEG95 codec-matched）：
M3 联合条件 macro **0.802** vs M0 无条件 0.732（bootstrap CI95
[+0.042,+0.094]），vs RACE NP-RSR 0.507；real FPR 5.7%（名义 5%±2pp
内）；7/7 生成器不退化——**预注册 GO 全部通过，是 v5 以来首个通过的
新机制路线**。

关键发现：增益几乎全来自**采集条件化**（M2≈M3，M1≈M0）；绝对性能仍
低于 UnivFD（0.908 同 subsample）；orig 混合格式协议下 RACE NP-RSR
反超（0.858 vs 0.652），CARF 优势限定于 codec-matched 协议。
详见 `reports/RESULTS.md` §16。P1（更大规模 + 消融 + 多种子）进行中。

## CoNTra（v10 主线，P0）：负结果——nuisance 与信号同向

按 `deep-research-report-v10.pdf` 第二部分的 CoNTra（反事实 nuisance
transport）完成 P0：`analysis/contra_p0.py`。配对反事实差分高度可识别
（近似秩一，跨 source cos≈0.99，mean-correction 消除 88% 跨 codec 漂移），
但**生成异常信号与采集 nuisance 同向**：投影校正后 fake−real 差异仅剩
2.5%–28%（η≪0.8），GO=false，按 v10 停止。

这是本项目最重要的理论性发现之一：在 Noiseprint++ 残差空间中，压缩
敏感性方向同时是采集漂移方向与真假判别方向，"先去除采集效应再检测"
不成立。详见 `reports/RESULTS.md` §18。
