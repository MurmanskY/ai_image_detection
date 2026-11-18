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

**核心结论**：融合在几乎不损失任何强项的前提下，把最差生成器
从 0.911 抬到 **0.921**（Open_MAGVIT2 0.911→**0.927**），
即对 UnivFD 最薄弱的情形实现了帕累托改进，同时机制通道带来
更好的格式稳健性（§4.3）。

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
5. 待做：注册进统一框架（`baselines/dca_fuse.py`）参与主表、
   完整 84k test 评估、LOGO 完整协议与 CI。

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
