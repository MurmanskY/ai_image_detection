# MGI 实验报告（实验设计、方法差异与结果细节）

> 本文档是 `reports/RESULTS.md` 的完整版：说明每组实验**如何开展**、
> **协议差异**和**结果细节**。所有数字以 fake 为正类（label=1）。
> 冻结结果以 `reports/RESULTS.md` 为准；本文档解释其来历与解读边界。

## 1. 实验总体设计

```
数据集 (ARForensics / DARG / GenImage)
   │  manifest (sha256/phash/generator 元数据) + audit (泄漏/标签/格式)
   ▼
Baseline (统一注册接口, score contract: 越大越像 fake)
   ▼
统一评估 engine/evaluator.py
   ├── 排序指标: AUROC / AP / TPR@0.1/1/5%FPR / FPR@95%TPR (免阈值)
   ├── 阈值指标: Acc / F1 / Balanced Acc (logit 用 0.5;
   │              anomaly 用 val Youden's J 校准后冻结)
   └── 置信区间: 分层 bootstrap 95% CI (--n_boot)
   ▼
归档 reports/runs/<run_id>/ (config+metrics+predictions.csv.gz+env+manifest)
```

### 1.1 数据集与协议

| 数据集 | 用途 | 协议 | 规模 |
|---|---|---|---|
| ARForensics test | 主测试 | 官方 test（7 生成器） | 每生成器 6000 real + 6000 fake，共 84,000 |
| ARForensics train/val | 训练/验证 | **自定义** `ARForensics-small-val-derived-v1` | 官方 validation（10k+10k）重划分：train 6000/类，val 1000/类 |
| DARG t2i/c2i | PRADA 外部测试 | `DARG-adapted-real-v1` | t2i 5 目录×1000；c2i 16 目录×10000 |
| GenImage | 传统检测器迁移 | legacy supplemental | 未做 manifest/audit，仅补充参考 |

**关键协议声明**（所有结果解读的前提）：

1. 训练协议为 val-derived 小规模协议，**与 D³QE 官方 100k/100k 不可直接比较**；
2. PRADA T2I real 用 ARForensics real 替代 RAISE-1k；C2I real 为 HF
   ImageNet 镜像子集（`Tsomaros/Imagenet-1k_validation`，1000 类×10），
   与官方子集不逐张一致；
3. ZED 每类 500 采样；DIRE test 为 subsample-500；PRADA 为 tiny-50 示例；
4. clean manifest（97,997 行）剔除了 3 组跨 split sha256 重复；当前冻结
   结果在清理前数据上得到（泄漏 3/98,000，影响可忽略）。

### 1.2 格式混杂控制实验

审计发现 **real 100% JPEG / fake 100% PNG**。为区分"格式捷径"与"生成痕迹"，
构建 `data/ARForensics_reencode`：所有图片统一 decode→RGB→JPEG Q95，
用同一 checkpoint 重评估（目前完成 UnivFD，7000 张对照）。

## 2. 各方法实验细节

### 2.1 训练式 black-box 检测器（统一框架，ARForensics test 84k）

训练：train 6000 real + 6000 fake（LlamaGen）/ val 1000+1000，AdamW + cosine LR，
best checkpoint 按 val AP 选取。评估：`eval.py --split test`。

| 方法 | 设计差异 | overall Acc / AP / AUC | per-generator 特点 |
|---|---|---|---|
| CNNDetection | ResNet-50 + blur/JPEG 增广（自实现） | 0.699 / 0.855 / 0.819 | 仅 LlamaGen (0.986) 好；RAR 0.031 崩 |
| FreqSpec | DCT 频谱 + 分类头（自实现） | 0.660 / 0.802 / 0.786 | 整体弱，频域痕迹对 AR 不稳定 |
| NPR-inspired | 最近邻上下采样残差×4 + ResNet | 0.748 / 0.890 / 0.860 | LlamaGen/Janus 好，其余弱 |
| UnivFD | 冻结 CLIP ViT-L/14 + 线性头 | 0.802 / 0.911 / 0.913 | 最强 black-box；但重编码后 AUC 0.889 |
| LGrad | 预训练 CNN 梯度幅值（GAN 迁移） | 0.713 / 0.852 / 0.834 | 迁移测试，不作 AR 结论 |
| DIRE | 20-step DDIM 反演残差 + ResNet50（Diffusion 迁移） | val 0.933/0.982/0.979；test sub500 0.743/0.863/0.859 | val→test 落差大（分布 shift）；LlamaGen 0.97 / RAR 0.17 |
| D³QE-lite | VQ histogram + 4 标量 + MLP（简化版） | 0.633 / 0.696 / 0.689 | 明确 lite，非 D³QE 复现 |

**结论**：通用检测器只对训练同源（LlamaGen）和部分 T2I 模型有效，
对 RAR/Open-MAGVIT2/Infinity 普遍失效——跨生成器泛化是核心难点。

### 2.2 Training-free 方法（val Youden's J 校准阈值）

| 方法 | 设计 | Acc / AP / AUC / TPR@1%FPR | 说明 |
|---|---|---|---|
| RIGID | DINOv2 clean/noisy 余弦相似度 ×(-10) | 0.675 / 0.742 / 0.741 / 0.0997 | 阈值 -9.914；修复了 sigmoid+0.5 误用 |
| ZED | SReC NLL−entropy gap (abs_delta01) | 0.564 / 0.564 / 0.581 / 0.015 | 阈值 0.0451；**每类 500**，低 FPR 指标统计功效不足 |

### 2.3 AR 专用方法（官方代码）

**D³QE 官方**（`third_party/D3QE`，训练协议为本地 val-derived）：
主测试 avg acc 0.7535 / AP 0.8269；鲁棒性 JPEG 95→50 acc 0.715→0.641，
crop 0.9→0.5 acc 0.667→0.567。CSV 在 `third_party/D3QE/results/D3QE/`。

**PRADA 官方**（`third_party/PRADA`，官方 predict.py + 官方评分函数，tiny-50 示例）：

- T2I（DARG-adapted-tiny50）：Infinity 0.9988 / Janus 0.9908 / LlamaGen 0.9980 /
  Switti 0.9996 AUROC——与论文 T2I 性能一致，**官方算法在适配协议下复现成功**；
- C2I（8 个代表模型）：0.6448–0.8204 AUROC，低于 T2I；
  解读受 real 子集替换与 50 张/类小样本限制。

**Data Provenance**（`third_party/DataProvenanceIAR` = sprintml 上游）：
LlamaGen/VAR 各 1000 样本已跑，输出 `outputs/dataprov_*_1000.csv`；
目前只有 CSV 产物，未汇总 AUROC/TPR 指标。

### 2.4 格式混杂对照（v3 新增）

| UnivFD（7000 张同协议） | AUC | Acc |
|---|---|---|
| 原始（real JPEG / fake PNG） | 0.9530 | 0.861 |
| 统一 JPEG95 重编码 | 0.8893 | 0.741 |

约 6 个 AUC 点来自格式捷径；其余判别力是真实生成痕迹。
**后续主表须同时报告两组。**

### 2.5 Token 统计分析（`analysis/token_stats.py` v2）

LlamaGen VQ tokenizer，7 生成器 × 500/类：单变量统计（token entropy、
gini、2D transition 条件熵/互信息、量化误差矩）对 real/fake 的区分
AUC 在 0.37–0.63 且**方向跨生成器翻转**。结论：bag-of-tokens 统计受
tokenizer affinity 支配，不能单独作为通用 AR 指纹。
产物：`outputs/analysis_test_v2/`（summary + per_image_features.csv + 图）。

## 3. 结果可信度分层

| 层 | 内容 | 可引用性 |
|---|---|---|
| A. 官方代码+官方权重 | PRADA T2I tiny、D³QE 官方 | 算法级复现；数据协议适配须随文声明 |
| B. 统一协议 benchmark | CNN/Freq/NPR/UnivFD/RIGID/ZED/D³QE-lite | 框架内横向比较有效；不与原论文数字直接比 |
| C. 架构迁移 | DIRE、LGrad | 只说明迁移性能，不说明原领域有效性 |

## 4. 已知局限

1. 训练协议非官方（val-derived 6k vs 官方 100k）；
2. PRADA real 集替换；DP 指标未汇总；GenImage 未审计；
3. 低 FPR 指标在小样本方法（ZED 500/类）上统计功效不足；
4. 格式混杂仅对 UnivFD 做了重编码对照，其余方法待补；
5. third_party 无 git 元数据，upstream commit 未 pin。
