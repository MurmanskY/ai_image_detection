# MGI 结果归档 (RESULTS.md)

只列出已完成并冻结的结果。正在运行的任务单独标注 `Status: RUNNING`，不作为最终结果。
所有指标正类为 fake (label=1)。训练协议除注明外均为 `ARForensics-small-val-derived-v1`
（官方 validation 重划分，非 D³QE 官方 100k/100k 协议，不可直接比较）。

## 1. ARForensics test（统一框架，7 生成器 × 6000 real + 6000 fake）

| 方法 | 状态 | Acc | AP | AUC | 备注 |
|---|---|---|---|---|---|
| CNNDetection | FROZEN | 0.6988 | 0.8554 | 0.8195 | `outputs/cnn_detection_arforensics/results_test.json` |
| FreqSpec | FROZEN | 0.6596 | 0.8020 | 0.7862 | `outputs/frequency_genimage` 等 |
| NPR (inspired) | FROZEN | 0.7478 | 0.8898 | 0.8604 | 简化版，非官方 NPR |
| UnivFD | FROZEN | 0.8017 | 0.9112 | 0.9127 | CLIP ViT-L/14 + 线性头 |
| LGrad | FROZEN | 0.7126 | 0.8522 | 0.8342 | GAN 方法迁移测试 |
| D³QE-lite | FROZEN | 0.6328 | 0.6958 | 0.6885 | 明确简化版 |

## 2. Training-free 方法（v1 P0 修正后，val 校准阈值）

| 方法 | 状态 | Acc | F1 | AP | AUC | TPR@1%FPR | 备注 |
|---|---|---|---|---|---|---|---|
| RIGID | FROZEN | 0.675 | 0.655 | 0.7421 | 0.7407 | 0.0997 | 阈值 -9.914 (Youden's J)；有 predictions.csv |
| ZED | FROZEN | 0.564 | 0.483 | 0.5639 | 0.5806 | 0.015 | 阈值 0.0451；**每类 500 采样子集** |
| ZED v2 (abs_delta02) | FROZEN | 0.514 | 0.525 | 0.5133 | 0.5218 | 0.002 | 阈值 0.0394；val 选择的统计量在 test 上未泛化 |

注：ZED 每类仅 500 张，1%FPR 对应 5 个负样本、0.1%FPR 不足 1 个——其 `TPR@0.1%FPR` 无统计解释力，`TPR@1%FPR` 也须连同 N_real=500 一并解读，不能与 6000/42000 负样本的方法直接并列。

**ZED 专项核查结论（见 `baselines/reproduction/zed.md`）**：实现与 SReC 协议逐层核对无误（NLL≈entropy 的数值合理）；val 上统计量对比曾显示 abs_delta02 优于 abs_delta01（0.77 vs 0.68），但 test 上反而更差（AUC 0.52 vs 0.58）——说明 ZED 的信号选择对生成器不稳定，**其差性能更可能来自方法本身对 AR 图像的弱适用性，而非实现错误**。

修正前 sigmoid+0.5 阈值下 RIGID/ZED 的 Acc/F1 无效，已从报告中移除；AUROC/AP 不受单调变换影响。

## 3. D³QE 官方代码（third_party/D3QE）

- 主测试：avg acc 0.7535 / AP 0.8269
- 鲁棒性 JPEG 95→50：avg acc 0.715→0.641 / AP 0.789→0.731
- crop 0.9→0.5：acc 0.667→0.567 / AP 0.682→0.586
- 输出：`third_party/D3QE/results/D3QE/`（CSV 已写出）
- 状态：FROZEN（训练数据协议与官方不同，见 `baselines/reproduction/d3qe.md`）

## 4. DIRE（Diffusion 方法迁移测试，20-step DDIM）

| 划分 | 状态 | Acc | AP | AUC | TPR@1%FPR |
|---|---|---|---|---|---|
| val | FROZEN | 0.933 | 0.9818 | 0.9790 | 0.825 |
| test subsample-500 示例 | FROZEN | 0.743 | 0.8627 | 0.8587 | 0.241 |
| test 全量 84k | FROZEN | 0.715 | 0.8383 | 0.8300 | 0.235 |

- test 全量 84k 归档 `reports/runs/dire_arforensics_test_full`
- test 为每生成器 500 张示例子集（7000 张，标注 `subsample-500-示例`），归档 `reports/runs/dire_arforensics_test_subsample500`
- per-generator：LlamaGen 0.970 / Janus_Pro 0.954 较好；RAR 0.170 / Open_MAGVIT2 0.312 / Infinity 0.366 较差——DIRE 为 Diffusion 专用方法，在 AR 上属迁移测试，不应解读为其在原领域无效

- 权重：`outputs/dire_arforensics/best.pth`；val 结果 `results_val.json` + predictions.csv
- 训练：2 epochs（best val AP 0.9818）

## 5. PRADA 官方代码（DARG）

### T2I tiny-50 示例（FROZEN，标注 `tiny-50-示例`）

每目录 50 张示例子集（symlink，不动原数据），官方预训练评分函数（train_size=0）。
输出：`outputs/prada_t2i_tiny/2026-11-17_14-53-33/`

| fake 生成器 | PRADA AUROC |
|---|---|
| infinity_2b | 0.9988 |
| janus_1b | 0.9908 |
| llamagen_xlstage2 | 0.9980 |
| switti_1024 | 0.9996 |

与论文 T2I 检测性能一致，复现成功。单特征基线（gt_llh_diff / icas / quant_err / aeroblade）见同目录 CSV。

### T2I 全量（每目录 1000 张，FROZEN）

官方 predict.py，官方预训练评分函数（train_size=0），real 为
DARG-adapted-real-v1（ARForensics real 1000 张替代 RAISE-1k，协议偏差同上）。
特征缓存 20/20：`outputs/prada_t2i/features/`；
结果目录：`outputs/prada_t2i/2026-11-24_17-28-25/`
（运行过程三次因服务器其他用户进程挤占显存 OOM，最终经
`scripts/run_prada_t2i_when_free.sh` 看门狗在空闲窗口完成，缓存复用无重复计算）。

| fake 生成器 | PRADA AUROC | gt_llh_diff | icas | quant_err | aeroblade |
|---|---:|---:|---:|---:|---:|
| infinity_2b | 0.9941 | 0.7672 | 0.5435 | 0.4793 | 0.8450 |
| janus_1b | 0.9840 | 0.9712 | 0.9296 | 0.9976 | 0.9587 |
| llamagen_xlstage2 | 0.9860 | 0.2374 | 0.2751 | 0.9961 | 0.9631 |
| switti_1024 | 0.9949 | 0.4589 | 0.8620 | 0.4335 | 0.5842 |
| **macro** | **0.9898** | 0.6087 | 0.6525 | 0.7267 | 0.8378 |

结论：PRADA 全量 T2I 检测 macro AUROC 0.9898，与 tiny-50 示例（0.99+）
一致，为目前最强的 AR 专用检测参照（跨 T2I 生成器稳定 ≥0.98）。
单特征基线均不稳定（如 gt_llh_diff 在 llamagen 上仅 0.237、方向因生成器
而异），印证 PRADA 多特征似然比评分函数的必要性。
归因混淆矩阵见 `cfm_prada_test.pdf`。

### C2I tiny-50 示例（FROZEN，标注 `tiny-50-示例`）

每目录 50 张示例子集，官方预训练评分函数（train_size=0）。
输出：`outputs/prada_c2i_tiny/2026-11-17_17-35-14/`、`outputs/prada_c2i_tiny_b2/2026-11-17_17-35-14/`

| fake 生成器 | PRADA AUROC | | fake 生成器 | PRADA AUROC |
|---|---|---|---|---|
| hmar_d16 | 0.786 | | hmar_d24 | 0.6448 |
| llamagen_b256 | 0.6504 | | rar_l | 0.8204 |
| rar_b | 0.746 | | rar_xl | 0.782 |
| var_d16 | 0.7456 | | var_d36 | 0.660 |

注：C2I 模式 8 个代表模型分两批跑完。AUROC 低于 T2I 模式（0.99+），可能因素：每类仅 50 张、real 为 HF ImageNet 镜像子集（非官方子集）、模型规模差异；归因混淆矩阵见 cfm_prada_test.pdf。
- 本地补丁（非算法改动）：`predict.py` 特征缓存 `map_location="cpu"` + 每 pass `gc.collect()`/`empty_cache()`（修复 wrapper 累积 OOM）；`external/rar/data.py` 推理 stub
- 协议偏差：T2I real 为 ARForensics 替代 RAISE-1k；C2I real 为 HF ImageNet 镜像子集（见 `baselines/reproduction/prada.md`）

## 6. Data Provenance（采样）

- LlamaGen / VAR 各 1000 样本：`outputs/dataprov_llamagen_original_1000.csv`、`outputs/dataprov_var_original_1000.csv`
- 状态：FROZEN（采样评估，非全量）

## 7. Token 统计分析（analysis/token_stats.py v2）

- 输出：`outputs/analysis_test_v2/`（token_stats_summary.json + per_image_features.csv + token_stats.png）
- 结论：单变量 token 统计（entropy/gini/transition/q-error）在 7 个生成器上 AUC 0.37–0.63 且方向跨生成器不一致 → bag-of-tokens 统计受 tokenizer affinity 影响，不能单独作为通用 AR 指纹
- 状态：FROZEN

## 8. 数据审计

- ARForensics manifest（98,000 行，含 sha256/phash）：`data/manifests/arforensics.csv`；审计报告 `reports/dataset_audit/arforensics.json`
- 标签/generator 元数据/重复路径：0 错误
- 精确重复：17 组，其中 **3 组跨 train/test**（`train/0_real` 与 Switti/LlamaGen/Infinity `0_real` 各 1 张 ImageNet 图 sha256 相同）——真实但极小规模（3/98,000）的 train-test 泄漏，源自 ImageNet 图像内容重复，结果解读时需声明
- 近似重复（pHash ≤ 8/64）：1,238 候选对，多为同 generator 同类别生成图，需人工复核，不直接判为重复
- DARG manifest：`data/manifests/darg_t2i.csv`（6,000 行）、`darg_c2i.csv`（160,000 行）
- 状态：FROZEN

### 8.1 格式/压缩偏差与统一重编码对照（v3 P1-D）

**发现**：ARForensics real 100% 为 JPEG（48,997 张），7 个生成器的 fake 100% 为 PNG
（`reports/dataset_audit/format_bias_arforensics.json`）。检测器可能利用
"JPEG vs PNG" 的格式痕迹而非生成痕迹——这是必须控制的混杂因素。

**对照实验**（UnivFD，同一 500/类 采样协议，7000 张）：
把所有图片统一 decode→RGB→JPEG Q95 重编码（`data/ARForensics_reencode`）后重评估：

| 指标 | 原始（JPEG real / PNG fake） | 统一 JPEG95 后 | Δ |
|---|---|---|---|
| overall AUC | 0.9530 | 0.8893 | **-0.064** |
| overall acc | 0.8614 | 0.7414 | -0.120 |
| real acc | 0.9260 | 0.9454 | +0.019 |

解读：格式统一后 AUC 仍达 0.889，说明 UnivFD 大部分判别力来自生成痕迹而非纯格式捷径；
但约 6 个 AUC 点的差距表明显式格式混杂确实抬高了原始数字。
**后续 benchmark 主表应同时报告原始与统一重编码两组结果。**

### 8.2 跨 split 泄漏清理（v3 P0-B）

数据审计发现的 3 组 train/test 精确重复已按 sha256 剔除（保留 test 侧，删除 train 侧），
生成 `data/manifests/arforensics_clean.csv`（97,997 行）与
`data/manifests/leakage_exclusions.csv`。注意：已冻结结果是在清理前数据上得到的，
3/98,000 的泄漏对指标影响可忽略，但正式论文结果必须基于 clean manifest。

## 9. DCA-Fuse 全量 test（84k，α=0.7，阈值 0.805 val 校准）

归档：`reports/runs/dca_fuse_arforensics_test_full/`（含 predictions.csv.gz）。

| 生成器 | UnivFD AUC | DCA-Fuse AUC | Δ |
|---|---|---|---|
| LlamaGen | 0.9599 | 0.9612 | +0.001 |
| RAR | 0.9697 | 0.9706 | +0.001 |
| VAR | 0.9560 | 0.9490 | -0.007 |
| Janus_Pro | 0.9362 | 0.9360 | ~0 |
| Switti | 0.8507 | 0.8431 | -0.008 |
| Open_MAGVIT2 | 0.8863 | **0.9053** | **+0.019** |
| Infinity | 0.8305 | 0.8244 | -0.006 |
| **overall** | 0.9127 | **0.9128** | ~0 |

全量结论：融合整体与 UnivFD 持平，Open_MAGVIT2 +1.9 点（机制通道在其
自身 tokenizer 家族信号最强），Infinity/Switti 略降。注：sub500 分析中
观察到的 worst-case 改进在全量上未复现（样本协议差异），如实记录。

## 10. 进行中的全量实验

- DIRE test 全量 84k：已完成，overall AUC 0.830 / AP 0.838（迁移测试，见第 4 节）
- PRADA T2I 全量（官方 predict.py，DARG-adapted-real-v1）：**已完成**，macro AUROC 0.9898（见第 5 节）
- CTRF 关系结构 MVP（v5 指南主线）：已验证，**负结果**——Mahalanobis 关系版 macro 0.694 vs S_min 0.695 持平，未过 GO 门槛（macro +0.03 / worst-gen +0.02），按预注册标准停止该主线。详见 `docs/method_dca_fuse.md` 末节。

## 11. DCA-Fuse 路径配对重评估（84k，α=0.7）

归档：`reports/runs/dca_fuse_arforensics_test_full_recalibrated/`；
原始复评结果：`outputs/dca_fuse/results_test_full_recalibrated.json`；
逐样本分数：`outputs/dca_fuse/results_test_full_recalibrated_predictions.csv`。
84,000 行预测按图像路径连接 UnivFD 与 CTAD，包含 42,000 real 与 42,000 fake；
overall 只计入一次 real pool。阈值在当前 α 对应的 validation fused scores 上
重新用 Youden's J 校准（0.80462）。

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

Overall CI 跨 0，故不能声称总体 AUROC 提升。Open_MAGVIT2 上的收益与
Switti/VAR/Infinity 上的退化同时存在，支持“生成器相关的互补信号”，
而非整体 SOTA。若多个路径代表相同底层图像，后续应使用内容/来源级
cluster bootstrap。

同日完成的经验尾部 CTAD pilot（500 fit real + 500 calibration real，500/类 test）
macro AUROC 为 0.7003（旧 `S_min`: 0.6952），最差生成器 0.5585（旧
`S_min`: 0.5594）；nominal 5% conformal p-value 在 test real 上观测 FPR
为 3.89%。此结果仅属探索性，不作为 open-set claim。

## 12. PARD P0 MVP（v6 指南主线，subsample-500 示例）：负结果，按预注册停止

脚本：`analysis/pard_mvp.py`（+ 诊断消融 `analysis/pard_ablation.py`）；
结果：`outputs/pard/pard_mvp.json` / `pard_ablation.json`；
特征缓存：`outputs/pard/pard_mvp_feats.npz`。

设计（v6 §2 的 MVP 简化）：DINOv2 ViT-S/14 冻结（与 RIGID 同骨干，隔离
"响应建模"贡献），5 类受控扰动 × 2 强度（JPEG95/75、Resize 0.9/0.75、
Gaussian Blur、Noise 0.05/0.10、Crop-Resize），4 个中间层 patch 特征的
一阶响应 Δ、二阶响应 K 与四象限空间统计构成 370 维描述子；
val real 1000（500 fit / 500 cal）real-only 建模，LedoitWolf Mahalanobis /
kNN(k=10) 异常评分；test 每生成器 500 real + 500 fake（seed=0，
**subsample-500 示例协议**）。RIGID 分数在同批图像、同骨干上同步计算。

| 生成器 | PARD-kNN | PARD-Mahalanobis | RIGID（同子集） |
|---|---:|---:|---:|
| Infinity | 0.663 | 0.534 | 0.689 |
| Janus_Pro | 0.720 | 0.574 | 0.658 |
| LlamaGen | 0.649 | 0.580 | 0.560 |
| Open_MAGVIT2 | 0.551 | 0.485 | 0.662 |
| RAR | 0.606 | 0.550 | 0.676 |
| Switti | 0.600 | 0.562 | 0.634 |
| VAR | 0.553 | 0.524 | 0.518 |
| **macro** | **0.620** | **0.544** | **0.628** |
| worst | 0.551 | 0.485 | 0.518 |

预注册 GO 判定（相对同子集 RIGID）：macro 增益 -0.008（要求 ≥+0.03），
worst 增益 +0.032（要求 ≥+0.05），不差于 RIGID 的生成器 3/7（要求 ≥4）
——**三条均未通过，GO=false，按 v6 停止条件不扩大规模**。

诊断性消融（post-hoc，不作为 GO 依据；kNN macro）：
first-order 0.626 / second-order 0.631 / patch-only 0.629 /
global-CLS-dynamics-only 0.476（多扰动全局动态不如 RIGID 单噪声）；
按扰动：noise 0.631 > crop 0.620 > blur 0.607 > jpeg 0.574 > resize 0.498；
按层：layer2 0.635 > layer5 0.628 > layer8 0.591 > layer11 0.572
（低层特征略优）。**没有任何子块超过 RIGID +0.03**，排除"特征选择不当"
作为主要败因。

结论：在 DINOv2 冻结特征空间中，受控扰动的局部响应动力学经 real-only
统计建模后，不提供超越 RIGID 全局相似度的稳定跨生成器 AR 检测信号。
结合 CTRF 负结果，real-only/免训练的冻结特征统计建模路线在 ARForensics
上饱和于 RIGID 量级（0.63–0.74），显著低于监督式 UnivFD（0.913）。
PARD 作为独立方法主线停止；若未来重启，需先解决参考分布对内容/格式的
敏感性（val real 与 test real 的来源差异）这一混杂因素。

## 13. MPID P0（v7 指南主线，DARG T2I LOGO-4）：负结果，按预注册停止

脚本：`analysis/mpid_p0.py`；结果：`outputs/mpid/mpid_p0.json`；
缓存：`mpid_p0_targets.npz`（PRADA 教师分数）、`mpid_p0_clip_feats.npz`。

设计：PRADA 完整分数（max over 4 models，离线复算与官方
`auroc_prada_test.csv` 完全一致，0.984–0.995）作为训练阶段特权教师；
学生 = CLIP ViT-L/14 冻结特征 + MLP(768→256→1)，A=BCE-only vs
B=BCE+MSE(λ=1) 机制蒸馏，同骨干/数据/预算；LOGO-4（每折留 1 个 T2I
生成器完全不参与训练）；800/200 train/test（seed 0，每目录 1000 张，
**mpid-p0-1000 示例协议**）。

| held-out 生成器 | A: BCE-only | B: +机制蒸馏 | 增益 | genprobe A→B |
|---|---:|---:|---:|---:|
| infinity_2b | 0.9989 | 0.9988 | -0.0001 | 0.935→0.967 |
| janus_1b | 0.9965 | 0.9963 | -0.0003 | 0.941→0.960 |
| llamagen_xlstage2 | 0.9979 | 0.9978 | -0.0001 | 0.947→0.978 |
| switti_1024 | 0.9987 | 0.9983 | -0.0004 | 0.940→0.972 |
| **macro** | **0.9980** | **0.9978** | **-0.0002** | 0.94→0.97 |

预注册 GO 判定：macro 增益 -0.0002（要求 ≥+0.02）；提升折数 0/4
（要求 ≥3）；JPEG95 增益 -0.0001（要求 >0）——**三条均未通过，
GO=false，按 v7 停止条件终止 MPID 主线**。

两个更重要的诊断发现：

1. **教师分数不可蒸馏（P0-1）**：Ridge 探针从冻结 CLIP 特征回归 PRADA
   分数，held-out 生成器上 Spearman ρ 中位数仅 **0.026**
   （-0.07 ~ +0.11），单特征目标同样不可预测。PRADA 的机制证据
   （似然比/量化误差）不线性编码在 CLIP 语义表征中，且该不可解码性
   跨生成器成立——蒸馏前提本身不成立，而非蒸馏方法问题。
2. **天花板效应**：BCE-only 学生在 DARG T2I LOGO 上已达 0.9965–0.9989，
   协议接近饱和，即便机制信号可蒸馏也无 +0.02 的余量；DARG T2I 的
   real/fake 对 CLIP 学生而言已被简单真假线索分开。
3. 副产物：机制蒸馏**提高**了生成器身份探针准确率（0.94→0.97），
   与 generator-invariance 目标相反，印证 v7 对"蒸馏可能把
   fingerprint 带给学生"的担忧。

结论：在"CLIP 冻结特征 + 轻量学生 + DARG T2I"设置下，机制特权蒸馏
既不必要（已饱和）也不可行（机制证据不可从视觉表征线性解码）。
至此 v5–v7 三条新机制主线（CTRF、PARD、MPID）均完成 P0 并如实记录
负结果；仓库最有价值的资产仍是严格复现的 baseline 矩阵、PRADA T2I
强参照（0.9898）与数据审计。若未来重启机制蒸馏，需要：(a) 换用更接近
机制信号的表征（如 tokenizer 域特征）而非语义 CLIP；(b) 选择未饱和的
评测协议（更弱的 backbone 或更难的 real/fake 分布）。

## 14. v8 复核与 STEM-AR 诊断（MPID 修正版 + DARG 审计 + STEM-AR）

### 14.1 MPID 严格 teacher holdout 重跑（v8 §2.2 修正）

脚本：`analysis/mpid_p0_strict.py`；结果：`outputs/mpid/mpid_p0_strict.json`。
修正：(a) 每折教师目标改为 max over 3 个非 held-out PRADA 模型（原版
教师集合包含 held-out 模型）；(b) 生成器身份探针改在独立 test 折评估
（原版为训练集内评估）。其余协议不变。

| 指标 | BCE-only | +机制蒸馏（严格教师） |
|---|---:|---:|
| LOGO macro AUROC | 0.9980 | 0.9979（增益 -0.0002） |
| 提升折数 | — | 1/4（要求 ≥3） |
| JPEG95 增益 | — | -0.0001（要求 >0） |
| genprobe（独立测试） | 0.869 | 0.885 |

**结论不变：GO=false。** v8 指出的训练集内探针高估成立（0.94→0.87），
但蒸馏仍使独立测试 genprobe 上升（0.869→0.885），方向性结论保持；
严格教师 holdout 下教师可预测性更低（Spearman 中位 0.010）。

### 14.2 DARG T2I 难度审计（v8 §2.3，P0-B）

脚本：`analysis/darg_audit.py`；结果：`outputs/mpid/darg_audit.json`。

- **近重复泄漏：无。** train/test 目录内 pHash≤6 近重复对 0（janus 1 对
  边界值 hamming=6）；real↔fake 跨目录 0 对（min hamming ≥10）。
- **元数据/格式 shortcut：极显著。** 仅用 [log文件大小, 宽, 高, 元数据
  键数, 颜色模式] 的逻辑回归 5 折 CV，real vs 每个生成器 AUC 全部
  = **1.0000**。原因：real 为 JPEG/RAISE 风格多种分辨率，全部 fake 为
  PNG 固定方形分辨率（1024²/384²/512²）。
- 但像素级 codec-matched（双方 JPEG95，CLIP 管线统一到 224）下 MPID
  BCE-only 仍 0.9974——**简单性主要来自内容域差异**（RAISE 真实照片
  vs T2I 合成内容的语义域差），而非压缩格式本身。
- 结论：DARG T2I LOGO 0.998 是"当前数据划分下的表现"（v8 原话），
  不能解释为开放世界未见生成器检测已解决；后续主实验应采用
  codec+domain 匹配协议。DARG 原始 prompt 本地不可得，prompt 级审计
  以图像级近重复替代（边界记录）。

### 14.3 STEM-AR 机制图可迁移性诊断（v8 §9 P0-C）：负结果

脚本：`analysis/stem_ar_p0.py`；结果：`outputs/mpid/stem_ar_p0.json`。

token-to-grid 验证（通过）：infinity_2b（13 scales 全平方，取原生
16×16）、switti_1024（14 scales 全平方，取原生 16×16）、
llamagen_xlstage2（32×32→2×2 池化，精确）、janus_1b（24×24→面积
插值，近似已记录）；光栅顺序为 VAR/AR 标准生成顺序。

教师图 = 各生成器自身模型的 e(k,t)=cond−uncond 最细尺度 16×16 证据图；
学生特征 = DINOv2 ViT-S/14 depth-11 patch tokens（256×384）；探针 =
per-cell 闭式 ridge（对偶形式，λ=1）；LOGO-4，仅 fake；shuffled
位置对照；orig 与 jpeg95 分别训练评估。

| codec | held-out | ρ(real) | ρ(shuffled) |
|---|---|---:|---:|
| orig | infinity_2b | 0.029 | 0.029 |
| orig | janus_1b | 0.035 | 0.035 |
| orig | llamagen_xlstage2 | -0.004 | -0.004 |
| orig | switti_1024 | 0.021 | 0.021 |
| jpeg95 | （四折同型） | ≈0.018 macro | 完全相同 |

**所有折、两种 codec 下 ρ(real) 与 ρ(shuffled) 完全相等（margin=0.0）**：
DINOv2 局部 patch 特征对 token 级机制证据图没有任何可迁移的空间预测
能力（微弱 ρ≈0.02 为探针噪声，与位置对齐无关）。预注册 GO 三条全部
失败，**按 v8 停止条件终止 STEM-AR**。

与 MPID 联合解读：PRADA 机制证据既不能作为标量从 CLIP 全局特征解码
（MPID，ρ≈0.03），也不能作为空间图从 DINOv2 局部特征解码（STEM-AR，
margin=0）。机制证据与冻结视觉表征之间的不可解码性在两种粒度上均成立，
机制蒸馏路线（v7/v8 候选）整体终止。

## 15. RACE 复现（v8 §9 P0-A，arXiv:2611.27161，预印本无官方代码）

脚本：`analysis/race_p0.py`（+诊断 `race_cfe_diagnose.py`、缓存迁移
`race_reextract_clip.py`）；结果：`outputs/race/race_p0.json`、
`race_cfe_diagnose.json`；复现说明：`baselines/reproduction/race.md`。
**复现级别：reimplementation-approximate**（无官方代码；门控/投影头结构、
子损失定义、NP-RSR 描述子统计均为按论文文字的最直接解释，逐条记录在
脚本头与 race.md）。协议：ARForensics real（train+val 7000 张，划分为
3500 train / 1000 dev / 2500 cal）替代论文 mixed authentic dataset；
test 7 生成器 × 1000/类（seed 0，**race-p0-1000 示例**），orig+JPEG95。

### 总体结果（本复现，非论文数值）

| 指标 | orig | JPEG95 |
|---|---:|---:|
| macro AUROC | 0.744 | 0.547 |
| worst（VAR） | 0.282 | 0.420 |
| FPR@τ（校准名义 5%） | 4.1–5.7% | **29–33%** |

分量 AUROC（orig）：NP-RSR 0.65–0.985（最强）；PRP 0.58–0.93；
**CFE 检索 0.096–0.461（全生成器反向）**。校准在 codec 内成立
（orig 测试 real 的 FPR≈名义 5%），JPEG95 后真实分数分布整体漂移、
校准阈值失效——real-anchored 统计校准对 real 侧 codec 漂移脆弱。

### 关键发现：真实参考检索在本协议下反向

- 冻结 DINOv2 depth-12 KNN 检索：macro **0.304**（反向）；
  冻结 CLIP pooled：0.611；拼接：0.592；CFE 训练后学习码仍反向
  （α 权重最终 DINO 0.910 / CLIP 0.090）。
- 解读：ARForensics 的 AR 生成图（ImageNet 内容/提示驱动）在冻结视觉
  空间中**比 ImageNet-test 真实图更接近 ImageNet-val 派生的真实参考库**
  ——"真实邻近性"在该数据集上不是有效的真假证据，这是数据/协议属性，
  不是 CFE 实现问题（冻结变体同样反向）。
- 因此本协议无法验证论文 ARForensics 94.69% 的数值；差距主要来源是
  真实图像池与 benchmark 协议差异，其次才是近似复现的实现偏差。

### 参照（同 subsample-1000，orig）

UnivFD（best.pth head 于同一批 CLIP 特征）：macro **0.908**；
RIGID（84k 全量归档）：0.7407；RACE 本复现：0.744（≈RIGID 量级，
远低于 UnivFD）。paired bootstrap 未做（记录在后续工作）。

### 复现过程事件（透明记录）

初版自定义 CLIP 中间层前向误用 LND 布局（当前 open_clip 为 batch-first），
导致 CLIP 分支特征为常量、首轮 CFE 无效；已修复（修复后 pooled 与官方
前向 cos=1.0）并重提特征重跑，本表为修复后结果。该 bug 仅影响
`analysis/race_p0.py` 自定义前向，不影响 MPID/PARD（官方前向）。

## 16. CARF P0（v9 指南主线）：**GO=true**（首个通过预注册标准的路线）

脚本：`analysis/carf_p0.py`；结果：`outputs/carf/carf_p0.json`。
协议：与 RACE P0 同 manifest（3500 参考 real / 2500 校准 real / 1000/类
test，race-p0-1000 示例）；A=全部统一 JPEG95（codec-matched，主协议），
B=orig as-is（对照）。语义条件 c=CLIP pooled（仅用于匹配参考，不作检测
分数）；采集条件 a=[log 文件大小, log 宽, log 高]；取证残差 r24=NP-RSR
末端残差 24 维 / r960=中间统计 960 维；M0 无条件收缩高斯，M1 内容核，
M2 采集核，M3 联合核；ESS<50 退回 M0；cal real conformal 校准（名义
FPR 5%）；test real 不参与任何拟合/带宽/阈值。

### 主协议 A（codec-matched JPEG95）r24 逐生成器

| 生成器 | M0 p(r) | M1 p(r\|c) | M2 p(r\|a) | M3 p(r\|c,a) | RACE NP-RSR |
|---|---:|---:|---:|---:|---:|
| Infinity | 0.890 | 0.890 | 0.903 | **0.904** | 0.578 |
| Janus_Pro | 0.836 | 0.836 | 0.938 | **0.938** | 0.507 |
| LlamaGen | 0.734 | 0.734 | 0.853 | **0.853** | 0.516 |
| Open_MAGVIT2 | 0.629 | 0.630 | 0.679 | **0.680** | 0.422 |
| RAR | 0.617 | 0.617 | 0.665 | **0.665** | 0.467 |
| Switti | 0.778 | 0.777 | 0.876 | **0.875** | 0.559 |
| VAR | 0.642 | 0.643 | 0.697 | **0.697** | 0.497 |
| **macro** | **0.732** | **0.733** | **0.802** | **0.802** | **0.507** |

### GO 判定（预注册，协议 A / r24）

| 条件 | 要求 | 实际 | 判定 |
|---|---|---|---|
| M3 > M0 | +CI 不含 0 | +0.070 [+0.042, +0.094] | ✓ |
| M3 > RACE NP-RSR | +CI 不含 0 | +0.295 [+0.241, +0.356] | ✓ |
| real FPR | ∈[3%,7%] | 5.7% | ✓ |
| 不退化生成器 | ≥5/7 | 7/7 | ✓ |

**GO=true。**（bootstrap 以生成器为单元的 cluster 级；逐图 bootstrap 列为
后续工作。）

### 诚实解读（重要）

1. **增益来源几乎全部在采集条件**：M1(仅内容)≈M0，M2(仅采集)≈M3——
   v9 的"内容条件化"假设未获支持，"采集条件化"假设获得支持。
2. **绝对性能仍低于监督基线**：M3 0.802 << UnivFD 同 subsample 0.908。
   CARF 的贡献定位应是"无真假训练数据下的条件化残差检测 + 可靠校准"，
   而非超越 UnivFD。
3. **协议 B（混合格式）下反转**：RACE NP-RSR 0.858 > CARF M3 0.652——
   当 real/fake 格式不同时 AE 重建误差吸收了格式捷径，CARF 的优势仅在
   codec-matched 协议下成立。
4. 覆盖率 ≈1.0（ESS 中位 ~3000），参考库规模在当前协议下充足。
5. r960 条件化同样有效（0.588→0.678）但整体弱于 r24。

### 下一步（v9 P1）

P0 通过，按 v9 进入 P1。计划：更大规模验证（3000/类，标注
carf-p1-3000 示例）、带宽/ESS/收缩强度消融、3 种子；84k 全量视 P1-lite
稳定性决定。若 P1 稳定，论文定位：条件化真实取证分布 + 组内校准的
可靠性贡献（配合 RACE 反向/校准漂移的协议分析）。

### 16.1 CARF P1-lite：3000/类规模复核 + 超参消融（carf-p1-3000 示例）

脚本：`analysis/carf_p1.py`；结果：`outputs/carf_p1/carf_p1.json`。

**规模复核**（协议 A，r24，3000/类，参考/校准集不变）：

| 模型 | macro AUROC | worst | FPR@5%cal |
|---|---:|---:|---:|
| M0 无条件 | 0.736 | 0.620 | 5.0% |
| M2 采集条件 | 0.806 | 0.667 | 5.5% |
| M3 联合 | 0.806 | 0.667 | 5.5% |

P0 结论在 3 倍规模上精确复现：ΔM3−M0 = **+0.070**（与 P0 的 +0.070
一致），7/7 生成器不退化，FPR 5.5% 在名义区间内。逐生成器提升与 P0
同构（Janus +0.104、Switti +0.097、LlamaGen +0.124 最大）。

**超参消融**（P0 规模，M2 分量）：
- 核带宽：σ×0.5 → 0.825 / σ×1.0 → 0.802 / σ×2.0 → 0.779（更窄带宽
  更局部化有益；默认 σ 为中位数启发式预设，未在 test 上调参，
  σ×0.5 的增益仅作敏感性记录）
- ESS_MIN {20,50,100}：0.803/0.802/0.801（不敏感）
- 收缩 λ {0.05,0.1,0.5}：0.805/0.802/0.779（小 λ 稳健）

**结论**：CARF 的采集条件化收益对规模和超参稳健。按 v9 决策树，84k
全量与多种子属 P1 后续可选项；鉴于结论在 1000→3000 规模间已稳定且
绝对性能（0.806）仍低于 UnivFD（0.908 同协议），论文贡献定位保持为
"无真假训练数据的条件化残差检测 + 可靠校准"，而非更强检测器。

## 17. CARF-A 协议修复版（v10 第一部分：5 项审计修复）

脚本：`analysis/carf_a.py`；结果：`outputs/carf_a/carf_a.json`
（逐图分数 `scores_*.npz`）。修复：(1) 带宽 σ 只在 reference 上估计并
冻结（原按 calibration/test 查询批次分别估计，属 transductive 评分）；
(2) M2/M3 各自独立校准阈值；(3) 真 Mondrian 分辨率桶分组校准；
(4) 逐图分数 + 逐图 paired bootstrap（替代 7 点 cluster bootstrap）；
(5) 采集条件变量消融与标签泄漏审计。协议 A（全 JPEG95）r24，1000/类。

### 条件变量标签泄漏审计（v10 §3.5 要求的审计，重要）

条件变量本身做逻辑回归（5 折 CV）区分 real vs fake 的 AUROC：
**full [size,w,h] 0.995 / wh_only 0.958 / size_only 0.766 /
size_per_px 0.995**。采集条件变量携带显著标签信息（分辨率与文件大小
分布在 real/fake 间不同），M2 增益不能全部归于 nuisance 匹配。

### 修复后结果（固定带宽、独立校准、逐图 bootstrap）

| 采集变量变体 | 泄漏 AUC | M2 macro | ΔM2−M0 (95% CI) | 不退化 | FPR |
|---|---:|---:|---:|---:|---:|
| full [size,w,h] | 0.995 | 0.813 | +0.081 [+0.078, +0.084] | 6/7 | 5.6% |
| wh_only | 0.958 | 0.755 | +0.023 [+0.021, +0.026] | 4/7 | 5.0% |
| size_only | 0.766 | 0.824 | +0.092 [+0.088, +0.096] | 7/7 | 5.3% |
| size_per_px | 0.995 | 0.840 | +0.108 [+0.104, +0.112] | 6/7 | 5.0% |

**GO（full 变体，预注册）：通过**——Δ>0 且 CI 不含 0；FPR 5.6% ∈
[3%,7%]；6/7 ≥5。修复协议（inductive 固定带宽）下结论保持，且略优于
原 transductive 版本（0.813 vs 0.802）。

### 分解解读（对 v9/v10 结论的重要修正）

- **分辨率条件贡献最小**（wh_only +0.023），**文件大小条件贡献最大**
  （size_only +0.092）——增益主要来自"内容复杂度匹配"（JPEG95 文件
  大小 ∝ 图像熵/纹理复杂度），而非分辨率/格式匹配。残差统计对图像
  复杂度敏感（平坦图 vs 纹理图的残差分布不同），按复杂度匹配真实
  参考降低了真实分布方差——这是真实的条件化机制，但"采集条件"应
  如实称为"可观测编码属性 + 内容复杂度代理"（v10 §2 已预警）。
- 泄漏审计显示条件变量本身可分真假（0.77–1.0），故 +0.08~0.11 是
  混合了真实条件化收益与分布差异的上限；wh_only 的 +0.023 是较干净
  的下限（其泄漏 0.958 仍不完全干净）。
- Mondrian 逐桶 FPR 与逐图分数已保存，供论文可靠性分析。

## 18. CoNTra P0（v10 第二部分）：负结果，nuisance 与信号同向

脚本：`analysis/contra_p0.py`；结果：`outputs/contra/contra_p0.json`。
设计：3500 真实参考 × 6 反事实视图（JPEG QF95/85/75/50 + resize
0.9/0.75，canonical=QF95）的 NP-RSR r24 残差；source-disjoint
fit(2500)/heldout(1000)；配对差分 δ_i(a)=r_i(a)−r_i(a0)；SVD 低秩基
(k=6)；跨 codec MMD 漂移；信号保留率 η=||P_⊥U·Δ_fake||²/||Δ_fake||²。

### GO 判定（预注册）

| 条件 | 要求 | 实际 | 判定 |
|---|---|---|---|
| 子空间稳定性 | per-view cos ≥0.9 | min 0.989（SVD 修复后主角度均值仅 1.9°） | ✓ |
| held-out 可预测性 | cos 中位 ≥0.5 | 0.939 | ✓ |
| 跨 codec 漂移下降 | MMD −≥30% | **−88.2%**（0.692→0.082） | ✓ |
| 信号保留率 η | ≥0.8 | **0.283 / 0.040 / 0.025** | **✗** |

**GO=false，按 v10 停止 CoNTra 路线。**

### 核心发现（对本项目最重要的理论性结果之一）

配对反事实实验干净地证明：真实图像的采集 nuisance response **高度可
识别**（近似秩一，第一主成分解释 97.7% 方差，跨 source 稳定 cos≈0.99，
mean-correction 可消除 88% 跨 codec 真实分布漂移）。**但 AR 生成图像
的异常信号几乎完全落在同一个方向上**——将残差投影到 nuisance 子空间
的正交补后，fake−real 差异仅剩 2.5%–28%。即：**压缩敏感性方向同时
是"采集 nuisance"方向和"真假判别"方向，两者在该残差空间中不可分离**。
canonical transport 在消除 codec 漂移的同时也消除了检测信号。

跨 codec AUROC 佐证（Infinity）：canonical QF95 0.900 → QF85 raw 0.659 /
mean 0.861 / lowrank 0.813 → QF50 raw 0.352 / mean 0.669。mean-correction
恢复了部分鲁棒性，但恢复后仍显著低于 canonical——校正与信号的对价
（tradeoff）是结构性的，不是调参问题。

### 对 CARF-A 的回溯解释

这解释了 CARF-A 的工作机制：条件化不是"消除 nuisance"，而是在复杂度
匹配的参考子集中利用**同一共享方向的幅度差异**做判别。v10 设想的
"先去除采集效应再检测"在此残差空间中不成立——采集效应就是检测信号。

### 复现过程事件

初版 SVD 误取样本空间左奇异向量（12500 维）为基，导致 71° 假不稳定
与下游 shape 错误；修正为残差空间右奇异向量 Vt（24×6）后，子空间实
际高度稳定（主角度均值 1.9°）。初版 30° 主角度 GO 判据对近似秩一响
应过严，修正为 per-view mean response 余弦判据（≥0.9），原始判据保
留为诊断量。两次修正均记录在 `outputs/contra/contra_p0.json`。

## 19. SACR P0（v11）：**GO=false，按预注册标准停止方法扩展**

脚本：`analysis/sacr_p0.py`；测试：`tests/test_sacr_p0.py`；结果：
`outputs/sacr/sacr_p0.json`、`canonical.json`、`complexity_audit.json`。
逐图预测：`scores_seed{0,1,2}.npz` 与
`cross_scores_seed{0,1,2}_{qf85,qf75}.npz`。本节为
**sacr-p0-1000 / cross-300 示例**，不是 84k 全量结果。

### 19.1 方法与严格协议

SACR 不再投影删除 CoNTra 的 nuisance 方向，而是保留
`z_parallel=U^T r` 并建模其条件真实分布。U 由 reference real 的
JPEG/resize 配对差分估计（k=6）；C 仅评共享方向，D 评
`z_parallel + λ z_perp`。条件变量包括文件大小、解码像素复杂度、codec、
像素+codec；评分器包括条件 Gaussian/Mahalanobis、conditional kNN、
条件分位数。模型分支、评分器和 λ 只在独立 validation 上选择。

- clean manifest 剔除旧 P0 中 3 张已知跨 split 重复真实图：reference
  3500、calibration 2498；validation 999 real + 1000 fake；三个 seed
  真正重划 reference/calibration，而非仅更换 bootstrap 随机数。
- 测试为 7 生成器 × 1000 real/fake，统一 JPEG95；跨 codec 为
  Infinity/Open_MAGVIT2/VAR × 300 real/fake，JPEG85/75。
- 像素复杂度从统一解码后的三个 128×128 patch 计算：多尺度梯度能量、
  局部熵、Haar 高频能量、DCT 熵、边缘密度。不是文件元数据。
- ESS<50 输出 unsupported/abstain；按 reference 局部熵中位数组内
  conformal 校准。AUROC 在 supported 样本上计算，同时报告 coverage。
- 预注册主标准：pixel SACR 相对每 seed 较强的 CARF-A(full/size_only)
  三种子平均 Δmacro≥+0.02、逐图 paired bootstrap CI 下界>0、每 seed
  Δ>0；≥5/7 不退化；FPR∈[3%,7%]；coverage≥95%；JPEG85/75 仍有收益。
  仅 file-size 有效或复杂度自身标签 AUC≥0.80 均不能通过 GO。

三个 seed 的 pixel 条件均独立选择 **D + conditional kNN + λ=0.25**；
没有选择测试集上的最优配置。

### 19.2 Canonical JPEG95：三种子平均 AUROC

| 生成器 | M0 | CARF-A full | CARF-A size-only | SACR file-size | SACR pixel | SACR codec |
|---|---:|---:|---:|---:|---:|---:|
| Infinity | 0.890 | 0.870 | 0.894 | 0.928 | 0.851 | 0.864 |
| Janus_Pro | 0.837 | 0.928 | 0.945 | 0.930 | 0.817 | 0.821 |
| LlamaGen | 0.733 | 0.841 | 0.868 | 0.920 | 0.802 | 0.747 |
| Open_MAGVIT2 | 0.631 | 0.692 | 0.718 | 0.758 | 0.756 | 0.647 |
| RAR | 0.619 | 0.680 | 0.706 | 0.759 | 0.753 | 0.631 |
| Switti | 0.776 | 0.896 | 0.889 | 0.823 | 0.809 | 0.791 |
| VAR | 0.643 | 0.716 | 0.742 | 0.769 | 0.781 | 0.653 |
| **macro** | **0.733** | **0.803** | **0.823** | **0.841** | **0.796** | **0.736** |

SACR file-size 虽达到 0.841，但平均 coverage 仅约 92.7%，且违反“核心
方法不能依赖文件大小”的预设，不能作为成功结果。SACR pixel coverage 为
99.7%，FPR 三种子为 5.4%/5.7%/6.2%，但 macro TPR@1%FPR 仅
0.26%–0.29%；校准合格不能弥补判别力不足。

### 19.3 GO 判定

| 条件 | 实际 | 判定 |
|---|---|---|
| 相对强 CARF-A 平均 Δ≥+0.02 | **−0.0275**，95%CI [−0.0351,−0.0280] | ✗ |
| 三个 seed 各自 Δ>0 | −0.0115 / −0.0392 / −0.0319 | ✗ |
| ≥5/7 生成器不退化 | 3/7、3/7、3/7 | ✗ |
| FPR∈[3%,7%]、coverage≥95% | FPR 5.4%–6.2%，coverage 99.7% | ✓ |
| JPEG85/75 收益与 FPR | AUROC 有提升，但部分 seed FPR 7.15%/7.25% | ✗ |
| 复杂度不是明显标签捷径 | validation AUC 0.836；within-source 0.817 | ✗ |

**GO=false。** 主 pixel 路线在严格协议下未超过 CARF-A，按 v11 停止
SACR 方法扩展，不进入 84k、外部 benchmark 或额外模块堆叠，也不开展
RA-Det/DEAR/Forensic Twins 的 SACR 同协议后续对比。

### 19.4 跨 codec 分解：增益主要来自同质量参考，而非像素条件

| 方法 | 参考条件 | JPEG85 macro | JPEG75 macro |
|---|---|---:|---:|
| CARF-A size-only | QF95 reference | 0.546 | 0.477 |
| SACR pixel | QF95 reference | 0.348 | 0.312 |
| SACR codec-only | 同质量 reference | 0.768 | 0.690 |
| SACR pixel+codec | 同质量 reference + 像素复杂度 | 0.786 | 0.711 |

pixel+codec 相对 CARF-A 的表观提升为 +0.240/+0.234，但 codec-only 已
解释其中绝大部分。相对 codec-only，三种子增量在 JPEG85 为
−0.009/+0.024/+0.040，在 JPEG75 为 +0.012/+0.013/+0.039；增益不稳定，
且不是 canonical 上所需的 +0.02 可重复优势。同质量真实参考本身是重要
的协议变量，不能将其贡献归因于 SACR 的像素条件创新。

### 19.5 结论与论文定位

1. CoNTra 的“nuisance–signal overlap”科学发现仍成立，但 SACR 的
   “干预子空间 + 像素复杂度 + 条件评分”没有转化为超过 CARF-A 的检测器。
2. CARF-A 的收益主要来自文件大小/内容复杂度代理；SACR 的可解释像素
   统计并没有消除标签关联（组内审计 AUC 仍约 0.82），也没有取得更好
   的条件分布估计。这是机制层面的负结果，不是调参不足。
3. 现阶段保留 CARF-A 为唯一通过协议修复后 GO 的新方法基线；论文主线
   应回到**条件分布错配、nuisance–signal coupling、校准与参考协议的
   系统性分析**，不再为了 ICML 方法论文继续叠加检测模块。
4. 验证：37 项项目测试通过；另外从逐图 NPZ 独立重算 126 组 canonical
   AUROC/coverage，全部与 JSON 一致；6 个跨 codec 文件均含 15 种方法
   × 600 张的完整逐图分数。

## 20. v12 P0/P1：跨特征、跨 codec 与 DARG 外部复核

脚本：`analysis/carf_protocol_p1.py`、`analysis/forensic_bank.py`；理论：
`docs/nuisance_signal_coupling.md`；结果：
`outputs/carf_protocol/carf_protocol_p1.json`；外部重复审计：
`external_duplicate_audit.json`。本节为
**carf-p1-arforensics-1000/cross-300 + darg-t2i-300 示例**，不是全量评估。

### 20.1 固定协议与独立表示

- ARForensics 沿用 SACR clean manifest：3500 reference、2498 calibration、
  1000/类 test，三个真正重划 reference/calibration 的 seed。
- 外部数据为本地 DARG T2I：1000 RAISE real 每 seed 分为
  400 reference/300 calibration/300 test；5 个 AR 生成器各 300 fake。
  2500 张使用图像中 SHA256 重复=0、pHash 完全重复=0、真实图跨 split
  pHash Hamming≤4 候选=0。
- 两种表示：(1) 既有 Noiseprint++ NP-RSR r24；(2) 独立固定高通滤波器组
  （水平/垂直二阶差分与 Laplacian），每个滤波器提取分布、自相关和径向谱
  统计，共 24 维。后者是 SRM-inspired 固定表示，不是完整 SRM 官方模型。
- CARF-A full 为唯一主配置，size-only 为诊断；没有新增条件特征或调整
  SACR。JPEG85/75 评估 QF95-only、same-quality、mixed-codec 三种参考池，
  每个 quality 独立校准；same-quality 明确属于已知 codec 条件，不是
  未知 codec zero-shot。

### 20.2 Canonical 检测复核（三种子平均 macro AUROC）

| 数据集 | 表示 | M0 | CARF-A full | CARF-A size-only |
|---|---|---:|---:|---:|
| ARForensics | NP-RSR | 0.733 | 0.803 | 0.823 |
| ARForensics | 固定高通 | 0.592 | 0.650 | 0.670 |
| DARG T2I | NP-RSR | 0.733 | 0.799 | 0.812 |
| DARG T2I | 固定高通 | 0.561 | 0.595 | 0.615 |

full−M0 的逐图 paired bootstrap 与生成器覆盖：

| 数据集/表示 | ΔAUROC（seed 0/1/2） | CI 下界 | 不退化 | 严格复核 |
|---|---|---|---|---|
| ARF / NP | +0.074 / +0.065 / +0.072 | 均>0 | 6/7、6/7、6/7 | 通过 |
| ARF / HP | +0.064 / +0.053 / +0.059 | 均>0 | 7/7、6/7、7/7 | 通过 |
| DARG / NP | +0.077 / +0.051 / +0.069 | 均>0 | 5/5、5/5、5/5 | FPR 范围未全通过 |
| DARG / HP | +0.034 / +0.041 / +0.028 | 均>0 | 3/5、3/5、3/5 | 未通过 |

DARG/NP 的 full FPR 为 4.18%/3.14%/1.05%，后两个 seed 低于预设 3%
（偏保守而非误报过高）；DARG/HP 的 seed0 FPR=8.71%，且绝对 AUROC 接近
随机。因此 DARG/NP 的判别增益稳定存在，但不能把严格协议标记写成通过；
DARG/HP 不具备稳定检测效用。

### 20.3 Nuisance–signal coupling：跨数据成立，跨特征不成立

| 数据集/表示 | nuisance 最低稳定性（3 seeds） | codec MMD 平均下降 | 平均信号保留 η |
|---|---|---|---:|
| ARF / NP | 0.981–0.999 | QF85 86.8%–89.6%；QF75 87.1%–88.5% | 0.100–0.115 |
| DARG / NP | 0.997–0.999 | QF85 89.4%–91.0%；QF75 90.0%–91.9% | 0.063–0.078 |
| ARF / HP | **0.193–0.516** | 多数设置下降失败或 MMD 增大 | 0.005 |
| DARG / HP | **0.024–0.415** | 高度不稳定，部分设置 MMD 增大 | 0.004–0.006 |

NP-RSR 上，DARG 五生成器 η 为 0.013–0.135，ARF 七生成器 η 为
0.047–0.239：真实 codec 响应稳定且可消除漂移，但 fake–real 均值差也主要
位于相同方向，跨真实来源和生成器集合复现了 coupling。

固定高通表示没有复现关键前提：U 的跨 source 稳定性低，mean correction
不能稳定降低真实漂移。虽然其 η 很低，但在 U 不稳定时不能把它解释为同一
个可迁移 nuisance 子空间。**因此 coupling 是 NP-RSR 表示相关现象，当前
不能声称是跨取证特征的普遍规律。**

### 20.4 参考池分解（CARF-A full，三种子 macro AUROC）

| 数据集/表示 | QF85：QF95-only / same / mixed | QF75：QF95-only / same / mixed |
|---|---|---|
| ARF / NP | 0.496 / 0.732 / 0.728 | 0.443 / 0.672 / 0.710 |
| ARF / HP | 0.535 / 0.545 / 0.523 | 0.552 / 0.540 / 0.560 |
| DARG / NP | 0.639 / 0.751 / 0.757 | 0.575 / 0.704 / 0.708 |
| DARG / HP | 0.602 / 0.583 / 0.568 | 0.593 / 0.544 / 0.574 |

NP-RSR 的 same/mixed reference 明显优于 QF95-only，说明参考 codec 分布是
核心协议变量；但 mixed 并非总是一致优于 same，且 HP 不呈现同样规律。
这些结果支持“参考分布匹配影响评分”，不支持“已知质量条件即可解决未知
codec 泛化”。

### 20.5 外部条件变量捷径与最终判断

DARG within-source 标签审计：full 条件 AUC=**0.995**，size-only=
**0.977**，pixel complexity=**0.814**。即使 RAISE 与 fake 都统一重编码，
原始文件大小和分辨率仍携带强标签信息。故 DARG/NP 的 +0.05~0.08 条件增益
不能全部归因于 nuisance 条件化，也不能作为 CARF-A 机制已解决因果混淆的
证据。

综合判断：

1. **得到支持的科学结论**：NP-RSR 的 nuisance–signal coupling 可跨
   ARForensics 与 DARG 复现；条件参考池和逐 codec 校准显著影响结果。
2. **未得到支持的推广**：coupling 未在独立固定高通表示上复现；CARF-A
   在外部 HP 上只有 3/5 生成器改善且绝对性能弱，条件变量还存在严重标签
   关联。
3. **CARF-A 定位不变**：可作为 real-only 条件参考基线，不能作为创新已
   充分、机制已普适验证的新检测算法。
4. **论文路线**：理论命题与正负证据适合“系统性评测与失效机制研究”；
   目前仍不足以支撑 ICML 方法论文。不新增 SACR-v2 或继续堆叠特征。
5. **验证**：41 项测试通过；从逐图 NPZ 独立重算 1080 组 canonical 与
   参考池 AUROC，全部与 JSON 一致。

## 21. v13 P0：反捷径匹配、参考池统计与表示条件化理论

脚本：`analysis/carf_mechanism_p0.py`、`analysis/dct_forensics.py`；结果：
`outputs/carf_mechanism/carf_mechanism_p0.json`；理论更新：
`docs/nuisance_signal_coupling.md` §8–10。实验为
**v13 matched-analysis + reference-statistics + representation-ci 示例**，
不是 84k 或新检测方法评测。

### 21.1 受控分辨率/codec 与条件匹配

所有图像统一 resize 256×256 BICUBIC + JPEG95；条件文件大小在本次 JPEG
编码后的字节流上计算，不再使用原始 PNG/JPEG 元数据。test 为 ARF 7×300/类
和 DARG 5×300/类。real/fake 以 reference 标准化条件做一对一最优分配，
caliper=2.5 在运行前冻结；分别做全 8 维、仅文件大小、仅像素复杂度匹配。
CARF-size/CARF-pixel 均使用既有核加权 Mahalanobis，没有新增评分器。

三种子平均 macro AUROC：

| 数据集 | 匹配协议 | M0 | CARF-size | CARF-pixel |
|---|---|---:|---:|---:|
| ARF | 全部受控图 | 0.550 | 0.551 | 0.562 |
| ARF | 全条件匹配 | 0.563 | 0.556 | 0.569 |
| ARF | 文件大小单独匹配 | 0.549 | 0.550 | 0.561 |
| ARF | 像素复杂度单独匹配 | 0.558 | 0.552 | 0.566 |
| DARG | 全部受控图 | 0.531 | 0.551 | 0.547 |
| DARG | 全条件匹配 | 0.524 | 0.545 | 0.541 |
| DARG | 文件大小单独匹配 | 0.531 | 0.550 | 0.547 |
| DARG | 像素复杂度单独匹配 | 0.525 | 0.546 | 0.542 |

ARF 文件大小单独匹配后的 size−M0 三种子为
+0.0016/+0.0007/−0.0009，CI 均跨零；像素匹配后为
−0.0047/−0.0056/−0.0075。CARF-pixel 虽保留约 +0.008~0.013 的小幅正增益，
但绝对 AUROC 仍仅约 0.56。原生分辨率下 NP-RSR 的 M0/CARF-A 结果不能
迁移到该受控协议，尺度/分辨率与内容分布是重要失效因素。

DARG 文件大小单独匹配后 size−M0 为 +0.019/+0.022/+0.018，CI 下界均>0；
说明该小规模受控增益不是完全由一维文件大小失衡造成。但其 AUROC 仍约
0.55，且部分生成器的像素条件标签 AUC 仍高（Switti 约 0.85），不能进一步
声称已排除内容分布捷径。全条件匹配也未充分平衡：ARF Infinity 的组合
条件 AUC 仍为 0.874，因此“排除全部可观测捷径”的假设未获支持。

### 21.2 Same-quality 与 mixed-codec：ESS 增加不等于估计更好

对固定真实查询，以全真实参考构造 proxy-oracle，将三种子条件均值误差
精确分解为 proxy bias² + seed variance。该 oracle 是经验代理，不是真实
总体均值。NP-RSR 结果：

| 数据/视图 | 参考池 | proxy bias² | seed variance | proxy MSE | ESS 中位 |
|---|---|---:|---:|---:|---:|
| ARF/QF85 | same | 0.00027 | 0.00104 | 0.00131 | 2769 |
| ARF/QF85 | mixed | 0.00502 | 0.00113 | 0.00615 | 8225 |
| ARF/QF75 | same | 0.00034 | 0.00118 | 0.00152 | 2769 |
| ARF/QF75 | mixed | 0.02924 | 0.00113 | 0.03037 | 8225 |
| DARG/QF85 | same | 0.00005 | 0.00073 | 0.00078 | 312 |
| DARG/QF85 | mixed | 0.00484 | 0.00081 | 0.00564 | 932 |
| DARG/QF75 | same | 0.00005 | 0.00074 | 0.00078 | 312 |
| DARG/QF75 | mixed | 0.02924 | 0.00081 | 0.03004 | 932 |

mixed-codec 的 ESS 约增至 3 倍，但条件偏差和 MSE 显著更大，跨 seed
方差没有下降。多个 codec 视图来自相同 source，ESS 未计入视图相关性，
不能解释为独立真实来源增加。已知 codec 时，same-quality 的条件偏差优势
可以大于 mixed pooling 的样本量收益；mixed pooling 不是无成本的方差降低。
HP 表示也呈现 mixed MSE 更高的趋势。

### 21.3 表示依赖 coupling：先识别，再解释 η

v13 固定判据：最低 per-view mean-response cos≥0.9，且 QF85/QF75 的
MMD reduction≥0.3，才将低 η 解释为可迁移 nuisance 子空间上的信号重叠；
否则标记 unidentified。稳定性、MMD 和 η 均补做 source/image bootstrap CI。

| 数据/表示 | nuisance 稳定性 | MMD reduction | 平均 η | 判定 |
|---|---|---|---:|---|
| ARF/NP | 0.981–0.999 | QF85 86.8%–89.6%；QF75 87.1%–88.5% | 0.124–0.132 | coupling |
| DARG/NP | 0.997–0.999 | QF85 89.4%–91.0%；QF75 90.0%–91.9% | 0.063–0.078 | coupling |
| ARF/DCT | 0.993–0.998 | QF85 6.6%–56.3%且CI跨零；QF75 72.8%–82.4% | 0.014–0.016 | unidentified |
| DARG/DCT | 0.984–0.996 | QF85 41.2%–63.3%；QF75 70.7%–82.6% | 0.005–0.008 | coupling |
| ARF/HP | 0.193–0.516 | 不稳定 | 0.001 | unidentified |
| DARG/HP | 0.024–0.415 | 不稳定 | 0.004–0.005 | unidentified |

DCT/DARG 各生成器 η 的 bootstrap 上界均远低于 0.8；ARF/DCT 虽然 η 同样
很低，但 QF85 的真实漂移校正未通过可识别性判据，不能据此声称 coupling。
这说明 coupling 不是简单的“只有 NP-RSR 存在”，而是由**表示、数据分布
和 codec 条件共同决定**。低 η 在 nuisance 子空间不可识别时没有机制含义。

### 21.4 对 v13 三个核心问题的回答

1. **能否排除文件大小/内容捷径？不能。** 受控协议使 ARF 的原增益基本
   消失；DARG 的一维文件大小匹配后仍有小增益，但内容分布未平衡且绝对
   性能接近随机。观察性匹配不能提供因果排除证明。
2. **same 与 mixed 的统计差异是什么？** mixed 提高名义 ESS，却增大条件
   偏差与 MSE，且相关视图没有降低跨 seed 方差；same-quality 在已知 codec
   下具有更明确的统计优势。
3. **表示依赖性能否成为研究贡献？可以，但必须带可识别性前提。** NP 在
   两个数据集稳定，DCT 只在 DARG/部分 codec 条件通过，HP 不通过；这一
   正负证据链比宣称普遍 coupling 更准确，也更具可证伪性。

论文定位仍为条件参考分布、参考池统计与失效机制的系统性研究，而非以
AUROC 提升为唯一目标的新检测器。首次运行仅在缓存加载时遇到变量名
笔误，修复后断点续跑，不影响数值协议。验证：46 项测试通过；独立重算
108 组受控全图及 216 组单独匹配 AUROC，全部与逐图分数一致；16 组
bias–variance 分解误差均小于 1e-5。

## 22. S3R P0（v14）：**GO=false，same-quality 被一致选择，停止扩展**

正式指南：`deep-research-report-v14.md`（用户附带的 v4 文件实际为旧 DCA
指南，未作为 S3R 依据）。脚本：`analysis/s3r_p0.py`；结果：
`outputs/s3r/s3r_p0.json`；理论：`docs/nuisance_signal_coupling.md` §11。
本节为 **s3r-p0-300 source-aware controlled views 示例**，不是全量评估。

### 22.1 协议与四种参考池

所有视图统一 256×256 后 JPEG95/85/75，条件为重编码 log 文件大小。
ARF 每 seed 使用 3000 fit、500 real validation、500 real calibration；
DARG 使用 300 fit、100 real validation、300 real calibration；test 300/类。
γ 只在真实 validation source 上按 Gaussian NLL 从
{0,0.1,0.2,0.3,0.5,0.75,1} 选择，fake 不参与。

比较：

1. same-quality：目标质量单视图，same-only 标准化与带宽；
2. naive mixed：混合 QF95/85/75，视图级均值/协方差，source cluster ESS；
3. source-weighted mixed：source 内摘要 + 跨 source 平均权重，
   between-source + within-source 总协方差；
4. hierarchical S3R：same 与 source-mixed 的均值/协方差分层收缩，
   两个分量分别使用独立 reference-only 标准化与带宽。

主 GO：S3R 相对 mixed 显著降低 proxy-MSE 且不超过 same 的 1.10 倍；
相对每 seed 较强 same/mixed 的 ΔAUROC>0 且 paired CI 下界>0；至少
5/7 ARF、4/5 DARG 不退化；FPR∈[3%,7%]、coverage≥95%；未知 codec
直接 abstain。

### 22.2 真实 validation 一致选择 γ=0

所有 **12 个 dataset×seed×quality 组的 γ 均为 0**。修复后的 S3R 在逐图
AUROC 上严格等于 same-quality；真实 held-out source 一致拒绝引入
mixed-codec，因此 hierarchical S3R 没有形成新的估计器。

三种子平均 macro AUROC：

| 数据集/质量 | same | mixed | source-mixed | S3R (γ=0) |
|---|---:|---:|---:|---:|
| ARF/QF85 | 0.528 | 0.511 | 0.511 | 0.528 |
| ARF/QF75 | 0.551 | 0.587 | 0.587 | 0.551 |
| DARG/QF85 | 0.556 | 0.517 | 0.517 | 0.556 |
| DARG/QF75 | 0.541 | 0.512 | 0.512 | 0.541 |

source-weighted mixed 与 naive mixed 的检测输出几乎一致，说明仅修正
ESS 计数和协方差分解并不会自动改善检测排序。受控协议下绝对 AUROC 仍
接近随机，不能包装为高性能 detector。

### 22.3 MSE 与检测 GO 分解

S3R=γ0 后，其 bias²、variance、MSE 与 same 完全相等。相对 mixed 的
proxy-MSE 差均显著为负：ARF/QF85 −0.00312、ARF/QF75 −0.00960；
DARG/QF85 −0.00289、DARG/QF75 −0.00872，source bootstrap CI 均不含 0。
**这只是继承 same-quality 的性质，不是 S3R 的增量贡献。**

| 数据集/质量 | S3R 相对较强 same/mixed 的 ΔAUROC | 不退化 | 判定 |
|---|---|---:|---|
| ARF/QF85 | 0 / 0 / 0，CI=[0,0] | 7/7 | 无正向增益，失败 |
| ARF/QF75 | −0.038 / −0.035 / −0.037 | 1/7 | 失败 |
| DARG/QF85 | 0 / 0 / 0，CI=[0,0] | 5/5 | 无正向增益，失败 |
| DARG/QF75 | 0 / 0 / 0，CI=[0,0] | 5/5 | 无正向增益，失败 |

MSE 标准通过，但检测标准失败。ARF S3R 的 FPR 范围 3.86%–7.29%，两组
高于 7%；DARG 为 2.36%–5.07%，三组低于 3%，故校准标准也未全部通过。
coverage 均约 99%–100%，满足 ≥95%；未知 codec 按协议 100% abstain。

**最终 GO=false，按 v14 停止 S3R，不开展全量或新增模块。**

### 22.4 理论意义与复现事件

source-level ESS 的防膨胀不变量成立：复制完整视图集不改变 source 内摘要、
平均 cluster 权重和 ESS，naive view ESS 则可虚增。分层收缩的误差式允许
γ=0 成为合法最优解；真实 validation 不保证测试 MSE，更不保证 AUROC。

复现过程中，结果打印键名和 JSON seed 键问题均在汇总阶段修复；另发现
hierarchical 初版让 same 分量共享 mixed 带宽，导致 γ=0 不严格等于 same。
该实质问题已修复为两个分量独立带宽，旧结果隔离在
`outputs/s3r/invalid_pre_bandwidth_fix/`，本节全部来自修正后的正式重跑。
12 个 checkpoint 可逐组恢复。

验证：53 项测试通过；独立重算最终 288 组逐图 AUROC/coverage，全部与
JSON 一致；所有 MSE=bias²+variance 分解通过数值校验。S3R 的负结果进一步
支持将论文定位为参考池统计与失效机制研究，而不是新检测算法。

## 23. v15：Source-ESS、条件校准与估计—判别关系统计分析

脚本：`analysis/reference_pool_analysis.py`；测试：
`tests/test_reference_pool_analysis.py`；结果：
`outputs/reference_analysis/v15_analysis.json` 与 `calibration_*.npz`；理论：
`docs/nuisance_signal_coupling.md` §12。全程复用 S3R checkpoint 和逐图分数，
未开发新检测器或调整阈值。

### 23.1 S3R 失败原因与保留的方法

S3R GO=false 的直接原因是：12 组实验全部 γ=0，严格退化为 same-quality；
没有超过较强基线的 ΔAUROC，且部分 FPR 超出预设区间。MSE 优于 mixed 只是
继承 same-quality 的性质，不能计作 S3R 增益。

当前保留 **Same-quality Conditional Real-reference Residual Scoring**
（same-quality CARF-A）作为实验基线，而非新算法：统一受控视图，按已知
codec/quality 选择真实参考，以 source 为独立单位估计条件残差均值/协方差，
Mahalanobis 评分，再用独立 real calibration 做组内 conformal；未知条件
abstain。其适用范围仅限已知质量、存在同质量真实参考且 source-disjoint
的协议，不是未知 codec zero-shot。

与相关方法的区别：UnivFD 84k AUROC 0.9127 但使用真假监督；CARF-A 原生
协议有正条件增益但受分辨率/内容代理影响；same-quality 的 proxy-MSE 优于
mixed 但受控 AUROC 仅约 0.51–0.59；S3R/source-mixed 未带来额外检测收益。
不同训练信息与协议不能仅按 AUROC 排名。

### 23.2 Source-ESS 模拟与真实权重不变量

模拟使用 `r_iq=√ρ·b_i+√(1−ρ)·e_iq`，使 ρ 精确等于视图相关系数；
128 组配置覆盖 source 数、视图数、相关性和均匀/倾斜权重，每组 500 次
Monte Carlo。

| 100 sources、等权 | Q=1 | Q=2 | Q=4 | Q=8 |
|---|---:|---:|---:|---:|
| ρ=0：MSE | 0.01125 | 0.00525 | 0.00248 | 0.00123 |
| ρ=1：MSE | 0.01040 | 0.00934 | 0.01048 | 0.01011 |
| 任意 ρ：naive view-ESS | 100 | 200 | 400 | 800 |
| 任意 ρ：source-ESS | 100 | 100 | 100 | 100 |

独立视图可降低估计方差，完全相关视图不能；view-ESS 却在两种情况下都
线性增长。source-ESS 是有意的 source 独立量，在独立视图存在时较保守，
不能独自表示所有视图信息。

真实参考权重复制完整视图集（ARF/DARG 各 100 查询）：view-ESS 均约翻倍，
source-ESS 比值约 1；均值最大变化 1.43e-6、协方差最大变化 2.98e-6、
source-ESS 最大变化 4.89e-4（浮点量级）。不变量在真实权重下成立。

### 23.3 条件校准：阈值离散性、FPR CI 与 risk–coverage

对 same-quality 的每个已知质量组保存 real calibration 分数、阈值、有效
样本数、有限样本秩和相邻阈值间隔，并对测试 real FPR 做 source bootstrap。

- ARF 各组 macro FPR 的 95% CI 整体约为 **3.05%–8.39%**，部分上界超过
  7%；点估计有两组超过 7%。
- DARG 各组 macro FPR 的 95% CI 整体约为 **1.62%–6.22%**，部分点估计
  低于 3%。
- ARF 的 calibration 每组 500 个支持样本，DARG 每组约 297–298 个；阈值
  秩和间隔已逐组保存，不能只报告一个阈值数字。
- ESS 阈值从 20 提升至 200 时，DARG coverage 从约 99% 降至约 **75%**。
  部分 FPR 下降，但不满足 coverage≥95%，构成明确的 risk–coverage 代价。
- 未知 codec 继续 100% abstain，不能算作 zero-shot 检测。

12 个 calibration 文件中的阈值、逐生成器 FPR 与 macro FPR 均已从逐图
分数独立重算一致。校准通过与否、coverage 和 AUROC 必须分别报告。

### 23.4 MSE、ESS 与 AUROC 的非单调关系

使用 same、naive mixed、source-mixed 的逐 seed 数据，排除重复的
S3R=γ0 点，共 36 个 dataset×seed×quality×method 点：

| 相关性 | Spearman |
|---|---:|
| 全局 MSE vs macro AUROC | −0.070 |
| 全局 source-ESS vs macro AUROC | −0.047 |
| 全局 source-ESS vs MSE | +0.300 |
| ARF/QF75 MSE vs AUROC | +0.510 |
| ARF/QF85 MSE vs AUROC | −0.824 |
| DARG/QF85 MSE vs AUROC | −0.933 |
| DARG/QF75 MSE vs AUROC | −0.966 |

全局 Pearson MSE–AUROC 为 +0.221。相关点有限且非独立随机样本，以上仅作
描述性证据，不作普遍独立性的统计检验。可支持的结论是：**较低条件均值
误差、更大 ESS、更高 AUROC 和合格 FPR 是不同目标，没有简单单调等价关系。**

### 23.5 已完成工作、下一步与 ICML 定位

已完成：统一 baseline 矩阵、AR 专用方法复现、数据/格式/重复审计、
CARF-A 协议修复、CoNTra/SACR/S3R 预注册负结果、ARF/DARG 外部小规模复核、
NP/DCT/HP 表示依赖 coupling、same/mixed bias–variance、source-ESS 模拟与
真实不变量、条件校准 CI 和 risk–coverage、逐图预测与 checkpoint。

下一步不再开发 S3R-v2、SACR-v2 或 CoNTra 新模块。优先工作是：扩大真实
来源与生成家族、同协议比较 Forensic Self-Descriptions、完善 source-aware
均值/协方差误差界、研究支持不足时的风险—覆盖保证，并将未知条件作为独立
风险问题，而非强行检测。

**ICML 判断：S3R 和 CARF-A 均不足以单独作为强方法论文。** 当前最合适的
定位是 *Reference-Pool Statistics and Failure Mechanisms for Real-Only
AI-Generated Image Detection*：以来源相关性、codec 条件、参考池偏差—方差、
表示依赖和条件校准失效为贡献。若无法补足更强理论保证和独立外部验证，
应考虑下一轮 ICML 或 CVPR/ICCV/TIFS，而不是继续堆叠检测模块。

验证：56 项测试通过；12 组 calibration 阈值与 FPR 独立复核一致；
模拟相关系数公式已修正为精确 ρ 后重跑全部统计分析。

## 24. SCALE / QSCALE / DART：token 序列分布建模的三次预注册检验

方法文档：`docs/method_scale_dart.md`。代码：`analysis/scale_p0.py`、
`analysis/qscale_p0.py`、`analysis/dart_p0.py`。数据均来自既有 DARG T2I
逐 token 条件/无条件似然缓存，不重新运行生成模型；三 seed、每生成器
300 fake，真实参考/校准/测试独立划分。

### 24.1 SCALE：逐位置经验尾概率

SCALE 用真实参考估计每个 token 位置的双侧经验尾概率，scale 内 Fisher 或
Higher Criticism 聚合，再做 model-bank max 的序列级真实校准。优点是 real-only
且无需 token 独立假设；缺点是丢失偏差方向和幅度。

| 方法 | macro AUROC（3 seeds） | GO |
|---|---|---|
| SCALE-Fisher | 0.729 / 0.717 / 0.740 | false |
| SCALE-HC | 0.413 / 0.396 / 0.397 | false |
| 官方 PRADA | 0.9997 / 0.9993 / 0.9996 | 强基线 |

### 24.2 QSCALE：signed quantile-function 空间

QSCALE 保留每个 scale 的 16 个 signed quantiles，以近似 1-Wasserstein 的
quantile-function 几何替代逐位置尾概率，使用真实参考 Mahalanobis/kNN。

- Mahalanobis：0.731 / 0.715 / 0.723；
- kNN：0.784 / 0.767 / 0.777；
- 仍显著低于 PRADA，GO=false。

这说明 signed 分布表示本身不能替代监督式 token 证据。

### 24.3 DART：变量 scale 的 Deep Sets + LOGO

DART 使用 quantile、scale length/index 和 attention pooling，严格
leave-one-generator-out；held-out fake 不参与训练或提前停止。

| held-out | DART AUROC（3 seeds） | PRADA | QSCALE-kNN |
|---|---|---|---|
| Infinity | 0.922–0.930 | 约1.000 | 0.772–0.788 |
| Janus | 0.657–0.791 | 约0.999 | 0.909–0.941 |
| LlamaGen | 0.179–0.293 | 约0.999 | 0.701–0.714 |
| Switti | 0.743–0.837 | 约1.000 | 0.685–0.701 |

macro 仅 0.629–0.706。LlamaGen 严重方向反转，说明共享分类器不能消除模型
特异 likelihood direction，LOGO GO=false。SCALE/QSCALE/DART 当前均不构成
ICML 方法贡献；后续仅保留模型特异 DART 的 C2I 小规模检验，若无稳定增益
则停止 token-distribution 路线。项目验证：64 tests passed。

## 25. 完整循环一致性 DART：C2I 多家族显著增益（小规模，GO 仍=false）

实现：`analysis/dart_p0.py`、`analysis/dart_c2i_p0.py`；方法文档：
`docs/method_scale_dart.md`；结果：`dart_c2i_cycle_consistent.json`、
`dart_c2i_external.json`、`dart_c2i_external_fullbank.json` 及
`significance*.json`。全部为 **dart-c2i-250 示例协议**，不是全量结果。

### 25.1 机制与逐步证据

完整 DART 将 AR 图像是否位于 tokenizer–decoder–likelihood 的不动点附近，
表示为变量 scale 的联合分布：signed likelihood-difference quantile、
scale 平均 quantization error、decoder reconstruction error 和 scale
length/index，再由 Deep Sets attention 学习尺度贡献。它不是简单增加通用
视觉特征，也不是 real-only 检测：每个参考模型使用真实与对应家族 fake
进行模型特异训练，真实 calibration 仅用于分数标准化与阈值校准。

逐步消融（VAR_d16 / LlamaGen-B256，3 seeds）：

| 版本 | VAR_d16 | LlamaGen-B256 | macro AUROC |
|---|---:|---:|---:|
| likelihood-only | 0.525–0.554 | 0.969–0.998 | 0.747–0.776 |
| + quantization | 0.587–0.642 | 0.994–0.999 | 0.791–0.819 |
| 完整循环一致性 | **0.811–0.876** | **0.994–0.999** | **0.905–0.938** |
| 官方 PRADA | 0.477–0.509 | 0.993–0.997 | 0.737–0.752 |

说明 likelihood quantile 对 LlamaGen 已足够，而 decoder cycle evidence 是
覆盖 VAR 等其他 AR 家族的关键，不是单纯增加分类器容量。

### 25.2 外部 RAR/HMAR 与参考家族覆盖

初始四模型 bank（VAR_d20/d30、LlamaGen-B/L256）中没有任何 RAR 模型，
RAR-B 仅 0.532–0.574。加入同家族 RAR-L 后，RAR-B 达 **0.880–0.922**。
这证明 model-aware 方法需要参考家族覆盖，缺失参考时不能据此否定机制。

完整五模型 bank 的结果（每生成器 250 张，三 seed）：

| 生成器 | DART AUROC | 同 bank PRADA | paired ΔAUROC（95%CI） |
|---|---|---|---|
| VAR_d16 | 0.770–0.863 | 0.478–0.509 | +0.293/+0.354/+0.292，均为正 |
| LlamaGen-B256 | 0.994–0.999 | 0.993–0.997 | 不显著 |
| RAR-B | **0.880–0.922** | 0.520–0.594 | +0.373/+0.286/+0.378，均为正 |
| HMAR-D16 | 0.755–0.848 | 0.484–0.542 | +0.331/+0.307/+0.271，均为正 |
| **macro** | **0.855–0.898** | 0.619–0.660 | seed0 +0.259 [+0.221,+0.296] |

LlamaGen 已饱和，主要增益来自 VAR、RAR 和 HMAR。当前基线是**同五模型
bank 的 PRADA**，不是官方完整八模型 bank，报告时不扩大比较口径。

### 25.3 GO、校准和未解决问题

预注册 GO 仍为 false：FPR 分别为 3%/4%/2%，最后一个 seed 低于 3% 下界。
这是偏保守的校准结果，不能事后修改阈值或 GO。部分生成器在 TPR@1%FPR
仍接近零，说明高 AUROC 不等于低 FPR 工作点实用。

目前尚缺：更大 calibration source 的 FPR 稳定性、跨 codec、更多真实来源、
官方完整 bank 对照、不同数据规模和更低 FPR 工作点验证。完整 DART 是
当前唯一具有显著多家族增益的候选方法，但尚不能称为成熟 ICML 方法或
未知生成器 zero-shot detector。后续只针对这些证据缺口验证，不再增加新
特征或网络模块。项目测试：64 passed。


## 26. DART 1000-real P1：AUROC 增益稳定，校准显著改善（9 seeds）

实现仍为 `analysis/dart_c2i_p0.py`；结果：`dart_c2i_real1000.json`、
`dart_c2i_real1000_s3_8.json`、`significance_real1000.json`。全部属于
**dart-c2i-real1000-fake250 示例协议**，不是全量实验。

### 26.1 评估协议修正与旧结果复核

本轮修复了会影响大规模真实集评估的工程问题：

- real 与 fake 的缓存样本数分别按 1000/250 检查，`pack()` 对 DART 和
  PRADA 使用相同的样本截断，避免旧 1000 行似然与 250 行循环特征错位；
- fake 训练/验证/测试固定为互斥的 100/50/100，消除旧代码验证 fake 可能与
  训练 fake 重叠的模型选择问题；
- 逐图分数文件增加配置名和 `n_real`，避免不同实验互相覆盖；
- RAR 分片只用于 batch=1 的相同计算。并发分片出现少量不一致后未采用，
  改用独占重算；最终 conditional/unconditional 前 250 行与旧缓存逐值一致。

使用互斥 fake 划分重跑原 250-real 完整五模型 P0，macro AUROC 为
0.893/0.898/0.856，FPR 仍为 3%/4%/2%，与旧结果几乎一致。因此旧的多家族
AUROC 结论不是由验证集重叠驱动，但旧 GO=false 不变。结果见
`dart_c2i_external_fullbank_corrected.json`。

### 26.2 P1 设置与前三 seed GO

真实集按 400 reference / 300 calibration / 300 test 独立划分，fake 每类
250；模型、特征、网络和 conformal 规则均不变。前三个 seed：

| seed | DART macro AUROC | 同 bank PRADA | ΔAUROC | DART FPR |
|---|---:|---:|---:|---:|
| 0 | 0.902 | 0.655 | +0.247 | 5.33% |
| 1 | 0.913 | 0.668 | +0.245 | 4.67% |
| 2 | 0.904 | 0.643 | +0.261 | 7.00% |

macro、增益、4/4 生成器不退化和 FPR 四项预注册标准均通过，故前三 seed
**GO=true**。四模型诊断在缺少 RAR-L 时 macro 仅 0.809–0.820，但 FPR 三个
seed 均为 4.67%，再次说明参考家族覆盖和校准稳定性是两个独立问题。

### 26.3 全部 9 seeds 与统计核验

| 生成器 | DART AUROC 均值 | 9-seed 范围 | TPR@1%FPR 范围 |
|---|---:|---:|---:|
| VAR_d16 | 0.845 | 0.780–0.891 | 0–0.01 |
| LlamaGen-B256 | 0.998 | 0.992–1.000 | 0.89–1.00 |
| RAR-B | 0.942 | 0.905–0.957 | 0.06–0.58 |
| HMAR-D16 | 0.824 | 0.767–0.851 | 0–0.01 |

- DART macro AUROC：**0.902 ± 0.015**；最小 seed 为 0.873；
- 同 bank PRADA macro：0.621–0.668；DART 最小 macro 增益为 +0.245；
- FPR 依次为 5.33%、4.67%、7.00%、6.00%、6.00%、3.67%、5.00%、3.00%、
  7.33%，相较旧协议的 0–13% 明显改善；
- seed 8 的 FPR=22/300=7.33%，略超 7% 上限，因此全部 9 seeds 的严格
  **GO=false**。不能事后放宽阈值或只汇报前三 seed。

对保存的逐图分数独立重算 72 个 AUROC，最大误差为 0。seed 0 的 5000 次
paired bootstrap：

| 生成器 | paired ΔAUROC | 95%CI |
|---|---:|---:|
| VAR_d16 | +0.315 | [+0.248,+0.383] |
| LlamaGen-B256 | +0.003 | [-0.002,+0.010] |
| RAR-B | +0.369 | [+0.312,+0.428] |
| HMAR-D16 | +0.300 | [+0.236,+0.366] |
| **macro** | **+0.247** | **[+0.220,+0.275]** |

### 26.4 结论与下一步

扩大真实参考与校准集同时提高了 macro AUROC 和 FPR 稳定性，支持 DART 的
完整生成循环一致性机制，而不是继续增加模块。当前仍不能称为通用检测器：
VAR/HMAR 的 TPR@1%FPR 接近零，RAR-B 在不同 seed 下也有较大波动；实验仍
限于小规模 C2I、模型特异监督、同五模型 bank 对照，尚无跨 codec、独立真实
来源和官方完整八模型 bank 验证。下一阶段只验证这些跨域与低 FPR 缺口。

项目测试：68 passed。第三方 TruFor 的全目录 pytest 收集仍缺少其独立依赖
`yacs`，与 DART 测试无关。


## 27. JPEG95 P2 smoke：宏观增益保留，VAR/HMAR 有压缩敏感性

实现：`analysis/dart_codec_p0.py`；结果：`dart_c2i_jpeg95_smoke.json`。
该实验仅使用 seed 0，训练和校准仍为原始图像，只把原 P1 的 300 张真实
测试图和每类 100 张 fake 测试图重编码为 JPEG95。没有压缩图训练、阈值
调整或测试集选择，属于**单 seed 跨 codec 示例诊断**。

| 生成器 | 原图 DART AUROC | JPEG95 DART | 下降 | JPEG95 PRADA |
|---|---:|---:|---:|---:|
| VAR_d16 | 0.836 | 0.753 | 0.084 | 0.500 |
| LlamaGen-B256 | 0.998 | 0.996 | 0.002 | 0.991 |
| RAR-B | 0.943 | 0.922 | 0.021 | 0.545 |
| HMAR-D16 | 0.831 | 0.766 | 0.065 | 0.503 |
| **macro** | **0.902** | **0.859** | **0.043** | **0.635** |

JPEG95 下 DART FPR=6%，仍处于预注册 3%–7% 区间；LlamaGen 的
TPR@1%FPR=0.82，RAR-B=0.07，VAR/HMAR 仍为 0。保存的 8 个压缩测试
AUROC 已独立重算，最大误差为 0。

结论是：DART 相对同 bank PRADA 的宏观增益在 JPEG95 后仍然保留，且校准
没有失效；RAR/LlamaGen 较稳健，但 VAR/HMAR 的 AUROC 下降较明显，不能
宣称 codec-invariant。后续需用独立真实来源和更多 codec/seed 验证该现象，
不通过新增模块事后修补这一单 seed 结果。项目测试：71 passed。


## 28. ImageNet-V2 独立真实来源：外部 FPR=4.67%，宏观增益保持

实现：`analysis/dart_external_real_p0.py`；结果：`dart_c2i_imagenetv2_smoke.json`。
数据来自 `clip-benchmark/wds_imagenetv2` 的 matched-frequency shard0，仅下载
一个约 320MB 分片，并确定性选择 300 张带 ImageNet 标签的真实图。保留完整
000–999 类映射，避免条件标签错配。训练、归一化、conformal 阈值均沿用原
ImageNet P1，不在 ImageNet-V2 上重新训练或调阈。该结果为**单分片、单 seed
外部真实 smoke**。

| 生成器 | DART AUROC | 同 bank PRADA | DART TPR@1%FPR |
|---|---:|---:|---:|
| VAR_d16 | 0.854 | 0.584 | 0.01 |
| LlamaGen-B256 | 0.999 | 0.997 | 0.98 |
| RAR-B | 0.953 | 0.637 | 0.53 |
| HMAR-D16 | 0.847 | 0.597 | 0.01 |
| **macro** | **0.913** | **0.704** | — |

DART 外部真实 **FPR=4.67%**，处于预注册 3%–7% 区间；PRADA 为 2.00%。
VAR/HMAR 在该外部真实来源上的 AUROC 没有下降，RAR/LlamaGen 也保持强结果。
8 个外部 AUROC 已独立重算，最大误差为 0。

这说明 P1 的校准不是只对原 ImageNet validation 分布有效，宏观增益也能迁移
到独立真实来源；但单分片结果不能替代更多来源和多 seed 验证，VAR/HMAR 的
低 FPR 检出率仍需单独报告。项目测试：73 passed。


## 29. 低 FPR 工作点与当前最终判断

`operating_points_real1000.json` 从保存的逐图分数独立计算 ROC 工作点。
下表为 1000-real P1 的 9-seed 平均 TPR（括号内为范围）：

| 生成器 | TPR@1%FPR | TPR@5%FPR | TPR@10%FPR |
|---|---:|---:|---:|
| VAR_d16 | 0.003（0–0.01） | 0.134（0.03–0.26） | 0.393（0.20–0.58） |
| LlamaGen-B256 | 0.954（0.89–1.00） | 0.997（0.98–1.00） | 1.000 |
| RAR-B | 0.387（0.06–0.58） | 0.733（0.61–0.85） | 0.861（0.74–0.92） |
| HMAR-D16 | 0.002（0–0.01） | 0.100（0.01–0.21） | 0.350（0.19–0.45） |

ImageNet-V2 seed 0 的 TPR@5%FPR 分别为 VAR 0.09、LlamaGen 1.00、RAR
0.74、HMAR 0.11；JPEG95 分别为 0、0.99、0.65、0.02。这再次说明宏观
AUROC 增益不能直接解释为所有家族在低误报下都实用。

当前结论：完整循环一致性 DART 已形成清晰的 ICML 方法主线。它具备机制
解释、多家族显著 AUROC 增益、9-seed 稳健性、JPEG95 保留增益和
ImageNet-V2 外部 FPR=4.67% 的证据；不应继续无目的地叠加模块。投稿前仍
应扩大跨域与低 FPR 实验，并明确模型特异监督、参考家族覆盖和小规模示例
协议的边界。现阶段不能声称通用检测、codec-invariant、未知生成器
zero-shot 或保证 ICML 录用。
