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
