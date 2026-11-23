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
- PRADA T2I 全量（官方 predict.py，DARG-adapted-real-v1）：**FAILED**（特征缓存完成后 BLIP2 prompt 提取阶段 OOM，特征已缓存可续跑）
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
