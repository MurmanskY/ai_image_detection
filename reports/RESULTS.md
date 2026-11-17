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
