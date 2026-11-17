# MGI: 生成图像鉴别实验框架

面向**自回归 (Autoregressive, AR) 生成图像鉴别**的统一 Benchmark 框架。
支持多种数据集与多种鉴别方法的统一训练/评估，目标是让任何研究者都能依据
仓库中的配置、数据清单和逐样本预测，独立重算并核验报告的实验结果。

冻结的实验结果总表见 **`reports/RESULTS.md`**；各方法复现细节与协议偏差见
`baselines/reproduction/`；数据审计报告见 `reports/dataset_audit/`。

## 目录结构

```
MGI/
├── train.py / eval.py        # 训练 / 评估入口 (eval 支持 --n_boot 输出 bootstrap CI)
├── configs/                  # 实验配置 (yaml)，一个文件对应一组实验
├── datasets/                 # 数据集层 (注册式)
│   ├── arforensics.py        # ARForensics (D³QE, ICCV 2025)，支持按 generator 过滤
│   ├── genimage.py           # GenImage (NeurIPS 2023)
│   ├── folder_dataset.py     # 通用 real/fake 文件夹数据集
│   ├── manifest.py           # 统一数据清单构建 (sha256/phash/generator 元数据)
│   └── audit.py              # 数据审计 CLI (跨 split 重复/标签/generator 校验)
├── models/                   # 骨干网络 (resnet/clip) 与分类头
├── baselines/                # baseline 方法 (注册式)
│   ├── registry.yaml         # 各方法论文/代码/实现状态/score 契约登记
│   ├── reproduction/         # 每方法一份复现说明 (协议、偏差、结果)
│   └── *.py                  # 方法实现
├── ours/                     # 我们的方法 (当前继承 CNNDetection 作为起点)
├── analysis/
│   └── token_stats.py        # VQ token 统计 v2 (2D transition/效应量/JS 散度/CI)
├── engine/                   # Trainer / evaluator / metrics
│                             # (Acc, F1, balanced Acc, AP, AUC, TPR@0.1/1/5%FPR,
│                             #  bootstrap 95% CI, Youden's J 阈值校准)
├── tests/                    # score 契约/指标/manifest/审计/bootstrap 单元测试
├── reports/
│   ├── RESULTS.md            # 冻结结果总表 (RUNNING 与 FROZEN 分开标注)
│   ├── dataset_audit/        # 数据审计报告
│   └── runs/<run_id>/        # 每次实验归档 (config/metrics/predictions.csv.gz/
│                             #  manifest.json/environment.json)
├── environment/              # system.json + pip_freeze.txt (环境锁定)
├── scripts/                  # 数据组织、manifest、归档、汇总脚本
├── third_party/              # 官方代码: D3QE / PRADA / DataProvenanceIAR
└── data/                     # 数据目录 (gitignore)
```

## 快速开始

```bash
pip install -r requirements.txt

# 训练 (以 ARForensics + CNNDetection 为例)
python train.py --config configs/cnn_detection_arforensics.yaml

# 评估 (输出 overall + per-generator 指标, 保存 results_<split>.json
#  与逐样本 predictions csv; --n_boot 1000 追加 bootstrap 95% CI)
python eval.py --config configs/cnn_detection_arforensics.yaml \
    --checkpoint outputs/cnn_detection_arforensics/best.pth --split test --n_boot 1000

# 运行单元测试
python -m pytest tests/ -q

# 生成数据清单并审计
python -m datasets.manifest --root data/ARForensics --dataset arforensics \
    --out data/manifests/arforensics.csv --with-hash
python -m datasets.audit --manifest data/manifests/arforensics.csv \
    --root data/ARForensics --check-split-overlap --check-exact-duplicates \
    --check-near-duplicates --check-labels --check-generator-metadata \
    --report reports/dataset_audit/arforensics.json

# 归档一次完成的实验
python scripts/archive_run.py --run_id <run_id> --config <cfg> \
    --results <results.json> --predictions <pred.csv> --method <m> --split <s>
```

## 如何扩展

**新增数据集**：在 `datasets/` 中新建文件，实现返回
`{"img", "label", "path", "generator"}` 的 Dataset，并用
`@DATASETS.register("名字")` 注册一个 build 函数，然后在配置的 `dataset.name` 中使用。

**新增方法**：在 `baselines/` 中新建文件，继承 `BaseMethod`，实现
`get_transform(train)` 与 `build_model()`，用 `@METHODS.register("名字")` 注册即可，
并在 `baselines/registry.yaml` 登记 score 契约与实现状态。

## 复现的 baseline（依据《调研自回归生成图像鉴别》）

| 方法 | 出处 | 状态 | 说明 |
|---|---|---|---|
| CNNDetection (ResNet-50) | CVPR 2020 | reimplemented | test acc 0.699 / AUC 0.819 |
| FreqSpec (log 幅度谱) | ICML 2020 | reimplemented | test acc 0.660 / AUC 0.786 |
| NPR (邻域像素关系) | CVPR 2024 | inspired 简化版 | test acc 0.748 / AUC 0.860 |
| UnivFD (CLIP ViT-L/14 + 线性头) | CVPR 2023 | faithful_reproduction | test acc 0.802 / AUC 0.913 |
| LGrad (梯度特征) | CVPR 2023 | reimplemented, GAN 迁移测试 | test acc 0.713 / AUC 0.834 |
| DIRE (DDIM 重建误差) | ICCV 2023 | reimplemented, Diffusion 迁移测试 | val acc 0.933; test sub500 acc 0.743 |
| RIGID (DINOv2 扰动敏感性) | 2024 | reimplemented, training-free | 校准阈值 acc 0.675 / AUC 0.741 |
| ZED (SReC NLL−Entropy) | ECCV 2024 | reimplemented, zero-shot | 每类 500 采样, acc 0.564 / AUC 0.581 |
| D³QE (离散分布差异+量化误差) | ICCV 2025 | 官方代码已跑 + lite 版 | 官方 avg acc 0.754 / AP 0.827, 含鲁棒性 |
| PRADA (条件/无条件概率比) | CVPR Findings 2026 | 官方代码已跑 | T2I tiny AUROC 0.99+; C2I tiny 0.64–0.82 |
| Data Provenance (Codebook 指纹) | ICLR 2026 | 官方代码已跑 | LlamaGen/VAR 各 1000 样本 |

各方法的结果均为 ARForensics test（7 生成器 × 6000 real + 6000 fake），
详细数字、per-generator 指标与协议偏差见 `reports/RESULTS.md` 与
`baselines/reproduction/*.md`。

## 实验数据

- **ARForensics** (D³QE, ICCV 2025): 首个 AR 生成图像检测基准，含 LlamaGen、VAR、
  Infinity、Janus-Pro、RAR、Switti、Open-MAGVIT2 共 7 种 AR 模型。
  **注意**: 当前 train/val 由官方 validation 重划分而来（协议名
  `ARForensics-small-val-derived-v1`，6k train + 1k val 每类），并非官方
  100k/100k 训练分卷，本仓库训练指标与 D³QE 论文数字不直接可比。
- **DARG** (PRADA, CVPR Findings 2026): 155,000 张、20 种 AR 生成器；
  官方代码与预训练评分函数位于 `third_party/PRADA`。
  协议偏差：T2I real 用 ARForensics real 替代 RAISE-1k；C2I real 为 HF ImageNet
  镜像子集（非官方逐张子集）。PRADA 全量评估计算量大，当前结果为 tiny-50 示例
  （每目录 50 张，特征缓存保留可续跑全量）。
- 数据清单：`data/manifests/`（ARForensics 98k 行、DARG t2i/c2i，含 sha256/phash）；
  审计发现 3 组跨 train/test 的 ImageNet 内容重复（3/98000，已在报告中声明）。

## 评估约定 (score contract)

- 进入公共 evaluator 的分数统一为"**越大越像 fake**"：
  训练式方法 (`score_type="logit"`) 经 sigmoid；training-free 方法
  (`score_type="anomaly"`, 如 RIGID/ZED) 直接用 raw score，不做 sigmoid。
- Acc/F1 等阈值指标：logit 方法用 0.5；anomaly 方法在**验证集**上以
  Youden's J 校准阈值后冻结（`eval.py` 自动执行），绝不在 test 上调阈值。
- 排序指标（AUROC/AP/TPR@FPR）一律基于连续分数，`--n_boot` 可输出
  bootstrap 95% CI。
- 每次评估同时输出 `results_<split>.json` 与 `results_<split>_predictions.csv`
  （逐样本 path/label/generator/score/prediction），任意指标可离线重算。
- `dataset.{train,val,test}_generators` 可按 generator 过滤样本，
  用于 leave-one-generator-out 协议。

## 当前研究结论摘要

- 通用检测器 (UnivFD/NPR) 对 LlamaGen/Janus 检测较好，但对 RAR/Open-MAGVIT2/
  Infinity 普遍失效，跨生成器泛化仍是核心难点。
- PRADA 官方代码在 DARG 上 T2I 复现成功（AUROC 0.99+），与论文一致。
- token 统计分析 (`analysis/token_stats.py` v2): 单变量 token 统计
  （entropy/gini/transition/q-error）无法跨生成器稳定区分 real/fake
  （AUC 0.37–0.63 且方向跨生成器翻转），说明 bag-of-tokens 统计受
  tokenizer affinity 影响，不能单独作为通用 AR 指纹。
