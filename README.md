# MGI: 生成图像鉴别实验框架

面向**自回归 (Autoregressive, AR) 生成图像鉴别**的论文实验代码框架。
支持多种数据集与多种鉴别方法的统一训练/评估，用于复现 baseline 并最终提出我们的方法。

## 目录结构

```
MGI/
├── train.py / eval.py        # 训练 / 评估入口
├── configs/                  # 实验配置 (yaml)，一个文件对应一组实验
│   ├── cnn_detection_genimage.yaml
│   ├── frequency_genimage.yaml
│   └── ours_ar.yaml
├── datasets/                 # 数据集层 (注册式, 便于扩展)
│   ├── folder_dataset.py     # 通用 real/fake 文件夹数据集 (自建 AR 数据)
│   └── genimage.py           # GenImage (NeurIPS 2023)
├── models/                   # 骨干网络 (resnet/clip) 与分类头
├── baselines/                # baseline 方法 (注册式)
│   ├── cnn_detection.py      # Wang et al. CVPR 2020
│   ├── frequency.py          # 频域 log 谱 (Frank et al. ICML 2020)
│   ├── npr.py                # NPR (CVPR 2024)
│   ├── univfd.py             # UnivFD (CVPR 2023)
│   ├── rigid.py              # RIGID (2024, training-free)
│   ├── d3qe_lite.py          # D³QE 轻量复现 (ICCV 2025)
│   ├── dire.py / lgrad.py / zed.py  # 骨架
├── ours/                     # 我们的方法
│   └── method.py             # OurMethod, 目前继承 CNNDetection 作为起点
├── analysis/                 # 分析脚本
│   └── token_stats.py        # D³QE 核心统计分析 (codebook 频率/熵/量化误差)
├── scripts/                  # 数据组织与批跑脚本
├── docs/experiments.md       # 实验记录
├── engine/                   # Trainer / evaluate / metrics
│                             # (Acc, F1, balanced Acc, AP, AUC, TPR@0.1/1/5%FPR)
├── utils/                    # registry / config / logger / device (MPS 支持)
├── splits/                   # 数据划分 manifest (path,label,generator,...)
└── data/                     # 数据目录 (结构见 data/README.md)
```

## 快速开始

```bash
pip install -r requirements.txt

# 训练 (以 GenImage + CNNDetection baseline 为例, 先修改配置中的 dataset.root)
python train.py --config configs/cnn_detection_genimage.yaml

# 评估 (输出 overall + per-generator 指标, 保存 results_<split>.json)
python eval.py --config configs/cnn_detection_genimage.yaml \
    --checkpoint outputs/cnn_detection_genimage/best.pth --split val
```

命令行可覆盖任意配置项:

```bash
python train.py --config configs/ours_ar.yaml --opts train.lr 5e-5 train.epochs 50
```

## 如何扩展

**新增数据集**：在 `datasets/` 中新建文件，实现返回
`{"img", "label", "path", "generator"}` 的 Dataset，并用
`@DATASETS.register("名字")` 注册一个 build 函数，然后在配置的 `dataset.name` 中使用。

**新增方法**：在 `baselines/` 中新建文件，继承 `BaseMethod`，实现
`get_transform(train)` 与 `build_model()`，用 `@METHODS.register("名字")` 注册即可。
`ours/method.py` 是我们方法的位置，当前继承 CNNDetection，可直接在其上迭代。

## 复现的 baseline（依据《调研自回归生成图像鉴别》）

| 方法 | 出处 | 状态 |
|---|---|---|
| CNNDetection (ResNet-50 + blur/JPEG 增广) | CVPR 2020 | 已实现 |
| FreqSpec (log 幅度谱) | ICML 2020 | 已实现 |
| NPR (邻域像素关系) | CVPR 2024 | 已实现 |
| UnivFD (CLIP ViT-L/14 冻结特征 + 线性头) | CVPR 2023 | 已实现 |
| RIGID (DINOv2 扰动敏感性, training-free) | 2024 | 已实现 |
| DIRE (ADM DDIM 重建误差) | ICCV 2023 | 已实现 |
| LGrad (StyleGAN 判别器梯度特征) | CVPR 2023 | 已实现 |
| ZED (SReC NLL−Entropy, zero-shot) | ECCV 2024 | 已实现 |
| D³QE (离散分布差异 + 量化误差) | ICCV 2025 | 轻量版 + 官方完整版 (`third_party/D3QE`) |
| Data Provenance (Codebook 几何指纹) | ICLR 2026 | 官方代码接入 (`third_party/DataProvenanceIAR`) |
| PRADA (条件/无条件概率比) | CVPR Findings 2026 | 官方代码接入 (`third_party/PRADA`) |

## 实验数据

- **ARForensics** (D³QE, ICCV 2025): 首个 AR 生成图像检测基准，含 LlamaGen、VAR、
  Infinity、Janus-Pro、RAR、Switti、Open-MAGVIT2 共 7 种 AR 模型。
  下载后用 `scripts/organize_arforensics.py` 组织目录（val 集重划分为 train/val，
  test 分卷按模型解压）。
  **注意**: 当前 train/val 由官方 validation 重划分而来 (协议名
  `ARForensics-small-val-derived`, 6k train + 1k val 每类)，并非官方
  100k/100k 训练分卷，因此本仓库训练出的指标与 D³QE 论文数字不直接可比。
  划分清单由 `scripts/make_manifest.py` 生成到 `splits/`。
- 当前服务器运行环境: `/home/yuanyi/anaconda3/envs/pytorch_pj/bin/python`（torch 2.9.1 + CUDA）。
- **DARG** (PRADA, CVPR Findings 2026): 155,000 张、20 种 AR 生成器；
  官方代码与预训练评分函数位于 `third_party/PRADA`。

## 评估约定 (score contract)

- 训练式方法 (`score_type="logit"`): 输出 BCE logit，评估时 `sigmoid` 后以 0.5 为阈值。
- Training-free 异常分数方法 (`score_type="anomaly"`, 如 RIGID/ZED): 评估时
  **不做 sigmoid**，直接对 raw score 计算 AUROC/AP/TPR@FPR; Acc/F1 使用在
  **验证集上以 Youden's J 校准**的阈值（`eval.py` 自动执行），而非固定的 0.5。
- 每次评估同时输出 `results_<split>.json` 与 `results_<split>_predictions.csv`
  (逐样本 path/label/generator/score/prediction)，任意指标可离线重算。
- `dataset.{train,val,test}_generators` 可按 generator 过滤样本，
  用于 leave-one-generator-out 协议。

## 实验设计建议（论文用）

- **跨生成器泛化**: 训练只用扩散类生成器，测试 AR 生成器 (LlamaGen/VAR/DALL·E 3)，
  利用 evaluator 输出的 per-generator 指标直接成表。
- **鲁棒性**: 测试集上加 JPEG 压缩 / resize / blur 扰动。
- **消融**: 在 `configs/` 中为每组消融建一个 yaml。
