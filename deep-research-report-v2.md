# MGI 实验框架规范化改造方案（deep-research-report-v2）

> **用途**：将本文档交给 Kimi Code，指导其对
> `MurmanskY/ai_image_detection` 仓库进行工程修改。\
> **当前阶段定位**：暂不开发或优化自己的检测方法（`ours/`），优先完善数据集、Baseline
> 忠实复现、统一评价、实验协议与可复现性。\
> **执行原则**：先审计、再修改；小步提交；不破坏正在运行的实验；不伪造结果；所有改动必须有测试。

------------------------------------------------------------------------

## 0. 给 Kimi Code 的总任务说明

请先完整阅读仓库，再按本文档实施改造。当前任务不是设计新的 AI
图像检测网络，而是把 MGI 建设成一个面向自回归（Autoregressive,
AR）生成图像鉴别的、可审计、可复现的 Benchmark Framework。

### 0.1 本轮工作范围

**必须完成的工作：**

1.  数据集接口与数据审计；
2.  Baseline 复现状态管理和论文协议对齐；
3.  统一预测分数语义和评价指标；
4.  统一实验协议与配置；
5.  实验结果、运行配置和环境信息的归档；
6.  自动化测试与文档。

**暂不处理：**

-   不设计新的检测网络；
-   不优化 `ours/method.py` 的模型结构；
-   不以提高某个 AUROC 为目标修改 Baseline；
-   不随意改变第三方官方代码；
-   不在未验证官方协议前，把自定义数据划分称为官方 Benchmark；
-   不把正在运行的实验中间结果作为最终测试结果。

### 0.2 Kimi Code 的执行方式

按以下顺序工作：

1.  检查当前 Git 状态、分支、commit SHA 和未提交改动；
2.  阅读相关源码、配置和现有实验结果；
3.  输出简短审计清单，指出需要修改的文件；
4.  先实现 P0 项，再实现 P1、P2；
5.  每个阶段运行测试；
6.  输出修改文件清单、测试命令和测试结果；
7.  未经用户明确授权，不执行大规模训练、不删除数据、不覆盖
    checkpoint、不改写历史实验结果。

**重要：**
不要假设仓库当前代码仍与之前审查时相同。以执行时的最新分支为准，先检查再决定修改。

------------------------------------------------------------------------

# 1. 项目目标与模块边界

建议将框架划分为四个模块：

1.  **Dataset
    Benchmark**：数据来源、标签、生成器元数据、官方划分、去重和预处理；
2.  **Baseline
    Implementations**：官方代码、忠实复现、重实现、简化版本的状态和协议；
3.  **Unified Evaluation**：统一 raw score、指标、阈值校准、置信区间；
4.  **Reproducibility**：配置、commit、环境、checkpoint
    标识、逐样本预测和结果归档。

基本数据流：

``` text
Dataset + Manifest
       |
       v
Baseline / Detector
       |
       v
Unified raw-score contract
       |
       v
Validation calibration (threshold metrics only)
       |
       v
Test evaluation (AUROC / AP / TPR@FPR / CI)
       |
       v
Run artifacts + report
```

所有方法必须通过统一接口输出逐样本连续分数。对于无法直接放进统一训练接口的第三方方法，可以通过
adapter 接入，但不得为了接口统一而改动其核心算法。

------------------------------------------------------------------------

# 2. P0：数据集与数据审计

## 2.1 数据集定位

建议按以下层次组织：

  --------------------------------------------------------------------------
  数据集                  定位                    使用方式
  ----------------------- ----------------------- --------------------------
  ARForensics             AR 检测主 Benchmark     主实验；严格保留官方 split

  DARG                    AR 生成器扩展数据集     独立外部测试；不要默认与
                                                  ARForensics 混合

  GenImage                通用 AI 图像检测补充    测试传统 GAN / Diffusion
                          Benchmark               检测器迁移到 AR 的能力
  --------------------------------------------------------------------------

数据集的实际可用性、许可、下载地址、版本和划分规则，必须以对应论文及官方发布页为准。不要仅凭
README 的二手描述认定某个 split 是官方 split。

## 2.2 官方划分与自定义划分必须分开

为每种数据集记录：

-   dataset name；
-   dataset version；
-   source URL / DOI；
-   official split definition；
-   local split definition；
-   image count by split；
-   image count by label；
-   image count by generator；
-   preprocessing protocol；
-   dataset manifest hash。

如果当前脚本把官方 validation 数据重新划分为
train/val，应将其标记为自定义划分，例如：

`ARForensics-val-derived-pilot-v1`

不能把这套结果报告为官方 ARForensics split 的结果。

建议保留两种模式：

-   `official`：完全遵循数据集原始协议；
-   `pilot`：允许用于调试、快速开发和预实验，但必须在结果中显式标记。

不得静默覆盖原始数据划分。

## 2.3 建立统一 Manifest

新增或完善：

``` text
datasets/
    manifest.py
    audit.py
    transforms.py
```

建议每张图片一行，至少包含：

``` csv
sample_id,path,sha256,phash,label,generator,generator_family,source,split,original_split,dataset
```

字段定义：

  字段                 含义
  -------------------- -------------------------------------
  `sample_id`          稳定唯一 ID
  `path`               相对数据根目录的路径
  `sha256`             文件级哈希
  `phash`              感知哈希，用于近似重复检查
  `label`              统一为 `0=real, 1=fake`
  `generator`          具体生成模型名称；真实图像用 `real`
  `generator_family`   `AR`、`GAN`、`Diffusion`、`real` 等
  `source`             真实图像来源或生成数据来源
  `split`              当前实验 split
  `original_split`     数据集发布时的原始 split
  `dataset`            数据集名称与版本

如果原始数据没有某字段，写入 `unknown`
或空值，并在文档中说明；不要推测补全。

## 2.4 数据审计脚本

新增 CLI，例如：

``` bash
python -m datasets.audit \
  --manifest data/manifests/arforensics.csv \
  --check-split-overlap \
  --check-exact-duplicates \
  --check-near-duplicates \
  --check-labels \
  --check-generator-metadata \
  --report reports/dataset_audit/arforensics.json
```

至少检查：

1.  Train / Val / Test 之间 SHA256 是否重复；
2.  Train / Val / Test 之间 pHash 近似重复；
3.  real/fake 标签是否与目录和元数据一致；
4.  fake 样本是否有正确 generator；
5.  real 样本的 generator 是否统一为 `real`；
6.  每个 split、label、generator 的样本数；
7.  文件是否损坏、无法解码或重复路径；
8.  分辨率、长宽比、格式、文件大小分布；
9.  是否存在 real/fake 的明显文件格式或压缩差异；
10. 同一真实图像是否被重复用于多个 generator 子集。

对于 pHash，阈值必须可配置，并在审计报告中记录。不要把 pHash
相似直接等同于重复；应输出候选对供人工复核。

## 2.5 数据变换规范

-   Train transform 和 Eval transform 分离；
-   所有方法的输入尺寸、插值方式、归一化方式应遵循各自原论文；
-   不得强制所有方法使用同一 transform，导致违反原论文设置；
-   统一评估的是同一批原始样本，但允许方法 adapter
    内部执行论文规定的预处理；
-   保存每种方法的 preprocessing description；
-   JPEG / resize / crop 等鲁棒性变换应由独立、确定性的 corruption
    pipeline 生成，并记录参数与随机种子。

------------------------------------------------------------------------

# 3. P0：Baseline 复现状态和论文匹配

## 3.1 建立 Baseline Registry

建议新增：

``` text
baselines/
    registry.yaml
    adapters/
    reproduction/
```

每个 Baseline 必须登记：

``` yaml
name: rigid
paper_title: "RIGID: A Training-free and Model-Agnostic Framework for Robust AI-Generated Image Detection"
paper_url: "..."
official_code_url: "..."
official_checkpoint_url: "..."
implementation_status: reimplementation
access_type: black_box
score_type: similarity
higher_is_fake: false
requires_training: false
input_resolution: null
preprocessing: "See reproduction/rigid.md"
protocol_reference: "..."
```

上面的 `null` 和 URL 仅作 schema 示例。Kimi Code
必须核验真实信息后填写，不得编造链接、权重或参数。

## 3.2 统一复现状态标签

每个方法只能使用以下状态之一或多个阶段状态：

  ----------------------------------------------------------------------------------
  状态                                定义
  ----------------------------------- ----------------------------------------------
  `vendored`                          已将第三方代码放入仓库

  `integrated`                        已接入 MGI 数据与评估 API

  `reimplemented`                     根据论文自行实现

  `faithful_reproduction`             关键结构、预处理、训练和评估协议均按论文实现

  `verified`                          已运行并与论文结果进行对照，偏差已解释

  `lite`                              明确简化版本，不等同于论文完整方法

  `inspired`                          仅借鉴思想，不应标为该论文的复现
  ----------------------------------------------------------------------------------

不要把"代码已复制"写成"已复现"。不要把简化版的结果与论文官方结果混在同一列。

## 3.3 Baseline 优先级

### AR 专用方法

  ------------------------------------------------------------------------------
  方法                    论文                    要求
  ----------------------- ----------------------- ------------------------------
  D³QE                    ICCV 2025               优先核验官方代码、权重和官方
                                                  split

  PRADA                   CVPR Findings 2026      保留官方执行环境和模型
                                                  wrapper；通过 adapter 接入

  Data Provenance         ICLR 2026               保留每个 generator
                                                  的原始依赖环境；通过 adapter
                                                  接入
  ------------------------------------------------------------------------------

### 通用 AI 图像检测方法

  --------------------------------------------------------------------------------------
  方法                    论文                    要求
  ----------------------- ----------------------- --------------------------------------
  CNNDetection            Wang et al., CVPR 2020  对齐训练数据、增强、backbone
                                                  和测试协议

  FreqSpec                Frank et al., ICML 2020 核验频谱处理、输入和分类头

  NPR                     CVPR 2024               优先核验官方代码；自行实现须标注
                                                  reimplementation

  UnivFD                  CVPR 2023               对齐 CLIP
                                                  backbone、预训练权重、特征和训练协议

  RIGID                   2024                    对齐扰动方式、DINO 特征、相似度和
                                                  score direction

  ZED                     ECCV 2024               使用官方 SReC checkpoint
                                                  和原始统计定义
  --------------------------------------------------------------------------------------

### 架构特定方法

  ---------------------------------------------------------------------------------
  方法                    原始适用领域            处理方式
  ----------------------- ----------------------- ---------------------------------
  DIRE                    Diffusion               标记
                                                  `architecture_specific`，说明在
                                                  AR 上属于迁移测试

  LGrad                   GAN                     标记
                                                  `architecture_specific`，说明在
                                                  AR 上属于迁移测试
  ---------------------------------------------------------------------------------

DIRE 和 LGrad 在 AR 上表现不佳，不应直接解释为它们在原适用领域无效。

## 3.4 每种方法建立 reproduction note

新增：

``` text
baselines/reproduction/
    cnn_detection.md
    frequency.md
    npr.md
    univfd.md
    rigid.md
    zed.md
    d3qe.md
    prada.md
    data_provenance.md
    dire.md
    lgrad.md
```

每份文件至少说明：

1.  论文引用与官方链接；
2.  官方代码 commit；
3.  权重来源及 SHA256（若有）；
4.  使用的 backbone；
5.  输入分辨率；
6.  图像预处理；
7.  数据增强；
8.  训练数据、验证数据和测试数据；
9.  训练轮数、optimizer、学习率等；
10. score 的原始定义及方向；
11. MGI 与原论文不一致的地方；
12. 当前复现状态；
13. 论文报告结果与 MGI 结果对比；
14. 未能复现的原因或待确认事项。

------------------------------------------------------------------------

# 4. P0：统一预测接口和 Score Contract

这是所有评价的基础。

## 4.1 统一 score 语义

每个 detector 必须显式声明：

``` python
score_type = "logit"  # logit | probability | anomaly | similarity | distance
higher_is_fake = True
```

建议再声明：

``` python
requires_calibration = False
access_type = "black_box"
```

统一规则：

**进入公共 evaluator 的 raw score 必须满足：数值越大，越倾向于 fake。**

例如：

-   fake logit：直接使用；
-   fake probability：直接使用；
-   anomaly score 越大越像 fake：直接使用；
-   similarity 越大越像 real：取负号；
-   distance 越大越像 fake：直接使用；
-   方向不明确：停止评估并要求方法 adapter 明确声明，不能默认猜测。

对于只返回两个类别 logits 的分类器，应明确 fake
类别索引，不能假定第二列一定是 fake。

## 4.2 不要对 AUROC/AP 先做 sigmoid 或阈值化

AUROC 和 AP 应使用连续 raw score。对于严格单调的
sigmoid，排序型指标理论上不变，但 raw score 更便于统一校验和审计。

不要将 raw score 截断成 0/1 后计算 AUROC/AP。

## 4.3 统一 API

建议 `engine/evaluator.py` 提供：

``` python
predict_scores(
    method,
    dataloader,
    device,
    split,
    amp=False,
) -> PredictionTable
```

`PredictionTable` 至少包含：

``` text
sample_id
path
label
generator
raw_score
fake_score
split
```

要求：

-   `raw_score` 保存方法原始输出；
-   `fake_score` 是方向归一化后的连续分数；
-   标签统一 `0=real, 1=fake`；
-   保留 generator 和 sample ID；
-   不在 evaluator 中隐式拟合模型；
-   不在 Test 上选择阈值。

所有第三方方法使用 adapter 输出该结构。

## 4.4 增加 Score Contract 单元测试

必须覆盖：

1.  `higher_is_fake=True` 不反转；
2.  `higher_is_fake=False` 正确反转；
3.  logits 与概率输出都能处理；
4.  两类 logits 能正确提取 fake 类；
5.  NaN / Inf 被检测并报告；
6.  预测数量与标签数量一致；
7.  sample_id 顺序与预测顺序一致；
8.  同一输入重复推理时，确定性方法的输出一致。

------------------------------------------------------------------------

# 5. P0/P1：统一评价指标和阈值校准

## 5.1 核心指标

主结果至少报告：

  指标          说明
  ------------- --------------------------
  AUROC         ROC 曲线下面积
  AP / AUPRC    Average Precision
  TPR@1%FPR     低误报区间的检出率
  TPR@0.1%FPR   更严格低误报区间的检出率

辅助指标：

-   Accuracy；
-   Balanced Accuracy；
-   Precision；
-   Recall；
-   F1；
-   FPR@95%TPR。

所有指标必须注明正类为 fake（label=1）。

注意：AP 会受正负样本比例影响。跨数据集或类别比例不同时，不能只比较 AP
而不报告样本分布。

## 5.2 TPR@FPR 的定义

使用 ROC 曲线计算 TPR@1%FPR 和 TPR@0.1%FPR。

必须明确插值或阈值选择方式，并固定实现，避免不同方法采用不同计算方式。

建议：

-   主指标使用 sklearn ROC 曲线及明确的插值规则；
-   保存 ROC points 或足以重算的逐样本 score；
-   对极低 FPR 指标报告真实样本数和误报数量；
-   样本数不足时标记为统计不稳定，不要过度解释。

## 5.3 阈值校准

阈值型指标必须在 Validation 上校准，再冻结到 Test。

推荐协议：

``` text
Train -> fit detector
Validation -> select threshold at target FPR
Test -> evaluate once using frozen threshold
```

建议主阈值规则：

-   `target_fpr=0.01`；
-   在 Validation 的真实图像上确定阈值；
-   保存阈值、校准 split、目标 FPR 和实际验证 FPR；
-   在 Test 上报告实际 FPR 和 TPR。

AUROC/AP 不需要阈值，直接使用 raw fake score。

对于训练式方法和训练自由方法，分别记录其阈值校准数据来源。训练自由不代表可以使用测试集调阈值。

## 5.4 置信区间与多次运行

建议增加：

-   AUROC 95% Bootstrap CI；
-   AP 95% Bootstrap CI；
-   TPR@1%FPR 的 Bootstrap CI；
-   可训练模型至少 3 个 seeds，资源允许时 5 个；
-   报告 mean ± std。

Bootstrap
应以样本为单位；如果同一真实图片与多个生成器样本存在配对关系，额外考虑按原始图像
ID 或 source group 聚类 bootstrap，避免把相关样本当作独立样本。

------------------------------------------------------------------------

# 6. P1：统一实验协议

至少支持四种 protocol。

## 6.1 In-distribution

-   训练、验证、测试划分固定；
-   可训练 Baseline 使用同一主数据划分；
-   保留各方法原论文要求的 transform 和训练设置；
-   所有方法测试同一批 sample IDs；
-   输出总体和 per-generator 指标。

## 6.2 Cross-generator

按生成器留出，而不是随机图片划分。

示例：

``` text
Train:
LlamaGen + VAR + RAR + Janus-Pro + Switti + Open-MAGVIT2

Test:
Infinity
```

轮换每个 generator，形成 Leave-One-Generator-Out（LOGO）结果。

Validation / threshold calibration 也不能使用 held-out generator
的标签或图像，除非协议明确将其定义为 few-shot target
adaptation，并单独报告。

## 6.3 Robustness

建议 corruption：

  类型         参数示例
  ------------ ---------------------------
  JPEG         quality 95 / 75 / 50 / 30
  Resize       0.75 / 0.5 / 0.25
  Blur         多个 Gaussian sigma
  Noise        多个 Gaussian noise level
  Crop         10% / 25% / 50%
  Screenshot   截图后重新编码

要求：

-   所有方法使用同一组 corruption 样本；
-   记录 corruption 参数和随机种子；
-   同时报告 clean 和 perturbed；
-   不在测试 corruption 上重新校准阈值；
-   可额外报告 clean-calibrated threshold 下的鲁棒性。

## 6.4 External dataset

-   ARForensics 和 DARG 分开报告；
-   不默认混合训练；
-   明确哪些 generator / source 在训练中出现过；
-   记录数据集版本和 manifest hash。

------------------------------------------------------------------------

# 7. P1：实验配置和结果归档

## 7.1 推荐目录

尽量在现有结构上增量改造，不要无必要地大规模重构。

``` text
MGI/
├── datasets/
│   ├── arforensics.py
│   ├── darg.py
│   ├── genimage.py
│   ├── manifest.py
│   ├── audit.py
│   └── transforms.py
├── baselines/
│   ├── registry.yaml
│   ├── adapters/
│   └── reproduction/
├── engine/
│   ├── trainer.py
│   ├── evaluator.py
│   ├── metrics.py
│   ├── calibration.py
│   ├── statistics.py
│   └── prediction.py
├── configs/
│   ├── datasets/
│   ├── baselines/
│   └── protocols/
├── experiments/
│   ├── in_distribution/
│   ├── cross_generator/
│   ├── robustness/
│   └── external_dataset/
├── reports/
│   ├── dataset_audit/
│   ├── runs/
│   └── RESULTS.md
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_scores.py
│   ├── test_splits.py
│   └── test_reproducibility.py
└── train.py / eval.py
```

## 7.2 输出目录与 Git 忽略策略

不要把大型 checkpoint、原始图片和中间 tensor 提交到 Git。

建议：

-   `outputs/`：继续忽略 checkpoint、缓存和大文件；
-   `reports/`：提交轻量、可审计的结果文件；
-   如逐样本预测过大，压缩成 CSV/Parquet，或发布到 Release / 外部
    artifact storage；
-   不要把私人数据、授权不明数据或受限数据上传到公开仓库。

每次完成的实验至少保存：

``` text
reports/runs/<run_id>/
    config.yaml
    metrics.json
    predictions.csv.gz
    manifest.json
    environment.json
    README.md
```

`manifest.json` 至少记录：

-   git commit SHA；
-   dataset name/version；
-   dataset manifest hash；
-   split；
-   method；
-   method source/status；
-   seed；
-   checkpoint identifier/hash；
-   sample count；
-   score semantics；
-   threshold calibration metadata；
-   start/end time。

## 7.3 结果状态

`RESULTS.md` 只列出已完成并冻结的结果。

正在运行的任务单独列出：

``` text
Status: RUNNING
Final test metrics: NOT REPORTED
```

禁止把中间 epoch 的 validation 指标、未校准的 accuracy
或论文作者报告的数字写成 MGI 复现结果。

------------------------------------------------------------------------

# 8. P1/P2：自动化测试与复现环境

## 8.1 必需测试

  测试                   目标
  ---------------------- ----------------------------------
  Dataset test           标签、路径、generator 元数据正确
  Split test             Train/Val/Test 无重复
  Score test             分数方向正确
  Metric test            与 sklearn 手工结果一致
  Calibration test       阈值只从 Validation 得到
  Prediction test        样本数和 sample_id 对齐
  Integration test       每个 Baseline 能完成小批量推理
  Reproducibility test   固定 seed 时结果可重复

测试指标时，构造一个很小的已知数组，手工计算并与实现比对。不要只测试代码"没有报错"。

## 8.2 环境锁定

第三方方法可能有不同 PyTorch / CUDA
依赖，不要强行让所有官方代码共享一个环境。

建议：

``` text
environment/
    base.lock
    d3qe.lock
    prada.lock
    data_provenance_<model>.lock
```

记录：

-   Python；
-   PyTorch；
-   torchvision；
-   CUDA；
-   GPU；
-   numpy；
-   scikit-learn；
-   timm / open_clip；
-   其他关键依赖版本。

对每个官方仓库保留其 license、引用信息和 upstream commit。

------------------------------------------------------------------------

# 9. 推荐执行顺序与验收标准

## Phase 0：仓库安全检查

执行前：

``` bash
git status --short
git branch --show-current
git rev-parse HEAD
```

要求：

-   不覆盖用户未提交修改；
-   不删除 outputs；
-   不清理 checkpoint；
-   不改写历史提交。

## Phase 1：P0 数据与评价链路

依次完成：

1.  核验当前数据划分和官方协议；
2.  增加 manifest；
3.  增加 dataset audit；
4.  修复 evaluator API；
5.  实现 score contract；
6.  实现 AUROC、AP、TPR@1%FPR、TPR@0.1%FPR；
7.  实现 Validation threshold calibration；
8.  保存逐样本 raw/fake score；
9.  增加单元测试。

**验收条件：**

-   小型合成数据测试通过；
-   手工 score 方向测试通过；
-   AUROC/AP 与 sklearn 结果一致；
-   验证集阈值不会读取 Test；
-   数据审计能发现人为注入的重复样本和错误 generator 标签；
-   `eval.py` 可完成小规模端到端评估。

## Phase 2：Baseline 对齐

优先顺序：

1.  CNNDetection；
2.  UnivFD；
3.  RIGID；
4.  ZED；
5.  D³QE；
6.  PRADA；
7.  Data Provenance；
8.  FreqSpec / NPR；
9.  DIRE / LGrad（架构特定迁移对照）。

每种方法完成：

-   reproduction note；
-   adapter；
-   score contract；
-   小批量推理；
-   结果归档；
-   与原论文协议差异说明。

**验收条件：**

-   不能只 import 成功；
-   至少完成一次完整 split 的推理；
-   有逐样本 score；
-   指标可独立重算；
-   方法状态标签准确。

## Phase 3：Benchmark 实验

依次完成：

1.  In-distribution；
2.  Per-generator；
3.  LOGO；
4.  Robustness；
5.  DARG external evaluation；
6.  多 seed 与置信区间。

**验收条件：**

-   每张结果表能追溯到 run_id；
-   每个 run_id 能找到配置和 manifest；
-   每个结果可从逐样本 score 重新计算；
-   不同方法使用一致的样本集合；
-   官方 split 与 pilot split 分开报告。

------------------------------------------------------------------------

# 10. Kimi Code 最终交付要求

完成修改后，请输出：

## A. 修改摘要

-   修改了哪些文件；
-   新增了哪些文件；
-   每项修改解决了什么问题；
-   哪些问题暂时没有解决及原因。

## B. 数据集审计结果

-   各数据集样本数；
-   split 样本分布；
-   generator 样本分布；
-   exact duplicate 数量；
-   near-duplicate 候选数量；
-   错误标签或元数据数量；
-   官方 split 与自定义 split 的差异。

没有真实数据时应标记 `NOT RUN`，不能编造统计结果。

## C. Baseline 对齐表

逐项报告：

-   方法；
-   论文；
-   官方代码；
-   权重；
-   implementation status；
-   score type；
-   higher_is_fake；
-   preprocessing；
-   与论文协议差异；
-   是否完成端到端测试。

## D. 评价系统测试

报告所有测试命令、通过/失败状态和关键输出。

## E. 实验结果归档

列出已完成、运行中、失败的 run_id。不要把未完成实验写成最终结果。

## F. 后续待办

按 P0 / P1 / P2 排序，说明依赖关系。

------------------------------------------------------------------------

# 11. 不可违反的约束

1.  不要修改或删除第三方官方实现的核心算法；必要适配放在独立 adapter。
2.  不要把简化实现标记为官方复现。
3.  不要把自定义 split 标记为官方 split。
4.  不要用 Test 数据选阈值、调参或选 checkpoint。
5.  不要把 logits、similarity、distance、anomaly score 的方向混为一谈。
6.  不要先二值化再计算 AUROC/AP。
7.  不要将论文报告数字写成自己的复现结果。
8.  不要提交未经确认的数据集、受限数据或大型 checkpoint。
9.  不要覆盖正在运行的实验或其输出文件。
10. 不要在本轮任务中开发新的 `ours` 模型。
11. 不要为了"统一预处理"而破坏原论文方法的输入协议。
12. 任何无法核实的信息都明确标记 `UNKNOWN` / `NOT VERIFIED`。

------------------------------------------------------------------------

# 12. 参考论文与官方资源

Kimi Code
应核验这些来源的标题、年份、版本和实现细节；不要仅依赖本文档中的概括。

-   **D³QE**, ICCV 2025：\
    https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_D3QE_Learning_Discrete_Distribution_Discrepancy-aware_Quantization_Error_for_Autoregressive-Generated_Image_ICCV_2025_paper.html\
    官方项目： https://ivg-yanranzhang.github.io/D3QE/

-   **PRADA**, CVPR Findings 2026：\
    https://arxiv.org/abs/2511.20068\
    官方代码： https://github.com/jonasricker/prada

-   **Data Provenance for Image Auto-Regressive Generation**, ICLR
    2026：\
    https://proceedings.iclr.cc/paper_files/paper/2026/hash/5a1667459d0cdeb2fe6b2f0dffc5cb9d-Abstract-Conference.html

-   **ZED: Zero-Shot Detection of AI-Generated Images**, ECCV 2024：\
    https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2665_ECCV_2024_paper.php\
    官方代码： https://github.com/grip-unina/ZED

-   **RIGID**, 2024：\
    https://arxiv.org/abs/2405.20112

-   **GenImage**, NeurIPS 2023：\
    https://github.com/GenImage-Dataset/GenImage

-   MGI 当前仓库：\
    https://github.com/MurmanskY/ai_image_detection

------------------------------------------------------------------------

## 最终执行目标

本轮改造完成后，MGI 应达到：

-   数据集来源和划分可追溯；
-   每个 Baseline 的实现状态和论文差异清楚；
-   所有方法输出方向统一的连续 fake score；
-   AUROC、AP、TPR@FPR 和置信区间计算规范；
-   阈值只在 Validation 上校准；
-   In-distribution、Cross-generator、Robustness、External dataset
    协议清楚；
-   每次实验有完整配置、环境、commit、manifest 和逐样本预测；
-   自动化测试能够发现数据、score 和 metric 的常见错误。

**本阶段的成功标准不是模型分数更高，而是任何研究者都能依据仓库中的配置、数据清单和预测结果，独立重算并核验
MGI 报告的实验结果。**
