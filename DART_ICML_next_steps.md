# DART：当前问题诊断与下一步科研计划

**研究方向：** 面向自回归（Autoregressive,
AR）生成图像取证的多尺度生成循环一致性建模\
**方法名称：** DART（Distribution-Aware Autoregressive Token Detector）\
**目标会议：** ICML 2027\
**仓库：** <https://github.com/MurmanskY/ai_image_detection>\
**主要方法文档：**
[`docs/method_scale_dart.md`](https://github.com/MurmanskY/ai_image_detection/blob/main/docs/method_scale_dart.md)\
**主要实验记录：**
[`reports/RESULTS.md`](https://github.com/MurmanskY/ai_image_detection/blob/main/reports/RESULTS.md)，第
24--29 节\
**分析基准提交：** `8d372bcd3e4587492bbe45409c4db0df0ff11a05`

------------------------------------------------------------------------

## 0. 执行摘要

DART 是目前项目中最值得继续发展的方法主线。它将 AR
生成链中的三类证据------token likelihood distribution、quantization
error 和 decoder reconstruction error------组织为多尺度联合特征，并使用
Deep Sets 与 attention pooling 学习不同 scale 的贡献。

现有实验已经显示出明显的宏观 AUROC 增益：

-   1000-real、9-seed 设置下，DART macro AUROC 为 **0.902 ± 0.015**；
-   同五模型 bank 的 PRADA macro AUROC 约为 **0.621--0.668**；
-   seed 0 paired macro ΔAUROC 为 **+0.247**，95% CI 为 **\[+0.220,
    +0.275\]**；
-   JPEG95 单 seed smoke 下 macro AUROC 为 **0.859**；
-   ImageNet-V2 独立真实来源 smoke 下 macro AUROC 为 **0.913**，外部真实
    FPR 为 **4.67%**。

但当前结果还不足以直接支撑成熟的 ICML 投稿。主要问题包括：

1.  **Calibration 数据复用：** 当前实现中，real calibration
    数据既用于模型分数 median/IQR 归一化，也用于 conformal
    阈值估计；并且参与 validation loss / checkpoint 选择。标准
    split-conformal 的有限样本保证因此不能直接套用。
2.  **Generator-family 泛化不足：**
    当前训练映射中，多个测试模型对应的生成器家族已经在训练 fake
    中出现。现有实验主要是图像级留出和有限的家族迁移，不是严格的
    unseen-family zero-shot。
3.  **Model-bank 覆盖影响明显：** 加入 RAR-L 后 RAR-B 的 AUROC 从约
    0.53--0.57 提升到约 0.88--0.92，说明模型 bank
    覆盖是方法性能的重要因素。
4.  **低 FPR 工作点不均衡：** VAR/HMAR 的 TPR@5%FPR 分别约为
    0.134/0.100，尽管 macro AUROC 较高，低误报场景下仍然较弱。
5.  **跨 codec 证据有限：** JPEG95 只有单 seed smoke，VAR/HMAR 的 AUROC
    分别下降约 0.084/0.065，尚不能声称 codec-invariant。
6.  **消融尚不完整：**
    已有逐步加入特征的消融，但仍需单特征、两两组合、聚合方式和 bank
    aggregation 的完整消融，以证明真正的互补机制。

**总体决策：** 暂停新增特征和网络模块。先修复 calibration
protocol，再完成严格 generator-family holdout、model-bank
coverage、完整消融、跨 codec 多 seed 和低 FPR
分析。所有结论必须依据修复后的协议重新确认。

------------------------------------------------------------------------

# 一、DART 方法回顾

## 1.1 方法定义

DART（Distribution-Aware Autoregressive Token
Detector）是一种模型特异的监督式 AR 图像检测方法。

它不只依赖单个 token likelihood 分数，而是联合建模 AR 生成过程中的：

1.  **Token likelihood distribution：** 条件与无条件 token
    log-likelihood 差值的 signed quantiles；
2.  **Quantization error：** 每个生成 scale 的平均量化误差；
3.  **Decoder reconstruction error：** 输入图像与对应 AR decoder
    重建图像之间的感知差异；
4.  **Scale metadata：** scale length 与 scale index。

每个 scale 的特征经共享 encoder 编码，再通过 attention pooling
聚合，最后由分类头输出检测分数。

当前完整特征维度为每 scale 18 维：16 个 signed quantiles，加 scale
length/index、quantization error 和 reconstruction error。

## 1.2 核心机制假设

DART 的研究假设是：一张图像是否符合特定 AR 生成机制，不应只通过孤立的
token likelihood 判断，而应同时考察 token likelihood、量化和 decoder
reconstruction 三个环节是否形成相互一致的统计模式。

该假设目前得到逐步消融结果的初步支持，但还需要更严格的单特征/组合消融和跨家族验证。不能把它表述为已经证明的普遍生成定律。

## 1.3 训练和推理设定

当前 DART 不是 real-only 或无需目标模型信息的 zero-shot 方法。每个参考
AR 模型独立训练一个 DART，训练中使用真实图像和对应 fake
图像；多个模型的分数经过 real calibration 归一化后，以 model-bank max
方式聚合。

因此当前准确定位是：

**Model-aware, supervised, multi-model-bank autoregressive forensic
detection.**

论文必须明确区分：

-   已知家族中的图像级留出；
-   同家族模型变体迁移；
-   完全未参与训练的生成器家族留出。

这三种任务不能混称为 unseen-generator zero-shot detection。

------------------------------------------------------------------------

# 二、当前实验结果

## 2.1 SCALE、QSCALE 与 Shared DART 的负结果

  ---------------------------------------------------------------------------------
  方法                 设计                     Macro AUROC / 结果 结论
  -------------------- --------------------- --------------------- ----------------
  SCALE-Fisher         逐位置经验尾概率 +             0.717--0.740 未达到强基线
                       Fisher 聚合

  SCALE-HC             Higher Criticism 聚合          0.396--0.413 失败

  QSCALE-Mahalanobis   Signed                         0.715--0.731 未达到 PRADA
                       quantile-function +
                       收缩 Mahalanobis

  QSCALE-kNN           Signed                         0.767--0.784 未达到 PRADA
                       quantile-function +
                       kNN

  Shared DART LOGO     跨生成器共享 Deep              0.629--0.706 LlamaGen
                       Sets                                        出现方向反转
  ---------------------------------------------------------------------------------

这些负结果支持一个重要观察：不同 AR 模型的 likelihood distribution
具有明显模型特异性，简单共享一个分类器无法可靠消除这种差异。

## 2.2 完整 DART 的逐步消融

  方法                   VAR-d16 AUROC   LlamaGen-B256 AUROC        Macro AUROC
  ----------------- ------------------ --------------------- ------------------
  Likelihood-only         0.525--0.554          0.969--0.998       0.747--0.776
  \+ Quantization         0.587--0.642          0.994--0.999       0.791--0.819
  Full DART           **0.811--0.876**      **0.994--0.999**   **0.905--0.938**
  PRADA                   0.477--0.509          0.993--0.997       0.737--0.752

VAR-d16 的增益尤其明显，说明 quantization 和 decoder reconstruction
evidence 可能补充了 likelihood-only 的不足。

但当前逐步消融不能单独证明三个组件之间存在非平凡交互。仍需完整的单特征与两两组合实验。

## 2.3 五模型 bank 结果

参考模型 bank：

-   VAR-d20；
-   VAR-d30；
-   LlamaGen-B256；
-   LlamaGen-L256；
-   RAR-L。

测试生成器：

-   VAR-d16；
-   LlamaGen-B256；
-   RAR-B；
-   HMAR-D16。

三 seed AUROC 范围：

  Generator                     DART   同五模型 bank PRADA
  --------------- ------------------ ---------------------
  VAR-d16               0.770--0.863          0.478--0.509
  LlamaGen-B256         0.994--0.999          0.993--0.997
  RAR-B                 0.880--0.922          0.520--0.594
  HMAR-D16              0.755--0.848          0.484--0.542
  Macro             **0.855--0.898**          0.619--0.660

加入 RAR-L 后，RAR-B 从约 0.532--0.574 提升至 0.880--0.922。这说明
model-bank coverage 会显著影响模型特异检测器的表现。

该结果不能被解释为完全未知家族泛化：RAR-L 与 RAR-B
属于同一生成器家族，VAR-d16 与训练中的 VAR fake 也存在家族关系。

## 2.4 1000-real、9-seed P1

协议：

  数据                    数量
  --------------------- ------
  Real reference           400
  Real calibration         300
  Real test                300
  每个 fake generator      250
  Seeds                      9

结果：

-   DART macro AUROC：**0.902 ± 0.015**；
-   同 bank PRADA：0.621--0.668；
-   最小 seed macro 增益：+0.245；
-   seed 0 paired macro ΔAUROC：+0.247；
-   seed 0 paired 95% CI：\[+0.220, +0.275\]。

各生成器：

  Generator         DART 平均 AUROC   TPR@1%FPR 范围   TPR@5%FPR 均值
  --------------- ----------------- ---------------- ----------------
  VAR-d16                     0.845          0--0.01            0.134
  LlamaGen-B256               0.998       0.89--1.00            0.997
  RAR-B                       0.942       0.06--0.58            0.733
  HMAR-D16                    0.824          0--0.01            0.100

9 个 seed 的 FPR 为 3.00%--7.33%，seed 8 为 7.33%，略高于预注册 7%
上限。因此，9-seed 严格 GO=false。不能只报告前三个 seed 的通过结果。

## 2.5 JPEG95 与 ImageNet-V2 外部真实来源

### JPEG95

单 seed smoke，训练与校准使用原图，只将测试图重编码为 JPEG95：

  Generator         原图 AUROC   JPEG95 AUROC    下降
  --------------- ------------ -------------- -------
  VAR-d16                0.836          0.753   0.084
  LlamaGen-B256          0.998          0.996   0.002
  RAR-B                  0.943          0.922   0.021
  HMAR-D16               0.831          0.766   0.065
  Macro                  0.902          0.859   0.043

JPEG95 FPR 为 6%。结果说明宏观增益仍保留，但 VAR/HMAR 对压缩更敏感。单
seed smoke 不足以支持 codec-invariant 结论。

### ImageNet-V2

使用 300 张独立真实图像，不重新训练或调整阈值：

-   DART macro AUROC：0.913；
-   同 bank PRADA：0.704；
-   DART 外部真实 FPR：4.67%；
-   PRADA 外部真实 FPR：2.00%。

这是独立真实来源的初步校准迁移证据，但仍是单分片、单 seed；fake
仍来自原有 DARG 测试生成器，不能称为完整跨数据集生成器泛化实验。

------------------------------------------------------------------------

# 三、必须优先修复的问题

## P0-1：Calibration data reuse

### 现象

当前 `analysis/dart_c2i_p0.py` 中，真实 calibration
数据用于计算每个模型分数的 median/IQR：

``` python
median = np.median(model_scores["real"][cal_idx])
iqr = (
    np.percentile(model_scores["real"][cal_idx], 75)
    - np.percentile(model_scores["real"][cal_idx], 25)
    + 1e-6
)
```

随后，同一批 calibration 数据又用于 conformal 阈值：

``` python
tau = conformal_tau(combined_dart["real"][cal_idx])
```

该数据还参与 DART validation loss / checkpoint 选择。

### 风险

最终 conformal score function 不是在 calibration set
之前固定的。因而不能直接将现有 FPR 结果解释为标准 split-conformal
的有限样本保证。

### 修复方案

将真实数据严格分成四个互斥集合：

  -----------------------------------------------------------------------
  集合                                用途
  ----------------------------------- -----------------------------------
  Reference / training real           输入特征标准化、训练 DART

  Model-selection / normalization     模型选择、每个模型分数 median/IQR
  real                                归一化

  Conformal calibration real          只计算最终 conformal 阈值

  Test real                           最终 FPR 评估
  -----------------------------------------------------------------------

具体要求：

1.  Conformal calibration 分数函数必须在该集合使用前固定；
2.  不允许用 conformal calibration 数据选择 checkpoint；
3.  不允许用 conformal calibration 数据拟合 median/IQR；
4.  model bank 必须在最终 calibration 前冻结；
5.  测试集不能参与阈值选择；
6.  修复后重新运行 9 seeds、JPEG95 和 ImageNet-V2 实验。

可选方案是严格 cross-fitting，但必须清晰定义每个 fold
中模型、归一化参数和校准样本的独立关系。

**验收标准：** 逐样本分数可独立重算；conformal calibration
与模型选择/归一化数据完全隔离；报告 FPR 点估计和置信区间。

## P0-2：严格区分生成器家族留出

### 当前问题

现有训练映射：

  DART 参考模型   训练 fake 来源
  --------------- ----------------
  VAR-d20         VAR-d16
  VAR-d30         VAR-d16
  LlamaGen-B256   LlamaGen-B256
  LlamaGen-L256   LlamaGen-B256
  RAR-L           RAR-B

因此，VAR-d16、LlamaGen-B256 和 RAR-B
在训练阶段已经出现；测试图像互斥不等于生成器家族未知。

### 必须建立三个独立协议

  ----------------------------------------------------------------------------------------
  协议                    训练与测试关系                           可以支持的结论
  ----------------------- ---------------------------------------- -----------------------
  Known-family image      同家族不同图像                           已知家族内检测
  holdout

  Held-out variant        训练中无该具体模型，但有同家族其他型号   同家族模型变体迁移

  Held-out family         训练中完全没有目标生成器家族             未见家族泛化
  ----------------------------------------------------------------------------------------

### 建议实验

-   训练：VAR、LlamaGen、RAR；
-   测试：HMAR、Infinity、Janus、Switti、Open-MAGVIT2 等完全留出的家族；
-   按真实生成器谱系核验，不仅按模型名称判断；
-   每个留出家族使用独立 fake test；
-   训练、验证、校准和测试的 fake 图像严格互斥。

**验收标准：** 论文中明确报告三种协议，不把 image-level holdout 误称为
unseen-family zero-shot。

## P0-3：低 FPR 工作点不足

### 当前问题

VAR/HMAR 的 TPR@5%FPR 分别只有约 0.134/0.100。

因此，高 macro AUROC 并不代表所有生成器在低误报场景下都有实用检测能力。

### 下一步

1.  对 VAR/HMAR 的 false negative 做逐样本分析；
2.  比较不同模型特异 detector 的分数相关性；
3.  比较 max、mean、固定权重和学习式 aggregation；
4.  报告 TPR@0.1%、1%、5%、10%FPR；
5.  使用独立 real calibration，给出 FPR 置信区间；
6.  同时报告 per-generator 与 macro 指标。

不要只报告 AUROC，也不要通过测试集选择 aggregation 方式。

------------------------------------------------------------------------

# 四、下一步方法与实验优化

## P1：Model Bank Coverage 系统研究

### 动机

RAR-L 加入 bank 后，RAR-B 的 AUROC 从约 0.53--0.57 提升到约
0.88--0.92。这表明 model-aware detector 的表现可能依赖参考 bank
对生成器家族的覆盖。

### 实验设计

  实验     Model bank
  -------- ---------------------------
  Bank-1   VAR only
  Bank-2   VAR + LlamaGen
  Bank-3   VAR + LlamaGen + RAR
  Bank-4   当前完整五模型 bank
  Bank-5   扩展到官方完整八模型 bank

### 分析指标

-   Bank size；
-   Generator-family coverage；
-   同家族参考模型数量；
-   不同模型 score correlation；
-   Max aggregation 对 FPR 的影响；
-   加入一个 bank 模型带来的边际性能变化；
-   Bank 缺失目标家族时的检测风险。

### 方法改进原则

暂时不新增网络特征。先在固定 detector outputs 上比较 aggregation 和 bank
composition，避免把 bank coverage 效应与特征/网络变化混在一起。

## P2：完整特征消融

当前已有逐步加入 likelihood、quantization、reconstruction
的结果，但还需完整组合消融。

  消融                            目的
  ------------------------------- --------------------------
  Likelihood-only                 基础 token evidence
  Quantization-only               量化机制贡献
  Reconstruction-only             decoder evidence 贡献
  Likelihood + Quantization       两者互补性
  Likelihood + Reconstruction     重建证据贡献
  Quantization + Reconstruction   后两阶段关系
  Full DART                       三种机制联合
  Full DART - Attention           attention 是否必要
  Full DART - Scale metadata      scale 元信息是否必要
  Mean pooling / Sum pooling      Deep Sets 聚合方式的贡献
  Max bank / Mean bank            模型 bank 聚合策略的贡献

重点不是只证明 Full DART 数字最高，而是证明：

1.  每个组件在不同生成器上的边际贡献；
2.  组件组合存在可重复的互补收益；
3.  增益不是单纯来自参数量或模型容量；
4.  attention 确实学习到有意义的 scale 权重。

建议增加参数量匹配的 MLP、mean pooling 和 sum pooling 对照。

## P3：跨 Codec 多 seed 实验

当前 JPEG95 只有单 seed。

建议固定训练、归一化、校准和测试 source split，开展：

  Codec      目标
  ---------- ----------
  Original   原始性能
  JPEG95     轻度压缩
  JPEG85     中度压缩
  JPEG75     较强压缩
  JPEG50     强压缩

分别报告三种协议：

1.  Original train / original calibration / compressed test；
2.  Original train / compressed calibration / compressed test；
3.  Compressed train / compressed calibration / compressed test。

三种协议分别回答测试分布偏移、校准迁移和压缩域适配问题，不能混为一个
robustness 结果。

建议至少 3--5 seeds，并报告 generator-wise paired ΔAUROC、FPR CI 和
TPR@FPR。

## P4：外部数据集验证

ImageNet-V2 当前仅为 300 real、单分片、单 seed 的真实来源 smoke。

下一步应补充：

-   更大规模独立真实来源；
-   至少一个独立生成图像 benchmark；
-   不参与 DART 训练的生成器家族；
-   统一图像预处理与 codec protocol；
-   独立 calibration/test source；
-   逐样本预测与可复现 manifest。

必须区分"外部 real calibration transfer"和"外部 fake generator
generalization"。

------------------------------------------------------------------------

# 五、Baseline 对比与公平性要求

当前最重要的 baseline 是 PRADA，但论文不能只比较一个 baseline。

## 5.1 必须保留的对照

  方法                            对照目的
  ------------------------------- ----------------------------------------------
  PRADA 官方评分                  AR conditional/unconditional likelihood 基线
  D³QE                            Codebook distribution + quantization error
  Data Provenance                 Codebook / decoder provenance
  DART likelihood-only            检验新增机制的增量
  DART full                       完整方法
  监督式通用检测器（如 UnivFD）   提供不同训练协议下的参考性能

## 5.2 公平比较要求

-   区分官方实现与本地复现；
-   区分官方完整 model bank 与同五模型 bank；
-   报告每个方法使用的训练信息；
-   对 DART 和可训练 baseline 使用相同 fake train/validation split；
-   对 real reference、normalization、calibration 和 test 做严格隔离；
-   同时报告 AUROC、AP、TPR@FPR、FPR；
-   paired bootstrap 必须基于相同 test images；
-   不把 PRADA 官方预训练分数与 DART
    监督训练协议描述成完全相同的训练条件。

------------------------------------------------------------------------

# 六、论文创新点的建议定位

## 6.1 推荐题目

**DART: Modeling Multi-Scale Generation-Cycle Consistency for
Autoregressive Image Forensics**

中文：

**DART：面向自回归图像取证的多尺度生成循环一致性建模。**

## 6.2 建议三项贡献

### Contribution 1：Generation-Cycle Consistency Representation

联合建模：

-   Token likelihood distribution；
-   Quantization error；
-   Decoder reconstruction error。

重点是将三个生成环节组织成多尺度联合证据，而不是声称单个特征本身全新。

### Contribution 2：Scale-aware Distribution Aggregation

通过 Deep Sets 和 attention pooling 处理可变数量的 AR
scales，学习不同尺度的相对贡献。

需要用参数量匹配对照、mean/sum pooling 和 attention ablation
证明其必要性。

### Contribution 3：Model-bank-aware AR Forensic Evaluation

系统研究 model-bank family coverage 对检测性能、低 FPR 和校准的影响。

该贡献需要通过 Bank-1 至 Bank-5 实验补足后才能作为已验证贡献。

## 6.3 理论分析方向

不建议为了论文强行加入与实验无关的复杂定理。

可以从以下问题建立理论：

设 AR 生成链为：

\[ z `\rightarrow `{=tex}Q(z)
`\rightarrow `{=tex}`\hat{x}`{=tex}=D(Q(z)) \]

构造：

\[ `\Phi`{=tex}(x)= \[L(x),Q_e(x),R_e(x)\] \]

其中：

-   (L(x))：token likelihood distribution；
-   (Q_e(x))：quantization error；
-   (R_e(x))：reconstruction error。

研究：

\[ P(`\Phi`{=tex}(x)`\mid `{=tex}`\text{real}`{=tex})
`\quad`{=tex}`\text{与}`{=tex}`\quad`{=tex}
P(`\Phi`{=tex}(x)`\mid `{=tex}`\text{fake}`{=tex}) \]

之间的可分性，以及三个证据的条件依赖关系。

可以尝试证明或分析：

1.  在明确假设下，联合证据相对单一证据具有额外判别信息；
2.  多尺度聚合对估计误差和判别能力的影响；
3.  model-bank 缺失目标生成家族时的风险变化；
4.  参考模型覆盖与检测风险之间的关系。

重要边界：真实图像并不一定来自当前被检测的 AR 模型，不能假设所有 real
images 都满足某个目标模型的 tokenizer--decoder
不动点。理论必须明确适用条件。

------------------------------------------------------------------------

# 七、ICML 投稿前的执行顺序

## 阶段 1：协议可信度（最高优先级）

-   [ ] 独立 normalization / model-selection / conformal calibration /
    test；
-   [ ] 重新训练与评估 9 seeds；
-   [ ] 重新计算 paired bootstrap；
-   [ ] 重新评估 JPEG95；
-   [ ] 重新评估 ImageNet-V2；
-   [ ] 保存逐样本分数、配置、manifest 和环境信息；
-   [ ] 独立复算所有 AUROC、FPR、TPR@FPR。

## 阶段 2：泛化协议

-   [ ] 整理真实 generator-family taxonomy；
-   [ ] 建立 known-family image holdout；
-   [ ] 建立 held-out variant；
-   [ ] 建立 held-out family；
-   [ ] 确认 fake train/validation/test 互斥；
-   [ ] 对完全未见家族单独报告结果。

## 阶段 3：方法机制

-   [ ] 完成三种单特征消融；
-   [ ] 完成三种两两组合消融；
-   [ ] 完成 Full DART；
-   [ ] 完成参数量匹配对照；
-   [ ] 完成 attention / mean / sum pooling 对照；
-   [ ] 完成 model-bank aggregation 对照。

## 阶段 4：鲁棒性和实际工作点

-   [ ] JPEG95/85/75/50 多 seed；
-   [ ] 至少一个独立外部生成图像 benchmark；
-   [ ] 扩大独立真实来源；
-   [ ] 报告 TPR@0.1/1/5/10%FPR；
-   [ ] 分析 VAR/HMAR false negatives；
-   [ ] 报告 per-generator FPR、AUROC 和置信区间。

## 阶段 5：论文写作

-   [ ] 明确任务设定为 model-aware supervised AR forensic detection；
-   [ ] 明确区分已知家族、模型变体和未知家族；
-   [ ] 以生成循环一致性为核心方法贡献；
-   [ ] 以 model-bank coverage 为重要分析；
-   [ ] 对比 PRADA、D³QE、Data Provenance；
-   [ ] 报告负结果和局限；
-   [ ] 完成理论假设与证明；
-   [ ] 完成匿名化、复现材料和 AI 使用披露。

------------------------------------------------------------------------

# 八、建议停止的工作

当前不建议：

-   新增更多 likelihood 变体；
-   无依据地增加视觉 backbone；
-   继续堆叠残差特征；
-   只为提升 macro AUROC 反复调整阈值；
-   将图像级留出宣传为未知家族 zero-shot；
-   将单 seed JPEG95 smoke 宣传为 codec-invariant；
-   将单分片 ImageNet-V2 real transfer 宣传为完整跨数据集检测；
-   只报告前三个通过的 seeds，而忽略全部九个 seeds 中的失败项。

DART
现有方法组件已经足以形成论文主线。下一阶段的重点是验证机制、修复协议和扩大泛化证据，而不是增加复杂度。

------------------------------------------------------------------------

# 九、建议论文最终叙事

建议围绕以下逻辑展开：

1.  **问题：** AR image detectors often rely on isolated token
    likelihood signals, which are model-specific and incomplete.
2.  **观察：** Likelihood-only evidence is strong for some AR families
    but weak for others.
3.  **机制假设：** AR generation involves coupled tokenizer,
    quantization, autoregressive likelihood, and decoder reconstruction
    processes.
4.  **方法：** DART jointly models signed likelihood distributions,
    quantization errors, and reconstruction errors across variable
    scales.
5.  **聚合：** Deep Sets attention learns scale-wise contributions and
    combines model-specific detectors through a model bank.
6.  **证据：** Ablations, multi-seed experiments, model-bank coverage,
    cross-codec tests, external source transfer, and strict
    held-out-family evaluation.
7.  **边界：** The method is model-aware and supervised; its performance
    depends on bank coverage and generator family, and low-FPR
    performance varies across families.

------------------------------------------------------------------------

# 十、最终决策

**DART 是当前最值得保留的 ICML 方法主线，但必须先修复 calibration data
reuse，再验证严格的 generator-family holdout。**

当前最有说服力的结果是：

-   1000-real、9-seed macro AUROC：0.902 ± 0.015；
-   相对同五模型 bank PRADA 的宏观增益：至少约 +0.245；
-   JPEG95 单 seed macro AUROC：0.859；
-   ImageNet-V2 独立真实来源 macro AUROC：0.913，FPR=4.67%。

当前最重要的风险是：

-   conformal calibration 数据复用；
-   generator-family 训练/测试重叠；
-   VAR/HMAR 低 FPR 检出率弱；
-   model-bank 覆盖影响较大；
-   跨 codec 和外部数据证据规模不足。

因此，下一阶段严格按以下顺序推进：

**修复校准协议 → 严格未知家族留出 → Model Bank Coverage → 完整消融 → 多
seed 跨 codec → 外部 benchmark → 理论与论文收敛。**

如果修复协议后 DART
仍然保持主要增益，并且在真正未见生成器家族上有可靠表现，就可以进一步形成有竞争力的
ICML
方法论文。当前结果值得继续投入，但不能提前宣称已经证明通用生成图像检测能力或保证
ICML 录用。
