# DART 最新实验分析报告（commit: 1af3041）

**研究对象：** DART（Distribution-Aware Autoregressive Token Detector）\
**分析依据：** - GitHub
commit：[`1af30412cba21c6bd07b0ee4d980b3df00207758`](https://github.com/MurmanskY/ai_image_detection/commit/1af30412cba21c6bd07b0ee4d980b3df00207758) -
[实验记录
RESULTS.md](https://github.com/MurmanskY/ai_image_detection/blob/main/reports/RESULTS.md) -
[DART
方法文档](https://github.com/MurmanskY/ai_image_detection/blob/main/docs/method_scale_dart.md) -
[严格未知家族实验](https://github.com/MurmanskY/ai_image_detection/blob/main/analysis/dart_family_p0.py) -
[跨 Codec
实验](https://github.com/MurmanskY/ai_image_detection/blob/main/analysis/dart_codec_p0.py)

------------------------------------------------------------------------

## 总体判断

这次更新非常重要。与上一轮相比，你已经完成了之前计划中最关键的四项工作：

1.  修复 calibration 数据复用问题，采用严格四划分协议。
2.  完成真正的未知生成器家族留出实验。
3.  完成完整特征消融和 Model-bank coverage 实验。
4.  完成 JPEG95、85、75、50 多 codec、多 seed
    实验，以及压缩校准和完整压缩域适配实验。

这使 DART 从一个具有较高 AUROC
的初步方法，进一步发展为一个具有**严格校准协议、未知家族泛化证据、生成机制消融和跨
codec 失效分析**的研究方案。

我的总体判断：

**DART 仍然是目前最值得作为 ICML 主线发展的方法。**

但现在论文的重点应该从单纯强调 0.906 的 Macro
AUROC，转向三个更有研究价值的问题：

1.  生成链证据如何实现跨 AR 家族迁移？
2.  Model-bank coverage 如何决定检测器的泛化能力？
3.  Codec shift
    为什么会破坏部分生成器的取证信号，而不仅仅是改变校准阈值？

------------------------------------------------------------------------

# 一、DART 方法是什么？

## 1.1 方法定义

DART（Distribution-Aware Autoregressive Token
Detector）是一种面向自回归生成图像的、多尺度生成链证据联合建模方法。

它的核心思想是：

**通过联合分析 AR 图像生成过程中的 Token Likelihood、Quantization Error
和 Decoder Reconstruction Error，判断输入图像是否具有特定 AR
生成模型的生成机制特征。**

与只使用 token likelihood 的 PRADA 不同，DART 不再将每个 token
独立地压缩为一个检测分数，而是构造每个 scale 的分布表示，再学习不同
scale 之间的关系。

## 1.2 三类核心证据

  ------------------------------------------------------------------------------
  证据                    含义                           作用
  ----------------------- ------------------------------ -----------------------
  Likelihood Distribution 条件与无条件 token likelihood  描述 token 概率分布形状
                          的 signed quantiles

  Quantization Error      每个 scale 的量化误差          描述 tokenizer/codebook
                                                         匹配情况

  Reconstruction Error    Decoder                        描述 decoder 重建一致性
                          重建与输入图像之间的感知误差
  ------------------------------------------------------------------------------

此外，还包括 scale length 和 scale index 等元信息。

每个 scale 的 18 维特征经过共享 MLP Encoder，再通过 Attention Pooling
聚合，最终输出 DART 检测分数。

## 1.3 当前存在两种不同的 DART 实验模式

这一点需要在论文中明确区分。

**模式 A：Model-specific DART**

在 C2I 实验中，每个参考 AR 模型独立训练一个 DART，最后进行 model-bank
aggregation。

**模式 B：Shared DART**

在严格未知家族 T2I 实验中，使用其他生成器家族训练共享
DART，并在测试时移除目标生成器家族及其对应参考模型。

这两种实验回答的问题不同：

-   Model-specific DART：已知参考模型体系下的检测能力。
-   Shared DART：目标生成器家族和对应参考模型均不可用时的迁移能力。

不能把两者的 AUROC 混在一起作为同一种泛化结果。

------------------------------------------------------------------------

# 二、Motivation 应该如何重新定位？

我建议将 DART 的 Motivation 从之前的：

"单一 token likelihood 不够，需要加入 quantization 和 reconstruction。"

升级为三个递进的科学问题。

## 2.1 问题一：AR 生成图像的取证证据具有生成阶段依赖性

AR 图像生成可以抽象为：

\[ z `\xrightarrow{Q}`{=tex} `\hat{z}`{=tex}
`\xrightarrow{\text{AR}}`{=tex} `\hat{z}`{=tex}\_{1:K}
`\xrightarrow{D}`{=tex} `\hat{x}`{=tex} \]

其中：

-   (Q)：Quantization；
-   AR：Autoregressive Token Modeling；
-   (D)：Decoder。

单独使用 likelihood 只能观察概率建模阶段。

但图像是否符合某个 AR 生成机制，还可能体现在 tokenizer 量化和 decoder
重建过程中。

因此可以提出：

**AR forensic evidence should be modeled jointly across the generation
cycle rather than relying on isolated token likelihood.**

这是 DART 最基本的研究动机。

## 2.2 问题二：不同生成器对不同证据的依赖不同

最新的完整消融实验进一步支持了这一点。

尤其是未知家族实验：

  特征                              Macro AUROC
  ------------------------------- -------------
  Likelihood                              0.640
  Quantization-only                       0.577
  Reconstruction-only                     0.746
  Likelihood + Quantization               0.657
  Likelihood + Reconstruction         **0.797**
  Quantization + Reconstruction           0.759
  Full DART                               0.790

这个结果揭示了一个值得重视的现象：

**对于当前未知生成器家族设置，Likelihood 与 Reconstruction
的联合证据比完整三特征组合更有效。**

Quantization 并没有带来稳定的额外收益。

这说明 DART 的真正研究重点可能不是简单地融合三个特征，而是：

-   哪些生成阶段的证据具有互补性？
-   哪些证据具有模型特异性？
-   哪些证据能够迁移到未见生成器家族？

这比最初的三特征融合动机更有深度。

## 2.3 问题三：Model-bank coverage 决定 model-aware detection 的泛化能力

这是本轮新增实验中非常有价值的发现。

在 C2I 实验中：

  Model bank         Macro AUROC 范围
  ---------------- ------------------
  VAR only               0.765--0.774
  VAR + LlamaGen         0.797--0.829
  完整 bank          **0.888--0.916**

加入 RAR 参考模型后，RAR-B 的检测能力显著提升。

这说明：

**Model-aware AR forensic detection
的性能不仅取决于检测网络，也取决于参考模型 bank
对目标生成机制的覆盖程度。**

但加入模型并不一定对所有生成器都有帮助。Max aggregation
可能改变不同模型分数之间的竞争关系，并影响校准。

因此，Model-bank coverage
应当成为论文中的独立研究问题，而不只是一个工程参数。

------------------------------------------------------------------------

# 三、实验具体完成情况

本轮实验可以整理为六个主要模块。

  实验模块                       核心目标                       完成情况
  ------------------------------ ------------------------------ ----------
  严格四划分                     修复 calibration 数据复用      已完成
  未知生成器家族                 验证真正的跨家族迁移           已完成
  特征组合消融                   研究生成链证据的互补性         已完成
  Model-bank coverage            研究参考模型覆盖影响           已完成
  多 Codec                       研究 JPEG 压缩引起的性能退化   已完成
  Codec calibration/adaptation   区分阈值漂移和检测信号损失     已完成

------------------------------------------------------------------------

# 四、实验结果分析

## 4.1 严格四划分协议：最重要的实验修复

之前 DART 的主要问题是：

真实 calibration 数据同时参与了模型分数归一化、模型选择和最终 conformal
threshold 估计。

本轮已经改为：

**真实数据：1000 张**

  子集            数量 用途
  ------------- ------ --------------------------------
  Reference        300 输入特征标准化和训练
  Selection        200 Checkpoint、median/IQR
  Calibration      200 只用于最终 conformal threshold
  Test             300 最终性能评估

Fake 数据：每个生成器 250 张，分别为 100 Train、50 Validation、100
Test。

现在：

-   Reference 用于输入特征标准化和训练；
-   Selection 用于 checkpoint、median/IQR；
-   Calibration 只用于最终 conformal threshold；
-   Test 只用于最终性能评估。

这是非常重要的进展。它使得本轮主要结果比旧版更加可信。

### 严格 9-seed 结果

  指标                                   结果
  ---------------------- --------------------
  DART Macro AUROC          **0.906 ± 0.013**
  Seed 0 paired ΔAUROC                 +0.233
  95% CI                   \[+0.206, +0.260\]
  最大 FPR                              5.33%
  超过 7% FPR 的 seed                       0

9 个 seed 的 FPR：

  Seed       FPR
  ------ -------
  0        5.33%
  1        5.00%
  2        4.33%
  3        5.00%
  4        4.33%
  5        2.33%
  6        5.00%
  7        2.67%
  8        4.67%

按照原有 3%--7% 双侧预注册标准，all-9 GO=false。

不过，这里需要区分两个概念：

-   从预注册实验标准看，确实没有完全通过 GO。
-   从误报风险看，低于 3% 的 FPR
    并不代表检测器不安全，而是校准结果偏保守。

下一轮实验建议把 FPR 的验收标准改为预先规定的单侧风险上界，例如 FPR 95%
置信上界不超过目标值，而不是要求 FPR 必须落在一个双侧区间。

但不能回头修改本轮 GO 结论。

另外，9 个 seed 是同一固定数据池上的不同划分，不能当成 9 个独立数据集。

**评价：严格校准协议已经修复，主要 AUROC
增益仍然保留。这是本轮最重要的正面结果。**

## 4.2 真正未知生成器家族实验：本轮最重要的新证据

使用 `analysis/dart_family_p0.py`，对 T2I 生成器进行了严格留出：

-   Infinity；
-   Janus；
-   LlamaGen-XL；
-   Switti。

目标 fake 不参与训练或验证。

同时设计了两种参考模型条件。

### 实验 A：目标生成器家族和对应参考模型同时移除

  方法      Macro AUROC
  ------- -------------
  DART        **0.790**
  PRADA           0.728

DART 对四个留出家族的平均 AUROC：

  Generator       DART AUROC
  ------------- ------------
  Infinity             0.800
  Janus                0.784
  LlamaGen-XL          0.785
  Switti               0.790

这说明，即使没有目标生成器对应的参考模型，Shared DART
仍然能够利用其他模型的生成链证据实现一定程度的跨家族检测。

这是目前支持 DART 泛化能力最重要的实验。

### 实验 B：目标 fake 留出，但对应参考模型仍然可用

  方法      Macro AUROC
  ------- -------------
  DART            0.855
  PRADA       **0.989**

这里 PRADA 明显更强。

这两个实验放在一起，揭示了一个很有价值的结论：

-   当目标家族及其参考模型同时缺失时，DART 比 PRADA 更有优势。
-   当目标模型参考信息可用时，PRADA 的 model-specific likelihood
    evidence 非常强。

因此，论文不应该宣称 DART 全面超过 PRADA。

更准确的研究定位是：

**DART improves detection under missing reference-family conditions,
while retaining competitive performance when the target family is
represented in the model bank.**

但还需要注意：0.790 的结果只是四个留出家族、三个 seed
的实验，属于初步跨家族证据，不能推广成对所有 AR 生成器的泛化保证。

## 4.3 特征消融：DART 的核心可能是 Likelihood + Reconstruction

  特征                              Macro AUROC
  ------------------------------- -------------
  Likelihood                              0.640
  Quantization-only                       0.577
  Reconstruction-only                     0.746
  Likelihood + Quantization               0.657
  Likelihood + Reconstruction         **0.797**
  Quantization + Reconstruction           0.759
  Full DART                               0.790

最重要的发现：

**Likelihood + Reconstruction 的组合效果最好。**

在新的 seeds 3--5 中：

-   L+R：0.805；
-   Full DART：0.799。

方向一致，但差距很小，目前不能宣称 L+R 显著优于 Full DART。

这意味着：

1.  Reconstruction 是非常重要的取证证据；
2.  Likelihood 与 Reconstruction 具有明显互补性；
3.  Quantization 在当前未知家族协议下没有表现出稳定的额外增益。

因此，我建议将 DART 的理论核心从：

*Three-component feature fusion*

调整为：

**Likelihood--Reconstruction Complementarity across Autoregressive
Generation Scales**

即自回归生成多尺度下的似然与重建证据互补性。

Quantization
可以保留在完整模型中作为辅助证据，但不要再把它作为核心创新的必要组成部分。

### Attention 是否真的必要？

参数量相同的 pooling 对比：

  特征     Attention    Mean     Sum
  ------ ----------- ------- -------
  L+R          0.797   0.796   0.775
  Full         0.790   0.795   0.759

Attention 没有表现出明显的 AUROC 优势。

不过 Mean 在部分 seed 的 FPR 达到 8%。

因此暂时保留 Attention 是合理的，但不能把 Attention Pooling
本身作为主要创新。

下一步把 L+R 作为候选简化版本，和 Full DART 做一次严格的多 seed、低
FPR、跨 codec 对比，再决定最终使用哪个版本。

## 4.4 Model-bank Coverage：重要研究发现

  --------------------------------------------------------------------------------------
  Bank              VAR-d16   LlamaGen-B256          RAR-B       HMAR-D16     Macro 范围
  ---------- -------------- --------------- -------------- -------------- --------------
  VAR only     0.896--0.927    0.792--0.798   0.452--0.496   0.885--0.920   0.765--0.774

  VAR +        0.849--0.900    0.999--1.000   0.476--0.530   0.846--0.887   0.797--0.829
  LlamaGen

  完整 bank    0.811--0.870    0.996--0.999   0.933--0.937   0.810--0.856   0.888--0.916
  --------------------------------------------------------------------------------------

这里有两个重要现象。

**第一，参考模型覆盖确实影响检测能力。**

增加 LlamaGen 参考模型后，LlamaGen 的检测能力明显提升。

增加 RAR 参考模型后，RAR-B 从约 0.45--0.53 提升到约 0.93。

**第二，增加 bank 并不是对所有模型单调有益。**

例如：

-   加入 LlamaGen 后，VAR 和 HMAR 的 AUROC 有所下降；
-   加入 RAR 后，RAR 的性能大幅提升；
-   Max aggregation 使不同模型的分数产生竞争关系。

这说明 DART 需要进一步研究：

\[ `\text{Detection Performance}`{=tex} =
f(`\text{Bank Coverage}`{=tex},`\text{Generator Similarity}`{=tex}) \]

这可能比继续优化 Attention 更有研究价值。

目前只能得出：Model-bank coverage 对 model-aware DART
的性能具有重要影响。还不能声称存在普遍的单调覆盖定律。

## 4.5 多 Codec 实验：DART 存在明确的压缩敏感性

本轮已经完成 JPEG95、JPEG85、JPEG75、JPEG50 的三 seed 固定测试集实验。

  Codec          Macro AUROC   相对原图下降
  ---------- --------------- --------------
  Original     0.893 ± 0.007            ---
  JPEG95       0.849 ± 0.007          0.044
  JPEG85       0.778 ± 0.011          0.115
  JPEG75       0.740 ± 0.005          0.153
  JPEG50       0.697 ± 0.002          0.197

这是一条比较稳定的退化曲线。

进一步看生成器：

  Codec      VAR-d16   LlamaGen   RAR-B    HMAR
  -------- --------- ---------- ------- -------
  JPEG95       0.736      0.995   0.908   0.757
  JPEG85       0.601      0.988   0.892   0.632
  JPEG75       0.544      0.976   0.851   0.590
  JPEG50       0.496      0.941   0.808   0.542

可以看到：

-   LlamaGen 对 JPEG 压缩比较稳定；
-   RAR 也具有一定鲁棒性；
-   VAR 和 HMAR 对 JPEG 压缩非常敏感；
-   JPEG50 下 VAR 已经接近随机检测。

这意味着 DART 的检测机制并不具有天然的 codec invariance。

这不是应该隐藏的负结果。相反，它可以成为论文中非常重要的失效机制分析。

## 4.6 JPEG85 三种适配协议：校准不能恢复已经丢失的检测信号

  协议                                 Macro AUROC     FPR
  ---------------------------------- ------------- -------
  原图训练/校准 → JPEG85 测试                0.767   6.33%
  原图训练/选择，JPEG85 校准和测试           0.767   5.00%
  JPEG85 完整训练/校准/测试                  0.780   7.00%

这里出现一个非常重要的区别：

**重新校准只改变阈值，不会改变 AUROC。**

也就是说：

-   JPEG85 校准可以把 FPR 从 6.33% 调整到 5%；
-   但是 AUROC 仍然是 0.767；
-   只有重新进行压缩域训练，才有可能恢复部分排序信号；
-   即使完整压缩训练，Macro AUROC 也只有 0.780。

VAR/HMAR 有一定恢复，但 RAR 略有下降。

因此可以提出：

> Calibration shift and detection-signal degradation are two distinct
> failure mechanisms.

中文：

校准偏移与检测信号退化是两种不同的失效机制。

这是本轮实验中非常适合写入论文 Discussion 的发现。

------------------------------------------------------------------------

# 五、对比 Baseline：目前 DART 的优势

需要分开讨论三种设置。

## 5.1 已知模型 bank 的 C2I 检测

最新严格协议下：

-   DART 9-seed Macro AUROC：0.906 ± 0.013；
-   Seed 0 paired ΔAUROC：+0.233；
-   95% CI：\[+0.206, +0.260\]。

这说明在当前五模型 bank 和已知训练家族设置下，DART 相比 PRADA
有明显增益。

但不能把这个结果解释为未知生成器 zero-shot。

## 5.2 目标家族完全未知

  方法      Macro AUROC
  ------- -------------
  DART        **0.790**
  PRADA           0.728

这是 DART 目前最有价值的跨家族结果。

但当目标参考模型仍可用时：

  方法      Macro AUROC
  ------- -------------
  DART            0.855
  PRADA       **0.989**

所以 DART
的优势是目标家族参考模型缺失时仍保留一定迁移能力，而不是所有场景都超过
PRADA。

## 5.3 与 D³QE、Data Provenance 的关系

  ----------------------------------------------------------------------------------------------
  Baseline                核心证据                                  DART 应证明的增量
  ----------------------- ----------------------------------------- ----------------------------
  PRADA                   Conditional/unconditional likelihood      Reconstruction evidence
                                                                    的互补性

  D³QE                    Codebook distribution、quantization error Likelihood--reconstruction
                                                                    的额外价值

  Data Provenance         Codebook proximity、decoder inversion     多尺度联合分布建模的增量

  DART                    Likelihood                                生成链证据的联合学习
                          quantiles、quantization、reconstruction
  ----------------------------------------------------------------------------------------------

目前你的消融已经开始说明 DART 的优势主要来自 L+R。

下一步应让 PRADA、D³QE、Data Provenance 和 DART
在尽可能一致的模型、数据划分和参考 bank 下比较。

特别是：

**不要将 DART 的监督训练结果直接与 PRADA
的官方预训练评分结果描述为相同训练信息下的全面优越。**

------------------------------------------------------------------------

# 六、DART 的创新性与 ICML 水准判断

目前的判断比上一轮更积极，但还没有到可以直接投稿的程度。

## 6.1 最值得保留的三个创新点

### 创新点一：Multi-scale generation-cycle consistency

联合分析 AR 生成过程中的：

-   Token likelihood distribution；
-   Decoder reconstruction；
-   Quantization error。

其中当前最有证据的是 Likelihood 与 Reconstruction 的互补性。

这是核心方法贡献。

### 创新点二：Generator-family transfer under missing model-bank coverage

在目标 fake 家族及其参考模型同时移除时，DART Macro AUROC 为 0.790，高于
PRADA 的 0.728。

这说明 DART 有机会解决 model-aware forensic detector
对参考模型覆盖的依赖问题。

这是当前最有价值的泛化贡献。

### 创新点三：Model-bank coverage 与检测性能的关系

不同 bank 的性能明显不同。

特别是加入 RAR 之后，RAR-B 检测显著改善。

这说明参考模型覆盖并不是简单的工程配置，而是影响模型特异取证检测能力的重要因素。

## 6.2 目前还不够的地方

  -------------------------------------------------------------------------------
  问题                    当前情况                   影响
  ----------------------- -------------------------- ----------------------------
  方法新颖性              主要特征分别已有相关工作   需要证明联合建模的新价值

  未知家族                4 个 T2I 家族、3 seeds     规模仍有限

  Model-bank coverage     已有三种 bank 对比         缺少更多家族和定量覆盖理论

  低 FPR                  VAR/HMAR 仍然较弱          实际检测能力不均衡

  Codec robustness        明确出现退化               不能宣称鲁棒

  理论深度                目前以实验分析为主         ICML 方法贡献还需加强
  -------------------------------------------------------------------------------

## 6.3 重新考虑 DART 的特征组合

从未知家族实验看：

\[ AUROC(L+R)=0.797 \]

\[ AUROC(Full)=0.790 \]

新 seeds 仍然是：

\[ 0.805 \> 0.799 \]

因此建议下一阶段正式研究一个简化版本：

**DART-LR：Likelihood--Reconstruction DART**

但现在不应直接宣布它优于 Full DART。

应该先进行：

-   至少 5 个 seeds；
-   相同训练集；
-   相同 model bank；
-   相同 calibration；
-   完整未知家族测试；
-   跨 codec 对比；
-   参数量和训练预算匹配。

如果 L+R
在这些实验中保持相当或更好的性能，并减少计算开销，那么它可能是更合理的最终方法。

------------------------------------------------------------------------

# 七、下一阶段改进建议

## 7.1 P0：完善统计协议与校准标准

严格四划分已经解决了之前的 calibration data reuse 问题。

接下来应该完善：

1.  对 9 个 seed 的 FPR 分别报告 Wilson 95% CI。
2.  报告 calibration threshold 的变化范围。
3.  使用更大的独立真实 calibration set 进行复核。
4.  将未来预注册 GO 改为 FPR 单侧风险上界约束。
5.  明确区分 split variability 与 independent dataset variability。

建议未来使用：

\[ FPR\_{`\mathrm{upper}`{=tex},95%}`\leq `{=tex}`\alpha`{=tex} \]

而不是要求 FPR 必须同时高于某个下限。

本轮旧 GO 仍然保持 false，不修改历史结论。

## 7.2 P1：扩大真正未知家族实验

现在只有四个严格留出的 T2I 家族：

-   Infinity；
-   Janus；
-   LlamaGen-XL；
-   Switti。

建议再增加至少 3--4 个不同家族，重点考虑：

-   Open-MAGVIT2；
-   其他独立 AR tokenizer/decoder 家族；
-   不同图像分辨率和生成策略的 AR 模型。

关键不是单纯增加 fake 数量，而是增加真正独立的生成机制。

建议建立以下实验矩阵：

  训练参考 bank         测试生成器   目的
  --------------------- ------------ -------------------------
  VAR + LlamaGen        Infinity     未知家族
  VAR + RAR             Janus        未知家族
  LlamaGen + RAR        Switti       未知家族
  完整 bank             新 AR 家族   完整覆盖条件
  不含目标家族的 bank   新 AR 家族   Missing-family transfer

每个设置至少三个 seeds，重要结果建议五个 seeds。

同时报告 DART 和 PRADA 的逐家族 AUROC、TPR@FPR 和置信区间。

## 7.3 P2：确定最终模型使用 Full 还是 L+R

  模型          已有未知家族 AUROC
  ----------- --------------------
  Full DART                  0.790
  DART-LR                    0.797

新 seeds：Full 0.799，L+R 0.805。

建议比较：

  维度                      需要验证
  ------------------------- --------------
  AUROC                     是否稳定保持
  TPR@1%FPR                 是否改善
  FPR                       是否更稳定
  Cross-family              是否保持泛化
  JPEG robustness           是否减少退化
  Inference time            是否降低
  Feature extraction cost   是否降低

如果 L+R 在多个独立实验中表现相当或更好，可以将它作为主模型，将 Full
DART 作为完整消融版本。

但不要只因为 0.007 的差距就提前更换模型。

## 7.4 P3：系统研究 Model-bank Coverage

目前已有三个 bank 设置，但还没有真正回答：

**需要多少个参考模型，才能获得稳定的未知生成器检测能力？**

建议定义一个参考模型覆盖指标：

\[ C(g,B)= `\max`{=tex}\_{m`\in `{=tex}B}
`\operatorname{Sim}`{=tex}(g,m) \]

其中：

-   (g)：目标生成器；
-   (B)：参考模型 bank；
-   (`\operatorname{Sim}`{=tex})：生成机制相似性。

Similarity 不能直接用检测结果定义，否则会产生循环论证。建议独立考虑
tokenizer、codebook、decoder architecture、scale structure 等模型属性。

然后研究：

\[ AUROC(g)=f(C(g,B),\|B\|) \]

重点分析：

1.  Bank size 对 AUROC 的影响；
2.  同家族模型是否比异家族模型更有价值；
3.  Bank coverage 与 FPR 的关系；
4.  Max aggregation 为什么可能损伤部分家族；
5.  是否存在更好的 coverage-aware aggregation。

如果这部分能够得到稳定规律，可能形成 DART
除生成链一致性之外的第二项重要贡献。

## 7.5 理论创新

### 理论方向一：联合证据的互补性

设：

\[ L=`\text{Likelihood evidence}`{=tex} \]

\[ R=`\text{Reconstruction evidence}`{=tex} \]

\[ Q=`\text{Quantization evidence}`{=tex} \]

标签为 (Y`\in`{=tex}{0,1})。

研究联合证据：

\[ P(L,R,Q`\mid `{=tex}Y) \]

相对于单独证据：

\[
P(L`\mid `{=tex}Y),`\quad `{=tex}P(R`\mid `{=tex}Y),`\quad `{=tex}P(Q`\mid `{=tex}Y)
\]

是否提供额外的判别信息。

可以从条件互信息出发：

\[ I(Y;R`\mid `{=tex}L) \]

以及：

\[ I(Y;L`\mid `{=tex}R) \]

研究 Reconstruction 在已知 Likelihood 后是否仍然提供额外判别信息。

当前消融支持 L+R 的互补性，但 AUROC
的提高本身不能证明条件互信息严格大于零。需要独立测试集上的统计估计、置信区间和相应假设。

### 理论方向二：Model-bank Coverage 与泛化风险

研究：

\[ R(g,B) \]

即目标生成器 (g) 在参考模型集合 (B) 下的检测风险。

尝试分析：

-   参考模型覆盖不足时的检测风险；
-   新增参考模型对最差生成器的影响；
-   Max aggregation 的误报概率；
-   不同 detector score 相关性对 bank aggregation 的影响。

尤其要研究为什么增加 bank 可能提高某个生成器的 AUROC，却降低其他生成器的
AUROC。

这部分可以形成明确的统计学习问题。

------------------------------------------------------------------------

# 八、ICML 论文定位建议

推荐题目：

**DART: Modeling Multi-Scale Generation-Cycle Consistency for
Autoregressive Image Forensics**

中文：

**DART：面向自回归图像取证的多尺度生成循环一致性建模**

建议论文结构：

  章节                            内容
  ------------------------------- -------------------------------------
  1\. Introduction                单一 token likelihood 的局限
  2\. Related Work                PRADA、D³QE、Data Provenance 等
  3\. Problem Formulation         AR 生成链取证问题
  4\. Method                      DART 多尺度联合建模
  5\. Mechanism Analysis          Likelihood 与 Reconstruction 互补性
  6\. Model-bank Generalization   未知家族和参考覆盖
  7\. Experiments                 多 seed、codec、外部验证
  8\. Limitations                 低 FPR、压缩敏感性和参考模型依赖

论文中一定要明确 DART 是 supervised、model-aware
的方法，不能将其包装为完全无监督或无需目标模型知识的 universal
detector。

------------------------------------------------------------------------

# 九、下一阶段科研路线

  优先级   工作                                       状态
  -------- ------------------------------------------ ----------
  P0       严格校准协议、独立 conformal calibration   已完成
  P1       Full DART 与 L+R 的严格多 seed 比较        下一步
  P2       扩大未知生成器家族至 7--8 个               下一步
  P3       Model-bank coverage 定量分析               下一步
  P4       联合证据互补性理论分析                     下一步
  P5       低 FPR 与跨 codec 失效机制分析             持续完善
  STOP     新增无依据特征、重新启动 S3R/SACR/CoNTra   停止

------------------------------------------------------------------------

# 十、最终评价

以下是研究成熟度的主观评价，不是 ICML 录用概率。

  评价维度             当前评价
  ------------------ ----------
  Motivation             8.5/10
  方法设计               8.5/10
  已知家族检测性能         9/10
  未知家族泛化           7.5/10
  消融实验                 9/10
  Model-bank 分析        8.5/10
  Codec 鲁棒性             6/10
  低 FPR 检测能力          6/10
  理论深度                 6/10
  ICML 准备程度        仍需完善

DART 已经有比较明确的 ICML 方法论文主线。

本轮最大的进步不是 Macro AUROC 从之前的数值变成 0.906，而是你补齐了：

-   严格校准；
-   真正未知家族；
-   生成链特征消融；
-   Model-bank coverage；
-   多 Codec 失效分析；
-   压缩校准与检测信号退化的区分。

这些使研究从单纯追求性能，逐渐转向机制、泛化和适用边界。

**我最希望你下一步完成的不是新网络，而是两个实验：**

1.  Full DART 与 L+R 的严格多 seed、未知家族配对比较。
2.  将未知家族从 4 个扩展到至少 7--8 个，并研究 model-bank coverage
    与泛化性能的关系。

如果这两个方向都得到稳定结果，DART 的论文说服力会明显增强。

83 tests passed
和大量逐图独立复算，证明了当前实现和结果归档具有较好的内部一致性，但不等于已经证明外部泛化。论文仍需要更大规模的独立生成器、真实来源和低
FPR 验证。

**现阶段建议保留 DART 作为唯一主线，不再启动 CARF-A、S3R、SACR 或 CoNTra
的新算法扩展。**
