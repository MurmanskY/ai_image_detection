# DART 七家族跨数据集验证与 Model-bank Coverage 分析报告

**方法：** DART（Distribution-Aware Autoregressive Token Detector）\
**GitHub commit：**
[`51d6d335d85f9db21dc1d3d1bd31972a95974053`](https://github.com/MurmanskY/ai_image_detection/commit/51d6d335d85f9db21dc1d3d1bd31972a95974053)\
**仓库：** <https://github.com/MurmanskY/ai_image_detection>\
**结果：** `reports/RESULTS.md` 第 34--35 节\
**主要实现：**
`analysis/dart_full_lr_comparison.py`、`analysis/dart_external_family_p0.py`、`analysis/dart_external_family_significance.py`、`analysis/dart_external_family_audit.py`

------------------------------------------------------------------------

## 执行摘要

本轮完成两组实验：

1.  Full DART 与 DART-LR 的严格 9-seed 配对比较；
2.  七个生成器家族的跨数据集评估与 model-bank coverage 枚举。

核心结论：

-   四家族 9-seed 中，DART-LR 的 macro AUROC 均值比 Full 高
    **0.00579**，但统计证据不一致：paired t-test (p=0.0425)，Wilcoxon
    (p=0.0547)，仅 6/9 seeds 为正，逐图 bootstrap 仅 3/9 seeds 的 CI
    完全大于零。因此不能据此替换 Full。
-   七家族跨数据集实验中，**Full DART 为 0.8456±0.0093**，高于 DART-LR
    的 0.8368±0.0063 和同 bank PRADA 的 0.6101。Full 相对 PRADA
    的平均逐图 paired ΔAUROC 为 +0.2354，五个 seed 的 bootstrap CI
    均为正。
-   七家族结果支持保留 Full：Full 相对 L+R 的平均 AUROC 优势为
    0.0088，5/5 seeds 都是 Full 更高，逐图 bootstrap CI 每个 seed
    均低于零。
-   Model-bank coverage 通常有帮助，但不是单调规律。不同目标家族对 bank
    组成的反应不同，max aggregation 也会造成模型间竞争和校准变化。
-   低 FPR 工作点仍是明显短板：Full 的宏平均 TPR@1%FPR 为
    0.145、TPR@5%FPR 为 0.339；部分家族 FPR 最高达到 8.67%。
-   本轮是示例子集：ARForensics real=1000、每个目标家族
    fake=250，并非完整数据集或大规模 benchmark。

**总体判断：DART
的方法论文证据链比上一轮更完整。当前最有价值的研究叙事是"多尺度生成循环证据 +
缺失参考家族下的迁移 + model-bank
组成对泛化的影响"。但尚不能宣称普遍未知生成器泛化，也没有解决低 FPR
和训练数据来源混杂等问题。**

------------------------------------------------------------------------

# 一、方法是什么？

## 1.1 DART 定义

DART 是一种面向自回归（AR）生成图像的监督式取证检测方法。它联合利用 AR
生成机制中的：

1.  **Token likelihood distribution：** 条件与无条件 token
    log-likelihood 差值的 signed quantiles；
2.  **Quantization error：** 每个生成 scale 的量化误差；
3.  **Decoder reconstruction error：** 输入图像与对应 AR decoder
    重建图像之间的感知误差；
4.  **Scale metadata：** scale length、scale index 等结构信息。

每个 scale 的特征经过共享 encoder，再通过 attention pooling 聚合不同
scale 的证据，最终由分类头产生检测分数。

DART 的核心不是单独使用某一种已有特征，而是**学习这些生成链证据在多个
scale 上的联合分布和互补关系**。

## 1.2 Full DART 与 DART-LR

-   **Full DART：** Likelihood + Quantization + Reconstruction；
-   **DART-LR：** Likelihood + Reconstruction，不使用 Quantization
    通道。

两者使用相同数据划分、训练预算、40 epochs、attention pooling 和独立
conformal calibration。参数量分别为 1234 和 1202，差异很小。

本轮最终建议：**保留 Full DART 作为主模型，DART-LR 作为简化消融版本。**

## 1.3 七家族实验的训练与测试设定

本轮七家族实验使用 DARG T2I fake 作为训练/验证阶段的生成图像来源，真实
reference/selection/calibration/test 来自 ARForensics corrected
test，目标 fake 来自 ARForensics 的七个家族：

-   Infinity
-   Janus-Pro
-   LlamaGen
-   Open-MAGVIT2
-   RAR
-   Switti
-   VAR

对于 DARG 训练集和四个 T2I
参考模型中已存在的重叠家族，实验同时排除目标家族训练 fake
和对应参考模型。Open-MAGVIT2、RAR、VAR 在本轮 DARG T2I 训练 fake 和四个
T2I 参考模型中均未覆盖。

这包括"重叠家族严格留出"和"新增家族测试"两类情况。论文中应分别报告，不能将两者混称为完全相同的
unseen-family 设置。

------------------------------------------------------------------------

# 二、Motivation

## 2.1 单一 token likelihood 不能完整表达 AR 生成机制

PRADA 表明 conditional/unconditional token likelihood 是有效的 AR
取证信号，但不同 AR 家族在 tokenizer、codebook、scale structure、token
ordering 和 decoder 上存在差异。单一 likelihood
统计量可能无法完整描述这些差异。

DART 构造联合证据：

\[ `\Phi`{=tex}(x)=`\left[L(x),Q(x),R(x)\right]`{=tex}\]

其中 (L) 为 token likelihood 分布证据，(Q) 为 quantization error，(R) 为
decoder reconstruction error。

核心问题是：这些不同生成阶段的证据能否互补，从而改善目标参考家族缺失时的检测能力？

## 2.2 Model-bank coverage 可能决定 model-aware detection 的泛化

实际部署时，目标生成器可能没有对应参考模型。本轮发现：

-   增加参考模型通常能改善部分家族；
-   完整 bank 不一定优于每个目标对应的最佳三模型子集；
-   bank 具体组成与 max aggregation 同样重要。

因此核心问题不只是 bank 越大是否越好，而是：

**Model-bank composition and aggregation jointly determine detection
performance under missing reference families.**

## 2.3 AUROC 提升不等于低误报工作点已解决

Full DART 七家族 macro AUROC 为 0.8456，但 TPR@1%FPR macro 只有
0.145，TPR@5%FPR macro 为 0.339，部分家族 FPR 最高达到 8.67%。

因此必须同时研究排序性能、低误报工作点和校准稳定性。

------------------------------------------------------------------------

# 三、本轮完成的实验

  -------------------------------------------------------------------------------------------
  实验模块                目标                           完成情况
  ----------------------- ------------------------------ ------------------------------------
  Full vs. L+R            检查 quantization              四家族、9 seeds，严格配对
                          通道是否提供稳定增益

  七家族跨数据集          检验缺失参考家族条件下的迁移   ARForensics 七家族示例子集、5 seeds

  Model-bank coverage     分析 bank size 和组成          seeds 0--2
                                                         枚举每个目标可用模型的所有非空子集

  显著性分析              paired ΔAUROC 与 bootstrap CI  已完成

  独立审计                逐图复算 AUROC、阈值、FPR、TPR 已完成
  -------------------------------------------------------------------------------------------

七家族协议：

  -----------------------------------------------------------------------
  项目                                设置
  ----------------------------------- -----------------------------------
  真实图像                            1000

  每个目标家族 fake                   250

  真实划分                            300 reference / 200 selection / 200
                                      calibration / 300 test

  每家族 fake test                    100

  Seeds                               5

  Baselines                           同可用 model bank 的 PRADA

  聚合                                Max aggregation
  -----------------------------------------------------------------------

这不是 84k 全量评估。

------------------------------------------------------------------------

# 四、实验结果

## 4.1 Full DART 与 DART-LR：四家族、9 seeds

  -------------------------------------------------------------------------------
  模型                         Macro      TPR@1%FPR      TPR@5%FPR       FPR 范围
                AUROC（均值±标准差）
  ----------- ---------------------- -------------- -------------- --------------
  Full DART          0.7854 ± 0.0243         0.0858         0.2242   2.33%--7.33%

  DART-LR        **0.7912 ± 0.0245**     **0.0878**     **0.2378**   2.67%--8.33%
  -------------------------------------------------------------------------------

L+R−Full：

-   均差：+0.00579；
-   seed 级 95% t 区间：\[+0.00025, +0.01133\]；
-   paired t-test：(p=0.0425)；
-   Wilcoxon：(p=0.0547)；
-   仅 6/9 seeds 方向为正；
-   逐图 paired bootstrap 仅 3/9 seeds 的 CI 完全大于零。

结论：L+R
在四家族协议上的均值略高，但统计证据依赖检验口径，不能称为稳定显著优于
Full。L+R 的 seed 8 FPR 达到 8.33%。

## 4.2 七家族跨数据集：最大严格可用 bank

  目标家族                  Full AUROC         DART-LR AUROC   同 bank PRADA
  -------------- --------------------- --------------------- ---------------
  Infinity             0.9430 ± 0.0051   **0.9467 ± 0.0029**          0.8911
  Janus-Pro            0.9211 ± 0.0129   **0.9273 ± 0.0134**          0.8346
  LlamaGen             0.7712 ± 0.0043       0.7738 ± 0.0072          0.6837
  Open-MAGVIT2     **0.8285 ± 0.0147**       0.8079 ± 0.0103          0.3399
  RAR              **0.8104 ± 0.0241**       0.7814 ± 0.0124          0.2688
  Switti           **0.8390 ± 0.0077**       0.8309 ± 0.0072          0.7973
  VAR              **0.8057 ± 0.0149**       0.7893 ± 0.0104          0.4554
  **Macro**        **0.8456 ± 0.0093**       0.8368 ± 0.0063          0.6101

Full 相对 PRADA 的平均逐图 paired Δmacro AUROC：

\[ `\Delta `{=tex}AUROC=+0.2354 \]

五个 seed 的 bootstrap 95% CI 均大于零。

解释边界：PRADA 在目标参考模型缺失时明显变弱，尤其 RAR 和
Open-MAGVIT2。因此这证明的是**当前 missing-reference-family 协议下 DART
的增益**，不是 DART 在所有条件下普遍优于完整参考 bank 的 PRADA。

## 4.3 七家族中 Full 与 L+R 的直接比较

七家族 L+R−Full：

\[ `\Delta `{=tex}AUROC=-0.00880 \]

-   5/5 seeds 都是 Full 更高；
-   seed 级 95% t CI：\[-0.01303, -0.00458\]；
-   每个 seed 的逐图 bootstrap CI 均低于零；
-   Wilcoxon 因仅 5 个 seed，(p=0.0625)。

结合四家族 9-seed 结果，建议保留 Full DART。

但 L+R 的宏平均 TPR@1%/5%FPR 为 0.150/0.343，略高于 Full 的
0.145/0.339，所以不能声称 Full 在所有低 FPR 工作点都更好。

## 4.4 低 FPR 工作点

Full DART 七家族宏平均：

-   TPR@1%FPR：0.145；
-   TPR@5%FPR：0.339。

  家族             TPR@1%FPR   TPR@5%FPR
  -------------- ----------- -----------
  Infinity             0.380       0.636
  Janus-Pro            0.172       0.508
  LlamaGen             0.076       0.264
  Open-MAGVIT2         0.052       0.236
  RAR                  0.068       0.182
  Switti               0.222       0.380
  VAR                  0.046       0.166

Full/L+R 逐配置 FPR 范围分别为 0.67%--8.67% 和 0.67%--7.00%。低 FPR
检出能力和 FPR 稳定性仍需改善。

## 4.5 Model-bank coverage

Full DART 描述性汇总：

  Bank size     配置数   平均 AUROC
  ----------- -------- ------------
  1                 72        0.778
  2                 90        0.824
  3                 48        0.832
  4                  9        0.814

观察：

1.  Bank 从 1 增加到 2--3 个模型通常有收益；
2.  最大 bank 并非始终最好；
3.  Open-MAGVIT2、RAR、VAR 从 1 到 3 个参考模型明显改善，但完整 4-model
    bank 相比各自 3-model 均值分别下降约 0.004、0.001、0.008；
4.  "是否包含相同生成范式"不能单独解释泛化；
5.  Max aggregation、bank 具体组成和 calibration 共同影响结果。

配对新增模型分析：

-   Full：加入同范式模型平均 ΔAUROC=+0.028；加入不同范式模型为 +0.038；
-   L+R：两者均约 +0.026。

因此当前证据不支持"同生成范式参考模型一定更有价值"，也不支持 bank size
与 AUROC 存在普遍单调关系。

## 4.6 工程验证

-   699 个 AUROC；
-   699 个阈值；
-   699 个 FPR；
-   2796 个 TPR 工作点；
-   独立复算最大绝对误差为 0；
-   88 tests passed，1 warning。

这些证明归档与逐图计算一致，不替代外部泛化证据。

------------------------------------------------------------------------

# 五、与 Baseline 的比较

## 5.1 同 bank PRADA

  方法              七家族 Macro AUROC
  --------------- --------------------
  Full DART                 **0.8456**
  DART-LR                       0.8368
  同 bank PRADA                 0.6101

Full−PRADA 平均 paired ΔAUROC=+0.2354，五个 seed 的 bootstrap CI
均为正。

应准确表述为：DART 在当前缺失目标参考家族的协议下优于同可用 bank 的
PRADA，而不是全面优于 PRADA。

## 5.2 与 D³QE、Data Provenance 的关系

  ----------------------------------------------------------------------------------------------------------
  方法                    已有核心                                  DART 应证明的增量
  ----------------------- ----------------------------------------- ----------------------------------------
  PRADA                   Conditional/unconditional token           多尺度联合建模和 decoder reconstruction
                          likelihood                                的增量

  D³QE                    Codebook 分布和 quantization error        Likelihood--reconstruction
                                                                    联合证据的额外价值

  Data Provenance         Codebook proximity、decoder inversion     多尺度生成链联合表示和缺失参考家族迁移

  DART                    Likelihood                                联合学习与跨家族、bank coverage 评估
                          quantiles、quantization、reconstruction
  ----------------------------------------------------------------------------------------------------------

不能把 quantization 或 decoder reconstruction
单独宣称为全新特征。真正需要证明的是联合表示、跨尺度聚合和缺失参考家族下的增量价值。

## 5.3 Baseline 公平性

当前七家族实验中，DART 使用 DARG fake 训练、ARForensics fake 测试；PRADA
使用对应可用参考模型的评分。两者训练信息不同。

论文应明确报告：

-   PRADA 是官方评分函数还是重新拟合版本；
-   DART 是否使用 fake labels；
-   每个模型可用 bank 的组成；
-   训练 fake 家族和测试 fake 家族；
-   calibration real 的来源；
-   图像预处理与测试图是否一致。

建议增加监督信息匹配的 baseline：在相同 DARG fake train/validation、相同
real split、相同 bank 输入下，训练简单 MLP/Deep Sets likelihood-only
detector，以证明增益来自生成链证据，而不只是监督训练信息。

------------------------------------------------------------------------

# 六、创新点与 ICML 水准判断

## 6.1 最有价值的三个贡献方向

### 创新点一：Multi-scale Generation-cycle Consistency

联合建模 token likelihood distribution、quantization error 和 decoder
reconstruction error，并通过 scale-wise representation 与 attention
aggregation 学习多尺度生成链证据。

这是 DART 的主要方法贡献。

### 创新点二：Missing-reference-family Transfer

在目标参考模型被移除的协议下，七家族 Full DART macro AUROC 为 0.8456，同
bank PRADA 为 0.6101。

这支持 DART 在当前 missing-reference-family
设置下具有较强迁移能力。但七家族中既有从 DARG 训练 fake
中严格排除的重叠家族，也有原始训练集和参考 bank
均未覆盖的新家族，两类结果应分别报告。

### 创新点三：Model-bank Composition Matters

Bank size 增加通常有帮助，但完整 bank
不一定最好；同范式覆盖也没有稳定优势。这是值得进一步研究的模型泛化现象，目前更适合作为系统性实验发现，而非已完成的理论贡献。

## 6.2 当前创新性不足之处

  --------------------------------------------------------------------------------------------
  问题                    当前证据                            影响
  ----------------------- ----------------------------------- --------------------------------
  特征新颖性              likelihood、quantization、decoder   必须证明联合建模的新价值
                          evidence 均有相关先前工作

  跨家族泛化              七家族示例子集，五 seeds            仍需更多独立家族和数据集

  Bank coverage           大量子集枚举                        目前是描述性统计，缺少可预测的
                                                              coverage 理论

  低 FPR                  Macro TPR@1% 仅 0.145               实用检测能力不均衡

  FPR 稳定性              最高 8.67%                          部分配置仍有校准风险

  数据来源                DARG fake 训练、ARForensics fake    需要排除 dataset/source shortcut
                          测试

  理论深度                以机制假设和实验分析为主            ICML 方法贡献仍需强化
  --------------------------------------------------------------------------------------------

## 6.3 ICML 判断

**DART 已经形成有竞争力的 ICML 方法论文候选，但尚未达到成熟投稿状态。**

正面因素：明确的生成机制动机、七家族结果、严格 Full/L+R
配对、缺失参考家族设置、bank 子集枚举和逐图独立审计。

主要风险：训练数据来源混杂、低 FPR 工作点弱、示例规模有限、bank
规律非单调且缺乏理论解释、baseline 监督信息未完全匹配。

------------------------------------------------------------------------

# 七、最重要的改进意见

## P0：排除训练数据来源与标签混杂

这是本轮最需要优先检查的问题。

当前七家族实验使用 DARG fake 训练，而真实图像来自
ARForensics。如果训练阶段真实图像只来自 ARForensics、fake 只来自
DARG，那么训练标签与数据集来源存在相关性。模型可能利用数据集差异，而不完全是
AR 生成机制差异。

这不代表当前结果一定由 shortcut 导致，但它是必须排除的替代解释。

建议新增对照：

  ----------------------------------------------------------------------------
  实验              Real train        Fake train        目的
  ----------------- ----------------- ----------------- ----------------------
  Current           ARF               DARG              当前协议

  Same-dataset      ARF               ARF 已知训练家族  控制跨数据集来源差异

  Cross-dataset     ARF + DARG real   ARF + DARG fake   平衡数据来源
  balanced

  Source-held-out   多来源 real/fake  不同来源测试      检验来源迁移
  ----------------------------------------------------------------------------

关键是让 real/fake
标签不再与数据集来源一一对应。如果无法获得足够的同源真实/生成样本，应明确限制结论。

## P1：扩大七家族实验

当前每家族 fake=250、测试 fake=100。建议：

1.  扩大真实来源数量；
2.  每个生成器增加独立测试图；
3.  至少增加一个独立 benchmark；
4.  增加不同 AR tokenizer/decoder 家族；
5.  按生成家族而不是模型版本进行分组统计。

避免把同一生成器家族的多个版本当作完全独立泛化证据。

## P2：研究低 FPR 失效

当前 Full DART：

-   Macro TPR@1%FPR=0.145；
-   Macro TPR@5%FPR=0.339。

重点分析 LlamaGen、Open-MAGVIT2、RAR、VAR 的 false negatives。

建议：

-   输出每个生成器的 real/fake score distribution；
-   研究不同 model detector 的 score correlation；
-   分析 max aggregation 是否提高 FPR；
-   比较 max、mean、median 和固定权重聚合；
-   使用独立 real calibration 评估 FPR 置信上界；
-   报告 TPR@0.1%、1%、5%、10%FPR。

不要仅通过调整阈值解决排序能力不足的问题。

## P3：改进 Model-bank Aggregation

比较：

1.  Max aggregation；
2.  Mean aggregation；
3.  Median aggregation；
4.  固定权重 aggregation；
5.  Selection-set 选择的 aggregation；
6.  Coverage-aware aggregation。

所有 aggregation 的选择必须在独立 selection set 完成，conformal
calibration 只负责最终阈值。

研究 bank size、bank composition、score correlation 与 FPR 的联合关系。

## P4：继续比较 Full 与 L+R

当前证据：

-   四家族 9-seed：L+R 均值高 0.00579，但统计证据不一致；
-   七家族 5-seed：Full 高 0.00880，5/5 seeds 支持 Full；
-   七家族 L+R 的宏平均 TPR@1%/5%FPR 略高。

建议保留 Full 为主模型，将 L+R
作为重要消融。只有更大规模、同源数据来源控制和严格配对实验支持 L+R
时，才考虑简化主模型。

## P5：建立可解释的 Model-bank Coverage 指标

可以尝试定义目标生成器 (g) 与参考 bank (B) 的覆盖程度：

\[ C(g,B)=`\max`{=tex}\_{m`\in `{=tex}B}`\operatorname{Sim}`{=tex}(g,m)
\]

Similarity 应由独立模型属性定义，例如 tokenizer/codebook 相似性、decoder
architecture、scale structure、token ordering 或生成机制。

不能直接用检测 AUROC 定义相似性，否则会循环论证。

进一步研究：

\[ AUROC(g,B)=f(C(g,B),\|B\|,`\text{aggregation}`{=tex}) \]

重点解释为什么某些 bank 扩展有帮助、另一些扩展会损害部分目标家族。

------------------------------------------------------------------------

# 八、推荐论文定位

## 8.1 推荐题目

**DART: Multi-Scale Generation-Cycle Consistency for Autoregressive
Image Forensics under Missing Reference Families**

中文：

**DART：面向参考家族缺失场景的自回归图像多尺度生成循环一致性取证**

## 8.2 建议论文贡献

1.  **方法贡献：** 提出多尺度 generation-cycle evidence
    representation，联合建模 token likelihood、quantization 和 decoder
    reconstruction。
2.  **泛化贡献：** 在目标参考家族缺失时，评估 DART 对七个 AR
    家族的迁移能力，并与同可用 bank 的 PRADA 对比。
3.  **系统分析贡献：** 通过 bank 子集枚举，分析 bank size、具体组成与
    max aggregation 对不同目标家族的影响。
4.  **鲁棒性分析：** 讨论低 FPR、codec shift
    和数据来源变化下的性能边界。

第 3、4 项需要避免夸大为已经建立的普遍理论规律。

## 8.3 建议论文结构

1.  Introduction：AR forensic detection 的参考模型缺失问题。
2.  Related Work：PRADA、D³QE、Data Provenance 等。
3.  Problem Formulation：model-aware 与 missing-reference-family
    任务定义。
4.  Method：DART 多尺度生成链联合建模。
5.  Experiments：七家族、Full/LR、PRADA 对比。
6.  Model-bank Coverage：bank 子集和组成分析。
7.  Robustness and Failure Analysis：低 FPR、codec 和数据来源控制。
8.  Limitations：模型 bank 依赖、示例规模和适用边界。

------------------------------------------------------------------------

# 九、下一阶段执行计划

  优先级   工作                                        目标
  -------- ------------------------------------------- --------------------
  P0       检查并控制 DARG/ARForensics 来源混杂        排除 shortcut
  P0       扩大七家族每类测试样本                      提高估计稳定性
  P1       增加独立 AR benchmark                       验证外部泛化
  P1       低 FPR false-negative 分析                  解释实用性能瓶颈
  P1       Max/mean/median/weighted aggregation 对比   改善 bank 聚合
  P2       Model-bank similarity/coverage 指标         建立可解释机制
  P2       完整 PRADA/D³QE/Data Provenance 公平对照    加强基线证据
  P3       理论分析与论文写作                          形成 ICML 完整叙事

### 应停止的工作

-   不再增加无明确假设的新特征；
-   不为追求单一 macro AUROC 反复调参；
-   不把 bank size 增加解释为普遍单调收益；
-   不把七家族示例子集称为完整外部 benchmark；
-   不把缺失参考家族实验称为所有生成器上的 zero-shot；
-   不只报告 AUROC 而忽略低 FPR 工作点；
-   不将 5 个 seed 当作 5 个独立数据集。

------------------------------------------------------------------------

# 十、最终结论

本轮实验最重要的进展：

1.  **Full DART 在七家族跨数据集示例实验中达到 0.8456±0.0093 macro
    AUROC**，高于同 bank PRADA 的 0.6101；
2.  Full 在七家族实验中比 L+R 高 0.0088，5/5 seeds 支持 Full，因此保留
    Full 为主模型；
3.  Model-bank 从 1 增至 2--3 个参考模型通常有收益，但完整 bank
    不一定最好；
4.  参考模型具体组成和 max aggregation 对不同目标家族产生不同影响；
5.  低 FPR 工作点仍弱，且部分配置 FPR 最高达到 8.67%；
6.  DARG fake 训练与 ARForensics fake 测试的来源差异需要进一步控制，避免
    dataset-origin shortcut。

**最终建议：继续以 Full DART
为唯一方法主线。下一阶段最重要的不是增加网络模块，而是控制训练数据来源混杂、扩大独立生成器家族测试、解决低
FPR 失效，并建立可解释的 model-bank coverage 分析。**

如果这些实验仍能支持当前结果，DART 将具备更完整的 ICML
方法论文证据链；但当前结果仍属于示例规模验证，不能据此保证通用检测能力或
ICML 录用。
