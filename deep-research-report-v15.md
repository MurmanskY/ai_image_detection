# 生成图像检测研究路线分析报告（v15）

**主题：S3R P0 失败后的研究收敛、参考池统计与表示依赖的失效机制分析**

-   项目仓库：<https://github.com/MurmanskY/ai_image_detection>
-   最新核查提交：`875e49391af7cfd06e4b1bad0b48ef7df8be7cc9`
-   主要结果归档：`reports/RESULTS.md` 第 22 节
-   核心实现：`analysis/s3r_p0.py`
-   测试：`tests/test_s3r.py`
-   核查日期：2026-11-25

------------------------------------------------------------------------

## 执行摘要

S3R（Source-aware Same-quality Reference Modeling）小规模预注册 P0
已完成，最终 **GO=false**。按照预注册规则，应停止 S3R，不开展 84k
全量实验，也不增加新模块。

最关键的结果是：12 个 dataset × seed × quality 实验组全部选择
(`\gamma=0`{=tex})。因此，所谓 hierarchical S3R 在最终实验中严格退化为
same-quality reference，没有形成新的估计器，也没有带来检测增益。

主要发现：

1.  **Same-quality 优于简单 mixed-codec pooling
    的估计偏差表现**，但这只证明在当前已知 codec/quality
    条件下的参考池统计优势，不证明未知 codec 泛化。
2.  Source-weighted mixed 与 naive mixed 的检测 AUROC 几乎一致；修正
    source-level ESS 和协方差分解，没有自动改善检测性能。
3.  S3R proxy-MSE 显著低于 mixed，但因 (`\gamma=0`{=tex})，其
    bias²、variance 和 MSE 与 same-quality 完全相同。这是继承
    same-quality 的统计性质，不是 S3R 的方法增益。
4.  校准与覆盖仍有失败组：ARF 有两组 FPR 超过 7%，DARG 有三组低于
    3%；未知 codec 100% abstain，不能称为 zero-shot。
5.  工程复核较完整：53 passed、1 warning；288 组逐图 AUROC/coverage
    独立重算一致；MSE=bias²+variance 数值校验通过；12 个 checkpoint
    均可恢复。

**最终决策：目前没有证据支持一个优于 same-quality CARF-A
的新检测算法。应停止 S3R、SACR 和 CoNTra
的算法扩展，把论文定位转向参考池统计、来源相关性、条件校准和表示依赖的失效机制研究。**

------------------------------------------------------------------------

# 一、应该选择什么新方法？设计思路是什么？

## 1.1 结论：目前不应宣称已经找到新的检测方法

本轮 S3R P0 的核心结果是：

  数据集/质量      same   mixed   source-mixed     S3R
  ------------- ------- ------- -------------- -------
  ARF / QF85      0.528   0.511          0.511   0.528
  ARF / QF75      0.551   0.587          0.587   0.551
  DARG / QF85     0.556   0.517          0.517   0.556
  DARG / QF75     0.541   0.512          0.512   0.541

12 个数据集 × seed × quality 实验组全部选择：

\[ `\gamma=0`{=tex} \]

S3R 的分层收缩形式为：

\[ `\hat{\mu}`{=tex}*{S3R}(q)
=(1-`\gamma`{=tex})`\hat{\mu}`{=tex}*{same}(q)
+`\gamma`{=tex}`\hat{\mu}`{=tex}\_{pool}(q) \]

当 (`\gamma=0`{=tex}) 时：

\[ `\hat{\mu}`{=tex}*{S3R}(q)=`\hat{\mu}`{=tex}*{same}(q) \]

协方差估计也同样退化为 same-quality 分量。

因此，S3R 并没有在实验中形成一个新的统计估计器。它只是由真实 validation
选择出 same-quality 作为最优端点。

**建议：将 same-quality CARF-A 作为当前参考池基线和协议，不要把 S3R
作为新方法投稿。**

## 1.2 现阶段可保留的方法框架

可以将现有方案描述为：

**Same-quality Conditional Real-reference Residual Scoring**

中文：同质量条件真实参考残差评分。

它由 CARF-A 的条件残差评分与 same-quality
参考池构成，但当前证据不足以将其包装为具有独立算法创新的新检测器。

流程如下：

1.  对图像进行统一的 256×256 处理，并按预定义 JPEG quality
    构造受控视图。
2.  提取固定取证残差表示（当前主实验为 NP-RSR）。
3.  根据已知 codec/quality，从真实参考集中选择同质量参考视图。
4.  以 source
    为单位识别同一真实图像的多个变换视图，避免把相关视图计为独立样本。
5.  在条件真实参考子集上估计残差均值和协方差。
6.  使用 Mahalanobis 距离计算条件异常分数。
7.  使用独立真实 calibration set 估计对应质量组的阈值。
8.  对未知 codec/quality 输出 unsupported 或
    abstain，而不借用已知组阈值。

## 1.3 数学形式

给定输入图像 (x)，取证残差为 (r(x))，质量条件为 (q(x))。

从同质量真实参考集中估计：

\[ `\hat{\mu}`{=tex}*{q} = `\sum`{=tex}*{i:q_i=q}`\tilde{w}`{=tex}\_i
r_i \]

\[ `\hat{\Sigma}`{=tex}*{q} = `\sum`{=tex}*{i:q_i=q}
`\tilde{w}`{=tex}*i(r_i-`\hat{\mu}`{=tex}*{q})(r_i-`\hat{\mu}`{=tex}\_{q})\^T
+`\lambda `{=tex}I \]

异常分数为：

\[ S(x)= (r(x)-`\hat{\mu}`{=tex}*{q})\^T `\hat{\Sigma}`{=tex}*{q}\^{-1}
(r(x)-`\hat{\mu}`{=tex}\_{q}) \]

其中权重应按独立 source 聚合，而不是把同一 source
的不同质量视图当作独立样本。

需要强调：该表达式是现有 CARF-A/same-quality 参考评分的整理，不代表 S3R
P0 已经提出并验证了新的检测算法。

## 1.4 当前方法的适用范围

目前可支持的表述是：

-   已知 codec/quality；
-   有同质量真实参考池；
-   source-disjoint reference/calibration/test；
-   在已评估的 ARF 与 DARG 小规模协议中比较 same 与 mixed 参考池。

目前不能声称：

-   未知 codec zero-shot；
-   任意平台传播链泛化；
-   对所有取证表示都有效；
-   S3R 优于 same-quality；
-   已解决数据集内容分布与标签代理问题。

------------------------------------------------------------------------

# 二、创新点是什么？

## 2.1 本轮实验真正支持的研究发现

### 发现一：Same-quality 参考池在当前协议中更可靠

在 12 个实验组中，validation 选择始终得到
(`\gamma=0`{=tex})。这说明在当前数据规模、残差表示和条件定义下，真实
validation 更偏好 same-quality，而不是引入 mixed-codec pooling。

### 发现二：增加名义 ESS 不等于增加独立信息

同一 source 的多个 codec 视图高度相关。source-weighted mixed 修正了名义
ESS 的膨胀，但其 AUROC 与 naive mixed 几乎一致。

这表明 source-level ESS 的统计修正本身并不会自动改善检测排序。

### 发现三：参考池统计质量与检测性能并非同一个指标

S3R proxy-MSE 显著低于 mixed，但 S3R 最终等于 same-quality，且没有超过
same-quality 的检测效果。

因此：

-   更低的参考均值估计误差不必然意味着更高 AUROC；
-   更大的 ESS 不必然意味着更低 MSE；
-   更低 MSE 不必然意味着更好的 fake-real 排序；
-   校准通过也不代表检测性能通过。

### 发现四：校准承诺必须与 coverage 一起解释

未知 codec 100% abstain。该系统没有对未知 codec 进行检测，而是拒绝处理。

这是可靠性边界，不是 zero-shot 泛化结果。

## 2.2 可以形成的论文贡献

当前最有依据的贡献是系统性分析，而非一个新 detector：

1.  **Reference-pool bias--variance analysis**：分析 same-quality 与
    mixed-codec 参考池的偏差、方差、MSE 和 source dependence。
2.  **Source-aware effective sample size**：指出多视图参考池的名义 ESS
    会高估独立 source 信息量，并给出 source-level 计算与不变量测试。
3.  **Conditional calibration failure
    analysis**：展示不同数据集和质量组的 FPR
    可能分别低于或高于目标区间，揭示组内校准的实际边界。
4.  **Representation-/codec-dependent forensic evaluation**：结合此前
    NP-RSR、DCT 和固定高通实验，说明 nuisance--signal coupling
    的可识别性取决于表示、数据集和 codec。

这些贡献需要进一步理论化和外部复现，才能成为有说服力的会议论文。

## 2.3 不能声称的创新

目前不能声称：

-   提出了优于 same-quality 的 S3R 检测算法；
-   mixed-codec pooling 通过 source weighting 后提升检测性能；
-   更大的 ESS 带来更好的检测效果；
-   proxy-MSE 下降意味着 AUROC 提升；
-   未知 codec zero-shot；
-   nuisance--signal coupling 是跨所有表示的普遍规律；
-   首次 real-only residual detection。

工作区的 **Forensic Self-Descriptions** 已经提出从真实图像学习多尺度
predictive residual filters，为每张图像形成 forensic
self-description，并使用真实图像描述符的 GMM 进行 zero-shot
detection；该工作还研究 open-set source attribution 和
clustering。因此，不能把 real-only residual modeling 本身作为新颖性。

------------------------------------------------------------------------

# 三、新方法相对老方法的优势和劣势

## 3.1 方法比较

  -------------------------------------------------------------------------------------------------------------------------------------
  方法                    主要优势                                     主要局限
  ----------------------- -------------------------------------------- ----------------------------------------------------------------
  UnivFD                  当前项目 84k 评估 AUROC 约                   使用真假监督训练，与 real-only 方法协议不同
                          0.9127，绝对检测性能较强                     

  RACE / NP-RSR           提供真实参考和取证残差检索思路               本地近似复现对 codec 和 calibration 较敏感

  CARF-A                  不需要生成器白盒访问或 fake                  增益受到分辨率、文件大小和内容分布影响；受控匹配后部分增益消失
                          training；条件参考在部分原生协议中有正增益   

  Same-quality CARF-A     已知质量时参考分布更匹配，proxy-MSE 优于     依赖已知 quality 和同质量真实参考池；不是未知 codec zero-shot
                          mixed                                        

  S3R                     修正了 source-level ESS 和 source            (`\gamma=0`{=tex})，没有形成新估计器，也没有检测增益
                          内协方差分解                                 
  -------------------------------------------------------------------------------------------------------------------------------------

## 3.2 Same-quality 的优势

-   在当前 12 个实验组中均被 validation 选中；
-   在 ARF/QF85、DARG/QF85 和 DARG/QF75 上优于 mixed；
-   proxy-MSE 与 S3R 相同，并显著优于 mixed；
-   避免把同一 source 的多个变换视图误当成独立样本；
-   便于建立已知 codec/quality 组内的校准协议。

## 3.3 Same-quality 的劣势

-   必须知道测试图像的 codec/quality；
-   同质量真实参考池不足时，估计可能不稳定；
-   未知 codec 目前直接 abstain；
-   不能把已知 quality 下的条件匹配结果称为 zero-shot；
-   受控小样本 AUROC 仍在约 0.51--0.59，检测效果有限。

------------------------------------------------------------------------

# 四、新方法能达到什么实验效果？

## 4.1 三种子平均 Macro AUROC

  数据集/质量      same   mixed   source-mixed     S3R
  ------------- ------- ------- -------------- -------
  ARF / QF85      0.528   0.511          0.511   0.528
  ARF / QF75      0.551   0.587          0.587   0.551
  DARG / QF85     0.556   0.517          0.517   0.556
  DARG / QF75     0.541   0.512          0.512   0.541

结论：

-   ARF/QF85：S3R 相对较强 baseline 的增益为 0，CI=\[0,0\]。
-   ARF/QF75：mixed 比 same 高约 0.035。
-   DARG/QF85、QF75：S3R 的相对增益均为 0。
-   Source-weighted mixed 与 naive mixed 几乎一致。

S3R 没有获得检测增益。

## 4.2 Proxy-MSE

S3R proxy-MSE 显著低于 mixed，但由于所有实验组均选择
(`\gamma=0`{=tex})，S3R 的：

-   bias²；
-   variance；
-   MSE；

与 same-quality 完全相同。

所以这只能证明 same-quality 的统计性质优于 mixed，不能算作 S3R
的方法增益。

## 4.3 校准与 coverage

  指标         实验结果       判断
  ------------ -------------- -------------------
  ARF FPR      3.86%--7.29%   两组超过 7%
  DARG FPR     2.36%--5.07%   三组低于 3%
  Coverage     约 99%--100%   达到至少 95% 要求
  未知 codec   100% abstain   不支持 zero-shot

虽然 coverage 达标，但部分组的 FPR
超出预设区间。因此校准标准没有完全通过。

**最终 GO=false。**

------------------------------------------------------------------------

# 五、目前已经完成的实验

## 5.1 方法与基线复现

已完成或建立结果归档的方向包括：

-   CNNDetection；
-   FreqSpec；
-   NPR / NP-RSR；
-   UnivFD；
-   LGrad；
-   DIRE；
-   RIGID；
-   ZED；
-   D³QE；
-   PRADA T2I；
-   Data Provenance；
-   RACE。

这些方法形成了监督式、real-only、频域、扩散重建、AR
机制证据和真实参考检索等不同路线的对照。

注意：本地 RACE 属于 approximate
reproduction，不应将本地结果等同于原论文官方结果。

## 5.2 失败路线和负结果

  -----------------------------------------------------------------------------------------
  研究路线                已完成工作                             结论
  ----------------------- -------------------------------------- --------------------------
  DCA-Fuse                84k 路径配对重评估                     整体没有稳定增益

  CTAD / CTRF             tokenizer affinity、关系结构等实验     未形成可靠检测方法

  PARD                    多扰动 real-only 检测                  未通过预注册标准

  MPID / STEM-AR          teacher holdout、机制图可预测性        未形成可迁移机制证据

  RACE                    CFE/PRP/NP-RSR 近似复现、codec         对 codec 协议敏感
                          校准审计                               

  CoNTra                  配对反事实干预、nuisance               漂移可校正，但 fake signal
                          子空间、signal retention               同时被损伤

  SACR                    pixel/file-size/codec 条件实验         未通过 GO

  S3R                     same/mixed/source-mixed/hierarchical   所有组
                          shrinkage                              (`\gamma=0`{=tex})，停止
  -----------------------------------------------------------------------------------------

## 5.3 CARF-A 和机制研究

已完成：

-   固定带宽；
-   独立真实校准集；
-   Mondrian 分组校准；
-   逐图 paired bootstrap；
-   文件大小、分辨率和内容复杂度标签审计；
-   ARForensics 与 DARG T2I 小规模外部复核；
-   NP-RSR、DCT、固定高通等表示的 coupling 识别；
-   same-quality 与 mixed-codec reference pool 比较；
-   受控 256×256/JPEG95 匹配实验。

## 5.4 S3R P0 工程验证

根据最新仓库归档：

-   53 passed，1 warning；
-   独立重算 288 组逐图 AUROC/coverage，全部与归档一致；
-   所有 MSE=bias²+variance 分解通过数值校验；
-   12 个实验 checkpoint 均可恢复；
-   初版 hierarchical 分量共享 mixed 带宽的问题已修复；
-   旧结果已隔离在 `outputs/s3r/invalid_pre_bandwidth_fix/`；
-   当前报告使用修正后的正式重跑结果。

这些结果支持实现的内部一致性，但不能替代外部泛化证据。

------------------------------------------------------------------------

# 六、下一步应该补充什么实验？

## 6.1 首要决策：停止 S3R 全量与模块扩展

S3R 的预注册核心检测标准失败，应该停止：

-   不做 84k 全量；
-   不继续增加 gamma 网格；
-   不增加新的 residual feature；
-   不再开发 S3R-v2；
-   不通过修改 GO 门槛挽救结果。

## 6.2 如果要继续研究，优先补充统计分析

### 实验 A：source-level ESS 的有效性

目标：验证 source-level ESS 是否正确反映独立信息量。

建议：

1.  对同一 source 复制或增加变换视图；
2.  比较 naive view-level ESS 与 source-level ESS；
3.  验证 source-level moments、covariance 和 ESS 在重复视图下保持不变；
4.  对不同 source 数量、不同视图数量和不同权重分布做模拟；
5.  报告 source-cluster bootstrap 置信区间。

当前单元测试已验证重复完整视图集不改变 source-weighted moments 和
ESS。下一步应补充统计模拟与真实数据验证。

### 实验 B：same-quality 的适用边界

建议区分三种任务：

  任务                  测试时是否知道 codec   参考池策略
  --------------------- ---------------------- --------------------------------
  Known-quality         是                     same-quality
  Mixed-known-quality   每张图质量可识别       按组选择 same-quality
  Unknown-codec         否                     abstain 或单独研究未知条件风险

不要把 Known-quality 结果写成 Unknown-codec zero-shot。

### 实验 C：校准有效性

当前 FPR 有组低于 3%、有组高于 7%。建议：

-   检查每组 calibration 样本数；
-   报告 conformal 阈值及其离散性；
-   增加 source-cluster bootstrap；
-   报告每组 FPR 的置信区间；
-   绘制 risk--coverage curve；
-   对未知 codec 明确报告 abstention。

不能通过测试集调整阈值。

### 实验 D：检测性能与估计误差的关系

S3R 显示 proxy-MSE 改善不必然提升 AUROC。

建议分析：

-   条件均值误差与 AUROC 的相关性；
-   条件协方差误差与 AUROC 的相关性；
-   source-level ESS 与检测性能的关系；
-   每个生成器上的条件分布偏离；
-   same/mixed 在不同 quality 下的排序变化。

这可能形成一项重要分析：**参考分布估计质量与检测判别质量之间并不存在简单的单调关系。**

## 6.3 与工作区论文的结合

### Forensic Self-Descriptions

该工作使用真实图像训练多尺度 predictive residual
filters，并为每张图像拟合紧凑的 forensic
self-description，再用真实图像描述符的 GMM 建模 real distribution。

它已经覆盖：

-   Zero-shot synthetic image detection；
-   Open-set source attribution；
-   Unsupervised source clustering。

因此，本项目不能把 real-only residual modeling 作为首次创新。

更有意义的比较是：

-   per-image self-description 与跨图像条件参考分布的差异；
-   source-level reference dependence；
-   codec/quality 分组下的条件分布估计；
-   calibration 和 coverage 的适用边界。

### UnivFD

作为强监督基线。应明确区分：

-   UnivFD：利用真假标签学习判别表示；
-   CARF-A/same-quality：使用真实参考残差分布进行异常检测。

两者的训练信息和评测协议不同，不能只按 AUROC 排名。

### RIGID / NPR / RACE

重点比较扰动响应、局部残差和真实参考检索机制。

需要说明本项目的贡献不是首次使用扰动或真实参考，而是对 reference pool 的
source dependence、codec 条件和 calibration failure 进行系统分析。

### PRADA / D³QE / Data Provenance

这些工作提供 AR tokenizer、量化误差、decoder likelihood 和 provenance
方向的机制证据。

可以作为 model-aware AR 方法对照，但不应把它们与 real-only reference
方法视为完全相同的任务协议。

------------------------------------------------------------------------

# 七、创新性是否足够 ICML？如何完善？

## 7.1 当前判断

**S3R 不足以作为 ICML 新方法投稿。**

原因：

1.  12 个实验组全部选择 (`\gamma=0`{=tex})，S3R 退化为 same-quality。
2.  相对 same-quality 的检测增益为零。
3.  FPR 部分组未达到预设区间。
4.  未知 codec 100% abstain。
5.  当前结果没有证明新的估计器优于简单 same-quality reference。

CARF-A 单独作为 detector 的方法创新性也偏弱。

## 7.2 当前最有价值的论文定位

建议把研究定位为：

**Reference-Pool Statistics and Failure Mechanisms for Real-Only
AI-Generated Image Detection**

中文：

**面向真实参考式 AI 生成图像检测的参考池统计与失效机制研究。**

核心研究问题：

> 在真实参考式生成图像检测中，参考池的 codec 条件、source
> 相关性和条件覆盖如何影响残差分布估计、检测性能与误报控制？

建议形成三条贡献：

1.  **Source dependence：** 证明同一 source 的多 codec
    视图不能作为独立参考样本，分析名义 ESS 膨胀。
2.  **Reference-pool bias--variance：** 量化 same-quality 与 mixed-codec
    的估计偏差、方差和 MSE 差异。
3.  **Conditional calibration：** 研究已知质量组的 FPR、coverage
    与未知条件 abstention 的边界。

这比把 S3R 包装成新检测器更符合目前的证据。

## 7.3 如果仍希望增加方法创新

不建议继续堆叠检测特征。真正值得探索的是：

### 方向一：Source-clustered reference estimator

把 source 作为统计独立单位，而不是 view。

理论上研究：

-   Cluster-aware ESS；
-   Source-level mean/covariance estimation；
-   重复视图下估计量不变性；
-   多视图相关性对有效样本量的影响。

### 方向二：Conditional reference selection with support guarantees

研究何时可以使用 same-quality reference，何时必须拒判。

可建立：

-   条件支持度估计；
-   有效独立 source 数；
-   参考池不足时的风险上界；
-   coverage-risk tradeoff。

### 方向三：Calibration-aware reference selection

不只在 validation 上最小化 NLL，还要研究：

-   条件分布估计误差；
-   校准阈值方差；
-   组间 FPR 偏差；
-   coverage 与 FPR 的联合约束。

但这些都只是下一阶段研究方向。必须先提出清晰的统计目标和理论保证，再设计实验验证。

------------------------------------------------------------------------

# 八、推荐论文题目与结构

## 8.1 候选题目

**Reference-Pool Statistics for Real-Only AI-Generated Image Detection:
Source Dependence, Codec Conditioning, and Calibration Failure**

中文：

**真实参考式 AI
生成图像检测的参考池统计：来源相关性、编码条件与校准失效。**

## 8.2 建议论文贡献

1.  **Source dependence analysis：** 分析同一真实 source 的多个 codec
    视图如何造成名义 ESS 膨胀。
2.  **Reference-pool bias--variance analysis：** 系统比较
    same-quality、mixed-codec 和 source-weighted mixed reference。
3.  **Representation-dependent mechanism analysis：** 结合 NP-RSR、DCT
    和固定高通，分析 nuisance--signal coupling 的可识别性边界。
4.  **Calibration failure analysis：** 在已知 codec 与未知 codec
    设置下，评估 FPR、coverage 和 abstention。

## 8.3 论文结构

1.  Introduction：真实参考式检测的参考池错配问题。
2.  Related Work：Forensic
    Self-Descriptions、UnivFD、NPR、RACE、RIGID、PRADA、D³QE、Data
    Provenance。
3.  Problem Formulation：source-correlated multi-view references
    与条件校准。
4.  Reference Pool Statistics：same-quality/mixed 的 bias--variance
    分析。
5.  Representation-dependent Coupling：NP-RSR、DCT、HP 的识别边界。
6.  Experiments：ARForensics、DARG、codec quality、source-level
    bootstrap。
7.  Calibration and Limitations：FPR、coverage、unknown codec
    abstention。
8.  Discussion：参考池统计与真实参考式检测的适用边界。

------------------------------------------------------------------------

# 九、最终科研执行计划

  优先级   工作                                               是否继续
  -------- -------------------------------------------------- ------------
  P0       冻结 S3R 负结果、代码和 checkpoint                 必须
  P0       完成 source-level ESS 与 bias--variance 理论说明   必须
  P1       校准 FPR/coverage/risk--coverage 分析              必须
  P1       补充 source-cluster bootstrap 和统计模拟           建议
  P1       Forensic Self-Descriptions 同协议对比              建议
  P2       额外数据集、更多生成器和真实来源                   视时间决定
  STOP     S3R-v2、SACR-v2、CoNTra 新模块、无依据特征堆叠     停止

**如果距离 ICML 截止期已经很近，不建议再进行大规模新方法研发。**
应优先完成统计分析、理论假设、强基线对照和论文叙事。如果无法形成足够强的理论与外部验证，建议考虑下一轮
ICML，或转向更适合系统性视觉取证研究的 CVPR/ICCV/TIFS。

------------------------------------------------------------------------

# 十、GitHub 代码与结果

-   [GitHub 仓库](https://github.com/MurmanskY/ai_image_detection)
-   [RESULTS.md](https://github.com/MurmanskY/ai_image_detection/blob/main/reports/RESULTS.md)
-   [S3R P0
    实现](https://github.com/MurmanskY/ai_image_detection/blob/main/analysis/s3r_p0.py)
-   [S3R
    单元测试](https://github.com/MurmanskY/ai_image_detection/blob/main/tests/test_s3r.py)
-   [Nuisance--signal coupling
    理论文档](https://github.com/MurmanskY/ai_image_detection/blob/main/docs/nuisance_signal_coupling.md)

## 版本与验证信息

-   最新提交：`875e49391af7cfd06e4b1bad0b48ef7df8be7cc9`
-   结果归档：`reports/RESULTS.md` 第 22 节
-   测试：53 passed，1 warning
-   独立重算：288 组逐图 AUROC/coverage，与归档一致
-   MSE=bias²+variance 分解：全部通过数值校验
-   Checkpoint：12 组均可恢复
-   旧版带宽错误结果：已隔离在 `outputs/s3r/invalid_pre_bandwidth_fix/`

------------------------------------------------------------------------

## 最终决策

**停止 S3R、SACR 和 CoNTra 的新检测算法扩展。**

当前没有证据表明 S3R 优于 same-quality reference。保留 same-quality
CARF-A 作为实验基线，把下一阶段的研究重点放在：

1.  Source-level ESS 与多视图相关性；
2.  Same-quality 与 mixed-codec 的偏差---方差关系；
3.  条件校准的 FPR、coverage 和未知条件拒判；
4.  Nuisance--signal coupling 的表示依赖边界。

只有当后续研究提出了真正新的统计估计方法，并且在检测性能、校准和外部泛化上同时通过预注册标准，才重新考虑将其作为
ICML 方法论文主线。
