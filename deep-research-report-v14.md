# 生成图像检测研究路线分析报告（v14）

**主题：受控匹配、参考池统计与表示条件化耦合分析**

-   GitHub：<https://github.com/MurmanskY/ai_image_detection>
-   最新核查提交：`d8840d9`
-   结果归档：`reports/RESULTS.md` 第 21 节
-   核查日期：2026-11-25

------------------------------------------------------------------------

## 核心结论

v13 的实验没有证明 CARF-A
原始增益可以排除分辨率、内容分布和文件大小代理。

在统一 256×256/JPEG95 并进行一对一条件匹配后：

-   ARForensics 的 CARF-size 增益基本消失。
-   DARG 仅保留约 +0.02 的小幅增益，绝对 AUROC 仍约为 0.55。
-   same-quality reference 的条件偏差和 proxy-MSE 明显低于 mixed-codec。
-   mixed-codec 的名义 ESS 虽约增至 3 倍，却没有降低 seed variance。
-   nuisance--signal coupling 表现出表示、数据集与 codec
    条件依赖：NP-RSR 在两个数据集成立，DCT 仅在 DARG 满足识别标准，HP
    未识别。

因此，建议：

1.  停止把 CARF-A 当作已经证实的通用新检测器。
2.  不再扩展 SACR 和 CoNTra。
3.  下一阶段重点研究 **S3R（Source-aware Same-quality Reference
    Modeling，来源感知的同质量参考建模）**。
4.  将研究重心转向真实参考池的偏差---方差权衡、source-level ESS
    和条件校准。

S3R 尚未实现或验证，不能作为已经完成的论文贡献。

------------------------------------------------------------------------

# 一、应该选择什么新方法？

## 1.1 推荐方法：S3R

**S3R：Source-aware Same-quality Reference Modeling**

中文：来源感知的同质量参考建模。

S3R 不是另一个残差特征提取器，而是针对 v13
发现的参考池偏差---方差问题，重新设计真实参考样本的选择、加权和校准。

CARF-A 保留为基线与核心评分器，但不再把其原始 AUROC
增益直接解释为已经排除数据捷径。

## 1.2 设计动机

目前实验揭示了三个重要问题。

**问题一：same-quality reference 更准确。**

same-quality reference 在已知 codec/quality 时具有更低的条件偏差。

mixed-codec 虽然增加名义
ESS，却因为不同质量视图的分布不匹配而引入明显偏差。

**问题二：同一 source 的多个视图并非独立样本。**

同一真实图像经过 JPEG、Resize
等变换后产生多个视图。这些视图高度相关，不能把每个视图都当作独立参考样本。

否则，名义 ESS 会高估真实独立信息量。

**问题三：条件参考模型必须考虑支持度。**

当目标 codec 已知时，优先使用同质量真实参考。

当目标 codec
未知或同质量参考不足时，不能直接混合全部视图，而应显式评估支持度和偏差风险。

因此，S3R 将以下因素纳入方法设计：

-   Reference support
-   Source dependence
-   Effective sample size
-   Coverage
-   Calibration validity

## 1.3 S3R 的算法流程

  ---------------------------------------------------------------------------------------------
  步骤                    操作                                         目的
  ----------------------- -------------------------------------------- ------------------------
  1\. 条件识别            读取已知 codec/quality；未知时保留 unknown   明确适用协议
                          状态                                         

  2\. Source 聚类         将同一真实图像的多 codec/resize 视图归为一个 避免重复视图膨胀样本量
                          source cluster                               

  3\. 同质量优先          目标质量组内建立真实参考子集，按 source      降低条件偏差
                          级权重聚合                                   

  4\. 支持度评估          计算 source-level ESS、条件距离和覆盖率      判断是否足以估计

  5\. 部分池化            同质量样本不足时，引入相邻质量组，并用真实   控制偏差---方差权衡
                          held-out source 选择 shrinkage               

  6\. 残差评分            使用 CARF-A 条件均值/协方差 Mahalanobis 分数 保留已有有效评分器

  7\. 独立校准            按已知 codec/quality 组做 real-only          控制误报并报告覆盖率
                          conformal；未知组 abstain                    
  ---------------------------------------------------------------------------------------------

## 1.4 数学设计

设：

-   (s)：真实图像 source；
-   (q)：codec/quality 组；
-   (r)：取证残差。

每个 source 的多个视图先聚合为 source-level summary。

对目标质量 (q\^\*)，构造部分池化均值：

\[ `\hat{\mu}`{=tex}(q\^\*) =
(1-`\gamma`{=tex})`\hat{\mu}`{=tex}\_{same}(q\^\*)
+`\gamma`{=tex}`\hat{\mu}`{=tex}\_{pool}(q\^\*) \]

其中：

\[ 0`\leq`{=tex}`\gamma`{=tex}`\leq1`{=tex} \]

协方差也采用相应的分层估计或收缩估计。

参数 (`\gamma`{=tex}) 必须：

-   通过独立真实 validation sources 选择；
-   使用 source-cluster bootstrap 评估；
-   不允许使用 fake test 选择。

如果 same-quality support 不足，则输出
`unsupported`，或者按照预注册规则执行部分池化。

最终异常分数仍采用：

\[ S(x)= (r(x)-`\hat{\mu}`{=tex}(q\^*))\^T
\[`\hat{\Sigma}`{=tex}(q\^\*)+`\lambda `{=tex}I\]\^{-1}
(r(x)-`\hat{\mu}`{=tex}(q\^*)) \]

**S3R 的潜在创新点是 source-aware reference estimator
与条件支持/风险控制，而不是 Mahalanobis 本身。**

它必须与 CARF-A、same-quality、mixed-codec
和简单分层收缩进行严格消融后，才能声称有效。

------------------------------------------------------------------------

# 二、v13 最新实验结果与科学解释

## 2.1 统一 256×256 / JPEG95 后的受控匹配

所有图像：

1.  Resize 为 256×256 BICUBIC。
2.  使用 JPEG95 重新编码。
3.  文件大小使用本次 JPEG 编码后的字节数。
4.  Real/Fake 根据 reference 标准化条件进行一对一最优匹配。
5.  Caliper=2.5 在实验运行前冻结。

测试规模：

-   ARForensics：7 个生成器。
-   DARG：5 个生成器。
-   每类、每生成器 300 张。

实验结果：

  数据集/协议                M0   CARF-size   CARF-pixel
  --------------------- ------- ----------- ------------
  ARF 全部受控图          0.550       0.551        0.562
  ARF 全条件匹配          0.563       0.556        0.569
  ARF 文件大小匹配        0.549       0.550        0.561
  ARF 像素复杂度匹配      0.558       0.552        0.566
  DARG 全部受控图         0.531       0.551        0.547
  DARG 全条件匹配         0.524       0.545        0.541
  DARG 文件大小匹配       0.531       0.550        0.547
  DARG 像素复杂度匹配     0.525       0.546        0.542

### 重要发现

ARF 文件大小匹配后，CARF-size 相对 M0 的三个 seed 增量为：

\[ +0.0016,`\quad `{=tex}+0.0007,`\quad -0.0009`{=tex} \]

置信区间均跨零。

像素复杂度匹配后，CARF-size 也没有正向增益。

这说明原生分辨率下 NP-RSR/CARF-A 的收益对尺度和内容分布高度敏感。

DARG 文件大小匹配后，size 相对 M0 的增量为：

\[ +0.019,`\quad `{=tex}+0.022,`\quad `{=tex}+0.018 \]

置信区间下界均大于零。

但是绝对 AUROC 仍然只有约 0.55，部分生成器的像素复杂度标签 AUC 约为
0.85。

此外，全条件匹配后，ARF Infinity 的组合条件标签 AUC 仍然为 0.874。

**结论：观察性一对一匹配只能提供受控证据，不能证明因果排除了所有文件大小和内容捷径。**

## 2.2 Same-quality 与 mixed-codec 的 Bias--Variance 分析

  数据/质量   参考池       Bias²   Seed variance   Proxy MSE    ESS
  ----------- -------- --------- --------------- ----------- ------
  ARF QF75    same       0.00034         0.00118     0.00152   2769
  ARF QF75    mixed      0.02924         0.00113     0.03037   8225
  DARG QF75   same       0.00005         0.00074     0.00078    312
  DARG QF75   mixed      0.02924         0.00081     0.03004    932

mixed-codec 的名义 ESS 约为 same-quality 的 3 倍，但：

-   Bias² 大幅增加。
-   Proxy MSE 大幅增加。
-   Seed variance 没有下降。

需要注意，proxy-oracle
是有限真实参考构造的经验代理，不是真实总体均值。因此，这个分解是诊断量，而不是总体偏差的无偏估计。

### 关键原因

mixed 的多个视图来自同一个真实 source。

ESS 没有考虑视图之间的相关性，因此更大的名义 ESS
不代表更多独立真实来源。

这支持一个重要结论：

**已知 codec 时，same-quality 的条件偏差优势可以大于 mixed pooling
的样本量收益。**

不能把 mixed pooling 当作无成本的方差降低方法。

## 2.3 表示条件化的 Nuisance--Signal Coupling

  ---------------------------------------------------------------------------------
  数据/表示     稳定性         MMD reduction                  平均 η 判定
  ------------- -------------- ------------------- ----------------- --------------
  ARF / NP-RSR  0.981--0.999   QF85                     0.124--0.132 coupling
                               86.8--89.6%；QF75                     
                               87.1--88.5%                           

  DARG / NP-RSR 0.997--0.999   QF85                     0.063--0.078 coupling
                               89.4--91.0%；QF75                     
                               90.0--91.9%                           

  ARF / DCT     0.993--0.998   QF85 6.6--56.3%，CI      0.014--0.016 unidentified
                               跨零；QF75                            
                               72.8--82.4%                           

  DARG / DCT    0.984--0.996   QF85                     0.005--0.008 coupling
                               41.2--63.3%；QF75                     
                               70.7--82.6%                           

  ARF / HP      0.193--0.516   不稳定                          0.001 unidentified

  DARG / HP     0.024--0.415   不稳定                   0.004--0.005 unidentified
  ---------------------------------------------------------------------------------

这修正了之前的研究结论：

-   Coupling 不只是 NP-RSR 现象。
-   DCT 在 DARG 中也满足可识别性判据。
-   ARF/DCT 的 QF85 MMD 改善置信区间跨零，不能把低 η 解释为 coupling。
-   HP 在两个数据集上的 nuisance 子空间均不稳定。

最准确的科学结论是：

**Nuisance--signal coupling 由表示、数据分布和 codec 条件共同决定。**

------------------------------------------------------------------------

# 三、创新点应该如何定位？

## 3.1 已经有证据支持的创新性发现

目前有四项比较有价值的发现。

**创新点一：受控匹配揭示数据捷径问题。**

CARF-A 原生分辨率下的增益不能直接迁移到统一尺度和编码协议。

分辨率、文件大小和内容分布代理会影响检测结论。

**创新点二：揭示参考池的偏差---方差权衡。**

名义 ESS 增大不等于估计质量提升。

同一 source 的多个 codec 视图存在相关性，mixed pooling
可能引入系统偏差。

**创新点三：提出表示条件化的 Coupling 识别准则。**

只有 nuisance 子空间稳定，而且真实漂移校正有效时，低 η 才有机制含义。

否则应标记为 `unidentified`。

**创新点四：建立表示条件化的正负边界证据。**

-   NP-RSR 在 ARF/DARG 中满足 coupling 判据。
-   DCT 在 DARG 中满足 coupling 判据。
-   HP 与 ARF/DCT 不满足。

这比简单宣称普遍 coupling 更准确，也更容易被实验检验。

## 3.2 不能声称的创新

论文中不能声称：

1.  已经排除文件大小和内容分布捷径。
2.  Mixed-codec 是无成本的方差降低方法。
3.  Coupling 是跨表示普遍规律。
4.  首次 real-only residual detection。
5.  首次扰动响应检测。
6.  首次 conformal calibration。
7.  首次 AR tokenizer forensic。
8.  S3R 已经优于 CARF-A。

## 3.3 与工作区论文的差异

  ---------------------------------------------------------------------------------------------------------------
  工作                    已有核心                                 本研究需要强调的差异
  ----------------------- ---------------------------------------- ----------------------------------------------
  Forensic                真实图像训练多尺度 predictive residual   强调 reference source
  Self-Descriptions       filters；每图拟合 forensic               dependence、codec-stratified reference
                          self-description；real GMM 做            estimation 与 bias--variance
                          zero-shot/open-set/clustering            

  UnivFD                  CLIP 语义特征监督真假分类                作为强监督基线；S3R/CARF-A 是 real-only
                                                                   reference 路线

  RIGID / ZED             训练自由的扰动/鲁棒性检测                扰动响应本身不是创新；重点在 nuisance
                                                                   子空间可识别性和参考池统计

  PRADA / D³QE / Data     AR                                       作为 model-aware AR 对照；S3R
  Provenance              likelihood、量化误差、codebook/decoder   不访问目标生成器内部概率
                          provenance                               

  NPR / RACE              取证残差与真实参考/扰动评分              需要明确参考池估计与校准的增量，不能重复简单
                                                                   real-only anomaly detection
  ---------------------------------------------------------------------------------------------------------------

特别需要注意：工作区的 Forensic Self-Descriptions 已经证明 real-only
forensic residual detection 可以用于 zero-shot detection、open-set
source attribution 和 clustering。

因此，不能把"只用真实图像建立参考分布"本身当作创新。

------------------------------------------------------------------------

# 四、新方法与老方法相比的优势和局限

当前应将 CARF-A 作为已实现方法，将 S3R 作为待验证的升级方案。

  ---------------------------------------------------------------------------------------------------
  方法                    优势                                 主要局限
  ----------------------- ------------------------------------ --------------------------------------
  UnivFD                  84k 同项目评估 AUROC 约              需要真假监督训练，和 real-only
                          0.9127，绝对检测性能强               方法协议不同

  RACE NP-RSR             真实参考与扰动响应，适合作机制基线   本地近似复现对 codec/calibration
                                                               敏感；JPEG95 FPR 曾明显漂移

  CARF-A                  不需要生成器白盒访问或 fake          统一 256/JPEG95 匹配后 ARF
                          training；NP-RSR                     增益基本消失；条件代理标签关联；依赖
                          上两个数据集原生协议有正增益         codec/reference 条件

  S3R（候选）             Source-aware ESS、same-quality       尚未实现/验证；已知 codec
                          优先、支持感知校准，有望减少 mixed   优势不能外推为未知 codec zero-shot
                          pooling 偏差                         
  ---------------------------------------------------------------------------------------------------

v13 受控协议绝对 AUROC 约 0.53--0.57，接近随机。

因此，不能把这组结果包装为高性能 detector。

更合理的效果主张是：

该实验揭示了原生协议性能对分辨率/内容分布的敏感性，并量化了参考池条件的偏差---方差代价。

------------------------------------------------------------------------

# 五、目前已经完成的实验

  --------------------------------------------------------------------------------------------------------------------------------
  模块                      已完成内容                                                              论文作用
  ------------------------- ----------------------------------------------------------------------- ------------------------------
  基线复现                  CNNDetection、FreqSpec、NPR-inspired、UnivFD、LGrad、DIRE、RIGID、ZED   统一比较矩阵
                            等                                                                      

  AR 专用方法               D³QE 官方代码、PRADA T2I、Data Provenance 采样                          Model-aware 对照

  数据审计                  SHA256/pHash、跨 split 重复、格式差异、DARG duplicate audit             数据泄漏/协议控制

  DCA-Fuse/CTAD/CTRF/PARD   84k 融合重评估、tokenizer affinity、关系统计、多扰动检测                负结果链

  MPID/STEM-AR              Teacher holdout 与机制图预测                                            机制蒸馏路线负结果

  RACE                      CFE/PRP/NP-RSR 近似复现与校准漂移分析                                   真实参考方法基线

  CARF-A                    固定带宽、独立校准、Mondrian、paired bootstrap、变量审计                当前主要 real-only
                                                                                                    条件参考实现

  CoNTra/SACR               配对干预、信号保留、三种子与复杂度审计                                  停止依据与 coupling 机制证据

  v12 P1                    DARG 外部复核、HP 独立表示、codec reference pool 分解                   跨数据/表示边界

  v13 P0                    256/JPEG95 控制、匹配审计、same/mixed proxy bias--variance、DCT         当前最有论文价值的新机制证据
                            coupling                                                                
  --------------------------------------------------------------------------------------------------------------------------------

工程验证：

-   46 tests passed。
-   1 warning。
-   108 组受控全图 AUROC 和 216 组单独匹配 AUROC 与逐图分数一致。
-   16 组 bias--variance 分解数值误差均小于 (10\^{-5})。

这些结果证明实现内部一致性，但不能替代独立数据集泛化。

------------------------------------------------------------------------

# 六、下一步实验计划：以 S3R P0 为唯一方法开发

## 6.1 P0：先复现统计发现并冻结协议

建议完成以下五项：

1.  冻结数据清单、source ID、codec quality、256/JPEG95 controlled
    views。
2.  所有 seed 重新按 source 划分 reference/calibration/test。
3.  重新计算 same/mixed 的 source-level ESS：同一 source
    的多视图先聚类，ESS 以 source cluster
    权重计算，而不是把每个视图当独立样本。
4.  对 same-quality、mixed-codec、source-weighted mixed、hierarchical
    partial pooling 四种 reference pool 做同一协议比较。
5.  用真实 held-out source 估计条件均值/协方差误差，明确 proxy-oracle
    是经验参考，不将其写成总体真值。

报告每个 codec 的：

-   Coverage
-   FPR
-   AUROC
-   Source-level ESS
-   Bias²
-   Variance
-   MSE

使用 source-cluster bootstrap 给出置信区间。

## 6.2 P1：S3R 的预注册 GO 标准

  -----------------------------------------------------------------------
  指标                                建议 GO 标准
  ----------------------------------- -----------------------------------
  偏差/MSE                            相对 mixed-codec 显著降低；相对
                                      same-quality 不恶化超过预设容忍值

  检测性能                            已知 codec 的跨质量测试中，paired
                                      ΔAUROC 相对 CARF-A 与 same-quality
                                      baseline 为正，CI 下界大于 0

  生成器稳健性                        至少 5/7 ARF 与 4/5 DARG
                                      生成器不退化

  校准                                每个已知质量组 real FPR
                                      落入预设区间；coverage ≥95%

  未知 codec                          不宣称 zero-shot；单独报告
                                      abstention/unsupported
                                      和风险覆盖曲线

  Source dependence                   Source-level ESS 与独立 source
                                      数一致，不因增加同一 source
                                      变换视图虚增
  -----------------------------------------------------------------------

如果 S3R 不能在 same-quality baseline
上带来稳定收益，则停止新方法扩展，论文转向参考池统计与失效机制分析。

不能为了通过 GO 事后修改阈值。

## 6.3 P2：论文完整性

如果 P1 通过，再补充：

1.  增加真实来源和生成器家族，至少一个额外公开 benchmark。
2.  比较 NP-RSR、固定 DCT、固定高通，严格遵守 representation
    identifiability 判据。
3.  对 Forensic Self-Descriptions 做可比复现或明确对照。
4.  报告 AUROC/AP、TPR@1%FPR、FPR、coverage、abstention、source-level
    ESS、推理时间和参考库大小。
5.  对已知 codec、未知 codec、mixed codec 三种任务分别定义协议，避免混淆
    condition-known 与 zero-shot。

------------------------------------------------------------------------

# 七、创新性是否足够 ICML？

**当前判断：还不够。**

v13 的科学分析比单纯 CARF-A detector
更有价值，但目前仍是有限数据上的系统性观察和诊断协议。

S3R 具有统计方法创新潜力，但尚未实现，也没有实验证据。

## 7.1 能提升到 ICML 水平的方向

**方向一：Source-correlated multi-view reference estimation**

将独立样本单位定义为 source，而不是 codec view。

给出：

-   Cluster-aware ESS。
-   条件均值估计误差界。
-   条件协方差估计误差界。

**方向二：条件参考池选择与部分池化**

在 same-quality bias 与 mixed-codec sample-size gain 之间，基于真实
held-out source 估计最优 pooling weight。

**方向三：Support-aware conformal risk--coverage**

-   已知组内给出有限样本 FPR 控制。
-   未知/低支持条件明确 abstain。
-   分析 coverage-risk trade-off。

**方向四：Representation-conditional identifiability**

只有 nuisance basis 稳定且 intervention drift 可校正时，才解释 η。

对 unidentified 表示不给 coupling claim。

**方向五：跨数据集和特征表示验证**

至少在两个数据集、多个独立 forensic
representations、不同生成家族上复现正负结果。

这些贡献必须形成真正的新统计方法和理论，而不只是把
same-quality、ESS、conformal 放进同一个 pipeline。

## 7.2 ICML 投稿判断

  ----------------------------------------------------------------------------------
  论文路线                            当前判断
  ----------------------------------- ----------------------------------------------
  CARF-A detector paper               创新性偏弱，不建议直接投稿

  SACR/CoNTra method paper            已按预注册标准停止

  S3R statistical method              值得做 P0，但尚未证明

  Reference-pool bias/variance +      当前最有证据的论文方向；需扩充理论和外部数据
  representation-dependent coupling   
  analysis                            
  ----------------------------------------------------------------------------------

如果距离 ICML 截止期不足以完成 S3R
P0、理论分析和外部复现，建议考虑下一轮 ICML 或更匹配的 CVPR/ICCV/TIFS。

当前不应通过夸大 0.55 左右的受控 AUROC 来制造方法贡献。

------------------------------------------------------------------------

# 八、建议论文主线

## 8.1 候选题目

**Source-Aware Conditional Reference Modeling for Reliable AI-Generated
Image Detection: Bias--Variance and Representation-Dependent Nuisance
Coupling**

中文：

**面向可靠 AI
生成图像检测的来源感知条件参考建模：偏差---方差权衡与表示依赖的干扰耦合。**

## 8.2 建议论文贡献

以下贡献需要等待 S3R 验证后才能最终确定。

1.  **发现：** 控制分辨率、编码和可观测内容条件后，原始 CARF-A
    增益显著收缩，揭示数据分布捷径对 detector evaluation 的影响。

2.  **统计分析：** Same-quality reference 相比 mixed-codec
    在条件偏差/MSE 上更优；重复变换视图造成名义 ESS 膨胀。

3.  **表示机制：** Nuisance--signal coupling 需要同时满足表示内 nuisance
    可识别性与数据/codec 条件；低 η 单独不足以证明 coupling。

4.  **方法：** 如果 P0 通过，提出 source-aware
    same-quality/partial-pooling reference estimator 与 coverage-aware
    calibration。

如果 S3R 未通过 GO，论文应去掉"新
detector"主张，改为系统性评测与失效机制研究。

------------------------------------------------------------------------

# 九、GitHub 与工作区来源

## GitHub 代码与结果

-   [GitHub
    最新结果：RESULTS.md](https://github.com/MurmanskY/ai_image_detection/blob/main/reports/RESULTS.md)
-   [机制实验：carf_mechanism_p0.py](https://github.com/MurmanskY/ai_image_detection/blob/main/analysis/carf_mechanism_p0.py)
-   [DCT
    表示：dct_forensics.py](https://github.com/MurmanskY/ai_image_detection/blob/main/analysis/dct_forensics.py)
-   [理论文档：nuisance_signal_coupling.md](https://github.com/MurmanskY/ai_image_detection/blob/main/docs/nuisance_signal_coupling.md)

## 工作区相关论文

-   Forensic Self-Descriptions
-   UnivFD
-   RIGID
-   PRADA
-   D³QE
-   Data Provenance
-   NPR
-   DIRE
-   GenDet
-   ZED
-   FreqSpec
-   LGrad
-   Manifold-induced Biases
-   Survey

## 版本与验证信息

-   GitHub 最新提交：`d8840d9`
-   RESULTS.md：第 21 节
-   测试：46 passed，1 warning
-   逐图重算：108 组受控全图 AUROC、216 组单独匹配 AUROC，与归档一致
-   Bias--variance 分解：16 组数值误差均小于 (10\^{-5})
-   结果规模：ARF 300/类、DARG 300 fake/生成器等示例协议，不是 84k
    全量。

------------------------------------------------------------------------

**最终决策：停止 SACR-v2、CoNTra 投影校正和无依据的特征堆叠。先完成 S3R
P0，验证 source-aware reference selection 是否真正优于 same-quality
baseline。若不能通过
GO，就将论文定位为参考池统计与表示依赖失效机制研究，而不是继续开发新检测器。**
