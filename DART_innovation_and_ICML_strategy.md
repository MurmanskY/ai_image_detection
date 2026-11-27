# DART 创新性分析与 ICML 方法升级建议

**研究对象：** DART（Distribution-Aware Autoregressive Token Detector）\
**目标会议：** ICML 2027\
**核心问题：**
将已有检测信号进行整合，是否足以构成方法创新？下一步如何形成真正有说服力的创新？

------------------------------------------------------------------------

## 核心判断

你的担忧是有道理的。这是目前 DART
研究中最值得认真讨论的问题，甚至比继续提高 AUROC 更重要。

**DART 目前确实存在"方法整合多于原理创新"的风险。仅仅把 PRADA 的
likelihood、D³QE 的 quantization error、Data Provenance 的 decoder
reconstruction 组合起来，再使用 Deep Sets 进行融合，并不足以自然构成
ICML 级别的方法创新。**

但是，这不意味着 DART 没有创新价值，也不意味着不能投稿 ICML。

关键在于：你究竟提出了一个新的检测信号，还是发现并解决了一个前人没有充分解决的机器学习问题？这两者是不同的创新路径。

------------------------------------------------------------------------

# 一、ICML 是否要求必须提出全新的检测信号？

不要求。

机器学习顶会的方法创新，并不一定意味着必须发明一个此前从未出现过的特征、信号或网络结构。

常见的方法创新大致有四种：

  -----------------------------------------------------------------------------------------------------
  创新类型                核心要求                                           举例
  ----------------------- -------------------------------------------------- --------------------------
  新信号                  发现以前没有利用过的信息                           新的生成机制取证信号

  新方法                  提出新的建模、学习或优化方式                       新的联合分布估计方法

  新理论                  揭示以前没有理解的机制或规律                       新的泛化界、可识别性理论

  新问题与新范式          发现一个重要但未被解决的问题，并提出有效解决方案   新的跨模型泛化学习框架
  -----------------------------------------------------------------------------------------------------

因此，DART 不一定要提出全新的 forensic signal 才有机会投稿。

但是，**如果只是将已有信号拼接起来，再训练一个普通分类器，通常很难说服
ICML 审稿人。**

因为审稿人很可能会问：

1.  PRADA 已经使用 likelihood，为什么还需要 DART？
2.  D³QE 已经研究 quantization，为什么再加入一次？
3.  Data Provenance 已经研究 decoder reconstruction，你的新东西是什么？
4.  Deep Sets 和 Attention Pooling 都是已有结构，为什么这构成方法创新？
5.  如果把三个已有特征简单拼接，再训练 MLP，是否也能得到类似结果？

如果这些问题回答不了，AUROC 再高也不一定能够弥补创新性不足。

------------------------------------------------------------------------

# 二、DART 算不算创新？

我的判断是：

**DART
有创新性，但目前的创新主要体现在已有生成机制证据的联合建模和跨模型应用上，而不是提出了全新的取证信号。它有
ICML 投稿潜力，但如果以目前的形式直接投稿，创新性很可能成为主要短板。**

这里有一个容易产生的误区：

一篇高水平机器学习论文，不一定必须发现一个全新的数据特征，但必须提出一个新的、有说服力的科学认识或解决问题的方式。

ICML 2027 的官方 Reviewer Instructions 也明确指出：

-   Originality 不要求必须提出完全全新的方法；
-   创造性地组合已有技术也可能具有原创性；
-   对已有方法进行新的评估，或者揭示此前没有充分理解的性质，也可以构成原创贡献。

因此，你的担忧是合理的，但不能简单地得出 DART 没有创新的结论。

真正需要判断的是：

**DART
是简单地把三个已有信号放在一起，还是发现了三个信号之间此前没有被利用的关系？**

这两者的学术价值差别很大。

------------------------------------------------------------------------

# 三、站在 ICML 审稿人的角度，如何评价目前的 DART？

假设审稿人研究生成图像检测、生成模型和机器学习方法，目前 DART
的方法可以概括为：

## 3.1 当前方法组成

  相关工作          已有证据                                 DART 中的使用
  ----------------- ---------------------------------------- -------------------------------
  PRADA             Token Likelihood Distribution            Likelihood 分布特征
  D³QE              Quantization Error / Codebook Evidence   量化误差
  Data Provenance   Decoder Reconstruction Evidence          Decoder 重建误差
  DART              多尺度联合建模                           Deep Sets + Attention Pooling

目前实验结果：

-   七家族 Macro AUROC：0.8456；
-   同 bank PRADA：0.6101。

## 3.2 审稿人可能提出的问题

  审稿问题                         当前 DART 的回答程度
  -------------------------------- ----------------------------------
  为什么需要三个信号？             有消融实验支持
  三个信号是否存在互补性？         有初步证据
  为什么需要 Deep Sets？           有一定实验，但优势不充分
  为什么需要 Attention Pooling？   目前没有证明显著优于其他聚合方式
  是否真正解决了未知生成器问题？   有七家族实验，但仍有范围限制
  是否有新的理论或统计机制？       目前不足
  性能提升是否来自新机制？         尚未完全排除训练数据来源混杂

这里最危险的是最后一个问题。

你目前七家族实验采用 DARG fake 训练、ARForensics fake
测试。需要进一步排除训练阶段 real/fake
标签与数据集来源之间的关联，否则审稿人可能质疑模型学到的是数据来源差异，而不是生成机制差异。

因此，0.8456
的结果很有价值，但不能仅凭这个数字证明生成循环一致性理论成立。

## 3.3 初步审稿结论

如果现在提交，可能出现这样的评价：

-   **Soundness：** 有较完整实验，但数据来源混杂需要控制。
-   **Significance：** 生成图像检测问题重要。
-   **Originality：** 三个已有信号的融合，方法新颖性不够突出。
-   **Empirical evidence：** 性能较好，但跨家族和低 FPR 仍有局限。

DART 目前最大的风险不是性能，而是：

**审稿人认为它是一项有效的工程组合，而不是一个新的机器学习方法。**

------------------------------------------------------------------------

# 四、最应该增加的不是新信号，而是新机制

这是下一阶段最重要的研究调整。

目前 DART 已有三个信号：

\[ L=`\text{Likelihood}`{=tex} \]

\[ Q=`\text{Quantization}`{=tex} \]

\[ R=`\text{Reconstruction}`{=tex} \]

但目前的核心建模方式，本质上还是：

\[ S(x)=f(L,Q,R) \]

即把三个特征输入神经网络，让网络自己学习如何分类。这属于 feature
fusion，还没有充分回答：

**为什么一张由 AR 模型生成的图像，其 Likelihood、Quantization 和
Reconstruction 之间应该具有某种特定的联合关系？**

这才是值得研究的创新。

------------------------------------------------------------------------

# 五、三个值得研究的真正创新方向

## 方向一：Generation-Cycle Interaction Modeling

`<badge>`{=html}最推荐方向`</badge>`{=html}

中文：生成循环交互关系建模。

### 5.1 核心思想

目前 DART
主要利用三个证据各自的数值。但是三个数值之间可能存在更重要的信息：

**一个 AR 模型生成的图像，其 token likelihood、quantization 和 decoder
reconstruction
不仅分别具有某种统计分布，而且它们之间可能存在特定的条件依赖关系。**

例如，两张图像具有相似的 token likelihood：

\[ L(x_1)`\approx `{=tex}L(x_2) \]

但是：

\[ R(x_1)`\ll `{=tex}R(x_2) \]

这说明它们在 token 概率方面相似，但 decoder reconstruction 表现不同。

可以进一步研究：

-   高 likelihood 是否应该对应某种 reconstruction error？
-   不同 quantization error 下，reconstruction error 应该如何变化？
-   不同 scale 的 likelihood 与 reconstruction 是否具有稳定的对应关系？

这些问题不是简单拼接特征能够直接回答的。

### 5.2 方法设计：Generation-Cycle Interaction Module

建议构建一个候选模块：

**Generation-Cycle Interaction Module（GCIM）**

对于第 (k) 个 scale，定义：

\[ z_k=\[L_k,Q_k,R_k\] \]

首先学习各阶段的基础表示：

\[ h_k\^L=f_L(L_k) \]

\[ h_k\^Q=f_Q(Q_k) \]

\[ h_k\^R=f_R(R_k) \]

然后显式建模交互：

\[ h_k\^{LR}=`\phi`{=tex}\_{LR}(h_k^L,h_k^R) \]

\[ h_k\^{LQ}=`\phi`{=tex}\_{LQ}(h_k^L,h_k^Q) \]

\[ h_k\^{QR}=`\phi`{=tex}\_{QR}(h_k^Q,h_k^R) \]

进一步建模跨尺度交互：

\[ h\^{cross}= `\operatorname{CrossScaleInteraction}`{=tex}
({h_k}\_{k=1}\^{K}) \]

最终：

\[ S(x)= f`\left`{=tex}( h^L,h^Q,h\^R, h^{LR},h^{LQ},h^{QR},h^{cross}
`\right`{=tex}) \]

真正值得研究的不是增加几个 MLP，而是明确建立一个假设：

> AR-generated images exhibit characteristic cross-stage statistical
> dependencies that are not fully captured by marginal likelihood,
> quantization, or reconstruction statistics.

中文：

AR 生成图像具有特定的跨生成阶段统计依赖关系，而这些关系无法被单独的
likelihood、quantization 或 reconstruction 分布充分表达。

### 5.3 如何证明创新？

需要设计以下对比：

  方法   使用的信息
  ------ ----------------------------
  M1     Likelihood-only
  M2     Reconstruction-only
  M3     Quantization-only
  M4     L+R concatenation + MLP
  M5     L+Q+R concatenation + MLP
  M6     L+R explicit interaction
  M7     L+Q+R explicit interaction
  M8     Cycle residual + L+R

所有模型必须：

-   使用相同训练数据；
-   使用相同数据划分；
-   参数量尽可能匹配；
-   使用相同优化预算；
-   使用相同 calibration；
-   在四家族和七家族设置下配对测试。

关键判定标准：

如果 M6/M7/M8 无法稳定超过
M4/M5，就没有足够证据证明显式交互机制带来了新的信息。

如果它们能够在未知家族、低 FPR、多个 codec
和独立真实来源上稳定改善，才有充分理由提出新的交互机制。

## 方向二：Generation-Cycle Consistency Residual

中文：生成循环一致性残差。

这是更有机会形成独立新检测分数的方向。

### 5.4 核心思想

目前 DART 使用：

-   Likelihood 的原始分布；
-   Quantization 的原始误差；
-   Reconstruction 的原始误差。

但可以研究：

**对于给定的 likelihood 和 quantization 状态，实际 reconstruction error
是否符合模型预期？**

不仅观察 reconstruction error
大不大，还观察它是否与其他生成阶段的证据相匹配。

### 5.5 候选新信号设计

先建立一个条件预测模型：

\[ `\hat `{=tex}R_k=g(L_k,Q_k) \]

其中 (`\hat `{=tex}R_k) 是根据 likelihood 和 quantization 预测的
reconstruction error。

定义：

\[ E_k\^{cycle} = R_k-`\hat `{=tex}R_k \]

这个量表示实际 decoder reconstruction 与根据前两个阶段预测的
reconstruction 之间的偏离。

进一步定义归一化残差：

\[ Z_k\^{cycle} = `\frac{R_k-\hat R_k}`{=tex}
{`\hat{\sigma}`{=tex}\_k(L_k,Q_k)+`\epsilon`{=tex}} \]

由此得到一个新的候选信号：

**Generation-Cycle Consistency Residual。**

注意：这目前是一个研究假设，不是已经证明的生成图像定律。真实图像与目标
AR decoder
之间也可能天然存在较大的重建误差，因此需要通过真假配对实验确认残差的方向和分布。

### 5.6 为什么这个方向可能有创新？

PRADA 研究 likelihood。

D³QE 研究 quantization。

Data Provenance 研究 decoder reconstruction。

而这里研究的是：

**给定前两个生成阶段的统计表现，第三个生成阶段是否符合预期？**

这从边缘信号检测转变成条件关系检测。

如果这种残差能够在未见生成器家族中稳定区分真实图像和生成图像，就有机会成为一项独立的取证信号。

### 5.7 必须完成的验证

1.  在 real train 上学习条件预测关系；
2.  在独立 real validation 上检查残差稳定性；
3.  在 fake validation 上检查残差是否存在系统性偏离；
4.  比较普通 reconstruction error 与 cycle residual；
5.  在未知生成器家族上验证；
6.  在 JPEG95/85/75 下验证；
7.  做 source-disjoint 配对实验。

关键消融：

\[ AUROC(R) \]

\[ AUROC(L,R) \]

\[ AUROC(E\^{cycle}) \]

\[ AUROC(L,R,E\^{cycle}) \]

如果 cycle residual 在 L+R 之外仍然提供稳定增益，才值得作为 ICML
论文中的新信号。

## 方向三：Coverage-Aware Autoregressive Forensics

中文：参考模型覆盖感知的自回归取证。

这是基于七家族实验产生的方向。

### 5.8 核心发现

  Bank               Macro AUROC
  ---------------- -------------
  VAR only            0.778 左右
  VAR + LlamaGen      0.824 左右
  三模型              0.832 左右
  四模型              0.814 左右

说明：

-   增加参考模型通常有帮助；
-   但更多模型不一定更好；
-   参考模型的具体组成会改变检测结果；
-   Max aggregation 可能造成不同模型之间的竞争。

### 5.9 方法设计

目前使用：

\[ S(x)=`\max`{=tex}\_{m`\in `{=tex}B}S_m(x) \]

Max 只选择最高分，没有显式考虑：

-   当前参考模型与目标图像的生成机制相似程度；
-   不同参考模型的证据相关性；
-   参考模型是否覆盖目标生成家族；
-   加入某个参考模型对真实误报的影响。

可以考虑：

\[ S(x)= `\sum`{=tex}\_{m`\in `{=tex}B} w_m(x)S_m(x) \]

其中：

\[ w_m(x)= `\frac{\exp(a_m(x))}`{=tex}
{`\sum`{=tex}\_{j`\in `{=tex}B}`\exp`{=tex}(a_j(x))} \]

但普通 Softmax 不是创新。

真正需要研究的是：如何基于独立可测量的生成机制属性，构造参考模型之间的
coverage 和 diversity 关系，并学习一个有风险控制的聚合方式。

例如：

-   Tokenizer 相似性；
-   Codebook 相似性；
-   Decoder 结构；
-   Scale structure；
-   Token ordering。

最终可以建立：

\[ `\text{Detection Risk}`{=tex} =
f(`\text{Coverage}`{=tex},`\text{Diversity}`{=tex},`\text{Aggregation}`{=tex})
\]

需要注意：目前七家族结果没有支持简单的同范式优于异范式规律。因此，不能预先假设同范式模型权重一定更大。

------------------------------------------------------------------------

# 六、三个方向应该如何选择？

不建议三个方向同时开展。

  ----------------------------------------------------------------------------------------
  方向               创新潜力       实验基础         主要风险               推荐程度
  ------------------ -------------- ---------------- ---------------------- --------------
  Generation-Cycle   高             较好             可能只是复杂特征融合   第一
  Interaction

  Cycle Consistency  高             需要新增实验     新残差可能没有判别力   第二
  Residual

  Coverage-aware     中高           已有七家族实验   可能只是集成优化       第三
  Forensics
  ----------------------------------------------------------------------------------------

**最推荐方向一和方向二结合，但只保留一个核心方法贡献。**

具体来说：

-   用 Generation-Cycle Interaction 作为主要建模框架；
-   用 Cycle Residual 作为一个具有明确物理/统计解释的候选新信号；
-   将 Model-bank Coverage 作为独立的泛化分析，而不是强行塞进主网络。

------------------------------------------------------------------------

# 七、最应该先做的关键实验：Feature Fusion vs. Mechanism Interaction

在真正开发新模块之前，建议先做这个实验。

目的：回答 DART 是否只是把已有信号整合起来。

  模型   设计
  ------ ----------------------------
  M1     Likelihood-only
  M2     Reconstruction-only
  M3     Quantization-only
  M4     L+R concatenation + MLP
  M5     L+Q+R concatenation + MLP
  M6     L+R explicit interaction
  M7     L+Q+R explicit interaction
  M8     Cycle residual + L+R

所有模型必须：

-   使用相同训练数据；
-   使用相同数据划分；
-   参数量尽可能匹配；
-   使用相同优化预算；
-   使用相同 calibration；
-   在四家族和七家族设置下配对测试。

`<badge>`{=html}关键判定标准`</badge>`{=html}

如果 M6/M7/M8 无法稳定超过
M4/M5，就没有足够证据证明显式交互机制带来了新的信息。

如果它们能够：

1.  在未知家族上稳定提升；
2.  在低 FPR 工作点改善；
3.  在多个 codec 下保持；
4.  在独立真实来源上校准有效；

那么就有机会提出一个比当前 DART 更有说服力的创新点。

------------------------------------------------------------------------

# 八、从 ICML 2027 投稿标准来看，应该如何定位？

ICML 2027 的论文贡献类型包括
General、Theory、Application-driven、Concept and Feasibility、Negative
Results、Datasets/Evaluations/Benchmarks 等。

官方明确指出：

-   Application-driven
    工作可以通过针对重要实际问题，对已有方法进行有创造性的组合而产生原创贡献；
-   Datasets/Evaluations/Benchmarks
    不一定需要新算法，也可以通过新评测协议和新的科学发现产生贡献；
-   但所有类型都要求高水平的科学质量。

对 DART 有两种投稿路线。

## 路线 A：General（更推荐）

前提：完成 Generation-Cycle Interaction 或 Cycle
Residual，形成真正的新方法。

论文重点：

-   新的生成循环交互机制；
-   新的条件一致性残差信号；
-   跨生成器家族泛化；
-   理论和实验验证。

## 路线 B：Datasets, Evaluations and Benchmarks

如果最终无法证明新的交互方法，但能够建立：

-   严格的 AR 生成器家族留出协议；
-   系统的 model-bank coverage benchmark；
-   跨 codec 的失效分析；
-   严格的 FPR/coverage 校准协议；
-   可复现的大规模评测资源；

可以考虑这一类型。

但它不能只是把现有实验整理成表格。必须有新的评测方法或能够改变领域认识的实质性发现。

------------------------------------------------------------------------

# 九、DART 现在能不能发 ICML？

## 9.1 当前版本

**有创新，但方法创新不足。**

-   三个核心信号都有前人工作基础；
-   目前主要是特征整合；
-   Deep Sets 和 Attention 不是新的方法原理；
-   有不错的跨家族实验；
-   有 model-bank coverage 的新发现。

## 9.2 改进后的版本

如果能够证明以下内容：

-   显式建模跨生成阶段的条件依赖；
-   提出新的 generation-cycle consistency residual；
-   证明交互证据超过简单特征拼接；
-   在严格未知家族上验证；
-   给出理论解释不同生成阶段证据的互补性；

那么方法创新性会明显增强。

这不是录用保证，但会改善论文的创新性论证。

------------------------------------------------------------------------

# 十、最终建议

考虑到 ICML 2027 完整论文截止日期为 2027 年 1 月 22 日，时间并不充裕。

不建议再花大量时间寻找一个全新的图像取证特征，而应集中回答：

**为什么 AR 生成图像的 Likelihood、Quantization 和 Reconstruction
之间存在值得利用的联合关系？**

希望最终论文能够从：

> We combine likelihood, quantization, and reconstruction features for
> AR image detection.

升级为：

> We identify and exploit cross-stage statistical dependencies in
> autoregressive image generation, showing that generation-cycle
> interactions provide forensic information beyond marginal likelihood,
> quantization, and reconstruction signals.

中文：

我们发现并利用了自回归图像生成过程中不同阶段的统计依赖关系，证明生成循环中的交互信息能够提供超出单独
likelihood、quantization 和 reconstruction 信号的取证信息。

**最终执行建议：**

1.  保留 Full DART 作为 baseline，不立即修改主模型。
2.  先用小规模、严格配对实验验证 Generation-Cycle Interaction。
3.  同时验证 Cycle Residual 是否提供 L+R 之外的独立信息。
4.  只有在未知家族、低 FPR、跨 codec
    和独立真实来源上稳定有效，才发展成新的主方法。
5.  如果交互机制没有独立增益，不要为了创新强行增加复杂模块；重新评估
    DART 方法论文或评测论文的投稿定位。

这比继续堆叠特征、增加网络层数或单纯追求更高 AUROC 更有价值。
