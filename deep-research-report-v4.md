# 面向 ICML 2027 的 DCA-Fuse 重构评估与中文论文初稿

## ICML 投稿标准与当前工作的定位

截至 2026 年 11 月 23 日，ICML 2027 的官方 Call for Papers 已经公布。主会论文要求“原创、严谨、对机器学习社区具有显著兴趣”，所有核心 claim 必须由可复现实验或可靠理论分析支持；投稿采用双盲评审，正文限 **8 页**，参考文献、Impact Statement 和 Appendix 不计页数，录用后正文可增加 1 页。ICML 2027 还新增了两阶段评审：第一阶段由 AC/SAC 判断论文是否具备进入完整评审的竞争力，因此“核心 idea 是否足够清晰、贡献是否一眼可辨、实验是否明显完整”比过去更重要。citeturn293164search0turn947861search5

ICML 的正式评审维度包括 **soundness、presentation、significance、originality**；Accept 级论文通常需要技术上扎实、实验和可复现性充分，并且对至少一个 AI/ML 子领域具有较高影响。ICML 2027 还要求作者选择贡献类型，包括 General、Theory、Application-driven、Concept and Feasibility、Datasets/Evaluations/Benchmarks 等。仅仅“把已有算法用于一个新视觉任务”通常不足以构成强 Application-driven 贡献，而原创性也可以来自新的问题理解、原理发现或有充分动机的新组合。citeturn947861search0turn947861search3

因此，**当前 DCA-Fuse 如果直接以“一个 AR 图像检测算法”投稿，我认为更接近 Weak Reject；若把核心重新提升为“多生成机制下的 tokenizer-affinity / minimum-normalized discrepancy 原理”，并完成真正 unseen family、跨数据集以及理论分析，则有机会达到 Weak Accept—Accept 区间。** 当前最大的 ICML 风险不是实现错误，而是 reviewers 很容易把它概括成：“UnivFD score + 五个 tokenizer 的 quantization-error minimum，再做线性加权”。这一概括一旦成立，originality 和 significance 会偏弱；仓库现有 full-scale 结果也尚不能支撑“总体性能提升”的 claim。该判断依据 ICML 的 originality/significance 标准以及当前仓库的全量结果。citeturn947861search3turn634291view4

## 六个问题的核心回答

1. **新方法的设计思路。** DCA-Fuse 建立在一个“生成器—tokenizer 亲和力”假设上：真实图像相对于不同 VQ tokenizer 的量化空间通常没有特别强的匹配关系，而由特定 AR 家族产生的图像可能在与该家族相关的 tokenizer 下出现异常低的 quantization error。方法因此准备多个冻结 tokenizer，对每张图计算 q-error，并用真实验证集为每个 tokenizer 建立均值/方差参考，得到 \(z_k(x)\)；随后以 \(s_\text{CTAD}(x)=-\min_k z_k(x)\) 作为机制通道得分。最终再将该得分与 UnivFD 的 CLIP 语义检测得分做 real-reference z-normalization，并以 \(\alpha=0.7\) 加权融合。当前 tokenizer bank 包含 LlamaGen、VAR、RAR、Open-MAGVIT2 和 Switti 五个 tokenizer。citeturn807771view0turn634291view0

2. **真正的创新点。** 最有价值的创新不是“两个 score 融合”，而是**把单 tokenizer 上 q-error 的跨生成器方向翻转重新解释为 tokenizer affinity，并通过多参考 tokenizer 的 order-statistic/min aggregation 将生成器特异信号转成统一异常分数**。仓库的 token 分析表明，entropy、gini、transition、q-error 等单变量在 7 个生成器上的 AUC 大约只有 0.37–0.63，且方向会翻转；DCA-Fuse 因而不再学习统一的 q-error 方向，而寻找“是否存在至少一个 tokenizer 使该样本异常贴合”。这一想法比“融合 UnivFD”更接近可写成 ICML 核心贡献的部分。citeturn634291view4turn807771view0

3. **相对老方法的优势与不足。** UnivFD 使用预训练视觉语言特征并做线性 probing，优势是通用、简单、跨生成器能力较强，但仍会依赖训练分布与低层数据 shortcut；仓库中统一 JPEG95 后其 7k 子集 AUC 从 0.953 降至 0.889，说明格式因素确实影响结果。D³QE 则显式利用单 tokenizer 的离散 codebook frequency discrepancy、quantization-error latent 和语义特征，并通过 discrepancy-aware transformer 建模；它是学习式 detector，并不是简单 q-error threshold。PRADA 更进一步直接利用目标 AR 模型的条件/无条件 token 概率比，因此信号极强，但需要访问候选生成模型的内部概率模型并进行模型特异校准。Data Provenance 则属于更明确的 source-attribution / white-box 路线，通过目标模型 decoder 的近似逆映射提取 QuantLoss 和 EncLoss，对来源模型做 provenance tracing。相比这些方法，DCA-Fuse 的优势在于**不需要完整 AR 生成模型，也不需要 fake 数据来训练 CTAD 机制通道，并能够同时兼容多个 tokenizer 家族**；但其缺点同样明显：tokenizer bank 本身仍包含与被测试生成家族高度相关的已知 tokenizer，因此目前不能把它描述成严格的“unknown-model open-set detection”。citeturn683047search0turn959856academia10turn683047search6turn683047search3

4. **目前真实能够支持的实验效果。** 7k 的 subsample-500 实验中，DCA-Fuse overall AUROC 为 **0.953，95% bootstrap CI [0.950, 0.956]**，和 UnivFD 的 0.953 基本一致；Open_MAGVIT2 从 0.911 提升到 0.927，且该小样本实验中 worst-generator AUROC 从 0.911 提升到 0.921。这个结果只能被称为探索性或机制验证结果。更关键的 full ARForensics 84k 实验中，UnivFD overall AUROC 为 **0.9127，DCA-Fuse 为 0.9128**，几乎没有总体提升；Open_MAGVIT2 从 0.8863 提高到 **0.9053**，但 VAR 0.9560→0.9490、Switti 0.8507→0.8431、Infinity 0.8305→0.8244 都发生下降。因此当前最准确的 claim 是：**DCA-Fuse 在特定 tokenizer-aligned generator 上带来显著互补收益，但尚未实现稳定的 overall improvement。**citeturn634291view1turn634291view4

5. **目前已经完成什么实验，以及下一步应该做什么。** 现有工作已经完成了较系统的 baseline 复现和审计：CNNDetection、FreqSpec、NPR-inspired、UnivFD、LGrad、RIGID、ZED、DIRE、D³QE 官方代码、PRADA 官方代码和 Data Provenance sampling；框架具备 AUROC、AP、低 FPR 指标、bootstrap CI、validation threshold calibration、逐样本 prediction 归档以及 SHA256/pHash 数据审计。DCA-Fuse 还做了 tokenizer affinity feasibility、LR/MLP 与 S_min 的 LOGO 对比、JPEG95 格式对照、聚合方式消融、tokenizer subset 消融、7k 定型实验和 84k full test。citeturn961017view0turn807771view2turn634291view1

6. **如果按 ICML 写论文，核心叙事必须改变。** 不建议把标题和贡献写成“DCA-Fuse: a better detector”，因为 full-scale 数据并不支持这个结论。更合理的科学问题是：**为什么单一 latent-space forensic statistic 在不同生成器间会发生 sign reversal？是否可以利用多个 reference quantizers，把 source-specific affinity 转化为 generator-agnostic evidence？** 这样论文的主要贡献从“融合模型”变成“发现并形式化一种 generative affinity phenomenon，并设计一个无需 fake training 的 order-statistic detector”，UnivFD fusion 可以退居为 downstream instantiation。这个定位更接近 ICML 对原创 insight 和广义机器学习价值的要求。citeturn947861search3turn634291view4

## 与 D³QE、Data Provenance 及相关工作的精准区分

D³QE 与 DCA-Fuse 都涉及 quantization error，但二者的研究对象实际上不同。D³QE 的核心出发点是**真实与 AR 生成图像在离散 codebook activation distribution 上存在系统差异**；它使用 dynamic codebook frequency statistics 改造 Transformer attention，再融合 semantic features 与 quantization-error latent，是一个经过 fake 数据训练的 discriminative detector。DCA-Fuse 当前没有 D³QE 的 discrepancy-aware attention、频率建模或 learned fusion；它更像是研究“一个样本相对于多个预训练 quantizer 的投影距离集合”这一跨模型几何现象。因此论文里不能写成“D³QE 的训练自由改进版”，而应强调问题设定不同：D³QE 学习 real/fake discrepancy，DCA-Fuse 试图寻找 reference-bank 中的 extremal affinity。citeturn683047search0turn683047search1

Data Provenance 与 DCA-Fuse 的重叠更值得警惕。Data Provenance 已经观察到：IAR 生成图像的 latent representations 往往离源模型 codebook entry 更近，并把这一现象定义为 QuantLoss；其方法进一步训练目标 decoder 的逆映射，并结合 EncLoss 和 token-search 来完成来源模型归因，论文报告的主要任务是 provenance tracing，而不仅仅是 binary detection。也就是说，“generated image 对源 tokenizer/codebook 更贴近”本身**不能再作为 DCA-Fuse 的独立 novelty claim**；真正需要证明的新东西是：**当不知道 source model 时，多 reference tokenizer 上的相对 affinity 是否具有可泛化的 statistical structure，以及这种 structure 是否能在 source tokenizer 缺失时仍成立。**citeturn683047search6turn683047search2

PRADA 同样给出了一个重要提醒：不同 AR 模型上的概率比信号并不具有统一方向或形状，PRADA 因而学习 model-specific scoring function；其 supplemental 明确展示了不同生成器需要非常不同的 token-wise score。DCA-Fuse 中观测到的 q-error sign reversal 与这个现象在概念上是相呼应的。区别在于 PRADA 使用目标生成器自身的 autoregressive likelihood，而 DCA-Fuse 试图用外部 tokenizer bank 逼近这种“source compatibility”。如果能够把这种联系形式化，这会明显增强论文的概念深度。citeturn959856search9turn959856academia10

从更广泛的生成图像检测文献来看，UnivFD 的核心经验是使用一个**并非专门为 fake detection 训练的通用视觉特征空间**能够提高 unseen-model generalization；Manifold-induced Biases 则进一步通过 pretrained diffusion score function 的 manifold geometry 建立 zero-shot 检测准则，明确把理论解释作为贡献；AIGIBench 说明很多 detector 在受控数据上看似很高的结果，在多源、退化和现实分布上会显著下降。ICML 审稿人因此很可能要求 DCA-Fuse 不仅“比 baseline 高一点”，而要证明它揭示的是一种可迁移的生成机制属性。citeturn683047search3turn959856search11turn959856search4

## 从 ICML 角度看，当前最需要补的实验

**必须补。** 第一是 **leave-one-tokenizer-family-out**：当测试 Open-MAGVIT2 时，tokenizer bank 中必须移除 Open-MAGVIT2；测试 RAR 时移除 RAR；测试 VAR 时移除 VAR。更严格的是按 family hold out，而不是单模型 hold out。如果移除同源 tokenizer 后仍然优于 UnivFD，才有资格支持“open-set affinity”的 claim；否则方法应被诚实地定义为“reference-bank assisted detection”。当前 bank 含 LlamaGen、VAR、RAR、Open-MAGVIT2、Switti，而测试集也包含这些家族，因此这是当前论文最关键的实验缺口。citeturn807771view0turn634291view2

第二是 **真正 unseen generator family + external dataset**。至少应在 DARG 上评估不参与 tokenizer bank/α 设计的 AR 模型，并最好额外构造一组新的 2026–2027 AR/IAR 模型。DARG 官方扩展版包含约 155k 图像、20 个 AR generators，非常适合作为 external validation；目前仓库只对 PRADA 做了 tiny-50 适配实验，还没有 DCA-Fuse 的系统跨数据集结果。citeturn959856search0turn959856search2turn634291view3

第三是把 **format/corruption robustness 提升到 full scale**。ARForensics 中真实图像全部为 JPEG、fake 全部为 PNG，这已经被仓库审计确认；目前统一 JPEG95 的 7k UnivFD 对照出现约 0.064 AUROC 降幅，所以所有主表都应同时报告 original、统一 JPEG95、JPEG75/50、resize、crop、blur、WebP、截图或社交媒体压缩版本。否则 ICML reviewer 很容易认为结果受数据 shortcut 污染。citeturn634291view4turn959856search4

第四是统计显著性。由于 full-scale DCA-Fuse 与 UnivFD 的 overall AUROC 是 0.9128 vs 0.9127，普通 bootstrap CI 并不能回答两种方法差异是否显著；应该基于**相同图片的 paired bootstrap**计算 \(\Delta\)AUROC 的置信区间，并补充 DeLong test。每个 generator 也应给 paired CI，而不是只给单方法 CI。若 \(\Delta\) 的 95% CI 跨 0，就不能声称 overall improvement。当前数据更合理的表述是“complementary redistribution of errors”。citeturn634291view4

**重要但可以放次优先级。** α=0.7 当前来自分析扫描，因此存在 test-informed hyperparameter selection 风险；正式实验应将 α 的确定完全限制在 train/val，或者采用无需 fake calibration 的解析融合规则。还应报告 α sensitivity、bank size K、min/top-k/softmin、不同 real reference size、不同 tokenizer initialization/版本，并做至少 3–5 个 seed 的 UnivFD head 重训。仓库自己的方法文档已经明确指出 α 是分析扫描值、当前阈值由 LlamaGen-only val 校准，这些都需要在论文版 protocol 中解决。citeturn634291view2

此外，应增加 **source-family attribution accuracy / top-k attribution**、ECE/Brier score、FPR@95%TPR、TPR@0.1%/1%/5%FPR、推理 FLOPs/latency/显存、bank storage 成本以及 failed cases。由于 DCA-Fuse 的 `argmin_k z_k` 天然产生一个 tokenizer-family prediction，如果这个 prediction 真能对应 source family，那么它将把论文从普通二分类器提升为更有解释力的 mechanism-aware model；如果归因准确率很低，也应该如实把它作为对 affinity 假设边界的检验。当前仓库已经具备低 FPR 评估和逐样本 score 归档，因此这些实验在工程上是可实现的。citeturn961017view0turn634291view0

## 如何把方法提升到更有 ICML 分量的核心算法

目前最值得升级的不是再加一个 backbone，而是把 CTAD 从 heuristic 的 `-min z` 提升成一个**可分析的 reference-distribution model**。可以令每个 tokenizer \(T_k\) 定义一个真实图像距离随机变量 \(Q_k(X)\)，用 robust location/scale 或 empirical CDF 将其转换为 \(u_k(x)=F_k(Q_k(x))\)。如果生成图像会在至少一个 compatible tokenizer 下进入真实分布的极端低尾，则检测问题可以写成多重异常检测：

\[
S(x)=\max_k[-\log \hat F_k(Q_k(x))].
\]

这种写法比 z-score 的 Gaussian 假设更稳健，而且可以研究在 null hypothesis“real image”下 \(S(x)\) 的极值分布，从而给出理论 FPR 控制或 conformal p-value。这样方法的 ML 贡献会从“经验 fusion”提升成“multi-reference latent-space anomaly detection”。这一方向与 ICML 偏好的 sound theoretical analysis 更契合。ICML 官方明确允许 claim 由实验或理论支撑，但对于一个性能仅和 UnivFD 持平的新方法，理论性质会显著提高贡献密度。citeturn293164search0turn947861search3

第二个可以形成真正 novelty 的方向是 **bank generalization**。不再把 tokenizer 当作五个手工 detector，而是把它们看成 latent quantization operators，从中学习 generator-independent invariants，例如 normalized reconstruction geometry、local curvature、multi-scale residual spectrum 或 cross-tokenizer rank pattern。如果一个 unseen source 即使没有自己的 tokenizer，也会表现为“在生成 tokenizer manifold 集合上比真实图像更接近”，那么就可以证明方法并非 closed-bank matching。这个问题与 ICLR 2025 的 manifold-bias 工作在科学风格上更接近：后者并非只是提出分数，而是解释为何生成样本偏向模型 manifold，并据此构建检测器。citeturn959856search11

如果以上两个方向来不及完成，我建议投稿时**降低 claim 强度**：把 DCA-Fuse 描述成“reference-bank-assisted AR detection and diagnostic study”，重点贡献改为发现 sign reversal、格式 shortcut、source affinity 以及现有 benchmark 的 protocol fragility。ICML 2027 的 “Datasets, Evaluations and Benchmarks” 类型允许高质量评估工作成为主要贡献，但那时就必须把 benchmark 覆盖、协议严谨性和可复现性做到非常强，而不能只围绕一个 fusion method。citeturn947861search0

## 中文论文初稿

**候选标题 A：** *From Sign Reversal to Source Affinity: Multi-Tokenizer Reference Modeling for Autoregressive Image Detection*

**候选标题 B：** *DCA-Fuse: Detecting Autoregressive-Generated Images via Cross-Tokenizer Affinity*

**候选标题 C，更适合 ICML：** *Multi-Reference Quantization Affinity for Open-World Autoregressive Image Detection*

**摘要。** 视觉自回归生成模型通过离散 token 预测实现高质量图像生成，但其快速迭代使依赖固定生成器训练数据的检测器难以泛化。我们首先对多个 AR 图像检测器进行系统分析，并发现一个此前未被充分研究的现象：真实图像与生成图像之间的 quantization-error discrepancy 并不存在跨生成器统一方向，而是随参考 tokenizer 与生成器家族的匹配关系发生显著翻转。基于这一观察，我们提出 DCA-Fuse，其核心机制通道 CTAD 利用多个冻结 VQ tokenizer 对输入图像进行量化，并相对于 real-only reference distribution 对量化误差进行标准化，再通过跨 tokenizer 的极端亲和力构造生成证据。该机制信号随后可与通用视觉语义检测器融合。ARForensics 全量实验表明，当前版本 DCA-Fuse 的整体 AUROC 为 0.9128，与 UnivFD 的 0.9127 基本持平，但在 Open-MAGVIT2 上由 0.8863 提升至 0.9053，说明机制亲和力能够补充视觉语义检测器的部分盲区；同时，小规模格式控制实验表明 CTAD 对 JPEG shortcut 的敏感度低于 UnivFD。我们进一步分析了单 tokenizer、supervised affinity fusion 和 multi-tokenizer aggregation 的失败模式，并讨论其在真正未知生成器上的适用边界。[待补实验：leave-one-tokenizer-family-out、DARG external validation、full-scale corruption robustness 和 paired significance tests]。citeturn634291view4turn634291view1

**引言。** 近年来，视觉生成从 GAN 和 diffusion 快速扩展至 autoregressive image generation。传统检测方法往往依赖纹理、频率、上采样 artifact 或判别式视觉表征；例如 UnivFD 使用预训练视觉语言特征提高跨模型泛化，NPR 利用上采样造成的邻域像素关系，而 LGrad 通过预训练模型的输入梯度构造广义 artifact 表征。然而，最新 benchmark 表明，高质量 detector 在数据域、图像退化和生成器发生变化后仍可能出现显著性能下降。citeturn683047search3turn959856search6turn683047search8turn959856search4

针对 AR 图像，D³QE 利用 codebook frequency discrepancy 和 quantization error，PRADA 利用 autoregressive conditional/unconditional likelihood ratio，而 Data Provenance 利用 source codebook proximity 和 decoder inversion。这些工作共同说明，AR 生成过程本身包含比通用视觉 artifact 更直接的 forensic signal，但通常需要 fake training data、特定 tokenizer 或目标生成器的白盒访问。citeturn683047search0turn959856academia10turn683047search6

我们在 ARForensics 上重新分析 token statistics 时发现，一个简单但重要的现象：同一个 quantizer 下，real–fake quantization-error 差异在不同生成器上可以发生符号翻转。单变量 entropy、gini、transition 和 q-error 的跨生成器 AUC 仅约 0.37–0.63，因此“一个 tokenizer + 一个全局方向”的检测规则难以泛化。进一步分析表明，当参考 tokenizer 与某些生成器家族具有较高结构兼容性时，生成图像更容易落入该 tokenizer 的低 quantization-error tail。citeturn634291view4turn807771view0

基于这一发现，我们提出多参考 tokenizer affinity 框架。与学习“fake 长什么样”不同，我们针对真实图像为每个 tokenizer 建模 reference distribution，并询问：一个测试图像是否对任意生成 tokenizer 表现出异常强的 latent-space affinity？这一视角将 source-specific、direction-dependent 的 q-error 转换为统一的 extremal anomaly evidence。当前实现进一步将该机制通道与 CLIP-based semantic detector 融合，以研究两类证据的互补性。citeturn807771view0

本文当前版本的贡献可以准确表述为：第一，我们系统揭示了 AR 检测中 quantization discrepancy 的 generator-dependent sign reversal，并将其解释为 cross-tokenizer affinity phenomenon；第二，我们提出基于 real-reference normalization 与 multi-tokenizer extremal aggregation 的 CTAD detector，该通道不使用 fake 图像训练；第三，我们通过 ARForensics、格式控制、LOGO 和 tokenizer ablation 分析其适用条件和失败模式；第四，我们发现该机制通道可以显著改善特定 tokenizer-aligned generator，但当前 full-scale 结果尚未产生总体 AUROC 提升，因此我们不宣称新的总体 SOTA。[待补实验后再决定是否加入“open-world”或“SOTA”表述]。citeturn634291view1turn634291view4

**相关工作。** 通用生成图像检测主要包括 discriminative artifact learning、预训练表示和生成模型反演三类路线。UniversalFakeDetect 表明，冻结的通用视觉语言表征加简单分类器能够比专门训练的 fake feature 更好地泛化至未知生成器；LGrad 和 NPR 则分别从输入梯度和上采样邻域结构中提取可迁移 forensic cue。另一方面，零样本工作开始利用 pretrained generative model 的 manifold geometry 构建检测统计量，强调对生成数据偏差的机制理解而非纯监督分类。citeturn683047search3turn683047search8turn959856search6turn959856search11

AR 专用检测进一步利用离散 latent structure。D³QE 将 dynamic codebook frequency statistics 引入 attention，并融合语义与 quantization-error representation；PRADA 通过条件与无条件 token probability ratio 为不同源模型学习 scoring function；Data Provenance 则从 source codebook proximity 出发，利用 inverse decoder、QuantLoss 和 EncLoss 进行来源检测与归因。我们的区别在于不使用 AR 模型的完整 autoregressive probability model，也不针对每个候选 source 训练逆 decoder，而是将多个现成 quantizer 视为 reference operators，研究跨 reference 的 relative affinity。[待补实验必须证明在 source tokenizer 缺失时仍有泛化能力，否则应将方法定位为 reference-assisted 而非 open-set。]citeturn683047search0turn959856academia10turn683047search6

**问题定义与方法。** 给定图像 \(x\) 和冻结 tokenizer bank \(\mathcal T=\{T_k\}_{k=1}^K\)，每个 tokenizer 将图像编码为连续 latent \(h_k(x)\)，并量化为 \(\hat h_k(x)\)。我们定义量化误差
\[
q_k(x)=\frac{1}{N_k}\|h_k(x)-\hat h_k(x)\|_2^2.
\]
对仅包含真实图像的 validation reference set \(\mathcal R\)，估计每个 tokenizer 的 \(\mu_k,\sigma_k\)，并计算
\[
z_k(x)=\frac{q_k(x)-\mu_k}{\sigma_k}.
\]
当前 CTAD 分数定义为
\[
s_{\mathrm{CTAD}}(x)=-\min_k z_k(x),
\]
即只要样本对某个 tokenizer 出现异常低的 quantization error，就产生较高 fake evidence。当前 bank 包含 LlamaGen、VAR、RAR、Open-MAGVIT2 和 Switti tokenizer。citeturn807771view0

语义通道采用 UnivFD 的 CLIP ViT-L/14 特征与线性 real/fake head。对语义得分 \(s_\mathrm{sem}\) 与机制得分 \(s_\mathrm{CTAD}\)，分别利用 real validation distribution 再次归一化，并定义
\[
s(x)=\alpha\,\tilde s_{\mathrm{sem}}(x)+(1-\alpha)\,\tilde s_{\mathrm{CTAD}}(x).
\]
当前实验使用 \(\alpha=0.7\)，分类阈值只在 validation 上由 Youden's J 确定。需要强调的是：CTAD 通道本身只利用 real reference，但完整 DCA-Fuse 依赖已经训练好的 UnivFD head，同时 \(\alpha\) 当前来自实验扫描，因此完整方法不能描述成“完全 training-free”。citeturn634291view0turn634291view2

**性质分析。** min aggregation 可以理解为一个 multiple-reference anomaly statistic：不同 generator 可能只在少数 compatible tokenizer 上产生异常低 q-error，因此均值聚合会被大量无信息通道稀释，而 minimum 保留稀疏的 affinity evidence。仓库中的消融结果与这一解释一致：min AUROC 0.661、top-2 mean 0.657、mean-low 0.635；加入一个近似中性的 Infinity VAE channel 后，6-channel AUC 由 5-channel 的约 0.695 降至 0.661。该结果说明 bank quality 比单纯增加 tokenizer 数量更重要。citeturn634291view2

[待补理论] 正式 ICML 版本建议以 empirical CDF 或 conformal tail probability 取代 Gaussian z-score，并推导在 real null hypothesis 下 extremal statistic 的 false-positive control。这样可以明确解释为什么 reference bank 扩大时需要校正 multiple comparisons，也可以避免不同 tokenizer q-error 分布非高斯导致的 z-normalization 偏差。

**实验设置。** 主数据集为 ARForensics。当前统一 test 包含七个生成器，每个生成器 6000 real 和 6000 fake，共 84,000 个 generator-conditioned test pairs；但仓库当前训练/验证协议是从官方 validation 重划分得到的 6000/类 train 与 1000/类 val，并非 D³QE 官方 100k real + 100k fake training protocol，因此与 D³QE 原论文数值不可直接比较。正式论文应基于 clean manifest 重新运行所有主表，并明确区分 official test 与自定义 train/val protocol。citeturn807771view2turn961017view0

我们报告 AUROC、AP、TPR@0.1%/1%/5%FPR、FPR@95%TPR 和 validation-calibrated threshold metrics，并保存逐样本 prediction。数据审计发现 3 个跨 train/test 的精确重复样本，clean manifest 已将其从训练侧移除；与此同时，ARForensics 中 real 为 JPEG 而 fake 为 PNG，因此正式主表必须加入统一编码实验。citeturn961017view0turn634291view4

**结果。** 在完整 84k test 上，UnivFD overall AUROC 为 0.9127，DCA-Fuse 为 0.9128，整体差异基本为零。分生成器来看，Open_MAGVIT2 从 0.8863 提升至 0.9053，而 VAR、Switti 和 Infinity 分别出现约 0.007、0.008 和 0.006 的下降。这说明当前 fusion 并不是全面优于 UnivFD，而是在不同 generator 间重新分配错误。citeturn634291view4

在 7k subsample-500 pilot 中，DCA-Fuse 和 UnivFD overall AUC 都约为 0.953，但 Open_MAGVIT2 从 0.911 提升至 0.927；该实验还观测到 worst-generator improvement。然而这一现象没有在 84k full-scale 中完全复现，因此只能作为 hypothesis-generation result，而不能作为最终性能 claim。citeturn634291view1turn634291view4

格式控制实验显示，在相同 7k 子集上，UnivFD 统一 JPEG95 后 AUC 从 0.953 降至 0.889，而 CTAD 从约 0.681 降至约 0.658，提示机制通道较少依赖 JPEG/PNG shortcut。不过当前 CTAD 结果仍是 subsample 级别，[待补实验] 需要完成全量多级 corruption benchmark 后才能形成稳健性结论。citeturn634291view1turn634291view4

**消融与失败案例。** 使用六个生成器训练 LR 或 MLP affinity combiner 并测试剩余生成器时，macro AUC 只有约 0.654–0.656，其中 MLP 在 Open_MAGVIT2 上降至 0.478；无监督 S_min 为约 0.681–0.695。这一结果支持“supervised combiner 会过拟合 observed generator-specific affinity”的解释，但它还不是严格的 unseen-tokenizer 证明，因为测试生成器自身的 tokenizer 可能仍在 reference bank 中。[待补实验：leave-one-tokenizer-family-out]。citeturn634291view1

**局限。** 第一，当前 tokenizer bank 与已知测试生成器存在直接家族对应，因而 open-set claim 尚不成立。第二，full-scale fusion 没有总体 AUROC 增益，说明固定 \(\alpha\) 不能稳定协调语义与机制通道。第三，ARForensics 本身存在明显格式混杂，当前部分鲁棒性结果只在 7k 子集上完成。第四，当前训练协议不同于 D³QE 官方训练协议。第五，完整方法依赖一个经过 fake 训练的 UnivFD head，因此只能称 CTAD 为 real-only calibration，而不能把整个 DCA-Fuse 称为 training-free。citeturn634291view2turn634291view4turn807771view2

**结论。** 本文研究了 AR 生成图像在不同 tokenization spaces 中的量化亲和力。实验发现，单 tokenizer 的 real/fake discrepancy 在不同生成器上具有显著 direction instability，而多参考 tokenizer 可以把这种 source-specific compatibility 转化为可检测的 extremal signal。当前 DCA-Fuse 在 full ARForensics 上并未超过 UnivFD 的总体 AUROC，但对 Open_MAGVIT2 等特定家族表现出明显互补性，并揭示了一个值得进一步研究的方向：利用生成模型族共享的 latent geometry，而非单一 artifact classifier，构建真正面向未知生成器的 forensic detector。[待补实验完成后再决定最终 ICML claim。]citeturn634291view4

## ICML 投稿结论与优先级判断

按照当前代码和已经冻结的结果，我会把论文的 ICML 2027 状态评为：**Soundness 约 3/4，Presentation 潜力 3/4，Significance 目前 2–3/4，Originality 目前 2–3/4，总体大约 Weak Reject。** 原因不是方法不可行，而是现有 strongest claim——multi-tokenizer affinity——仍被 tokenizer coverage confound；同时 full-scale overall improvement 几乎为零。ICML 的 Weak Accept 定义要求论文能够实质推进一个子领域并让后续研究者有明显可构建之处，而当前版本还差一个真正把“affinity phenomenon”提升为通用 ML 原理的步骤。citeturn947861search3turn634291view4

如果只能优先做三件事，我建议依次完成：**leave-one-tokenizer-family-out；DARG + 真正未见 generator family 外部验证；empirical-CDF/conformal affinity score + paired significance analysis。** 这三件事分别解决 novelty、generalization 和 soundness 三个最关键的 ICML reviewer concern。之后再补 full corruption、归因、复杂度、多 seed 和 calibration。完成这些后，论文应从“CLIP + 多 tokenizer min fusion”重新定位成 **“multi-reference latent affinity as a general mechanism for open-world generated-data detection”**；这会比单纯追求 ARForensics 上再提高 0.5–1 个 AUC 点更接近 ICML 的接受逻辑。citeturn947861search0turn947861search3