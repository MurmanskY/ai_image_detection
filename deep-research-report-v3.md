# MGI 当前复现实验框架深度审查

## 审查结论

本次审查只关注 `ai_image_detection` 当前 `main` 分支中的**数据集、baseline 复现、评估代码、实验归档和科学可比性**，不评价或设计 `ours`。审查基准为 **2026 年 11 月 17 日**。我重新读取了仓库当前公开的 `README.md`、`reports/RESULTS.md`、数据集组织与审计代码、`engine/metrics.py`、`engine/evaluator.py`、`eval.py`、baseline registry 和各方法的 reproduction 文档，并与 D³QE、PRADA、ZED、RIGID、UnivFD、CNNDetection 等官方论文/仓库交叉核对。当前 GitHub 页面能够确认审查对象是 `main`，但网页接口没有向我暴露可可靠引用的 HEAD commit SHA，因此本报告不能声称已经“以某个 SHA 锁定”；这本身也应该成为实验归档必须补上的元数据。citeturn398786view2turn501054view3

总体判断是：**相比上一版框架，科学规范性已经有明显提升，尤其是 score 方向、validation 阈值校准、Manifest、重复数据审计、baseline 状态说明、结果冻结归档等关键基础设施已经真正落地；但是目前还不能把大多数结果称为“论文严格复现结果”。** 最大的原因不是 evaluator，而是 **训练/测试协议与原论文不一致、上游代码版本没有 commit pin、部分 baseline 是自行重实现甚至 inspired 版本，以及 PRADA/DARG 的 real 集并非官方 real 集**。D³QE 官方代码接入和 PRADA 官方代码接入是当前最接近“official implementation”的两条线，但二者的数据协议仍有偏差。citeturn339188view4turn339188view5turn339188view6turn447726view0turn447726view1

我会把当前框架定性为：

> **一个已经具备较好 benchmark 工程骨架的“protocol-adapted AR image detection evaluation framework”，但尚未达到可以把所有 baseline 都称为 paper reproduction 的程度。**

如果目前准备写论文，最需要做的不是继续增加新方法，而是把以下三件事做实：**官方数据协议、baseline provenance、metric/statistics protocol**。

## 当前实验结果到底能说明什么

目前仓库中可公开核验的冻结结果主要集中在 `reports/RESULTS.md`。该文件明确说明：只列冻结结果，`RUNNING` 不作为 final；所有指标以 fake 为正类；除特别注明外，训练协议采用 `ARForensics-small-val-derived-v1`，即把官方 validation 重新划成 train/val，而不是 D³QE 论文的官方 100k fake + 100k real 训练集。因此这一整组结果适合作为**统一框架下的内部横向比较**，但不应拿数值与原论文 table 逐项直接比较。citeturn839054view0turn339188view3

当前可以从仓库直接核验的结果如下。表中的“性质”是我根据当前协议重新分类的，而不是简单照抄 README 状态。

| 方法 | 数据 / split | 样本规模 | 当前可核验指标 | 当前状态 | 科学解释 |
|---|---:|---:|---|---|---|
| CNNDetection | ARForensics test | 7 generator × 6000 real + 6000 fake | Acc 0.6988, AP 0.8554, AUC 0.8195 | FROZEN | 自行重实现 + AR 协议适配，不是 CVPR 2020 paper reproduction |
| FreqSpec | ARForensics test | 同上 | Acc 0.6596, AP 0.8020, AUC 0.7862 | FROZEN | 自实现频域版本，不是 Frank et al. 官方实现 |
| NPR | ARForensics test | 同上 | Acc 0.7478, AP 0.8898, AUC 0.8604 | FROZEN | 明确是 inspired 简化版 |
| UnivFD | ARForensics test | 同上 | Acc 0.8017, AP 0.9112, AUC 0.9127 | FROZEN | 架构较忠实，但训练协议已改 |
| LGrad | ARForensics test | 同上 | Acc 0.7126, AP 0.8522, AUC 0.8342 | FROZEN | GAN 方法迁移到 AR |
| D³QE-lite | ARForensics test | 同上 | Acc 0.6328, AP 0.6958, AUC 0.6885 | FROZEN | lite，不应叫 D³QE reproduction |
| RIGID | ARForensics test | 7 generator 全量测试 | Acc 0.675, F1 0.655, AP 0.7421, AUC 0.7407, TPR@1%FPR 0.0997 | FROZEN | training-free 重实现 |
| ZED | ARForensics test 子集 | 每类 500 | Acc 0.564, F1 0.483, AP 0.5639, AUC 0.5806, TPR@1%FPR 0.015 | FROZEN | 非全量，且官方代码尚未公开 |
| D³QE official code | ARForensics | 本地协议训练 | Avg Acc 0.7535, AP 0.8269 | FROZEN | 官方代码，但**不是官方训练协议** |
| DIRE | validation | 2000 左右的当前 val 协议 | Acc 0.933, AP 0.9818, AUC 0.9790, TPR@1%FPR 0.825 | FROZEN val | validation，不是最终 test |
| DIRE | test subsample | 每 generator 500，合计 7000 | Acc 0.743, AP 0.8627, AUC 0.8587, TPR@1%FPR 0.241 | FROZEN example | 示例子集，不是 84k full test |
| PRADA T2I | DARG tiny-50 | 每 fake 目录 50 | Infinity 0.9988, Janus 0.9908, LlamaGen 0.9980, Switti 0.9996 AUROC | FROZEN tiny | official code/weights，但 real 集不是官方 RAISE-1k |
| PRADA C2I | DARG tiny-50 | 每目录 50 | 0.6448–0.8204 AUROC | FROZEN tiny | official code/weights，但 real subset 不是论文精确 subset |
| Data Provenance | LlamaGen / VAR | 各 1000 | 仓库只列输出 CSV，没有公开汇总指标 | FROZEN sampled | 不能凭“已运行”推断性能 |

这些值都来自当前 `RESULTS.md` 或对应 reproduction 文档；其中 RIGID/ZED 的旧版 sigmoid+0.5 阈值 Acc/F1 已被明确废弃，现有数值是 validation 校准阈值后的版本，这个修正是正确的。citeturn447726view3turn793777view4turn793777view5

这里有三个尤其重要的解读。

第一，**UnivFD 的 AUC 0.9127 是当前统一自定义协议下表现最好的普通 black-box supervised baseline 之一，但这不等价于“成功复现 UnivFD 论文结果”**。当前实现冻结 CLIP ViT-L/14，只训练线性头，这与 UnivFD 的核心形式相符；但训练数据已经从原论文的生成器协议改成了 ARForensics val-derived 数据。因此更准确的名称应是 `UnivFD architecture-faithful, protocol-adapted`，而不是 `faithful_reproduction`。原论文确实基于大型预训练视觉语言模型的 feature space，并公开了官方代码，但原始实验协议与这里不同。citeturn793777view3turn538259search5turn538259search8

第二，**DIRE validation 0.979 AUC 不能拿来证明 DIRE 在 AR 上很好**。当前 DIRE 是 20-step DDIM 近似，而 reproduction 文档自己说明官方使用更完整的反演设置；更关键的是 test subsample 已经从 val AUC 0.979 降至 0.859，而且不同 AR generator 间差异非常大。这说明 val-derived 训练/验证分布与真正 AR test 的 generator 分布之间存在明显 shift。当前结果反而很好地证明了：你们未来主表必须优先报告 test/per-generator，而不能突出 val。citeturn793777view6turn447726view3

第三，**PRADA T2I 0.99+ 很漂亮，但现在不能写成“严格复现成功”**。PRADA 官方数据说明 T2I fake 使用 Synthbuster prompts，对应 real images 来自 RAISE-1k；当前仓库明确把 T2I real 换成 ARForensics real。C2I real 也不是作者的精确 ImageNet subset，而是另一个 HF mirror 的构造子集。因此你可以写“官方算法和权重在本地数据适配协议下 successfully executed”，但不能用它证明 paper table 被 reproduced。官方 DARG 包含 155,000 个 generated images、20 个 AR generators，这一点与当前 repo 的方法描述一致。citeturn447726view1turn160383search2

另外，目前 `Data Provenance` 只公开声明了 LlamaGen/VAR 各 1000 样本的 CSV 输出，却没有在 `RESULTS.md` 给出 AUROC、TPR 或 attribution accuracy，所以**这部分目前没有公开可审计的“实验性能结果”**，只能确认“任务被运行并产生 CSV”。仓库还把 upstream URL 写成 `UNKNOWN`，但目前公开的 `sprintml/DataProvenanceIAR` 仓库是可检索到的，且该论文已进入 ICLR 2026 Proceedings。因此这里应尽快补 upstream URL 与 vendor commit。citeturn447726view2turn160383search3turn160383search6

## 数据集协议审查

### ARForensics

这是目前最大的 P0 问题。

D³QE 官方仓库明确给出的 ARForensics protocol 是：**train = 100k LlamaGen + 100k ImageNet，validation = 10k LlamaGen + 10k ImageNet，test = 7 个 AR generator 的 balanced test set**。官方仓库还专门发布过紧急修正：最初 Hugging Face test 解压后错误包含 `MAR`，论文实际使用的是 `Switti`；正确 test generator 应是 Infinity、Janus-Pro、RAR、Switti、VAR、LlamaGen、Open-MAGVIT2。citeturn160383search0turn160383search1

你们当前 `scripts/organize_arforensics.py` 明确没有下载约 35 GB 的官方 train 分卷，而是把官方 10k+10k validation 的前一部分拿来作为 train、下一部分作为 val；当前默认是 train 每类 6000、val 每类 1000。这意味着它虽然保持了测试集，但**训练协议不再是 ARForensics official protocol**。代码已经正确把这个协议命名成 `ARForensics-small-val-derived-v1`，这一步非常好，但后续所有结果表必须继续坚持这个命名，不能只写“ARForensics”。citeturn339188view3turn339188view4

另一个问题是 split 的构造方式。当前脚本写的是“前 `n_train` 张”和“接下来 `n_val` 张”，不是基于固定 seed 的 shuffle，也不是基于内容 hash 的 deterministic partition。只要上游压缩包的文件排序、命名或解压行为变化，就可能改变 split。对于临时 pilot 协议可以接受，但用于论文不够稳。citeturn339188view3

数据审计本身比上一轮已经完善很多。Manifest 已包含 `sample_id,path,sha256,phash,label,generator,generator_family,source,split,original_split,dataset`，并明确 full-file SHA256、perceptual hash、`original_split=validation` 等 provenance 字段。Audit 检查了 split 间 exact hash、pHash candidate、标签合法性、generator metadata、重复路径、损坏图片和格式/大小概览。当前归档结果显示 98,000 行、17 组 exact duplicates，其中有 3 个 train–test exact overlaps，另有 1,238 个 pHash≤8 candidate pairs。citeturn339188view4turn339188view5turn447726view3

这里需要比当前报告再严格一步：**3 个 cross-split duplicates 即使只占 3/98,000，也建议在正式 protocol 中直接移除，而不是仅声明**。尤其是真实 ImageNet 图片可能在多个 generator test folder 中重复出现，任何 repeated real source 都会让普通 sample-level bootstrap 错把非独立样本当成独立观察值。后面统计部分我会给修改方案。citeturn447726view3

此外，当前 audit 文档说会输出“文件格式与大小分布”，但我没有在公开结果摘要中看到针对 **real/fake 分别统计 JPEG/PNG 比例、JPEG quality proxy、width/height、aspect ratio、file size** 的正式 table。对于 generated-image detection，这是必须做的，因为 detector 很容易利用 encoding artifact、resolution 或 compression 差异而非生成痕迹。当前代码具备基础，但报告层面还不够。citeturn339188view5turn398786view3

### DARG

当前 DARG 使用方式最大的科学问题不是 fake generator，而是 **real reference 不匹配**。PRADA 官方仓库明确说明 T2I 的真实图像对应 RAISE-1k，而 C2I 的构造与 ImageNet 类别对应；你们 reproduction 文档已经诚实记录 T2I 用 ARForensics real 替代、C2I 使用 HF ImageNet mirror subset。这些结果应定义为 `DARG-adapted-real-v1`，不要写成 DARG official reproduction。citeturn160383search2turn447726view1

`RESULTS.md` 里 DARG manifest 显示 T2I 6,000 行、C2I 160,000 行。官方仓库所谓 155,000 指的是 **generated images 总量**，所以 160k C2I / 6k T2I manifest 并不一定矛盾——manifest 很可能同时包含 real；但这个关系目前在结果报告中不够清楚。建议数据 manifest summary 必须分别报告 `n_real`、`n_fake`、`n_fake_per_generator`，否则读者很容易把 “155k official generated images” 与 “166k manifest rows” 误判为数据版本错误。citeturn160383search2turn447726view3

### GenImage

目前 GenImage loader 本质上是一个简单的 folder-based loader：按 split 目录读取 real/fake 文件，并没有像 ARForensics 那样在当前公开材料中显示 dataset revision、generator provenance、manifest hash 或 audit report。citeturn398786view0

因此，GenImage 目前不应该成为“已完成数据审计”的数据源。建议把它降级为 legacy supplemental benchmark，直到也生成与 ARForensics 一致的 manifest + audit。尤其你们目前主任务是 AR autoregressive detection，GenImage 更适合回答“传统 GAN/diffusion detector 能否迁移到 AR”，而不是核心 AR benchmark。

## Baseline 与论文及官方代码的一致性

这里我建议不要只用 `reimplemented / faithful_reproduction / vendored` 这几个标签，而是引入两个正交字段：

`code_origin = official | reimplementation | inspired`

`protocol_fidelity = official | adapted | partial`

这样会比现在单一 `implementation_status` 更科学。当前 registry 已经开始记录 paper URL、official code URL、access type、score type、higher_is_fake 等信息，这是对上一版建议的很好落实，但 upstream commit 普遍仍为空。citeturn339188view6

| 方法 | 当前实现本质 | 与论文/官方代码一致性判断 | 当前建议标签 |
|---|---|---|---|
| CNNDetection | 自己实现 ResNet-50 + blur/JPEG style augmentation | backbone 和思路相近，但 augmentation 未逐项核验，训练集完全不同 | `reimplementation / adapted` |
| FreqSpec | 自实现 DCT/频谱 + classifier | reproduction 文档自己承认频谱处理与分类头未与官方逐项对齐 | `reimplementation / adapted` |
| NPR | 简化 NPR residual ×4 + ResNet | 官方 NPR 模块和训练协议未完整复现 | `inspired / adapted` |
| UnivFD | frozen CLIP ViT-L/14 + linear head | 核心架构高度一致，但训练数据不是官方 protocol | `architecture-faithful / adapted` |
| RIGID | DINOv2 + noisy image similarity | 原理一致，但 backbone/perturbation/preprocess 未与官方脚本逐项验证 | `reimplementation / partial` |
| DIRE | 20-step DDIM residual + ResNet | 官方为 diffusion-specific pipeline，本地又做了 20-step approximate | `reimplementation / transfer` |
| LGrad | gradient representation + classifier | GAN 专用方法迁移，非论文实验协议 | `reimplementation / transfer` |
| ZED | SReC NLL/entropy gap | 论文思路可对齐，但官方 GitHub 目前 README 仍写 Code Coming Soon | `paper-reimplementation / partial` |
| D³QE official | vendored 官方仓库 | code origin 最可靠；训练数据不符合 100k/100k official protocol | `official-code / adapted-protocol` |
| PRADA | vendored 官方 `predict.py` + 官方权重 | algorithm/code 很接近官方；真实数据集替换导致 protocol 不一致 | `official-code / adapted-data` |
| Data Provenance | vendored code | upstream 尚未在本地记录；公开 upstream 已存在 | `official/vendor-candidate / unverified-pin` |

CNNDetection 的官方仓库确实提供了原论文代码与 pretrained model，但你们当前实现没有 vendor 官方 repo，也明确说明增强参数没有逐项核验。因此它不能标成 official reproduction。citeturn793777view0turn538259search0

FreqSpec 更明确：当前 reproduction 文档直接写了“频谱处理与分类头为自行实现，未与官方逐项对齐”，因此它的实验结果只能代表一个 frequency baseline，而不能直接作为 Frank et al. 方法的严格复现值。citeturn793777view1

NPR 目前分类是正确的。文档明确把它称为 inspired 简化版，并声明不得与官方 NPR 直接比较；这一点应继续保留，而且建议主论文表中方法名直接写成 `NPR-inspired`，避免读者只看到 NPR 三个字母。citeturn793777view2

UnivFD 是当前最值得修正命名的一项。官方论文/代码确实使用大型预训练视觉语言模型特征，并有 `UniversalFakeDetect` 官方项目；你们冻结 CLIP ViT-L/14 再训练线性头的框架基本符合其核心做法。但是当前训练集和 augmentation/protocol 已经发生变化，因此 `faithful_reproduction` 容易让人以为“paper setting reproduced”。建议改成 `faithful_architecture_reimplementation`，同时另设 `protocol_fidelity: adapted`。citeturn793777view3turn538259search5turn538259search8

RIGID 当前更适合写成 “paper-motivated reimplementation”。官方 IBM 仓库是存在的，因此如果目标是 benchmark 严谨性，最稳妥的方式不是继续自己猜 noise/std/backbone，而是**直接 vendor 官方 repo 的某个 commit，然后建立 adapter**。当前版本使用 DINOv2 ViT-S/14、noise_std=0.05、余弦相似度，并取负后统一为 higher=fake；方向转换在 evaluator 语义上没有问题，但 reproduction 文档也承认 perturbation 与官方对齐程度尚未逐项核验。citeturn793777view4turn538259search4

ZED 是一个特殊情况：官方 GRIP-UNINA 仓库存在，但目前公开 README 的 `Code` 仍然是 “Coming Soon”。因此现阶段不可能宣称“与官方 GitHub implementation 一致”，只能说“根据论文公式的 reimplementation”。这也是为什么你们 reproduction 文档中“其他与官方统计定义一致”这句话建议改成“与论文描述的统计量设计一致，尚无公开官方实现可逐行验证”。citeturn160383search5turn793777view5

D³QE 是目前最干净的一条 official-code baseline，但训练集必须补齐。官方明确要求 100k LlamaGen + 100k ImageNet 训练、10k+10k val，并且提醒 test 的 MAR→Switti 修正。因此要想得到真正的 `D3QE-official-reproduction`，最优先工作不是改模型，而是下载官方 train archive、锁定 corrected test revision，然后原样跑官方脚本。citeturn160383search0turn447726view0

PRADA 同理：官方代码已经接入，这是好事；当前最主要的阻碍是 dataset real side，而不是算法。只要恢复官方 RAISE-1k / paper ImageNet subset，并固定官方 weights、repo commit、DARG Zenodo version，就能很接近严格 reproduction。citeturn160383search2turn447726view1

Data Provenance 则需要先修 provenance metadata。当前本地文档写 upstream URL `UNKNOWN`，但 `sprintml/DataProvenanceIAR` 已公开存在，论文也已经在 ICLR 2026 Proceedings 中。因此应立刻把该 upstream 写入 registry，并记录 vendored HEAD 的 SHA；在此之前，“official code”最好写为 `vendored candidate, provenance not yet verified`。citeturn160383search3turn160383search6turn447726view2

## 评价指标与 evaluator 是否权威

这一部分目前已经从“明显有问题”提升到了“基本正确，但统计层面还不够”。

当前 `engine/evaluator.py` 已经建立统一 score contract：logit 方法 sigmoid 后输出；anomaly 方法保留 raw score；如果方法自身是 `higher_is_fake=False`，统一反向为“越大越 fake”。这是一个很好的框架设计。citeturn339188view1

`eval.py` 对非 logit / training-free 方法，如果 test 时没有显式 threshold，就重新构造 validation set，调用 `calibrate_threshold`；因此 RIGID/ZED 不再用错误的 `sigmoid(score)>=0.5`。当前实现确实做到了“阈值来自 validation，而不是 test”。citeturn339188view2

`metrics.py` 当前使用 `sklearn` 的 `roc_auc_score`、`average_precision_score`、`balanced_accuracy_score`、`f1_score` 和 `roc_curve`，定义上都是主流权威实现；`TPR@FPR` 是先构造 ROC，再在 `FPR <= target` 区域选择最大 TPR，当前 targets 是 0.1%、1%、5%。这种实现是一个合理的离散 ROC conservative definition，但必须在论文里明确“不做 interpolation”，因为不同论文可能用线性插值或 nearest operating point。citeturn339188view0

这里仍有四个 P1 问题。

**第一，缺少 FPR@95TPR。** 用户之前要求的这项指标当前不在 `FPR_TARGETS` 或 `compute_metrics` 中。很多 detection / OOD / forensic 工作会报告固定 TPR 下的误报率，因此建议增加 `fpr_at_tpr(labels, scores, target_tpr=0.95)`，并与 `TPR@1%FPR` 同时保留。citeturn339188view0

**第二，Bootstrap 目前是普通 sample bootstrap，不是 stratified 或 grouped bootstrap。** 当前实现每次从所有样本中有放回抽 n 个，然后如果碰巧只有单一类别就跳过。这在样本很多时 AUROC 通常没大问题，但不是最严谨的方案；更重要的是 ARForensics 的真实图片可能在 generator-specific folders 中重复，且已经检测出 3 个 cross-split exact duplicate，普通 bootstrap 会把相关样本错误视作独立 observation。建议至少改为 class-stratified bootstrap；如果能建立 `source_image_id / prompt_id / original_real_id`，进一步做 group bootstrap。citeturn339188view0turn447726view3

**第三，低 FPR 指标必须伴随有效负样本数。** ZED 目前每类只有 500 张，1% FPR 对应 5 个 false positives，而 0.1% FPR 连 1 个样本都不到；这种情况下 `TPR@0.1%FPR` 几乎没有统计解释力。即使 `TPR@1%FPR=0.015` 算法实现没有错，也应附 `N_real=500` 和 CI，不能与 6000 或 42000 个 real image 的结果并排无注释比较。citeturn793777view5turn339188view0

**第四，主结果表应该以 threshold-free metrics 为主。** AUROC/AP 可以直接比较 ranking 能力，而 Acc/F1/BA 会依赖 calibration protocol。当前 Youden J threshold 是合理的 validation-only 选择，但它并不是唯一权威 threshold；如果部署目标是控制误报，更有实际意义的是在 validation 上校准 `FPR=1%` 的 operating threshold，再在 test 固定 threshold 报 TPR/Accuracy。建议把 Youden 结果保留为 supplemental，同时新增 `val_fpr_1pct` calibration protocol。citeturn339188view0turn339188view2

因此我建议正式 benchmark 采用两套并行评价：

| 类型 | 主指标 | 是否需要阈值 |
|---|---|---|
| Ranking | AUROC, AP | 否 |
| Forensic operating point | TPR@1%FPR, TPR@0.1%FPR, FPR@95%TPR | 否，ROC 曲线直接计算 |
| Deployment | Accuracy, F1, Balanced Acc | 是，必须 validation calibration |
| Reliability | 95% CI | 否，但必须声明 bootstrap unit |

这里还要补充：当前 common baseline table 只报告 Acc/AP/AUC，没有将 TPR@1%FPR、CI、sample count 系统性补齐；RIGID/ZED 才有 TPR。要建立权威 benchmark，所有方法必须由同一个 predictions-level evaluator 重新计算，而不是有的方法抄官方 CSV、有的方法用统一 evaluator、有的方法只抄 paper style average。citeturn447726view3

## 与上一版改造建议相比，哪些已经完成

上一轮 `deep-research-report-v2.md` 的核心方向是先规范实验框架，而不是开发新方法。从当前 main 看，很多最重要的基础设施已经落地：Manifest、SHA256/pHash、dataset audit、baseline registry、reproduction notes、统一 score contract、validation 阈值校准、AUROC/AP/TPR、bootstrap CI、results archive、tests 以及 frozen/running 区分都已经进入仓库。这说明当前阶段已经不需要再做一次“大重构”，而应该进入**protocol hardening** 阶段。citeturn339188view4turn339188view5turn339188view6turn339188view0turn447726view3

我会把目前状态划分成：

| 上一版建议 | 当前状态 | 判断 |
|---|---|---|
| Manifest + SHA256 | 已完成 | 保留 |
| pHash / duplicate audit | 已完成 | 需要去除已知 leakage |
| generator metadata | 已完成 | 基本正确 |
| official/custom split 区分 | 已完成 | 做得很好 |
| score higher=fake contract | 已完成 | 正确 |
| training-free val calibration | 已完成 | 正确 |
| AUROC / AP | 已完成 | 正确 |
| TPR@FPR | 已完成 | 需补定义和 CI |
| FPR@95TPR | 未完成 | P1 |
| Bootstrap CI | 已完成基础版 | 需 stratified/grouped |
| baseline registry | 已完成 | upstream commit 尚未 pin |
| reproduction docs | 已完成 | 部分 status 命名过强 |
| official ARForensics train | 未完成 | **P0** |
| official DARG real protocol | 未完成 | **P0** |
| Data Provenance upstream provenance | 未完成 | **P0** |
| format/compression/resolution bias audit | 部分完成 | **P1** |
| per-generator full metric archive | 不完整 | **P1** |
| run-level Git SHA / env lock | 不够明确 | **P1** |

换句话说，**v2 的工程骨架已经大体建立；现在真正阻碍论文级可信度的不是“缺几个 Python 文件”，而是实验协议和 provenance 没有完全锁死。**

## 给 Kimi Code 的 P0 / P1 / P2 修改建议

### P0：在继续跑大规模实验前完成

**P0-A：把 ARForensics 分成两个完全独立的 protocol。**

必须新增：

```text
ARForensics-official-v1
ARForensics-small-val-derived-v1
```

其中 official 必须严格遵循 D³QE：

```text
train:
  real: 100000 ImageNet
  fake: 100000 LlamaGen

val:
  real: 10000 ImageNet
  fake: 10000 LlamaGen

test:
  Infinity
  Janus_Pro
  RAR
  Switti
  VAR
  LlamaGen
  Open_MAGVIT2
```

同时对 Hugging Face / ModelScope 下载保存 dataset revision、archive filename、archive SHA256，并运行一个 hard assertion：

```python
expected_generators = {
    "Infinity",
    "Janus_Pro",
    "RAR",
    "Switti",
    "VAR",
    "LlamaGen",
    "Open_MAGVIT2",
}
assert set(test_generators) == expected_generators
assert "MAR" not in test_generators
```

原因是官方确实曾经发布过 MAR/Switti 错误版本。citeturn160383search0turn160383search1

**P0-B：去掉 ARForensics 已知的 3 个 train/test exact leakage，并生成 clean manifest。**

不要只在 report 中声明。建议生成：

```text
arforensics_raw_manifest.csv
arforensics_clean_manifest.csv
leakage_exclusions.csv
```

正式论文结果只能从 clean manifest 运行；exclusion 应基于 SHA256，而不是 filename。当前已知确实有 3 组跨 split exact duplicate。citeturn447726view3

**P0-C：恢复 PRADA 的官方 real-image protocol。**

T2I 下载/准备真正的 RAISE-1k；C2I 必须从 PRADA 官方提供的 sample mapping / subset metadata 构建同一批 ImageNet real，而不是“每类选 10 张看起来等价”的 HF mirror。然后将目前 tiny 结果重命名为：

```text
PRADA-DARG-adapted-tiny50
```

只有严格官方数据的 full/subset protocol 才可以叫：

```text
PRADA-DARG-official
```

PRADA 官方仓库明确给出了 DARG generator 构成和 real-image provenance。citeturn160383search2

**P0-D：给所有 upstream code 加 immutable provenance。**

`baselines/registry.yaml` 新增：

```yaml
upstream_repo:
upstream_commit:
vendor_commit:
weight_source:
weight_sha256:
dataset_revision:
```

尤其立即修复 Data Provenance：

```yaml
official_code_url: https://github.com/sprintml/DataProvenanceIAR
```

并核验本地 `third_party/DataProvenanceIAR` 是否和某个 upstream commit byte-identical。当前公开 upstream 已存在。citeturn160383search3

**P0-E：重新定义 reproduction taxonomy。**

不要再只用一个 `faithful_reproduction`。推荐：

```yaml
code_origin:
  official
  reimplementation
  inspired

architecture_fidelity:
  exact
  faithful
  approximate

protocol_fidelity:
  official
  adapted
  transfer

verification_status:
  unverified
  runs
  matched_reference
```

例如：

```yaml
univfd:
  code_origin: reimplementation
  architecture_fidelity: faithful
  protocol_fidelity: adapted
  verification_status: runs
```

这样不会再产生“架构对齐 = 论文复现”的歧义。citeturn793777view3

### P1：建立论文级 benchmark 所需

**P1-A：补全 metric API。**

新增：

```python
fpr_at_tpr(labels, scores, target_tpr=0.95)
```

并输出：

```text
auc
ap
tpr_at_fpr_0.001
tpr_at_fpr_0.01
tpr_at_fpr_0.05
fpr_at_tpr_0.95
balanced_accuracy
f1
accuracy
```

所有 threshold-dependent metrics 必须额外写：

```text
threshold_value
threshold_source_split
threshold_rule
```

当前 metric 实现已经有前三类基础。citeturn339188view0

**P1-B：重写 bootstrap。**

从现在的 sample bootstrap：

```python
idx = rng.integers(0, n, n)
```

升级为至少 stratified bootstrap：

```python
real_idx = bootstrap(real_indices)
fake_idx = bootstrap(fake_indices)
```

再进一步支持：

```text
bootstrap_unit = sample | source_id | prompt_id
```

如果同一 real base image 被多个 generator paired test 复用，必须按 source-image group bootstrap，而不是把每次出现当独立样本。

**P1-C：每个 final result 必须提交 predictions-level artifact。**

每次 run 至少归档：

```text
config.yaml
run_meta.json
metrics.json
metrics_per_generator.json
predictions.parquet
environment.txt
git_sha.txt
dataset_manifest_sha256.txt
stdout.log
```

`predictions.parquet` 至少：

```text
sample_id
path
label
generator
raw_score
normalized_score
split
source_id
```

目前部分结果路径只指向 `outputs/...`，公开 GitHub 无法独立检查这些 artifact；例如 Data Provenance 只有 CSV 路径声明而没有汇总指标。citeturn447726view3turn447726view2

**P1-D：做真正的格式/压缩偏差报告。**

针对 real/fake 和每个 generator 分别输出：

```text
extension
decoded_format
width
height
aspect_ratio
file_size
estimated_jpeg_quality
has_exif
color_mode
```

然后至少画出：

```text
real vs fake resolution distribution
real vs fake JPEG/PNG distribution
real vs fake file-size distribution
```

如果发现 real 全是 JPEG、fake 大量 PNG，正式 benchmark 应增加一个“统一 re-encode”控制实验，例如所有图片统一 decode→RGB→PNG 或统一 JPEG Q=95，再重新评估 detector。当前 audit 已经有格式/大小检查入口，但结果归档还没达到这个层次。citeturn339188view5

**P1-E：所有 baseline 都统一跑 full per-generator test。**

最终表不要只有 overall AUC：

```text
method | LlamaGen | VAR | RAR | Infinity | Janus-Pro | Switti | Open-MAGVIT2 | macro
```

每格至少报告 AUROC + AP + TPR@1%FPR。`macro` 必须是 generator-level macro average，而不是简单把 84k 样本全部混起来后再算一个 pooled AUC，因为 pooled metric 可能被容易检测的 generator 主导。

### P2：在 benchmark 稳定后补充

P2 可以加入 robustness protocol、external dataset protocol、统一 docker/conda lockfile，以及 official-vs-adapted result table 自动生成器。这里不需要马上扩充新检测方法。

建议把最终论文结果拆成三个层次：

`Table A — Official-paper reproductions`  
只放真正满足官方代码/权重/数据协议的实验。

`Table B — Unified ARForensics benchmark`  
所有方法在**同一 ARForensics protocol** 下重新评估，允许 protocol-adapted，但必须统一。

`Table C — Architecture-transfer baselines`  
DIRE、LGrad 等原本不是 AR detector 的方法单独放，防止读者误解。

## 最终研究建议

现阶段最值得做的不是再跑更多 baseline，而是完成两个**golden reproduction**，作为整个框架的“验收测试”：

**第一优先：D³QE official reproduction。**

理由是它直接定义了 ARForensics，官方代码、官方数据、官方训练/验证/测试 protocol 都明确，而且官方还提供了 corrected test set。只要把 100k+100k train 下载完整，固定 corrected Switti 版本，vendor 某个 upstream commit，再跑出与论文接近的主表，你就获得了一个非常强的 framework sanity check。citeturn160383search0turn160383search1

**第二优先：PRADA official DARG reproduction。**

不是继续跑 tiny-50，而是先恢复 RAISE-1k 和官方 C2I real subset，再使用官方 weights/`predict.py` 跑论文规定的 reduced evaluation subset。PRADA 官方 repo 已经公开给出 DARG 构成和使用方式，因此这是第二个非常适合做 end-to-end verification 的基准。citeturn160383search2

只要这两个 golden baselines 能做到：

> upstream commit 固定 → dataset revision 固定 → manifest 固定 → prediction artifact 固定 → metric 统一重算 → paper reference result 对照

那么你们整个 MGI 框架的可信度会提升一个层级。

目前代码框架的方向已经基本正确，尤其 **Manifest、audit、score contract 和 validation calibration** 这些上一轮的关键问题已经修得不错。当前真正需要收紧的是“**什么叫复现**”：凡是数据、权重、预处理、源码或 protocol 有一个关键部分不一致，就不要叫 paper reproduction，而应明确叫 adapted evaluation、transfer evaluation 或 reimplementation。做到这一点之后，当前已有的实验数据才会真正成为后续研究可以信赖的基线。