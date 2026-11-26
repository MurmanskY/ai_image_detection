# SCALE / QSCALE / DART：自回归 token 序列分布检测

## 研究动机

PRADA 证明条件/无条件 token 似然比是强 AR 取证信号，但其检测器需要为每个模型学习 token-wise scoring function。D³QE 使用 codebook 频率与量化误差的监督融合，Data Provenance 使用 codebook 接近度和 decoder inversion。我们关注一个不同问题：**能否直接对整条 token 似然序列的分布形状建模，而不是把每个 token 独立压缩成标量？**

## SCALE：非参数逐位置尾概率

SCALE 对每个 token 位置的 conditional–unconditional log-likelihood difference，在真实参考上估计经验双侧尾概率，在 scale 内用 Fisher mean-surprisal 或 Higher Criticism 聚合，再对 model-bank max score 做独立真实序列级校准。

优点是不训练 fake 分类器、无需 token 独立假设；序列级 conformal 只要求真实 calibration/test 在组内可交换。缺点是逐位置尾概率会丢失偏差方向和幅度，max-scale 还可能选择噪声尺度。

DARG T2I 三 seed P0：Fisher macro AUROC 0.717–0.740，HC 0.396–0.413，官方 PRADA 0.9993–0.9997。**GO=false，不进入 C2I。**

## QSCALE：signed quantile-function 表示

QSCALE 将每个 scale 的 likelihood-difference 分布表示为 16 个 signed quantiles。quantile function 的 L2 几何对应经验 1-Wasserstein 几何，保留 SCALE 丢失的方向和分布形状。真实参考评分使用收缩 Mahalanobis 或固定 kNN。

三 seed 结果：Mahalanobis 0.715–0.731，kNN 0.767–0.784，仍明显低于 PRADA。**GO=false。** 说明 signed 分布表示本身并不足够，真实参考距离不能替代监督式 token 证据。

## DART：变量 scale 的 Deep Sets 检测器

DART 使用每个 scale 的 16 quantiles、log scale length 和 scale index，经共享 encoder 与 attention pooling 得到序列表示：

- scale encoder：18→32→16；
- attention：学习不同尺度的相对重要性；
- 分类器：16→1；
- 可处理不同 AR 模型的可变 scale 数量。

与 PRADA 的区别是：PRADA 学习逐 token 标量评分，DART 学习 quantile-function 的分布形状和 scale 间组合。与 D³QE 的区别是：DART 不使用 codebook frequency 或量化误差 latents，而使用条件/无条件似然的分布函数。

### LOGO 严格泛化实验

每折完全留出一个生成器的 fake，其他三个生成器用于训练与提前停止；held-out real 只用于输入标准化和阈值校准。三 seed × 4 folds：

| held-out | DART AUROC（3 seeds） | PRADA | QSCALE-kNN |
|---|---|---|---|
| Infinity | 0.922–0.930 | 约 1.000 | 0.772–0.788 |
| Janus | 0.657–0.791 | 约 0.999 | 0.909–0.941 |
| LlamaGen | 0.179–0.293 | 约 0.999 | 0.701–0.714 |
| Switti | 0.743–0.837 | 约 1.000 | 0.685–0.701 |

macro AUROC 仅 0.629–0.706。LlamaGen 出现严重方向反转，说明共享分类器不能把模型特异的 likelihood direction 转成通用证据。**LOGO GO=false。**

## 模型特异 DART 与完整生成循环一致性

模型特异 DART 的 C2I 实验揭示了清晰的分解：仅用 likelihood quantile 时，LlamaGen 已接近 0.99，但 VAR 接近随机；加入 scale-level quantization error 后 VAR 仅小幅改善。随后加入与 Data Provenance 一致的 decoder reconstruction evidence，形成 tokenizer–decoder–likelihood 的完整循环一致性表示。

完整版本在每个 scale 上使用：signed likelihood-difference quantile、scale 平均 quantization error、decoder reconstruction error，以及 scale length/index。它的创新点不是单独使用某一种已有特征，而是把 AR 生成过程的不动点约束表示为变量 scale 的联合分布，并由 Deep Sets 学习不同尺度的贡献。

三 seed、4 个参考模型、VAR_d16 与 LlamaGen-B256 各 250 张的小规模结果：

| 方法 | VAR_d16 | LlamaGen-B256 | macro AUROC |
|---|---:|---:|---:|
| DART（likelihood-only） | 0.525–0.554 | 0.969–0.998 | 0.747–0.776 |
| DART + quantization | 0.587–0.642 | 0.994–0.999 | 0.791–0.819 |
| 完整循环一致性 DART | **0.811–0.876** | **0.994–0.999** | **0.905–0.938** |
| 官方 PRADA | 0.477–0.509 | 0.993–0.997 | 0.737–0.752 |

VAR_d16 相对 PRADA 的逐图 paired ΔAUROC 为 +0.334 至 +0.351，95%CI 均大于 0；macro seed0 Δ=+0.176，95%CI [+0.131,+0.220]。LlamaGen 的差异不显著，说明主要创新收益来自补全其他 AR 家族，而不是继续提高已饱和模型。

预注册 GO 仍为 false：两个 seed 的 real FPR=2%，低于预设的 3% 下界。这是偏保守的校准结果，不是误报过高，但不能事后修改 GO 或阈值来宣称通过。

## 独立 RAR/HMAR 与参考家族覆盖

初始四模型 bank 的外部结果为：RAR-B 仅 0.532–0.574，HMAR-D16 为 0.785–0.866。加入同家族 RAR-L 后，RAR-B 提升至 **0.880–0.922**，说明 model-aware 方法不能把缺失参考家族解释为方法本身失效。

完整五模型 bank（VAR_d20/d30、LlamaGen-B/L256、RAR-L）结果：

| 生成器 | DART AUROC（3 seeds） | 同 bank PRADA | paired ΔAUROC（95%CI） |
|---|---|---|---|
| VAR_d16 | 0.770–0.863 | 0.478–0.509 | +0.293–0.354，均显著为正 |
| LlamaGen-B256 | 0.994–0.999 | 0.993–0.997 | 不显著 |
| RAR-B | **0.880–0.922** | 0.520–0.594 | +0.286–0.378，均显著为正 |
| HMAR-D16 | 0.755–0.848 | 0.484–0.542 | +0.271–0.331，均显著为正 |
| **macro** | **0.855–0.898** | 0.619–0.660 | seed0 +0.259 [+0.221,+0.296] |

这些结果支持“完整生成循环一致性 + quantile Deep Sets”在多个 AR 家族上的互补收益，而不是只靠 LlamaGen 的单一强点。参考模型覆盖是必要条件；缺少同家族参考时不能作公平的机制否定。

## 多 seed 与训练/校准规模敏感性

在 100 real train / 50 real calibration 的完整五模型 bank 上扩展至 9 seeds：DART macro AUROC 为 **0.884 ± 0.021**，同 bank PRADA 为 0.636，FPR 在 **0–13%** 间波动。将 real train 降至 50、calibration 增至 100 后，macro 变为 0.686–0.861、FPR 为 4%–11%，说明真实训练规模是敏感因素，单纯增加 calibration 不能修复。

随后完成 1000-real P1：400 reference / 300 calibration / 300 test，fake 每类 250，网络、阈值规则和五模型 bank 不变。9 seeds 结果为：

| 生成器 | DART AUROC 均值 | 9-seed 范围 | TPR@1%FPR 范围 |
|---|---:|---:|---:|
| VAR_d16 | 0.845 | 0.780–0.891 | 0–0.01 |
| LlamaGen-B256 | 0.998 | 0.992–1.000 | 0.89–1.00 |
| RAR-B | 0.942 | 0.905–0.957 | 0.06–0.58 |
| HMAR-D16 | 0.824 | 0.767–0.851 | 0–0.01 |

macro AUROC 为 **0.902 ± 0.015**，同 bank PRADA 各 seed 为 0.621–0.668；最小 macro 增益仍为 +0.245。前三个 seed 的 macro 为 0.902/0.913/0.904，FPR 为 5.33%/4.67%/7.00%，预注册 **GO=true**。9 seeds 的 FPR 为 3.00%–7.33%，其中 seed 8 为 22/300=7.33%，略超上限，因此全部 9 seeds 的严格 **GO=false**，不事后放宽标准。

独立逐图重算的 72 个 AUROC 与 JSON 完全一致。seed 0 paired macro ΔAUROC=+0.247，95%CI [+0.220,+0.275]；VAR/RAR/HMAR 的区间分别为 [+0.248,+0.383]、[+0.312,+0.428]、[+0.236,+0.366]，LlamaGen 不显著。结果见 `significance_real1000.json`。

## JPEG95 P2 smoke

`analysis/dart_codec_p0.py` 使用原图训练和校准，仅将 seed 0 的 300 张真实测试图和每类 100 张 fake 测试图改为 JPEG95。macro AUROC 从 0.902 降至 **0.859**（下降 0.043），FPR 为 6%。VAR/HMAR 分别下降 0.084/0.065，RAR/LlamaGen 仅下降 0.021/0.002；JPEG95 下同 bank PRADA macro 约 0.635。说明增益在压缩后仍保留，但 VAR/HMAR 有明显 codec 敏感性。8 个压缩测试 AUROC 已独立重算一致。该结果是单 seed smoke，不代替多 seed 跨 codec 结论。

## ImageNet-V2 独立真实来源

`analysis/dart_external_real_p0.py` 从 ImageNet-V2 matched-frequency shard0 确定性选择 300 张带标签真实图，保持原 ImageNet 训练、归一化和阈值不变。DART 外部真实 **FPR=4.67%**，macro AUROC=**0.913**；同 bank PRADA 分别为 2.00% 和 0.704。VAR/HMAR 在外部真实集上的 AUROC 反而为 0.854/0.847，RAR 为 0.953，LlamaGen 为 0.999。8 个外部 AUROC 独立重算一致。这支持校准迁移和跨真实来源增益，但仍是单分片、单 seed 结果。

## 当前研究边界

SCALE、QSCALE 和共享 DART 均未达到 PRADA，不能支持“无需监督即可替代 PRADA”。完整循环一致性 DART 已在多家族、小规模 C2I 上显示稳定 AUROC 增益，扩大真实集改善校准，ImageNet-V2 外部 FPR 也保持有效，但证据仍限于模型特异监督训练和少数生成器。它不等于未知生成器 zero-shot，也不保证 ICML 录用。VAR/HMAR 的 TPR@1%FPR 仍接近零，TPR@5%FPR 均值也仅为 0.134/0.100；RAR/LlamaGen 分别为 0.733/0.997，不能只报告 AUROC。工作点见 `operating_points_real1000.json`。更多 codec、多 seed 压缩与外部来源实验、完整数据规模和完整官方 bank 对照仍未完成。

## 投稿判断

方法已经具备明确机制创新和可重复的小规模性能证据，P1 前三 seed 通过预注册 GO，9-seed AUROC 增益稳定，JPEG95 和 ImageNet-V2 smoke 均保留宏观增益。主要剩余问题是 9-seed 严格校准一次边界失败、VAR/HMAR 低 FPR 检出率弱，以及跨域实验仍属小规模。当前可形成 ICML 方法论文的实验主线，但投稿前仍需补足多 seed 跨域、低 FPR 工作点和更完整基线对照，不能承诺录用。


## 严格四划分更新（优先于旧 P1 协议）

根据后续指南，当前评估已将真实数据拆为 300 reference/train、200 selection+median/IQR、200 独立 conformal calibration、300 test，fake 保持互斥 100/50/100。checkpoint、归一化和 model bank 在 conformal calibration 前冻结；calibration 只用于最终阈值。

9-seed macro AUROC 为 **0.906 ± 0.013**，seed 0 paired Δ=+0.233，95%CI [+0.206,+0.260]。FPR 没有超过 7%，但 seed 5/7 因偏保守、低于 3% 下限，严格 all-9 GO=false。逐样本 calibration/test 分数和阈值均已保存；72 个 AUROC、72 个阈值及 288 个工作点独立复算一致。

严格 JPEG95 seed 0 macro AUROC=0.844、FPR=5.67%；ImageNet-V2 seed 0 macro=0.902、外部 FPR=5.33%。结果分别为 `dart_c2i_jpeg95_strict.json` 和 `dart_c2i_imagenetv2_strict.json`。这些结果替代旧协议对有限样本 conformal 保证的解释，但仍不代表完整跨 codec 或跨数据集证明。


## 严格未知家族与 bank 覆盖更新

`analysis/dart_family_p0.py` 在 Infinity、Janus、LlamaGen-XL、Switti 上实施真正 held-out-family：目标 fake 不参与训练/验证。目标参考模型也移除时，共享完整循环 DART macro AUROC=0.790，高于同 bank PRADA 0.728；参考模型可用时 DART=0.855、PRADA=0.989，因此不能笼统宣称优于 PRADA。

完整消融显示未知家族主要由 likelihood + reconstruction 驱动（0.797），quantization-only 仅 0.577；新 seeds 3–5 的 L+R/full 为 0.805/0.799，差异很小。模型特异训练在严格排除时仅 0.631。max/mean/median 中 max 对 DART 最优，attention/mean/sum 未显示需要更换 attention 的稳定证据。

C2I bank 实验进一步确认覆盖效应：VAR-only、VAR+LlamaGen、完整 bank 的 macro 范围分别为 0.765–0.774、0.797–0.829、0.888–0.916。完整 bank 补齐 RAR，但 max 聚合对部分已有家族存在分数/校准权衡。所有家族结果与逐模型 bank 分数均已独立复算，见 `family_holdout_audit.json`。


## 多 codec 固定测试集复核

JPEG95/85/75/50 均已完成 3-seed 固定测试集评估。测试真实 300、每类 fake 100 在所有 seed 中固定，训练/选择/校准只在剩余样本中重排，保持严格隔离。平均 macro AUROC 分别为 **0.849/0.778/0.740/0.697**，相比原图均值 0.893 持续退化。JPEG85/95 的部分 seed 压缩 FPR 达 8–9%。

VAR/HMAR 是主要失效家族；JPEG50 时 VAR 已接近随机，LlamaGen/RAR 相对稳健。该协议只回答“原始训练与校准、压缩测试”的分布偏移，不与压缩校准或压缩训练混用。192 个 AUROC、96 个阈值和 768 个工作点已独立复算一致，见 `codec_matrix_audit.json`。


## JPEG85 校准迁移与完整域适配

三种协议已严格分开：原始训练/校准到压缩测试的 macro AUROC=0.767、FPR=6.33%；改用独立压缩 calibration 后 AUROC 不变、FPR=5.00%；完整压缩训练/选择/校准/测试后 macro=0.780、FPR=7.00%。

因此，阈值偏移可以由压缩校准修复，但检测信号退化不能靠阈值解决。完整压缩训练对 VAR/HMAR 有有限恢复，对 RAR 还略降，不能作为全面鲁棒性证据。实现见 `analysis/dart_codec_calibration.py` 与 `analysis/dart_codec_full_adaptation.py`，16 个 AUROC、16 个阈值和 64 个工作点已独立复算一致。
