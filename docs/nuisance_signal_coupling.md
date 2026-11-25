# Nuisance–signal coupling：理论命题、适用条件与可证伪预测

本文档对应 v12 的理论分析，不把经验观察提前写成普遍定理。结论均限定于给定残差表示、参考分布和校准协议。

## 1. 定义

设真实图像在条件干预 `a` 下的取证残差为 `r(a) ∈ R^d`，canonical 条件为 `a0`。用真实图像配对差分 `r(a)-r(a0)` 估计列正交矩阵 `U ∈ R^(d×k)`，令

- `P_U = UU^T`，`P_perp = I-P_U`；
- `Δ = E[r | fake] - E[r | real]`，且 `Δ ≠ 0`；
- 耦合比例 `ρ = ||P_U Δ||² / ||Δ||²`，因此 `0 ≤ ρ ≤ 1`。

`ρ` 是均值差能量的比例，不是分类 AUC、互信息或因果效应。当前 CoNTra 报告的信号保留率 `η = ||P_perp Δ||² / ||Δ||² = 1-ρ`。

## 2. 命题 A：不变投影的信号损失

**命题。** 对正交投影 `P_perp`，有

`||P_perp Δ||² = (1-ρ)||Δ||²`。

更一般地，对任意满足 `LU=0` 且算子范数 `||L||₂ ≤ 1` 的线性映射 `L`，有

`||LΔ||² ≤ (1-ρ)||Δ||²`。

**证明。** 由勾股定理，`||Δ||² = ||P_UΔ||² + ||P_perpΔ||²`。代入 `ρ` 即得第一式。因为 `LU=0`，所以 `LΔ=LP_perpΔ`；再由 `||L||₂≤1`，得到 `||LΔ||≤||P_perpΔ||`。证毕。

**边界。** 若取消非扩张约束，将 `L` 任意放大即可使结论失效；若 `Δ=0`，`ρ` 无定义。该命题只约束均值差，不直接给出 AUROC 下降量。

## 3. 命题 B：条件建模何时保留最优判别信息

保留完整坐标 `(z_parallel,z_perp,a)=(U^Tr, P_perp r, a)` 与保留 `r` 在信息上等价，因为 `r=Uz_parallel+z_perp`。因此，条件化本身不会像 `LU=0` 那样删除信息。

进一步假设在每个条件 `a` 内，真实和生成分布均为高斯，均值分别为 `μ0(a), μ1(a)`，共享协方差为 `Σ(a)`。对数似然比为

`log p(r|fake,a)/p(r|real,a)
 = (μ1(a)-μ0(a))^T Σ(a)^(-1)r + C(a)`。

在等先验、同协方差条件下，该分数是 Bayes 最优判别规则；若检测异常分数使用相应的条件 Mahalanobis 形式，则它保留条件内均值差，而不是把 `U^TΔ` 置零。

**边界。** 一般的 real-only Mahalanobis 并不等价于真实/生成似然比；非高斯、异协方差、均值/协方差估计误差或条件错配时，没有 AUROC 最优保证。这正说明“保留坐标”不能推出“条件评分一定优于无条件评分”，SACR 的负结果与该理论一致。

## 4. 命题 C：已知组内的有限样本误报控制

固定 reference 及其拟合得到的评分函数。对预定义组 `g`，设独立真实校准分数 `S_1,...,S_m` 与测试真实分数 `S_(m+1)` 在组内可交换。令

`τ_g = Quantile_{ceil((m+1)(1-α))/m}(S_1,...,S_m)`。

则

`P(S_(m+1) > τ_g | g) ≤ α`。

若进一步假设分数无并列，则还有 `α-1/(m+1) ≤ P(S_(m+1)>τ_g|g) ≤ α`。

**证明。** 可交换性使测试分数在 `m+1` 个分数的秩上均匀分布。只有当其秩位于最大约 `α(m+1)` 部分时才超过上述次序统计量，因此概率至多 `α`；无并列时可得到对应的下界。证毕。

**边界。** 该保证不覆盖未知条件组、跨 codec 分布漂移、用测试集选择评分器、参考与校准混用或组内不可交换。未知组必须标记 unsupported/abstain，不能借用其他组阈值后仍宣称有限样本保证。

## 5. 命题 D：条件估计误差对分数的确定性扰动界

设真实条件均值和精度矩阵为 `μ, A=Σ^(-1)`，估计为 `μ_hat=μ+δ, A_hat=A+E`，并令 `d=r-μ`。假设

`||d||₂ ≤ R, ||δ||₂ ≤ e_μ, ||E||₂ ≤ e_A, ||A||₂ ≤ B`。

定义 `S=d^TAd` 与 `S_hat=(d-δ)^T(A+E)(d-δ)`，则

`|S_hat-S| ≤ e_A R² + 2(B+e_A)R e_μ + (B+e_A)e_μ²`。

**证明。** 展开 `S_hat-S = d^TEd - 2δ^T(A+E)d + δ^T(A+E)δ`，对三项分别使用 Cauchy–Schwarz 和算子范数不等式，并由 `||A+E||₂≤B+e_A` 得证。

**解释。** 条件邻域有效样本量不足通常会增大 `e_μ,e_A`，从而同时影响排序和校准；coverage/abstention 是方法有效性的一部分，不能只报告支持样本上的 AUROC。该式是确定性上界，不是针对某一核回归估计量的收敛速度。

## 6. 可证伪预测

若 nuisance–signal coupling 是稳定机制而非 Noiseprint++ 的偶然性质，应观察到：

1. 在独立固定高通残差表示上，真实配对干预方向仍具有跨 source 稳定性；
2. `P_perp` 仍显著降低 fake–real 均值差能量，而非只降低真实 codec MMD；
3. same-quality reference 相对 QF95-only reference 改善跨 codec 结果，但该改善不能自动归因于新增条件变量；
4. 若 CARF-A 的条件收益能在独立表示和 DARG T2I 上复现，应表现为三种子 `full-M0` 配对 AUROC 增量为正、bootstrap CI 下界大于零且校准 FPR 处于预设范围；
5. 若条件变量标签审计接近随机失败、coverage 不足或独立数据上增量不能复现，则不能把 CARF-A 的收益解释为普适的条件参考机制。

这些预测全部允许实验否定当前机制解释。未通过时，应转向系统性评测与失效机制研究，而不是新增检测模块。

## 7. v12 P1 实证复核后的理论边界

实验结果见 `outputs/carf_protocol/carf_protocol_p1.json`：

- **NP-RSR 跨数据集复现成立。** ARForensics 三种子 nuisance 最低稳定性为 0.981–0.999，DARG T2I 为 0.997–0.999；两者 mean-correction 均消除约 87%–92% 的真实 codec MMD，而七生成器/五生成器的平均正交信号保留率仅约 0.10/0.07。命题 A 所描述的高耦合在两个真实来源与生成器集合上成立。
- **固定高通表示未复现稳定 nuisance 子空间。** 其最低稳定性仅为 ARF 0.193–0.516、DARG 0.024–0.415，mean-correction 经常增大而非减小 MMD。虽然正交均值差能量很低，但由于 U 本身不稳定，不能据此宣称存在同一个可迁移 nuisance 子空间。这一结果否定的是固定高通表示上的可识别性，而不是所有非 NP 表示都存在或不存在 coupling。
- **条件增益与机制解释必须分离。** CARF-A full 在 ARF 的 NP/HP 和 DARG 的 NP 上均有三种子正增量，但 DARG 文件大小标签 AUC=0.977、full 条件 AUC=0.995，且 DARG HP 只有 3/5 生成器改善。因此条件 Mahalanobis 的正增量不能单独证明命题 B 中的理想条件模型已经实现，更不能排除标签代理。
- **理论在论文中的合适作用。** 命题 A/C/D 可解释为什么不变投影、未知组校准和低覆盖评分会失效；命题 B 只提供高斯同协方差下的充分条件。实验不支持“任意残差空间均存在稳定耦合”或“条件评分普遍优于无条件评分”的更强表述。

## 8. 表示条件化的可识别性判据（v13）

将结论显式写成表示与数据分布的函数：对残差映射 `ψ ∈ F`，定义

`U_ψ = InterventionBasis(ψ(r(a))-ψ(r(a0)))`，
`Δ_ψ = E[ψ(r)|fake]-E[ψ(r)|real]`，

以及 `ρ_ψ = ||P_Uψ Δ_ψ||²/||Δ_ψ||²`。只有当 `U_ψ` 在 source-disjoint 重抽样中稳定，且真实 intervention drift 可被一致校正时，才将 `1-ρ_ψ` 解释为可迁移 nuisance 子空间上的信号保留率。

实验预注册的可操作判据为：最低 per-view mean-response cos≥0.9，且 QF85/QF75 的 MMD reduction≥0.3。判据不满足时标记 **unidentified**，而不是根据很低的 η 反推 coupling。

v13 的结果形成三种情况：

- NP-RSR：ARF 与 DARG 均满足可识别性，且各生成器 η 的 bootstrap 上界低于 0.8，支持 coupling。
- 固定 DCT：DARG 满足判据，η=0.005–0.008，支持 coupling；ARF 的 QF75 MMD 下降 72.8%–82.4%，但 QF85 仅 6.6%–56.3%且 bootstrap CI 跨零，因此整体仍为 unidentified。
- 固定高通：两个数据集的 nuisance 稳定性均不满足判据，保持 unidentified。

因此，`ρ_ψ` 不仅依赖表示，还依赖数据分布和 codec 条件；不能从单一表示或单一质量层级推出普遍结论。

## 9. Same-quality 与 mixed-codec 的 bias–variance 分解

对固定真实查询，设第 `t` 个 seed 的条件均值估计为 `μ_hat_t`，全真实参考得到的 proxy-oracle 为 `μ*`。记 `μ_bar=T^(-1)Σ_t μ_hat_t`，则有恒等式

`T^(-1)Σ_t ||μ_hat_t-μ*||²
 = ||μ_bar-μ*||² + T^(-1)Σ_t ||μ_hat_t-μ_bar||²`。

左侧是 proxy MSE，右侧分别是跨 seed 偏差平方和方差。`μ*` 是有限真实参考的经验代理，不是真实条件均值，因此该分解是诊断量而非总体偏差的无偏估计。

v13 中 mixed-codec 的 ESS 约为 same-quality 的 3 倍，但 NP-RSR 的 proxy MSE 明显更高。例如 ARF/QF75：same 的 bias²/variance/MSE 为 0.00034/0.00118/0.00152，mixed 为 0.02924/0.00113/0.03037。DARG/QF75 同样由 0.00078 增至 0.03004。mixed 的多视图来自相同 source，ESS 没有计入视图间相关性，因此更大的 ESS 不等于更多独立真实来源，也没有带来跨 seed 方差下降。

这支持一个有限结论：已知 codec 时，same-quality 的条件偏差优势可以大于 mixed-codec 的样本量收益；不能把 mixed pooling 当作无成本的方差降低。

## 10. 可观测捷径匹配的证据边界

统一分辨率与 codec 后，再做 real/fake 条件匹配，只能检验当前可观测文件大小和复杂度代理是否解释增益。匹配后标签 AUC 仍高时，说明重叠支持不足；匹配后 AUROC 变化也可能包含样本选择效应。因此该分析不能证明因果排除，只能给出以下可证伪结论：

- ARF 在 256×256/JPEG95 下 M0/CARF-size 仅约 0.55，size 增益的 CI 跨零，原生分辨率下的增益不能迁移到该受控协议。
- DARG 在文件大小单独匹配后仍有约 +0.02 的 CARF-size 增益，说明该小规模受控结果不是完全由一维文件大小失衡造成；但其绝对 AUROC 仍约 0.54–0.55，且像素复杂度标签 AUC 在部分生成器上超过 0.8，不能进一步宣称已排除内容分布捷径。
- 全条件匹配后 ARF 部分生成器的组合条件 AUC 仍约 0.87，故“完全排除文件大小与内容分布捷径”的假设未获支持。

## 11. S3R：source-aware 估计的端点与失败边界（v14）

### 11.1 Source-level ESS 不变量

设 source `i` 有视图权重 `w_iq`。source-weighted mixed 先在 source 内得到 `u_iq=w_iq/Σ_q w_iq` 与条件摘要，再以跨视图平均权重 `v_i ∝ (1/Q)Σ_q w_iq` 聚合 source。source-level ESS 为

`ESS_source = (Σ_i v_i)² / Σ_i v_i²`。

若将完整视图集复制 K 次，source 内摘要和跨视图平均权重不变，因此 `ESS_source` 不变；naive view-level ESS 在等权时可增至 K 倍。这就是实现和单元测试强制保持的防膨胀性质，但它不保证条件均值偏差更小。

### 11.2 Hierarchical shrinkage 的精确误差式

设 same 与 source-mixed 的条件均值为 `μ0,μ1`，经验 oracle 为 `μ*`，令 `μ_γ=(1-γ)μ0+γμ1`。则

`||μ_γ-μ*||² = ||e0+γ(e1-e0)||²`，其中 `e0=μ0-μ*、e1=μ1-μ*`。

若 oracle 已知，未约束最优权重为

`γ* = -<e0,e1-e0> / ||e1-e0||²`，再截断到 `[0,1]`。

实际 oracle 未知，S3R 仅用真实 held-out source 的 Gaussian NLL 选择离散 γ。该选择不保证测试 MSE，更不保证 AUROC 提升。γ=0 必须严格退化为 same-quality，γ=1 必须严格退化为 source-weighted mixed，包括各自独立的 reference-only 标准化、带宽、均值、协方差与 ESS。

### 11.3 P0 结果

所有 12 个 dataset×seed×quality 实验组均选择 **γ=0**。因此 S3R 在最终逐图输出上严格等于 same-quality；真实 validation 一致拒绝引入 mixed-codec。其相对 mixed 的 proxy-MSE 显著下降只是继承 same-quality 的性质，不是新方法增益。

GO=false：QF85 的相对较强基线 ΔAUROC=0；ARF/QF75 因 mixed 更强而退化；FPR 也有部分组超出预设区间。按预注册标准停止 S3R，不开展全量或新增模块。实现初期曾让 hierarchical 两个分量共享 mixed 带宽，导致 γ=0 不完全等于 same；该版本已隔离为 `invalid_pre_bandwidth_fix`，最终结果来自修复独立带宽后的正式重跑。
