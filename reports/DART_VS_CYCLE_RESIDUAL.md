# Full DART vs Cycle Residual：ICML 方法与实验对比

## 结论先行

Full DART 是当前最稳健的监督融合基线；Cycle Residual 是当前创新性最强、七家族
示例性能最高的候选方法。Residual 不能作为 Full 的通用附加模块：Full+residual
的 ARF GO=false。最合理的论文路线是研究独立条件残差方法，并把 Full 作为强监督
基线，而不是继续叠加模块。

## 方法差异

| 维度 | Full DART | Raw Cycle Residual |
|---|---|---|
| 输入 | L/Q/R 多尺度特征 | 同样的 L/Q/R，但先学习真实条件关系 |
| 核心计算 | 监督 Deep Sets 学习 `f(L,Q,R)` | `R-E[R|L,Q,metadata]` |
| 检测器训练 | 需要其他家族 fake labels | 不需要 fake 检测器训练 |
| 真实数据用途 | 归一化、训练、选择、校准 | 条件拟合、尺度估计、校准 |
| 创新对象 | 多尺度联合表示 | 新的条件一致性分数 |
| 理论性质 | 机制动机与消融 | 最小二乘投影/正规方程；无检测保证 |
| 主要风险 | 工程组合与监督信息差异 | 低 R²、聚合依赖、跨域校准与压缩退化 |

## 性能总表

### 七家族 ARF 示例

| 方法 | Macro AUROC | TPR@1%FPR | TPR@5%FPR | 备注 |
|---|---:|---:|---:|---|
| 同缺失 bank PRADA | 0.6101 | — | — | 参考覆盖缺失 |
| Full DART（原主协议） | 0.8456±0.0093 | 0.145 | 0.339 | 5 seeds |
| Full（参数匹配版） | 0.8444±0.0100 | — | 0.343 | 5 seeds |
| Full+residual | 0.8425±0.0083 | — | 0.336 | GO=false |
| 原始 R，max | 0.8590±0.0031 | 0.093 | 0.368 | 同尺度消融 |
| Raw residual，max | 0.8768±0.0048 | 0.180 | 0.417 | 9 seeds |
| Raw residual，mean | **0.8802±0.0030** | **0.223** | 0.407 | 9 seeds |

Raw max residual 相对 P0 Full 的逐图 paired ΔAUROC=+0.0236，相对 L+R 为
+0.0295；相对同尺度 R 的 max/mean/median 增益分别为
+0.0179/+0.0318/+0.0307，9/9 seeds 的 CI 均为正。

### 来源迁移

| 协议 | Full | Raw residual max |
|---|---:|---:|
| 当前 real=ARF → ARF | 0.8456 | 0.8768 |
| DARG real/fake → ARF | 0.8434 | 0.8827 |
| ARF+DARG → ARF | 0.8436 | 0.8814 |
| DARG → DARG | — | 0.7186，低 FPR/FPR 较弱 |

Residual 的跨来源宏 AUROC 保持，但不是所有测试域和低 FPR 工作点都更好。

## 创新性判断

Full 的 L/Q/R 分别已有 PRADA、D³QE 和 Data Provenance 基础，Deep Sets 与
attention 也是已有组件，因此主要贡献是联合建模与系统评估。Residual 把问题改写为
条件关系偏离，构造了新的分数，并有同尺度分解证明其超出原始 R，这是更强的
方法创新。

但 residual 的理论贡献目前有限：无正则最小二乘残差满足 `X^T residual=0`，ridge
版本只是收缩近似。这一正交性不能推出 AUROC 提升、因果识别、跨域校准或普遍
生成定律。真实 selection R² 较低，论文应表述为“经验有效的条件去相关残差”，
而不是“发现了强生成物理定律”。

## 稳健性与失效

- C2I max residual 在 original/JPEG95/85/75/50 为
  0.9022/0.8571/0.7908/0.7398/0.6830，不是压缩不变。
- ImageNet-V2 原阈值外部 FPR 约 8%–11%；150/150 外部 calibration/test 后，
  max residual FPR 均值 5.29%、AUROC 0.8848，但个别划分仍达 10%。
- Mean/median 在 C2I 和外部真实上可能明显弱于 max，聚合不能事后随意选择。
- 当前测试每家族 fake=250、test=100，不是完整 benchmark。

## ICML 推荐贡献结构

1. **方法贡献：** 提出真实参考条件下的 generation-cycle residual，将 AR 取证从
   边缘信号和特征融合推进到条件一致性偏离。
2. **实验贡献：** 证明 residual 在相同尺度下稳定超过 R，并在七家族、来源迁移、
   多 codec 和低 FPR 协议中给出配对结果。
3. **系统贡献：** 分析 model-bank coverage、聚合和外部 calibration 的失效边界。
4. **讨论贡献：** 区分条件投影的可证明性质与检测增益的经验性质。

推荐主线：**Learning Generation-Cycle Residuals for Autoregressive Image
Forensics**。Full DART 作为监督融合 baseline，PRADA 作为 model-aware baseline。

## 投稿前仍需完成

- 更大的独立 real/fake 测试集和独立 benchmark；
- 预注册聚合规则，禁止按测试结果选择 max/mean/median；
- 更大外部 calibration 集及 FPR 风险上界；
- 与 D³QE、Data Provenance 的监督信息和参考 bank 公平对照；
- 条件模型形式、逐尺度残差和 ridge 敏感性分析；
- 在新数据上一次性确认当前 residual 增益，避免示例集过拟合。

## 复现入口

- `analysis/dart_interaction.py`
- `analysis/dart_cycle_validation.py`
- `analysis/dart_cycle_decomposition.py`
- `analysis/dart_cycle_robustness.py`
- `analysis/dart_full_residual_comparison.py`
- `analysis/dart_external_calibration.py`
- `reports/RESULTS.md` 第 35–38 节
