# DART 与 Generation-Cycle Residual：方法、理论和 ICML 定位

## 核心区别

Full DART 学习 `f(L,Q,R)`，属于监督式多尺度特征融合。Cycle Residual 先学习真实
参考分布中的条件关系 `R_hat=E[R|L,Q,scale metadata]`，再使用 `R-R_hat`，属于
条件一致性建模。前者关注各阶段特征是否联合具有判别力，后者关注 reconstruction
是否偏离前两个生成阶段所对应的真实条件预期。

Raw residual 不需要 fake labels 来训练检测器；Full DART 需要真实与其他生成家族
的 fake 进行监督训练。二者比较时不能声称监督信息完全一致，应与原始 R、L+R、
Full 以及 Full+residual 同时比较。

## 条件残差定义

对参考模型 `m` 和尺度 `k`，输入包含 likelihood quantiles `L_k`、量化误差 `Q_k`、
尺度元数据 `c_k` 和重建误差 `R_k`。条件预测器为：

\[
\hat R_k=g_m(L_k,Q_k,c_k).
\]

当前实现使用固定 ridge=1 的线性条件模型，只在真实 reference 上拟合。真实
selection 估计残差 robust scale `s_m`：

\[
Z_k=\frac{R_k-\hat R_k}{s_m}.
\]

图像分数为尺度均值，再对 model bank 使用 max、mean 或 median 聚合。最终阈值只由
独立真实 calibration 估计，测试集不参与拟合、模型选择或阈值估计。

## 可证明但有限的性质

对不带正则化的最小二乘条件预测器，训练 reference 上的残差满足正规方程：

\[
X^\top(R-X\hat\beta)=0.
\]

因此残差与条件设计矩阵 `X` 的列空间正交，可解释为去除真实参考中可由
`L,Q,metadata` 线性预测的部分。ridge 版本是收缩估计，不严格满足无正则正规方程。
该性质不意味着：

- 残差与标签条件独立；
- 残差一定提高 AUROC；
- 真实和生成图像必然遵循不同生成循环关系；
- 跨 codec 或跨真实来源阈值自动保持有效。

论文应把这些经验问题交给配对实验，而不是把投影解释写成生成定律。

## 为什么可能超过原始 R

若 `R=R_pred+E`，且 `R_pred` 在真实参考中占据部分与真假标签弱相关的变化，则
去除该成分可以降低参考分布内的可预测变异。R̂ 单独不需要具有很高 AUROC；只要
它解释了一部分阻碍 R 判别的变化，残差就可能改善。真实 selection R² 较低说明
条件解释力有限，因此残差增益必须通过同尺度 R/R̂/residual 分解验证，而不能仅从
R² 推断。

## 当前实验证据

- 显式 L+R/Full interaction 没有稳定超过参数量匹配的融合模型。
- L+R+residual 相对 L+R 的七家族 P0 增益为 +0.00944，3/3 seeds 配对 CI 为正。
- Raw residual 的 current/DARG→ARF/balanced→ARF macro AUROC 为
  0.8768/0.8827/0.8814。
- 同尺度 residual 相对原始 R 在 max/mean/median 下分别提高
  +0.0179/+0.0318/+0.0307，9/9 seeds 的逐图 CI 均为正。
- R̂ 单独仅约 0.59，支持残差而非条件预测本身提供主要增益。

## ICML 创新性对比

| 维度 | Full DART | Cycle Residual |
|---|---|---|
| 科学问题 | 多阶段特征如何联合分类 | 给定前序阶段，重建是否符合真实条件关系 |
| 方法对象 | 特征表示与监督融合 | 条件预测、正交化与一致性偏离 |
| 新信号 | 主要复用已有 L/Q/R | 新构造的条件残差分数 |
| 网络创新 | Deep Sets/attention 均为已有组件 | Raw 版本不需要检测网络 |
| 训练信息 | 需要其他家族 fake labels | Raw 版本条件模型只需真实 reference |
| 理论解释 | 主要是机制动机与消融 | 有投影/正规方程解释，但无检测保证 |
| 主要风险 | 工程组合、监督信息不匹配 | R² 低、聚合依赖、跨域校准与压缩退化 |

最有说服力的论文主线不是“再增加一个特征”，而是：**真实参考中的条件生成关系
能否用于构造可迁移的取证残差，并证明它超出原始 reconstruction 和监督融合。**

## 当前不能声称的内容

不能声称通用未知生成器检测、完全无需目标知识、codec-invariant、零样本外部校准、
严格因果识别或已建立普遍生成定律。当前结果仍是固定测试图上的示例子集，多 seed
衡量划分和训练波动，不是独立数据集复证。

## 后续必须满足的条件

1. Full+residual 相对 Full 的严格参数匹配增量；
2. 外部真实 calibration/test 完全分离时 FPR 风险控制；
3. 更大独立 fake/real 测试集和低 FPR 置信区间；
4. 不同条件模型、尺度和聚合规则的敏感性分析；
5. 与 PRADA、D³QE、Data Provenance 在相同参考 bank 和监督信息下比较。


## Full+residual 与外部校准更新

五 seed 参数匹配实验显示，Full+residual 在 DARG 四家族上提升 +0.0080，但在
ARF 七家族上平均下降 −0.00197，TPR@5% 也下降，GO=false。因此不把 residual
堆叠进 Full，而将其作为独立候选方法与监督 Full 基线比较。

ImageNet-V2 采用 150/150 互斥 calibration/test 后，max residual 的外部 FPR
均值从 9.60% 降至 5.29%，AUROC 保持 0.8848 不变；原始 R 为 6.04%/0.7913。
这支持外部再校准，但不支持零样本阈值保证，小校准集下个别划分 FPR 仍达 10%。
