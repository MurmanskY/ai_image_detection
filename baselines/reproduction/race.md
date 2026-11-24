# RACE 复现说明

- 论文: Mu et al., "Real-Anchored Complementary Evidence for Generalizable
  AI-Generated Image Detection", arXiv:2611.27161v1 (2026-11-20 预印本,
  未宣称顶会接收)
- 官方代码: **无**（论文与 arXiv 页面均未提供代码链接）
- 复现级别: **reimplementation-approximate**（论文忠实近似版）
- 实现: `analysis/race_p0.py`（主流程）, `race_cfe_diagnose.py`（分量诊断）,
  `race_reextract_clip.py`（CLIP 特征缓存迁移）
- 结果: `outputs/race/race_p0.json`, `race_cfe_diagnose.json`;
  汇总见 `reports/RESULTS.md` §15

## 与论文的方法对齐点

- CFE: DINOv2-ViT-L/14 + CLIP-ViT-L/14 冻结, 深度 {6,12,18,24}, 有界残差
  自适应 (η=0.25, γ 初始化 0), shared 64-d / private 128-d 码, Eq.(2) 的
  7 项损失及权重, 40 epoch AdamW (3e-4, 1e-4); 检索 q=ν([√α_D d12; √α_C
  pr_C]) ∈ R^1152, K=16, α 由 dev 上 IQR 尺度估计
- PRP: Gaussian/Uniform/Haar-HH 三个固定探针 (σ=1/255, RGB 空间, clip),
  DINOv2 depth-12 余弦响应, dev median/IQR(数值下限) 归一, 无参数
- NP-RSR: Noiseprint++ (TruFor vendored 权重), 3 个原生 128×128 patch
  (center/top-left/bottom-right), 1968 维描述子, p=0.2 坐标掩码重建,
  仅对 24 个末端残差坐标计分; 融合 S=0.5·Z_sG+0.5·Z_sR, cal real 上
  conformal 阈值 (名义 FPR 5%)

## 近似点（论文未完全指定，均为最直接解释）

1. 门控/投影头/解码器结构: e_b=Linear(d,d) 跨深度共享, 门控 Linear(d,4)
   softmax, 投影头 MLP(d→512→out), 重建解码器 MLP(192→512→d)
2. 子损失定义: VICReg 风格 (view=MSE, align=跨编码器 shared MSE,
   var=relu(1-std), cov=off-diag cov², sep=sh/pr 交叉协方差²,
   rec=重建 CLIP 冻结锚点, geo=批量成对余弦几何 MSE)
3. CLIP 深度特征: resblocks {6,12,18,24} 的裸 CLS token (中间深度无
   ln_post/proj; 当前 open_clip 为 batch-first 布局, 已按官方前向对齐)
4. NP-RSR 描述子: conv6/11/16 taps 经 2×2 avgpool、去 16px 边界后每通道
   [mean,std,min,max,median] (320×3=960); 末端残差 24 维 = 4 分布统计 +
   4 方向自相关 + 4 相位象限能量 + 12 径向谱能量; 掩码 AE 为
   MLP 1968→1024→256→1024→1968
5. CFE 两视图: v0=全局 224 resize, v1=RRC(0.5-1.0)+hflip+ColorJitter
   (md5(path) 作 seed 保证 dino/clip 同视图)

## 数据协议偏差（必须随结果注明）

1. 真实图像池: 论文用 mixed authentic dataset; 本地用 ARForensics
   train(6000)+val(1000) real (ImageNet val 派生), 划分 train 3500 /
   dev 1000 / cal 2500 (论文校准 n=5000)
2. 测试: ARForensics test 7 生成器 × 1000/类 (seed 0, race-p0-1000 示例),
   orig + JPEG95
3. Noiseprint++ 来源: grip-unina/TruFor (clone, commit ae54475,
   `third_party/TruFor/`, 不纳入版本库, 权重随仓库自带)

## 主要结果与结论

- 本复现 macro AUROC: orig 0.744 / JPEG95 0.547 (论文 ARForensics
  报告 94.69%, 协议不同不可直接比较)
- 分量: NP-RSR 最强 (0.65–0.985), PRP 次之 (0.58–0.93),
  CFE 检索反向 (0.096–0.461)
- 关键协议发现: 冻结 DINOv2/CLIP KNN 检索在 ARForensics 上系统性反向
  (AR 生成图比 ImageNet-test real 更接近 ImageNet-val 派生参考库),
  真实参考邻近性在该数据集协议下不是有效证据
- 校准 codec 内成立 (orig FPR≈5%), JPEG95 后 real 分布漂移 FPR≈30%
- 同 subsample 参照: UnivFD 0.908, RIGID (84k) 0.7407

## 复现过程事件

初版自定义 CLIP 前向误用 LND 布局导致 CLIP 分支常量 (首轮 CFE 无效);
已修复并重提特征 (修复后 pooled 与官方前向 cos=1.0)。仅影响本目录
`race_p0.py`, 不影响其他实验。
