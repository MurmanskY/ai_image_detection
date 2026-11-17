# UnivFD 复现说明

- 论文: Ojha et al., "Towards Universal Fake Image Detectors That Generalize Across Generative Models", CVPR 2023
- 官方代码: https://github.com/WisconsinAIVision/UniversalFakeDetect
- 官方 commit: 未 vendored (结构对齐官方 fix_backbone 线性 probing)
- 权重: CLIP ViT-L/14 官方预训练 + 本地训练线性头 `outputs/univfd_arforensics/best.pth`
- Backbone: CLIP ViT-L/14 (冻结) + 线性分类头
- 输入分辨率: 224
- 预处理: CLIP 官方均值方差归一化; train 端 RandomResizedCrop(0.8-1.0)+翻转
- 数据: ARForensics-small-val-derived-v1
- 训练: 只训练线性头 (对齐官方 --fix_backbone)
- Score: BCE logit
- 与论文差异: 训练数据非官方 ProGAN 协议, 为 ARForensics val-derived
- 状态: faithful_reproduction (结构/预处理对齐, 数据协议不同)
- 结果: ARForensics test acc 0.8017 / AP 0.9112 / AUC 0.9127
- 未确认: 官方在 AR 数据上无报告值, 无法直接对照
