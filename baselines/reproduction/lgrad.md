# LGrad 复现说明

- 论文: Tan et al., "Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection", CVPR 2023
- 官方代码: https://github.com/chuangchuangtan/LGrad-DeepfakeDetection
- 官方 commit: 未 vendored (reimplemented)
- 权重: 本地训练 `outputs/lgrad_arforensics/best.pth`
- 模型: 预训练 CNN 的梯度幅值作为表示 + 分类头, 见 `baselines/lgrad.py`
- 输入分辨率: 224
- 数据: ARForensics-small-val-derived-v1
- Score: BCE logit
- 与论文差异: LGrad 为 GAN 专用方法, 在 AR 上属迁移测试
- 状态: reimplemented; architecture_specific
- 结果: ARForensics test acc 0.7126 / AP 0.8522 / AUC 0.8342
