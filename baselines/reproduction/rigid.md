# RIGID 复现说明

- 论文: He et al., "RIGID: A Training-free and Model-Agnostic Framework for Robust AI-Generated Image Detection", 2024 (arXiv:2405.20112)
- 官方代码: https://github.com/IBM/RIGID
- 官方 commit: 未 vendored (reimplemented)
- 权重: DINOv2 (torch.hub facebookresearch/dinov2), 无需检测器训练
- Backbone: DINOv2 ViT-S/14 (默认, 见 config)
- 输入分辨率: 224
- 预处理: ImageNet 归一化; 原图 vs 加高斯噪声(noise_std=0.05)图的 DINOv2 特征余弦相似度
- Score: logit = -similarity × scale (scale=10), 值越大越 fake; score_type=anomaly
- 阈值: 验证集 Youden's J 校准 (-9.914), 冻结后用于 test
- 数据: ARForensics test (7 生成器 × 12000)
- 与论文差异: 扰动方式与官方对齐程度未逐项核对; 官方以相似度直报, 本实现取负统一方向
- 状态: reimplemented
- 结果 (v1 P0 修正后, 校准阈值): acc 0.675 / f1 0.655 / AP 0.7421 / AUC 0.7407 / TPR@1%FPR 0.0997
- 注意: v1 修正前 sigmoid+0.5 阈值下的 Acc/F1 无效; AUROC/AP 不受 sigmoid 单调性影响
