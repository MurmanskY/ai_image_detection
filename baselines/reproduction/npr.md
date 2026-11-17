# NPR 复现说明

- 论文: Tan et al., "Rethinking the Up-Sampling Operations in CNN-Based Generative Network for Generalizable Deepfake Detection", CVPR 2024
- 官方代码: https://github.com/chuangchuangtan/NPR-DeepfakeDetection
- 官方 commit: 未 vendored (inspired 简化实现)
- 权重: 本地训练 `outputs/npr_arforensics/best.pth`
- Backbone: ResNet
- 输入分辨率: 224
- 预处理: NPR 残差 `x - nearest_up(nearest_down(x))` ×4
- 数据: ARForensics-small-val-derived-v1
- Score: BCE logit
- 与论文差异: 为 NPR-inspired 简化版; 官方 NPR 模块结构与训练协议未完整复现
- 状态: inspired
- 结果: ARForensics test acc 0.7478 / AP 0.8898 / AUC 0.8604
- 注意: 结果不能与官方 NPR 直接比较; 正式对比应运行作者官方代码/权重
