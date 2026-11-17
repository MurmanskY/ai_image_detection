# CNNDetection 复现说明

- 论文: Wang et al., "CNN-Generated Images Are Surprisingly Easy to Spot... for Now", CVPR 2020
- 官方代码: https://github.com/PeterWang512/CNNDetection
- 官方 commit: 未 vendored (reimplemented)
- 权重: 无官方权重; 本地训练 `outputs/cnn_detection_arforensics/best.pth`
- Backbone: ResNet-50 (ImageNet 预训练)
- 输入分辨率: 224
- 预处理: Resize+Crop, ImageNet 归一化; train 端随机裁剪/翻转/JPEG 模糊增强见 `baselines/cnn_detection.py`
- 数据增强: 见代码 (高斯模糊 + JPEG, 概率对齐官方风格, 未逐项核对)
- 训练/验证/测试: ARForensics-small-val-derived-v1 (train 6000+6000, val 1000+1000, test 7 生成器×12000)
- 训练: 配置 `configs/cnn_detection_arforensics.yaml`
- Score: BCE logit, sigmoid 后越大越 fake
- 与论文差异: 训练集非官方 ProGAN+真实图, 改为 ARForensics val-derived 小规模协议; 测试集为 AR 生成器
- 状态: reimplemented
- 结果: test acc 0.6988 / AP 0.8554 / AUC 0.8195 (ARForensics test, 统一 evaluator)
- 未确认: 数据增强参数未与官方逐项核对
