# FreqSpec (Frequency) 复现说明

- 论文: Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition", ICML 2020
- 官方代码: https://github.com/RUB-SysSec/GANCTDetection
- 官方 commit: 未 vendored (reimplemented)
- 权重: 本地训练 `outputs/frequency_genimage/best.pth` 等
- Backbone: DCT 频谱 + 分类网络, 见 `baselines/frequency.py`
- 输入分辨率: 224
- 预处理: 灰度 DCT 系数
- 数据: 见对应 config (genimage / arforensics)
- Score: BCE logit
- 与论文差异: 频谱处理与分类头为自行实现, 未与官方逐项对齐
- 状态: reimplemented
- 结果: ARForensics test acc 0.6596 / AP 0.8020 / AUC 0.7862
- 未确认: DCT 归一化方式与官方是否一致
