# ZED 复现说明

- 论文: Cozzolino et al., "Zero-Shot Detection of AI-Generated Images", ECCV 2024
- 官方代码: https://github.com/grip-unina/ZED
- 官方 commit: 未 vendored (reimplemented)
- 权重: 官方 SReC checkpoint (多分辨率无损压缩概率模型)
- 模型: SReC, NLL 与 entropy 编码代价差
- 统计量: 默认 abs_delta01 = |d0 - d1| (多分辨率 gap)
- Score: raw statistic, 越大越 fake; score_type=anomaly
- 阈值: 验证集 Youden's J 校准 (0.0451), 冻结后用于 test
- 数据: ARForensics test 每类 500 张采样子集 (非全量, 计算开销限制)
- 与论文差异: 测试为采样子集; 其他与官方统计定义一致
- 状态: reimplemented
- 结果 (校准阈值): acc 0.564 / f1 0.483 / AP 0.5639 / AUC 0.5806 / TPR@1%FPR 0.015
- 注意: 采样数为每类 500, 报告中必须注明
