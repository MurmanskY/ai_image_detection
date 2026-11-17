# D³QE 复现说明

- 论文: Zhang et al., "D³QE: Learning Discrete Distribution Discrepancy-aware Quantization Error for Autoregressive-Generated Image Detection", ICCV 2025
- 官方代码: https://github.com/Zhangyr2022/D3QE (vendored 于 `third_party/D3QE`)
- 官方权重: LlamaGen VQ `pretrained/vq_ds16_c2i.pt`

## d3qe_official (vendored)

- 运行方式: `third_party/D3QE` 官方脚本直接运行 (train + eval_model + 鲁棒性)
- 结果: 主测试 avg acc 0.7535 / AP 0.8269; 鲁棒性 JPEG 95→50 avg acc 0.715→0.641 / AP 0.789→0.731; crop 0.9→0.5 acc 0.667→0.567 / AP 0.682→0.586
- 输出: `third_party/D3QE/results/D3QE/` (CSV 已写出)
- 训练数据: 官方 train 分卷未下载, 训练协议与官方 100k/100k 不可直接比较 (ARForensics-small-val-derived-v1)
- 状态: vendored + 已运行 (训练数据协议不同, 结果非官方协议数字)

## d3qe_lite (lite)

- 实现: `baselines/d3qe_lite.py` — LlamaGen VQ token histogram + entropy/unique/q-error 标量 + MLP
- 与官方差异: 无 D³-Attention, 无 semantic feature fusion, 无动态 codebook statistics
- 状态: lite (明确简化版本, 不得标为 D³QE 复现)
- 结果: ARForensics test acc 0.6328 / AP 0.6958 / AUC 0.6885

## token 统计分析 (analysis/token_stats.py v2)

- 单变量 token 统计 (entropy/gini/transition/q-error) 在 7 个生成器上 AUC 0.37–0.63 且方向跨生成器不一致, 说明 bag-of-tokens 统计受 tokenizer affinity 影响, 不能单独作为通用 AR 指纹
- 输出: `outputs/analysis_test_v2/` (summary + per_image_features.csv + 图)
