# DIRE 复现说明

- 论文: Wang et al., "DIRE for Diffusion-Generated Image Detection", ICCV 2023
- 官方代码: https://github.com/ZhendongWang6/DIRE
- 官方 commit: 未 vendored (reimplemented)
- 权重: ADM `pretrained/256x256_diffusion_uncond.pt` (官方公开权重) + 本地训练 ResNet50 `outputs/dire_arforensics/best.pth`
- Backbone: 扩散模型反演重建残差 + ResNet50
- 输入分辨率: 256
- 预处理: 20-step DDIM 反演 + 重建, 取 |x - x_recon| 残差
- 数据: ARForensics-small-val-derived-v1
- 训练: 2 epochs, 见 `configs/dire_arforensics.yaml` (batch 4, lr 1e-4)
- Score: BCE logit
- 与论文差异: 官方为完整步数 DDIM, 此处 20 步近似; DIRE 为 Diffusion 专用方法, 在 AR 上属迁移测试
- 状态: reimplemented; architecture_specific
- 结果: val acc 0.933 / AP 0.9818 / AUC 0.9790 / TPR@1%FPR 0.825; test 结果待 84k 全量评估完成
- 注意: 反演开销大 (~4.2s/img), test 84k 张需较长时间
