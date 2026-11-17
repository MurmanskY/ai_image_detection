# PRADA 复现说明

- 论文: Ricker et al., "PRADA: Probability-Ratio-Based Attribution and Detection of Autoregressive-Generated Images", CVPR Findings 2026 (arXiv:2511.20068)
- 官方代码: https://github.com/jonasricker/prada (vendored 于 `third_party/PRADA`)
- 官方权重: `third_party/PRADA/prada_weights/` (12 个评分函数), `model_weights/` (janus/switti/llamagen/var/rar/hmar/infinity/blip2 全套)
- 运行方式: 官方 `predict.py`, mode=t2i / c2i
- 数据: DARG (`data/darg/darg/{t2i,c2i}`)
- Score: 官方 log 概率比, 越大越 fake

## 协议偏差 (必须随结果注明)

1. T2I real: 官方使用 RAISE-1k; 本地用 ARForensics real 1000 张替代, 与官方子集不逐张一致
2. C2I real: 官方为特定 ImageNet 子集; 本地用 HF 镜像 `Tsomaros/Imagenet-1k_validation` 10000 张/1000 类子集 (`scripts/prepare_imagenet_subset_hf.py`), 与官方子集不逐张一致
3. C2I fake: 使用 8/16 个目录 (hmar_d16/d24, llamagen_b256, rar_b/l/xl, var_d16/d36)

## 本地修复

- `src/external/rar/data.py`: 上游 RAR 仓库的 data 模块未被 vendor, 添加推理用 stub (train-only 类)
- `src/external/Janus/janus/models/modeling_vlm.py`: `params: AttrDict = field(default_factory=AttrDict)` ×5 + `all_tied_weights_keys` property
- LlamaGen/Infinity: 设备与 SDPA 回退补丁

## 状态

- vendored; T2I/C2I 已端到端运行 (特征缓存于 `outputs/prada_{t2i,c2i}/features/`)
- 未接入统一 eval.py, 结果由官方脚本产出
