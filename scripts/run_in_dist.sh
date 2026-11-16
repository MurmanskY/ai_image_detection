#!/bin/bash
# 顺序运行全部训练式方法的训练 + 同域评估 (ARForensics val 重划分子集)
# 用法: bash scripts/run_in_dist.sh
set -x
PY=/opt/anaconda3/envs/ML/bin/python3

for cfg in cnn_detection_arforensics npr_arforensics d3qe_lite_arforensics univfd_arforensics; do
  $PY train.py --config configs/$cfg.yaml
  $PY eval.py --config configs/$cfg.yaml \
      --checkpoint outputs/${cfg}/best.pth --split val
done

# RIGID: training-free, 直接评估
$PY eval.py --config configs/rigid_arforensics.yaml --split val

# D3QE 核心统计分析 (real vs LlamaGen, val 子集)
$PY analysis/token_stats.py --root data/ARForensics --split val --n 500 \
    --vq_ckpt pretrained/vq_ds16_c2i.pt --out outputs/analysis

echo "ALL_EXPERIMENTS_DONE"
