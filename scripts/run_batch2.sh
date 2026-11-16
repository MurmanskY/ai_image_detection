#!/bin/bash
# 第二轮批跑: 按优先级排序, 控制单实验时长
set -x
PY=/opt/anaconda3/envs/ML/bin/python3

# 1. RIGID (training-free, 快)
$PY eval.py --config configs/rigid_arforensics.yaml --split val

# 2. D3QE token 统计分析 (real vs LlamaGen, val 子集)
$PY analysis/token_stats.py --root data/ARForensics --split val --n 500 \
    --vq_ckpt pretrained/vq_ds16_c2i.pt --out outputs/analysis

# 3. CNNDetection (经典基线, 5 epochs)
$PY train.py --config configs/cnn_detection_arforensics.yaml
$PY eval.py --config configs/cnn_detection_arforensics.yaml \
    --checkpoint outputs/cnn_detection_arforensics/best.pth --split val

# 4. UnivFD (3 epochs, 只训练线性头)
$PY train.py --config configs/univfd_arforensics.yaml --opts train.epochs 3
$PY eval.py --config configs/univfd_arforensics.yaml \
    --checkpoint outputs/univfd_arforensics/best.pth --split val

# 5. D3QE-lite (3 epochs)
$PY train.py --config configs/d3qe_lite_arforensics.yaml --opts train.epochs 3
$PY eval.py --config configs/d3qe_lite_arforensics.yaml \
    --checkpoint outputs/d3qe_lite_arforensics/best.pth --split val

echo "BATCH2_DONE"
