#!/bin/bash
# 跨生成器评估: 在 ARForensics test 划分 (7 种 AR 模型) 上评估所有方法
# 训练只用 LlamaGen (train 划分), test 中的其它模型即为未见过的生成器
# 用法: bash scripts/run_cross_gen.sh [每类最大样本数, 默认 500]
set -x
PY=/opt/anaconda3/envs/ML/bin/python3
N=${1:-500}

for cfg in cnn_detection npr d3qe_lite univfd; do
  ckpt=outputs/${cfg}_arforensics/best.pth
  if [ -f "$ckpt" ]; then
    $PY eval.py --config configs/${cfg}_arforensics.yaml \
        --checkpoint $ckpt --split test \
        --out outputs/${cfg}_arforensics/results_test.json \
        --opts dataset.max_per_class $N
  else
    echo "跳过 $cfg (无权重 $ckpt)"
  fi
done

# RIGID: training-free
$PY eval.py --config configs/rigid_arforensics.yaml --split test \
    --out outputs/rigid_arforensics/results_test.json \
    --opts dataset.max_per_class $N

# D3QE token 统计分析: 跨模型对比
$PY analysis/token_stats.py --root data/ARForensics --split test --n $N \
    --vq_ckpt pretrained/vq_ds16_c2i.pt --out outputs/analysis_test

echo "CROSS_GEN_DONE"
