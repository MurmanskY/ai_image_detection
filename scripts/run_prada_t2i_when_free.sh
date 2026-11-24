#!/bin/bash
# PRADA T2I 收尾看门狗: 等待任一 GPU 空闲显存 >= NEED_MB 后运行 predict.py。
# 仅剩 dataset=switti_1024_model=switti_1024.pt 一个特征 + 最终评估。
# predict.py 失败 (如被其他进程挤占 OOM) 时自动回到等待循环重试。
set -u
cd /data/yuanyi/ai_image_detection/third_party/PRADA

PROJ=/data/yuanyi/ai_image_detection
TARGET="$PROJ/outputs/prada_t2i/features/dataset=switti_1024_model=switti_1024.pt"
NEED_MB=13000
PY=/home/yuanyi/anaconda3/envs/pytorch_pj/bin/python

while true; do
  if [ -f "$TARGET" ]; then
    echo "[$(date '+%F %T')] 目标特征已存在, 无需再跑"
    exit 0
  fi
  best_gpu=-1; best_free=0
  for g in 0 1; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g")
    free=$((total - used))
    if [ "$free" -gt "$best_free" ]; then best_free=$free; best_gpu=$g; fi
  done
  echo "[$(date '+%F %T')] 最空闲 GPU$best_gpu: ${best_free}MiB 可用 (需要 ${NEED_MB}MiB)"
  if [ "$best_free" -ge "$NEED_MB" ]; then
    echo "[$(date '+%F %T')] 显存足够, 在 GPU$best_gpu 启动 predict.py"
    CUDA_VISIBLE_DEVICES=$best_gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HTTPS_PROXY=http://10.177.73.90:7890 HTTP_PROXY=http://10.177.73.90:7890 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    "$PY" predict.py \
      --real_image_dir "$PROJ/data/darg/darg/t2i/real" \
      --fake_image_dirs "[\"$PROJ/data/darg/darg/t2i/infinity_2b\",\"$PROJ/data/darg/darg/t2i/janus_1b\",\"$PROJ/data/darg/darg/t2i/llamagen_xlstage2\",\"$PROJ/data/darg/darg/t2i/switti_1024\"]" \
      --output_root "$PROJ/outputs/prada_t2i" \
      --mode t2i --train_size 0 \
      --prada_weight_dir prada_weights \
      --model_weight_dir model_weights \
      --num_workers 4
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$TARGET" ]; then
      echo "PRADA_T2I_FULL_DONE"
      exit 0
    fi
    echo "[$(date '+%F %T')] predict.py 退出码 $rc 且特征未完成, 回到等待循环"
  fi
  sleep 240
done
