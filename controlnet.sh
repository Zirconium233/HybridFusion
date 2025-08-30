#!/usr/bin/env bash
set -euo pipefail

# -------- 基本配置（参考 pretrain_vae.py） --------
PRETRAINED="runwayml/stable-diffusion-v1-5"   # 可替换为本地权重路径
OUTPUT_DIR="./checkpoints/controlnet_4ch_msrs"
SEED=42

# 分辨率（保持 MSRS 原始 480x640，需为 8 的倍数）
HEIGHT=480
WIDTH=640

# 训练超参
BATCH=4
EPOCHS=10
LR=1e-5
ACCUM=1
WORKERS=4
MIXED_PRECISION="bf16"   # 可改为 "fp16" 或 "no"

# 可选：已存在的 4 通道 ControlNet 继续训练
CONTROLNET_PRETRAIN=""

# 可选：xFormers（如已安装）
USE_XFORMERS=0

# 可选：断点恢复
RESUME=""   # 设为 "latest" 或具体 checkpoint 路径，如 "./checkpoints/controlnet_4ch_msrs/checkpoint-5000"

# -------- 环境准备建议 --------
# pip install -U accelerate diffusers transformers torchvision
# accelerate config  # 首次使用需交互配置

cd "$(dirname "$0")/.."

CMD=(accelerate launch Image/train_controlnet_fusion.py
  --pretrained_model_name_or_path "$PRETRAINED"
  --output_dir "$OUTPUT_DIR"
  --dataset_name MSRS
  --height "$HEIGHT"
  --width "$WIDTH"
  --seed "$SEED"
  --train_batch_size "$BATCH"
  --num_train_epochs "$EPOCHS"
  --learning_rate "$LR"
  --gradient_accumulation_steps "$ACCUM"
  --dataloader_num_workers "$WORKERS"
  --mixed_precision "$MIXED_PRECISION"
  --report_to tensorboard
  --logging_dir logs
  --checkpointing_steps 1000
  --checkpoints_total_limit 5
  --allow_tf32
)

# 可选：继续训练已有的 4 通道 ControlNet
if [[ -n "$CONTROLNET_PRETRAIN" ]]; then
  CMD+=(--controlnet_model_name_or_path "$CONTROLNET_PRETRAIN")
fi

# 可选：xFormers
if [[ "$USE_XFORMERS" == "1" ]]; then
  CMD+=(--enable_xformers_memory_efficient_attention)
fi

# 可选：断点恢复
if [[ -n "$RESUME" ]]; then
  CMD+=(--resume_from_checkpoint "$RESUME")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"