#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
if [ "$HF_USER" = "Not logged in" ];then
    HF_USER=$USER
fi

DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAMES=$HF_USER/so100_bimanual_clothes_5
POLICY_NAME=act
OUTPUT_DIR=outputs/train/act_so100_bimanual_clothes_5
JOB_NAME=act_so100_bimanual_clothes_5
DEVICE=cuda
STEPS=300000
SAVE_FREQ=20000
BATCH_SIZE=8
ENABLE_IMAGE_TRANSFORM=false
RESUME=false
USE_AMP=false
# PRETRAINED_PATH=outputs/train/act_so100_bimanual_transfer_5/checkpoints/last/pretrained_model
PRETRAINED_PATH=

if [ -z $PRETRAINED_PATH ]; then
  if [ $RESUME = true ]; then
    echo "继续训练。"
    python lerobot/scripts/train.py \
    --resume=true \
    --config_path=$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ \
    --use_amp=$USE_AMP 
  else
    echo "从头开始训练。"
    python lerobot/scripts/train.py \
      --dataset.repo_id=$DATASET_NAMES \
      --dataset.root=$DATA_ROOT/$DATASET_NAMES \
      --dataset.image_transforms.enable=$ENABLE_IMAGE_TRANSFORM \
      --policy.type=$POLICY_NAME \
      --output_dir=$OUTPUT_DIR \
      --job_name=$JOB_NAME \
      --device=$DEVICE \
      --use_amp=$USE_AMP \
      --batch_size=$BATCH_SIZE \
      --steps=$STEPS \
      --save_freq=$SAVE_FREQ
  fi
else
  echo "开始微调：$PRETRAINED_PATH"
  python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_NAMES \
    --dataset.root=$DATA_ROOT/$DATASET_NAMES \
    --dataset.image_transforms.enable=$ENABLE_IMAGE_TRANSFORM \
    --policy.path=$PRETRAINED_PATH \
    --output_dir=$OUTPUT_DIR \
    --job_name=$JOB_NAME \
    --device=$DEVICE \
    --use_amp=$USE_AMP \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ
fi