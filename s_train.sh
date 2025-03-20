#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
if [ "$HF_USER" = "Not logged in" ];then
    HF_USER=$USER
fi

DATA_ROOT=/home/hhws/projects/robot_datasets
#DATASET_NAMES=$HF_USER/so100_bimanual_clothes_5
DATASET_NAMES=$HF_USER/so100_bimanual_transfer_6
POLICY_NAME=pi0
#POLICY_NAME=act
OUTPUT_DIR=outputs/train/pi0_so100_bimanual_transfer_1
JOB_NAME=pi0_so100_bimanual_clothes_6
DEVICE=cuda
STEPS=20000
SAVE_FREQ=5000
BATCH_SIZE=8
ENABLE_IMAGE_TRANSFORM=false
RESUME=false
# PRETRAINED_PATH=outputs/train/act_so100_bimanual_transfer_5/checkpoints/last/pretrained_model
PRETRAINED_PATH=lerobot/pi0

if [ -z $PRETRAINED_PATH ]; then
  if [ $RESUME = true ]; then
    echo "继续训练。"
    python lerobot/scripts/train.py \
    --resume=true \
    --config_path=$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ
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
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ
fi
