#!/bin/bash

HF_USER=hhws
DATA_ROOT=/home/hhws/projects/robot_datasets
#DATASET_NAMES=$HF_USER/so100_bimanual_clothes_5
DATASET_NAMES=$HF_USER/so100_bimanual_cube_1
# POLICY_NAME=act
POLICY_NAME=pi0
OUTPUT_DIR=outputs/train/pi0_so100_bimanual_cube_1
JOB_NAME=pi0_so100_bimanual_cube_1
STEPS=6000
SAVE_FREQ=500
BATCH_SIZE=8
OPTIMIZER_LR=2.5e-5
ENABLE_IMAGE_TRANSFORM=false
RESUME=true
# PRETRAINED_PATH=outputs/train/act_so100_bimanual_transfer_5/checkpoints/last/pretrained_model
# PRETRAINED_PATH=lerobot/pi0
PRETRAINED_PATH=""

if [ -z $PRETRAINED_PATH ]; then
  if [ $RESUME = true ]; then
    echo "继续训练。"
    python lerobot/scripts/train.py \
    --resume=true \
    --config_path=$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ \
    --policy.optimizer_lr=$OPTIMIZER_LR 
  else
    echo "从头开始训练。"
    python lerobot/scripts/train.py \
      --dataset.repo_id=$DATASET_NAMES \
      --dataset.root=$DATA_ROOT/$DATASET_NAMES \
      --dataset.image_transforms.enable=$ENABLE_IMAGE_TRANSFORM \
      --policy.type=$POLICY_NAME \
      --policy.optimizer_lr=$OPTIMIZER_LR \
      --output_dir=$OUTPUT_DIR \
      --job_name=$JOB_NAME \
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
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ
fi
