#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
if [ "$HF_USER" = "Not logged in" ];then
    HF_USER=$USER
fi

DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAMES=$HF_USER/so100_bimanual_transfer_7
POLICY_NAME=act
OUTPUT_DIR=outputs/train/act_so100_bimanual_transfer_7
JOB_NAME=act_so100_bimanual_transfer_7
DEVICE=cuda
STEPS=80000
SAVE_FREQ=10000
BATCH_SIZE=8
ENABLE_IMAGE_TRANSFORM=false
RESUME=false
PRETRAINED_PATH=


if [ -z $PRETRAINED_PATH ]; then
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
    --save_freq=$SAVE_FREQ \
    --resume=$RESUME 
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
    --save_freq=$SAVE_FREQ \
    --resume=$RESUME 
fi