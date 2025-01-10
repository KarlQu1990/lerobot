#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
if [ "$HF_USER" = "Not logged in" ];then
    HF_USER=$USER
fi

DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAMES=$HF_USER/so100_bimanual_clothes_2
POLICY_NAME=act_so100_bimanual_real
ENV_NAME=so100_bimanual_real
RUN_DIR=outputs/train/act_so100_bimanual_clothes_2
JOB_NAME=act_so100_bimanual_clothes_2
DEVICE=cuda
OFFLINE_STEPS=100000
SAVE_FREQ=10000
BATCH_SIZE=8
LR=0.00001
LR_BACKBONE=0.00001
WEIGHT_DECAY=0.0001
VISION_BACKBONE=resnet18
PRETRAINED_BACKBONE_WEIGHTS=ResNet18_Weights.IMAGENET1K_V1

DATA_DIR=$DATA_ROOT python lerobot/scripts/train.py \
  "++dataset_repo_id=[${DATASET_NAMES}]" \
  policy=$POLICY_NAME \
  env=$ENV_NAME \
  hydra.run.dir=$RUN_DIR \
  hydra.job.name=$JOB_NAME \
  device=$DEVICE \
  wandb.enable=false \
  training.offline_steps=$OFFLINE_STEPS \
  training.save_freq=$SAVE_FREQ \
  training.batch_size=$BATCH_SIZE \
  training.lr=$LR \
  training.lr_backbone=$LR_BACKBONE \
  training.weight_decay=$WEIGHT_DECAY \
  policy.vision_backbone=$VISION_BACKBONE \
  policy.pretrained_backbone_weights=$PRETRAINED_BACKBONE_WEIGHTS