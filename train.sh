#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAME=koch_test3
POLICY_NAME=act_koch_real
ENV_NAME=koch_real
RUN_DIR=outputs/train/act_koch_test3 
JOB_NAME=act_koch_test
DEVICE=cuda

DATA_DIR=$DATA_ROOT python lerobot/scripts/train.py \
  dataset_repo_id=${HF_USER}/$DATASET_NAME \
  policy=$POLICY_NAME \
  env=$ENV_NAME \
  hydra.run.dir=$RUN_DIR \
  hydra.job.name=$JOB_NAME \
  device=$DEVICE \
  wandb.enable=false