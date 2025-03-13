#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
if [ "$HF_USER" = "Not logged in" ];then
    HF_USER=$USER
fi

DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAME=$HF_USER/so100_bimanual_clothes_4

python lerobot/scripts/visualize_dataset_html.py \
  --root $DATA_ROOT/$DATASET_NAME \
  --repo-id $DATASET_NAME