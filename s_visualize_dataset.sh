#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
if [ "$HF_USER" = "Not logged in" ];then
    HF_USER=$USER
fi

DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAME=$HF_USER/so100_bimanual_clothes_1

python lerobot/scripts/visualize_dataset_html.py \
  --root $DATA_ROOT \
  --repo-id $DATASET_NAME