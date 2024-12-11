#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
if [ "$HF_USER" = "Not logged in" ];then
    HF_USER=$USER
fi

DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAME=koch_test3

python lerobot/scripts/visualize_dataset_html.py \
  --root $DATA_ROOT \
  --repo-id ${HF_USER}/$DATASET_NAME