#!/bin/bash

HF_USER=hhws
DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAME=$HF_USER/so100_bimanual_transfer_6

python lerobot/scripts/visualize_dataset_html.py \
  --root $DATA_ROOT/$DATASET_NAME \
  --repo-id $DATASET_NAME