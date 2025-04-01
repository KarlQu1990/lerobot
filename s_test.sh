#!/bin/bash

ROBOT_TYPE=so100_bimanual
FPS=30
INFERENCE_TIME_S=300
DEVICE=cuda

# POLICY_NAME=act
POLICY_NAME=pi0
PRETRAINED_PATH=outputs/train/pi0_so100_bimanual_cube_1/checkpoints/last/pretrained_model
# PRETRAINED_PATH=outputs/train/act_so100_bimanual_transfer_8/checkpoints/last/pretrained_model
# PRETRAINED_PATH=outputs/train/act_so100_bimanual_clothes_5/checkpoints/last/pretrained_model
TASK="把桌面上的方块堆叠起来。"

python lerobot/scripts/control_robot.py \
  --robot.type=$ROBOT_TYPE \
  --control.type=test_policy \
  --control.task=$TASK \
  --control.name=$POLICY_NAME \
  --control.fps=$FPS \
  --control.inference_time_s=$INFERENCE_TIME_S \
  --control.device=$DEVICE \
  --control.pretrained_policy_name_or_path=$PRETRAINED_PATH