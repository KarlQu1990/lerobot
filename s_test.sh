#!/bin/bash
DATA_ROOT=/home/hhws/projects/robot_datasets
ROBOT_PATH=lerobot/configs/robot/so100_bimanual.yaml
PRETRAINED_PATH=outputs/train/act_so100_bimanual_transfer_3/checkpoints/last/pretrained_model
FPS=30
INFERENCE_TIME_S=300
DEVICE=cuda

python lerobot/scripts/control_robot.py test_policy \
  --robot-path $ROBOT_PATH \
  -p $PRETRAINED_PATH \
  --fps $FPS \
  --inference-time-s $INFERENCE_TIME_S \
  --device $DEVICE