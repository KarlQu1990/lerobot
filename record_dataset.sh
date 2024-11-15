#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
DATA_ROOT=/home/hhws/projects/robot_datasets
ROBOT_PATH=lerobot/configs/robot/koch.yaml
DATASET_NAME=koch_test3
FPS=30
WARMUP_TIME_S=5
EPISODE_TIME_S=60
RESET_TIME_S=30
NUM_EPISODES=50

python lerobot/scripts/control_robot.py record \
  --robot-path $ROBOT_PATH \
  --fps $FPS \
  --root $DATA_ROOT \
  --repo-id ${HF_USER}/$DATASET_NAME \
  --push-to-hub 0 \
  --tags experiment \
  --warmup-time-s $WARMUP_TIME_S \
  --episode-time-s $EPISODE_TIME_S\
  --reset-time-s $RESET_TIME_S \
  --num-episodes $NUM_EPISODES
