#!/bin/bash
HF_USER=$(huggingface-cli whoami | head -n 1)
if [ "$HF_USER" = "Not logged in" ];then
    HF_USER=$USER
fi

DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAME=$HF_USER/so100_test1
ROBOT_PATH=lerobot/configs/robot/so100.yaml
EPISODE=0
FPS=30

python lerobot/scripts/control_robot.py replay \
    --robot-path $ROBOT_PATH \
    --fps $FPS \
    --root $DATA_ROOT \
    --episode $EPISODE \
    --repo-id $DATASET_NAME