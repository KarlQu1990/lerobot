#!/bin/bash

DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAME="lerobot/debug"
EPISODE=0
FPS=30

python lerobot/scripts/control_robot.py teleoperate \
    --fps $FPS \
    --root $DATA_ROOT \
    --episode $EPISODE \
    --repo-id $DATASET_NAME