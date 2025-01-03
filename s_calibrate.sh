#!/bin/bash

ROBOT_PATH=lerobot/configs/robot/so100.yaml

python lerobot/scripts/control_robot.py calibrate \
    --robot-overrides "~cameras" \
    --robot-path $ROBOT_PATH