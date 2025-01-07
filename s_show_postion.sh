#!/bin/bash

DISPLAY_CAMERAS=1
ROBOT_PATH=lerobot/configs/robot/so100_bimanual.yaml
# ROBOT_PATH=lerobot/configs/robot/so100.yaml

python lerobot/scripts/control_robot.py show_position \
    --robot-path $ROBOT_PATH