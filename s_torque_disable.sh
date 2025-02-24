#!/bin/bash
ROBOT_PATH=lerobot/configs/robot/so100_bimanual.yaml
# ROBOT_PATH=lerobot/configs/robot/so100.yaml

python lerobot/scripts/control_robot.py torque_disable \
    --robot-path $ROBOT_PATH