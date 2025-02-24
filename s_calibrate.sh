#!/bin/bash

ROBOT_PATH=lerobot/configs/robot/so100.yaml
ARMS="main_follower main_leader"

# ROBOT_PATH=lerobot/configs/robot/so100_bimanual.yaml
# ARMS="left_follower right_follower left_leader right_leader"
# ARMS="right_follower right_leader"
# ARMS="left_follower left_leader"

python lerobot/scripts/control_robot.py calibrate \
    --robot-overrides "~cameras" \
    --robot-path $ROBOT_PATH \
    --arms $ARMS