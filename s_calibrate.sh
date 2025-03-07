#!/bin/bash

ROBOT_TYPE="so100_bimanual"
# ARMS='["left_follower","left_leader","right_follower","right_leader"]'
# ARMS='["right_follower","right_leader"]'
ARMS='["left_follower","left_leader"]'

python lerobot/scripts/control_robot.py  \
    --robot.type=$ROBOT_TYPE \
    --control.type=calibrate \
    --control.arms=$ARMS