#!/bin/bash

ROBOT_TYPE="so100_bimanual"

python lerobot/scripts/control_robot.py \
    --robot.type=$ROBOT_TYPE \
    --control.type=show_position