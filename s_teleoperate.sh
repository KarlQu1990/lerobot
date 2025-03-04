#!/bin/bash

FPS=30
DISPLAY_CAMERAS=true
ROBOT_TYPE="so100_bimanual"

python lerobot/scripts/control_robot.py \
    --robot.type=$ROBOT_TYPE \
    --control.type=teleoperate \
    --control.fps $FPS \
    --control.display_cameras $DISPLAY_CAMERAS
