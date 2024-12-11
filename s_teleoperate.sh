#!/bin/bash

FPS=30
DISPLAY_CAMERAS=1
ROBOT_PATH=lerobot/configs/robot/so100.yaml

python lerobot/scripts/control_robot.py teleoperate \
    --fps $FPS \
    --display-cameras $DISPLAY_CAMERAS \
    --robot-path $ROBOT_PATH