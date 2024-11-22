#!/bin/bash

FPS=30
DISPLAY_CAMERAS=0

python lerobot/scripts/control_robot.py teleoperate \
    --fps $FPS \
    --display-cameras $DISPLAY_CAMERAS