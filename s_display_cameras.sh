#!/bin/bash

FPS=25
PIXEL_FORMAT=MJPG
COLOR_MODE=bgr

python lerobot/scripts/display_cameras.py \
    --fps $FPS \
    --pixel-format $PIXEL_FORMAT \
    --color-mode $COLOR_MODE 