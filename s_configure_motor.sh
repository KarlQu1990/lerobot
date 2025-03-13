#!/bin/bash

ID=$1

python lerobot/scripts/configure_motor.py  \
    --port /dev/ttyUSB3 \
    --brand feetech \
    --model sts3215 \
    --ID $ID