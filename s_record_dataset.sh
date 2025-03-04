#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
if [ "$HF_USER" = "Not logged in" ];then
    HF_USER=$USER
fi

ROBOT_TYPE=so100_bimanual
DATA_ROOT=/home/hhws/projects/robot_datasets
DATASET_NAME=$HF_USER/so100_bimanual_transfer_4
TASK_DESC="抓取方块传递给另一个机械臂，然后放到料框里。"
FPS=30
WARMUP_TIME_S=5
EPISODE_TIME_S=90
RESET_TIME_S=30
NUM_EPISODES=50
PRETRAINED_PATH=

if [ -z $PRETRAINED_PATH ]; then
    echo "不带预训练权重录制。"
    python lerobot/scripts/control_robot.py \
    --robot.type=$ROBOT_TYPE \
    --control.type=record \
    --control.root=$DATA_ROOT/$DATASET_NAME \
    --control.fps=$FPS \
    --control.single_task=$TASK_DESC \
    --control.repo_id=$DATASET_NAME \
    --control.tags='["so100","tutorial"]' \
    --control.warmup_time_s=$WARMUP_TIME_S \
    --control.episode_time_s=$EPISODE_TIME_S \
    --control.reset_time_s=$RESET_TIME_S \
    --control.num_episodes=$NUM_EPISODES \
    --control.push_to_hub=false
else
    echo "带预训练权重录制。"
    python lerobot/scripts/control_robot.py \
    --robot.type=$ROBOT_TYPE \
    --control.type=record \
    --control.root=$DATA_ROOT/$DATASET_NAME \
    --control.fps=$FPS \
    --control.single_task=$TASK_DESC \
    --control.repo_id=$DATASET_NAME \
    --control.tags='["so100","tutorial"]' \
    --control.warmup_time_s=$WARMUP_TIME_S \
    --control.episode_time_s=$EPISODE_TIME_S \
    --control.reset_time_s=$RESET_TIME_S \
    --control.num_episodes=$NUM_EPISODES \
    --control.push_to_hub=false \
    --control.policy.path=$PRETRAINED_PATH

fi