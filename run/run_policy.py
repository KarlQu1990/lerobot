import logging
import time

import torch
import tqdm

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.record import busy_wait, predict_action
from lerobot.robots.bi_sam01_follower import BiSAM01Follower, BiSAM01FollowerConfig
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

# ========= 基本参数 =========
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 180
TASK_DESCRIPTION = "拆垛"

# ========= 相机&机器人 =========
camera_config = {
    "high": OpenCVCameraConfig(index_or_path="high", width=640, height=480, fps=FPS),
    "low": OpenCVCameraConfig(index_or_path="low", width=640, height=480, fps=FPS),
    # "left": OpenCVCameraConfig(index_or_path="left_wrist", width=640, height=480, fps=FPS),
    # "right": OpenCVCameraConfig(index_or_path="right_wrist", width=640, height=480, fps=FPS),
}

robot_config = BiSAM01FollowerConfig(
    left_arm_port="left_follower",
    right_arm_port="right_follower",
    id="robot_arm_follower",
    cameras=camera_config,
)
robot = BiSAM01Follower(robot_config)

# ========= 策略 =========
policy = ACTPolicy.from_pretrained(
    "C:/Users/chlai/projects/lerobot_trained_models/act_sam01_luosi_fenjian_load/checkpoints/last/pretrained_model"
)
policy.eval()
policy.reset()

obs_features_only = hw_to_dataset_features(robot.observation_features, prefix="observation")

# ========= 事件监听 & 可视化（仅显示，不写盘）=========
from lerobot.utils.control_utils import init_keyboard_listener

_, events = init_keyboard_listener()
_init_rerun(session_name="inference_only")


def inference_loop(
    robot,
    events: dict,
    fps: int,
    policy,
    control_time_s: int,
    single_task: str | None = None,
    display_data: bool = False,
):
    """
    纯推理循环：obs -> predict_action -> send_action
    无数据集、无视频写入。
    """
    timestamp = 0.0
    timestamp_int = 0
    start_episode_t = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tqdm.tqdm(total=control_time_s, desc="推理进度") as pbar:
        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()

            if events.get("exit_early", False):
                events["exit_early"] = False
                break

            observation = robot.get_observation()

            observation_frame = build_dataset_frame(obs_features_only, observation, prefix="observation")

            with torch.no_grad():
                action_values = predict_action(
                    observation_frame,
                    policy,
                    device,
                    policy.config.use_amp,
                    task=single_task,
                    robot_type=robot.robot_type,
                )

            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}

            sent_action = robot.send_action(action)

            if display_data:
                log_rerun_data(observation, sent_action)

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1.0 / fps - dt_s)

            timestamp = time.perf_counter() - start_episode_t
            if int(timestamp) > timestamp_int:
                timestamp_int = int(timestamp)
                pbar.update(1)

        if timestamp_int < control_time_s:
            pbar.update(control_time_s - timestamp_int)


# ========= 主流程 =========
robot.connect()
log_say("开始纯推理")

try:
    for ep in range(NUM_EPISODES):
        log_say(f"推理评估：第 {ep + 1}/{NUM_EPISODES} 条（任务：{TASK_DESCRIPTION}）")
        if hasattr(robot, "reset"):
            try:
                robot.reset()
            except Exception as e:
                logging.info(f"reset 跳过：{e}")

        inference_loop(
            robot=robot,
            events=events,
            fps=FPS,
            policy=policy,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        if events.get("exit_early", False):
            log_say("收到退出指令，提前结束。")
            break
finally:
    robot.disconnect()
    log_say("推理完成（未录制视频/未写入数据集）。")
