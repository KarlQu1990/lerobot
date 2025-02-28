import sys
import signal

from pathlib import Path
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.scripts.control_robot import record
from lerobot.common.utils.usb_utils import USBDeviceManager


robot = None


def terminate_handler(signum, frame):
    global robot
    print(f"Receive teminate signal: {signum}")

    if robot and robot.is_connected:
        robot.disconnect()

    print("Exit now.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, terminate_handler)
    signal.signal(signal.SIGBREAK, terminate_handler)

    robot_overrides = None
    data_root = Path(__file__).resolve().parents[1] / "lerobot_datasets"
    username = "hhws"

    robot_path = "lerobot/configs/robot/so100_bimanual.yaml"
    dataset_name = "so100_bimanual_transfer_4"
    fps = 30
    warmup_time_s = 5
    episode_time_s = 90
    reset_time_s = 30
    num_episodes = 1
    pretrained_path = None

    # 加载USB信息
    USBDeviceManager().load()

    init_logging()
    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    repo_id = f"{username}/{dataset_name}"
    record(
        robot,
        data_root,
        repo_id,
        fps=fps,
        pretrained_policy_name_or_path=pretrained_path,
        warmup_time_s=warmup_time_s,
        episode_time_s=episode_time_s,
        reset_time_s=reset_time_s,
        num_episodes=num_episodes,
        push_to_hub=False,
        tags="experiment",
        display_cameras=True,
    )
