import sys
import signal

from pathlib import Path
from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.control_robot import record
from lerobot.common.utils.usb_utils import USBDeviceManager
from lerobot.common.robot_devices.control_configs import RecordControlConfig, PreTrainedConfig


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
    init_logging()

    # 加载USB信息
    USBDeviceManager().load()

    # 参数配置
    data_root = Path(__file__).resolve().parents[1] / "lerobot_datasets"
    username = "hhws"
    robot_type = "so100_bimanual"
    dataset_name = "so100_bimanual_transfer_4"
    single_task = "抓取方块后传递给另一个机械臂，然后放到料筐里。"
    fps = 30
    warmup_time_s = 5
    episode_time_s = 90
    reset_time_s = 30
    num_episodes = 1
    push_to_hub = False
    pretrained_path = None

    # 加载机器人和配置
    repo_id = f"{username}/{dataset_name}"
    robot = make_robot(robot_type)

    policy = None
    if pretrained_path:
        policy = PreTrainedConfig.from_pretrained(pretrained_path)

    config = RecordControlConfig(
        repo_id=repo_id,
        single_task=single_task,
        root=data_root / repo_id,
        policy=policy,
        fps=fps,
        warmup_time_s=warmup_time_s,
        episode_time_s=episode_time_s,
        reset_time_s=reset_time_s,
        num_episodes=num_episodes,
        push_to_hub=push_to_hub,
        tags="experiment",
        display_cameras=True,
    )

    record(robot, config)
