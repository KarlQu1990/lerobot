import sys
import signal

from pathlib import Path
from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.control_robot import replay
from lerobot.common.utils.usb_utils import USBDeviceManager
from lerobot.common.robot_devices.control_configs import ReplayControlConfig


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

    data_root = Path(__file__).resolve().parents[1] / "lerobot_datasets"
    username = "hhws"
    robot_type = "so100_bimanual"
    dataset_name = "so100_bimanual_transfer_4"
    fps = 30
    episode = 0

    # 加载机器人和配置
    repo_id = f"{username}/{dataset_name}"
    robot = make_robot(robot_type)
    config = ReplayControlConfig(repo_id=repo_id, root=data_root / repo_id, fps=fps)

    replay(robot, config)
