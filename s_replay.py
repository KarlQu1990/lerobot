import sys
import signal

from pathlib import Path
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.scripts.control_robot import replay
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
    robot_path = "lerobot/configs/robot/so100_bimanual.yaml"
    data_root = Path(__file__).resolve().parents[1] / "lerobot_datasets"
    username = "hhws"

    dataset_name = "so100_bimanual_transfer_4"
    fps = 30
    episode = 0

    # 加载USB信息
    USBDeviceManager().load()

    init_logging()
    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    repo_id = f"{username}/{dataset_name}"
    replay(
        robot,
        episode,
        fps=fps,
        root=data_root,
        repo_id=repo_id,
    )
