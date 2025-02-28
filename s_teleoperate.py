import sys
import signal

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.scripts.control_robot import teleoperate
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
    arms = "left_follower right_follower left_leader right_leader"
    fps = 30
    display_cameras = True

    # 加载USB信息
    USBDeviceManager().load()

    init_logging()
    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    teleoperate(robot, fps=fps, display_cameras=display_cameras)
