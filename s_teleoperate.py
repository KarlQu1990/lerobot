import sys
import signal

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.control_robot import teleoperate
from lerobot.common.utils.usb_utils import USBDeviceManager
from lerobot.common.robot_devices.control_configs import TeleoperateControlConfig


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

    robot_type = "so100_bimanual"
    fps = 30
    display_cameras = True

    # 加载机器人和配置
    robot = make_robot(robot_type)
    config = TeleoperateControlConfig(fps=fps, display_cameras=display_cameras)

    teleoperate(robot, config)
