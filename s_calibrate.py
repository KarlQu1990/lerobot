import sys
import signal

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.control_robot import calibrate
from lerobot.common.utils.usb_utils import USBDeviceManager
from lerobot.common.robot_devices.control_configs import CalibrateControlConfig

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

    robot_type = "so100_bimanual"
    arms = ["left_follower", "right_follower", "left_leader", "right_leader"]
    # arms = ["right_follower", "right_leader"]
    # arms = ["left_follower", "left_leader"]

    # 加载USB信息
    USBDeviceManager().load()

    # 加载机器人和配置
    robot = make_robot(robot_type)
    config = CalibrateControlConfig(arms=arms)

    calibrate(robot, config)
    if robot.is_connected:
        robot.disconnect()
