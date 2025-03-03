import sys
import signal

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.control_robot import test_policy
from lerobot.common.utils.usb_utils import USBDeviceManager
from lerobot.common.robot_devices.control_configs import TestPolicyConfig

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
    pretrained_path = "outputs/train/act_so100_bimanual_transfer_3/checkpoints/last/pretrained_model"
    fps = 30
    inference_time_s = 120
    device = "cuda"

    robot = make_robot(robot_type)
    config = TestPolicyConfig(
        fps=fps, inference_time_s=inference_time_s, pretrained_policy_name_or_path=pretrained_path, device=device
    )

    test_policy(robot, config)
