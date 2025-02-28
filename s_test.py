import sys
import signal

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.scripts.control_robot import test_policy
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
    pretrained_path = "outputs/train/act_so100_bimanual_transfer_3/checkpoints/last/pretrained_model"
    fps = 30
    inference_time_s = 120
    device = "cuda"

    # 加载USB信息
    USBDeviceManager().load()

    init_logging()
    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    test_policy(
        robot, fps=fps, inference_time_s=inference_time_s, device=device, pretrained_policy_name_or_path=pretrained_path
    )
