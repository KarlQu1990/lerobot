import sys
import signal

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.train import train
from lerobot.common.utils.usb_utils import USBDeviceManager
from lerobot.common.robot_devices.control_configs import TestPolicyConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

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
    username = "hhws"
    dataset_name = "so100_bimanual_transfer_4"
    pretrained_path = None
    fps = 30
    inference_time_s = 120
    device = "cuda"
    policy_type = "act"
    output_dir = "outputs/train/act_so100_bimanual_transfer_1"
    use_amp = True
    seed = 1234
    batch_size = 8
    steps = 100_000
    eval_freq = 20_000
    log_freq = 200
    save_freq = 20_000
    optimizer_type = "adam"
    lr = 0.001

    repo_id = f"{username}/{dataset_name}"
    # DATASET_NAMES=$HF_USER/so100_bimanual_transfer_3
    # POLICY_NAME=act_so100_bimanual_real
    # ENV_NAME=so100_bimanual_real
    # RUN_DIR=outputs/train/act_so100_bimanual_transfer_3
    # JOB_NAME=act_so100_bimanual_transfer_3
    # DEVICE=cuda
    # OFFLINE_STEPS=100000
    # SAVE_FREQ=10000
    # BATCH_SIZE=8
    # LR=0.00001
    # LR_BACKBONE=0.00001
    # WEIGHT_DECAY=0.0001
    # VISION_BACKBONE=resnet18
    # PRETRAINED_BACKBONE_WEIGHTS=ResNet18_Weights.IMAGENET1K_V1
    # ENABLE_IMAGE_TRANSFORM=false
    # RESUME=false

    config = TrainPipelineConfig(
        fps=fps, inference_time_s=inference_time_s, pretrained_policy_name_or_path=pretrained_path, device=device
    )

    train(robot, config)
