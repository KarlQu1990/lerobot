import logging
import sys
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import torch
import tqdm

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class
from lerobot.record import busy_wait, predict_action
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_sam01_follower,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


@dataclass
class TestPolicyConfig:
    robot: RobotConfig
    policy: PreTrainedConfig
    task: str | None = None
    fps: int = 30
    inference_time_s: int = 60
    pretrained_name_or_path: str | None = None
    display_data: bool = True


@parser.wrap()
def test(cfg: TestPolicyConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="recording")

    policy_class = get_policy_class(cfg.policy.type)

    if not cfg.pretrained_name_or_path:
        logging.error("预训练权重不能为空。")
        return

    robot = make_robot_from_config(cfg.robot)
    if not robot.is_connected:
        robot.connect()

    policy = policy_class.from_pretrained(cfg.pretrained_name_or_path)
    if cfg.policy.type == "act":
        policy.config.n_action_steps = cfg.policy.n_action_steps

    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.reset()
    policy.to(device)

    obs_features_only = hw_to_dataset_features(robot.observation_features, prefix="observation")

    timestamp = 0.0
    timestamp_int = 0
    start_t = time.perf_counter()
    timestamp = 0
    control_time_s = cfg.inference_time_s

    with tqdm.tqdm(total=control_time_s, desc="推理进度") as pbar:
        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()

            observation = robot.get_observation()

            observation_frame = build_dataset_frame(obs_features_only, observation, prefix="observation")

            with torch.no_grad():
                action_values = predict_action(
                    observation_frame,
                    policy,
                    device,
                    policy.config.use_amp,
                    task=cfg.task,
                    robot_type=robot.robot_type,
                )

            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}

            sent_action = robot.send_action(action)

            if cfg.display_data:
                log_rerun_data(observation, sent_action)

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1.0 / cfg.fps - dt_s)

            timestamp = time.perf_counter() - start_t
            if int(timestamp) > timestamp_int:
                timestamp_int = int(timestamp)
                pbar.update(1)

        if timestamp_int < control_time_s:
            pbar.update(control_time_s - timestamp_int)

    robot.disconnect()


def main():
    test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("##############################")
        import rerun as rr

        rr.rerun_shutdown()
        sys.exit(0)
