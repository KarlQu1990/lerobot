# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

- For remote controlled robots like LeKiwi, run this script on the robot edge device (e.g. RaspBerryPi):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.

- Train on this dataset with the ACT policy:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import logging
import os
import platform
import signal
import sys
import time
from dataclasses import asdict
from pprint import pformat

import rerun as rr

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    ShowPositionConfig,
    TeleoperateControlConfig,
    TestPolicyConfig,
    TorqueDisableConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    is_headless,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.usb_utils import USBDeviceManager
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

########################################################################################
# Control modes
########################################################################################
robot: Robot = None


def terminate_handler(signum, frame):
    global robot
    print(f"Receive teminate signal: {signum}")

    if robot and robot.is_connected:
        robot.disconnect()

    print("Exit now.")
    sys.exit(0)


@safe_disconnect
def calibrate(robot: Robot, cfg: CalibrateControlConfig):
    # TODO(aliberts): move this code in robots' classes
    if robot.robot_type.startswith("stretch"):
        if not robot.is_connected:
            robot.connect()
        if not robot.is_homed():
            robot.home()
        return

    arms = robot.available_arms if cfg.arms is None else cfg.arms
    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    available_arms_str = " ".join(robot.available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:
        raise ValueError(f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`.")

    for arm_id in arms:
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:
        robot.disconnect()

    if robot.robot_type.startswith("lekiwi") and "main_follower" in arms:
        print("Calibrating only the lekiwi follower arm 'main_follower'...")
        robot.calibrate_follower()
        return

    if robot.robot_type.startswith("lekiwi") and "main_leader" in arms:
        print("Calibrating only the lekiwi leader arm 'main_leader'...")
        robot.calibrate_leader()
        return

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    robot.to_teleoperate_mode()

    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_data=cfg.display_data,
    )


@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # TODO(rcadene): Add option to record logs
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    if not robot.is_connected:
        robot.connect()

    robot.to_teleoperate_mode()

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("初始化录制", cfg.play_sounds)
    warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_data, cfg.fps)

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    recorded_episodes = dataset.num_episodes
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        log_say(f"录制第{dataset.num_episodes + 1}个视频", cfg.play_sounds)
        record_episode(
            robot=robot,
            dataset=dataset,
            events=events,
            episode_time_s=cfg.episode_time_s,
            display_data=cfg.display_data,
            policy=policy,
            fps=cfg.fps,
            single_task=cfg.single_task,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and ((recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]):
            log_say("重新布置环境", cfg.play_sounds)
            reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

        if events["rerecord_episode"]:
            log_say("重新录制视频", cfg.play_sounds)

            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset._wait_image_writer()
            dataset.clear_episode_buffer()
            time.sleep(1)
            continue

        logging.info("start to save episode...")
        dataset.save_episode()
        logging.info("episode saved.")

        events["rerecord_episode"] = False
        events["exit_early"] = False

        recorded_episodes += 1

        if events["stop_recording"]:
            break

    log_say("停止录制", cfg.play_sounds, blocking=True)
    stop_recording(robot, listener, cfg.display_data)

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("录制结束", cfg.play_sounds)
    return dataset


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
    actions = dataset.hf_dataset.select_columns("action")

    if not robot.is_connected:
        robot.connect()

    robot.to_teleoperate_mode()

    # log_say("播放视频", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


@safe_disconnect
def show_position(robot: Robot, cfg: ShowPositionConfig):
    if not robot.is_connected:
        robot.connect()

    robot.torque_disable()

    for name, arm in robot.leader_arms.items():
        pos = arm.read("Present_Position")
        pos_str = ", ".join([f"{x:.2f}" for x in pos])
        logging.info(f"主臂{name}, 关节角: {pos_str}")

    for name, arm in robot.follower_arms.items():
        pos = arm.read("Present_Position")
        pos_str = ", ".join([f"{x:.2f}" for x in pos])
        logging.info(f"从臂{name}, 关节角: {pos_str}")


@safe_disconnect
def test_policy(robot: Robot, cfg: TestPolicyConfig):
    import torch

    from lerobot.common.policies.factory import get_policy_class

    policy_class = get_policy_class(cfg.policy_name)

    if not cfg.pretrained_name_or_path:
        logging.error("预训练权重不能为空。")
        return

    if not robot.is_connected:
        robot.connect()

    robot.to_follower_control_mode()

    policy = policy_class.from_pretrained(cfg.pretrained_name_or_path)
    policy.config.n_action_steps = cfg.n_action_steps
    policy.reset()

    policy.to(cfg.device)

    time_cost = 0
    count = 0
    log_interval = 1
    last_log_t = 0

    while True:
        start_time = time.perf_counter()

        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()

        if not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)

        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(cfg.device)

        if cfg.task:
            observation["task"] = [cfg.task]

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)
        # Remove batch dimension
        action = action.squeeze(0)
        # Move to cpu, if not already the case
        action = action.to("cpu")
        # Order the robot to move
        robot.send_action(action)

        dt_s = time.perf_counter() - start_time
        time_cost += dt_s
        count += 1

        if time_cost > cfg.inference_time_s:
            break

        if time.time() - last_log_t > log_interval:
            logging.info(f"average fps: {count / time_cost:.1f}")
            last_log_t = time.time()

        elapse = 1 / cfg.fps - dt_s
        if elapse > 0:
            busy_wait(elapse)


@safe_disconnect
def torque_disable(robot: Robot, cfg: TorqueDisableConfig):
    if not robot.is_connected:
        robot.connect()

    robot.torque_disable()


def _init_rerun(control_config: ControlConfig, session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop.

    Args:
        control_config: Configuration determining data display and robot type.
        session_name: Rerun session name. Defaults to "lerobot_control_loop".

    Raises:
        ValueError: If viewer IP is missing for non-remote configurations with display enabled.
    """
    if (control_config.display_data and not is_headless()) or (
        control_config.display_data and isinstance(control_config, RemoteRobotConfig)
    ):
        # Configure Rerun flush batch size default to 8KB if not set
        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

        # Initialize Rerun based on configuration
        rr.init(session_name)
        if isinstance(control_config, RemoteRobotConfig):
            viewer_ip = control_config.viewer_ip
            viewer_port = control_config.viewer_port
            if not viewer_ip or not viewer_port:
                raise ValueError(
                    "Viewer IP & Port are required for remote config. Set via config file/CLI or disable control_config.display_data."
                )
            logging.info(f"Connecting to viewer at {viewer_ip}:{viewer_port}")
            rr.connect_tcp(f"{viewer_ip}:{viewer_port}")
        else:
            # Get memory limit for rerun viewer parameters
            memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "20%")
            rr.spawn(memory_limit=memory_limit)


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    global robot
    robot = make_robot_from_config(cfg.robot)

    # TODO(Steven): Blueprint for fixed window size

    if isinstance(cfg.control, CalibrateControlConfig):
        calibrate(robot, cfg.control)
    elif isinstance(cfg.control, TeleoperateControlConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_teleop")
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_record")
        record(robot, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        replay(robot, cfg.control)
    elif isinstance(cfg.control, RemoteRobotConfig):
        from lerobot.common.robot_devices.robots.lekiwi_remote import run_lekiwi

        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_remote")
        run_lekiwi(cfg.robot)
    elif isinstance(cfg.control, ShowPositionConfig):
        show_position(robot, cfg.control)
    elif isinstance(cfg.control, TestPolicyConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_test")
        test_policy(robot, cfg.control)
    elif isinstance(cfg.control, TorqueDisableConfig):
        torque_disable(robot, cfg.control)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    if platform.system() == "Windows":
        signal.signal(signal.SIGINT, terminate_handler)
        signal.signal(signal.SIGBREAK, terminate_handler)
    else:
        signal.signal(signal.SIGTERM, terminate_handler)
        signal.signal(signal.SIGKILL, terminate_handler)

    USBDeviceManager().load()
    control_robot()
