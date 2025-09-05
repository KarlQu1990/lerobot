from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
from lerobot.robots.bi_sam01_follower import BiSAM01Follower, BiSAM01FollowerConfig
from lerobot.teleoperators.bi_sam01_leader.bi_sam01_leader import BiSAM01Leader
from lerobot.teleoperators.bi_sam01_leader.config_bi_sam01_leader import BiSAM01LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener, sanity_check_dataset_robot_compatibility
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

# ========= 运行参数 =========
NUM_EPISODES = 30
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "收纳螺丝螺母"
RESUME = False

# ========= 机器人与遥操作配置 =========
camera_config = {
    "high": OpenCVCameraConfig(index_or_path="high", width=640, height=480, fps=FPS),
    "left": OpenCVCameraConfig(index_or_path="left_wrist", width=640, height=480, fps=FPS),
    "right": OpenCVCameraConfig(index_or_path="right_wrist", width=640, height=480, fps=FPS),
}
robot_config = BiSAM01FollowerConfig(
    left_arm_port="left_follower",
    right_arm_port="right_follower",
    id="robot_arm_follower",
    cameras=camera_config,
)
teleop_config = BiSAM01LeaderConfig(
    left_arm_port="left_leader",
    right_arm_port="right_leader",
    id="robot_arm_leader",
)

robot = BiSAM01Follower(robot_config)
teleop = BiSAM01Leader(teleop_config)

# ========= 数据集特征（与机器人/策略对齐）=========
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

REPO_ID = "hhws/shou_na"
ROOT_DIR = "C:/Users/chlai/projects/lerobot_datasets"
THREADS_PER_CAMERA = 4  # 参考推荐值
num_cameras = len(getattr(robot, "cameras", {}))
total_image_writer_threads = max(1, THREADS_PER_CAMERA * max(1, num_cameras))

# ========= 创建/恢复数据集 =========
if RESUME:
    # 恢复已有数据集，并启动图片写入
    dataset = LeRobotDataset(
        REPO_ID,
        root=ROOT_DIR,
        batch_encoding_size=1,  # 按需：你也可以改成更大的批量做视频编码
    )
    if num_cameras > 0:
        dataset.start_image_writer(
            num_processes=0,  # 按需可调：>0 会启多个进程
            num_threads=total_image_writer_threads,
        )
    # 校验数据集与机器人/特征/FPS 的一致性
    sanity_check_dataset_robot_compatibility(dataset, robot, FPS, dataset_features)
else:
    # 新建数据集
    dataset = LeRobotDataset.create(
        root=ROOT_DIR,
        repo_id=REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=total_image_writer_threads,
        # 如果你想用多进程写图，可加：image_writer_processes=1
    )

# ========= 键盘监听 + 可视化（仅显示）=========
_, events = init_keyboard_listener()
_init_rerun(session_name="recording")

# ========= 连接设备 =========
robot.connect()
teleop.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"正在录制第 {episode_idx + 1} 个数据集")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # 环境重置（最后一条或重录时跳过/另处理）
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("重新布置环境")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# ========= 清理 =========
log_say("停止录制")
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()
