from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("sam01_follower")
@dataclass
class SAM01FollowerConfig(RobotConfig):
    port: str

    disable_torque_on_disconnect: bool = True

    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    use_degrees: bool = False

@RobotConfig.register_subclass("sam01_follower_end_effector")
@dataclass
class SAM01FollowerEndEffectorConfig(SAM01FollowerConfig):
    urdf_path: str | None = None
    target_frame_name: str = "link_7"
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50

    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
        }
    )
