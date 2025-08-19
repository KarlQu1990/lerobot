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
