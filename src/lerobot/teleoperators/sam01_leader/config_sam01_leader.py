from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("sam01_leader")
@dataclass
class SAM01LeaderConfig(TeleoperatorConfig):
    port: str