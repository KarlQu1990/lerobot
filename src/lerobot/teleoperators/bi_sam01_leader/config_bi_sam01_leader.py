#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_sam01_leader")
@dataclass
class BiSAM01LeaderConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str

