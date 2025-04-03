#!/usr/bin/env python

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

import logging

from lerobot.configs.train import TrainPipelineConfig


class TensorboardLogger:
    """Primary logger object. Logs either locally or using Tensorboard"""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.tensorboard
        self.log_dir = cfg.output_dir

        if not cfg.tensorboard.enable:
            self._tb = None
        else:
            from torch.utils.tensorboard import SummaryWriter
            
            self._tb = SummaryWriter(str(self.log_dir))
        
    def log_dict(self, d, step, mode="train"):
        assert mode in {"train", "eval"}
        # TODO(alexander-soare): Add local text log.
        if self._tb:
            for k, v in d.items():
                if not isinstance(v, (int, float, str)):
                    continue

                if self._tb:
                    self._tb.add_scalar(k, v, step)  