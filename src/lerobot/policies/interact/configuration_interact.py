#!/usr/bin/env python
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("interact")
@dataclass
class InterACTConfig(PreTrainedConfig):
    """Configuration for InterACT policy (compatible with LeRobot ACT API)."""

    # ===== IO / chunking =====
    n_obs_steps: int = 1
    chunk_size: int = 100  # sequence length decoded per forward
    n_action_steps: int = 100  # how many steps to pop per env.step()

    # ===== Normalization modes =====
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ===== Backbone / transformer =====
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"

    # Encoder/Decoder depth
    num_blocks: int = 3  # InterACT encoder blocks (segment-wise + cross-segment)
    n_decoder_layers: int = 1  # action decoder layers

    # InterACT-specific structure
    num_cls_tokens_arm: int = 3
    num_cls_tokens_image: int = 3
    use_av_arm: bool = False  # optional 3rd arm segment

    # Inference helpers
    temporal_ensemble_coeff: float | None = None  # if set, use temporal ensembling (requires n_action_steps==1)

    # Training/loss
    dropout: float = 0.1
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # ===== Required by PreTrainedConfig (delta schedule) =====
    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        # Predict actions at every position in the chunk
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None

    # ===== Optional presets & validation =====
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        """Validate feature selections (aligned with ACT expectations)."""
        # Require at least one observation source
        has_images = bool(self.image_features)
        has_env = bool(self.env_state_feature)
        has_robot = bool(self.robot_state_feature)
        if not (has_images or has_env or has_robot):
            raise ValueError(
                "InterACTConfig: require at least one of image_features, env_state_feature, robot_state_feature."
            )
        # Require action feature
        if self.action_feature is None or getattr(self.action_feature, "shape", None) is None:
            raise ValueError("InterACTConfig: action_feature must be defined with a valid shape.")
        # Basic consistency
        if self.n_action_steps > self.chunk_size:
            raise ValueError("InterACTConfig: n_action_steps cannot exceed chunk_size.")

    def __post_init__(self):
        super().__post_init__()
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "Temporal ensembling requires n_action_steps == 1 because the policy must be queried every step."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size.")
        if self.n_obs_steps != 1:
            raise ValueError(f"Multiple observation steps not handled yet. Got n_obs_steps={self.n_obs_steps}")
        # At least one of images or environment state must be present
        if not self.image_features and not self.env_state_feature:
            raise ValueError("Provide at least one image or the environment state among the inputs.")
