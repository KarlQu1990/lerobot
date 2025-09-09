#!/usr/bin/env python
import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.policies.act.modeling_act import (
    ACTDecoder,  # 直接复用通用层实现风格
    ACTDecoderLayer,
    ACTEncoderLayer,
    ACTSinusoidalPositionEmbedding2d,
    create_sinusoidal_pos_embedding,
    get_activation_fn,
)
from lerobot.policies.interact.configuration_interact import InterACTConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy


class InterACTPolicy(PreTrainedPolicy):
    """InterACT policy adapted to LeRobot ACT API."""

    config_class = InterACTConfig
    name = "interact"

    def __init__(
        self,
        config: InterACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        self.model = InterACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[k] for k in self.config.image_features]
        actions = self.model(batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    def preprocess(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[k] for k in self.config.image_features]
        batch = self.normalize_targets(batch)
        return batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        actions_hat = self.model(batch)
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()
        loss_dict = {"l1_loss": l1_loss}
        # 可选：如果你需要 L2：
        # l2 = (F.mse_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)).mean()
        # loss_dict["l2_loss"] = l2
        loss = l1_loss
        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat([
                self.ensembled_actions_count,
                torch.ones_like(self.ensembled_actions_count[-1:]),
            ])
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class InterACT(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.config = config

        self.use_robot_state = bool(self.config.robot_state_feature)
        self.use_images = bool(self.config.image_features)
        self.use_env_state = bool(self.config.env_state_feature)
        self.use_av_arm = config.use_av_arm
        self.num_cls_tokens_arm = config.num_cls_tokens_arm
        self.num_cls_tokens_image = config.num_cls_tokens_image

        # ----- CLS tokens & fixed positional encodings (device-agnostic) -----
        if self.use_robot_state:
            self.cls_input_arm1 = nn.Embedding(1, config.dim_model)
            self.cls_input_arm2 = nn.Embedding(1, config.dim_model)
            if self.use_av_arm:
                self.cls_input_av = nn.Embedding(1, config.dim_model)

            num_arm_input_token_encoder = self.num_cls_tokens_arm + 7  # 每臂 7 维
            self.register_buffer(
                "arm1_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_arm_input_token_encoder, config.dim_model).unsqueeze(0),
                persistent=False,
            )
            self.register_buffer(
                "arm2_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_arm_input_token_encoder, config.dim_model).unsqueeze(0),
                persistent=False,
            )
            if self.use_av_arm:
                self.register_buffer(
                    "av_encoder_pos_enc",
                    create_sinusoidal_pos_embedding(num_arm_input_token_encoder, config.dim_model).unsqueeze(0),
                    persistent=False,
                )

        if self.use_images:
            self.cls_input_image = nn.Embedding(1, config.dim_model)
            self.register_buffer(
                "image_encoder_pos_enc",
                create_sinusoidal_pos_embedding(self.num_cls_tokens_image, config.dim_model).unsqueeze(0),
                persistent=False,
            )

        if self.use_av_arm:
            total_cls = 3 * self.num_cls_tokens_arm + self.num_cls_tokens_image
        else:
            total_cls = 2 * self.num_cls_tokens_arm + self.num_cls_tokens_image
        self.register_buffer(
            "cls_encoder_pos_enc",
            create_sinusoidal_pos_embedding(total_cls, config.dim_model).unsqueeze(0),
            persistent=False,
        )

        # ----- Backbone -----
        if self.use_images:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # ----- Transformer encoder / decoder -----
        self.encoder = InterACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # robot state -> hidden
        if self.use_robot_state:
            self.encoder_robot_state_input_proj = nn.Linear(1, config.dim_model)

        # decoder pos
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # action head
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        # batch[OBS_IMAGES] 是 List[Tensor]，与 ACT 保持一致
        if ACTION in batch:  # 训练时由 policy.preprocess 注入
            pass

        if self.use_images:
            assert OBS_IMAGES in batch
        batch_size = (
            batch[OBS_IMAGES][0].shape[0] if self.use_images else batch["observation.environment_state"].shape[0]
        )

        # ----- robot state tokens -----
        encoder_in_tokens = []
        encoder_in_pos_embed_list = []

        if self.use_robot_state:
            arm1_state = batch["observation.state"][:, :7]
            arm2_state = batch["observation.state"][:, 7:14]
            arm1_state_proj = self.encoder_robot_state_input_proj(arm1_state.unsqueeze(-1))
            arm2_state_proj = self.encoder_robot_state_input_proj(arm2_state.unsqueeze(-1))

            cls_token_arm1 = (
                self.cls_input_arm1.weight.repeat(self.num_cls_tokens_arm, 1).unsqueeze(0).repeat(batch_size, 1, 1)
            )
            cls_token_arm2 = (
                self.cls_input_arm2.weight.repeat(self.num_cls_tokens_arm, 1).unsqueeze(0).repeat(batch_size, 1, 1)
            )

            encoder_in_tokens.append(torch.cat([cls_token_arm1, arm1_state_proj], dim=1))
            encoder_in_tokens.append(torch.cat([cls_token_arm2, arm2_state_proj], dim=1))

            encoder_in_pos_embed_list.extend(list(self.arm1_encoder_pos_enc))
            encoder_in_pos_embed_list.extend(list(self.arm2_encoder_pos_enc))

            if self.use_av_arm:
                av_state = batch["observation.state"][:, 14:21]
                av_state_proj = self.encoder_robot_state_input_proj(av_state.unsqueeze(-1))
                cls_token_av = (
                    self.cls_input_av.weight.repeat(self.num_cls_tokens_arm, 1).unsqueeze(0).repeat(batch_size, 1, 1)
                )
                encoder_in_tokens.append(torch.cat([cls_token_av, av_state_proj], dim=1))
                encoder_in_pos_embed_list.extend(list(self.av_encoder_pos_enc))

        # ----- image features -----
        if self.use_images:
            all_cam_features = []
            all_cam_pos_embeds = []
            cls_token_image = (
                self.cls_input_image.weight.repeat(self.num_cls_tokens_image, 1).unsqueeze(0).repeat(batch_size, 1, 1)
            )

            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            all_cam_features = torch.cat(all_cam_features, dim=-1)  # 按宽拼接
            encoder_in_tokens.append(
                torch.cat([cls_token_image, einops.rearrange(all_cam_features, "b c h w -> b (h w) c")], dim=1)
            )

            all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, dim=-1)
            encoder_in_pos_embed_list.append(
                torch.cat(
                    [
                        list(self.image_encoder_pos_enc)[0],
                        list(einops.rearrange(all_cam_pos_embeds, "b c h w -> b (h w) c"))[0],
                    ],
                    dim=0,
                )
            )

        encoder_in_tokens = torch.cat(encoder_in_tokens, dim=1)  # (B, S, D)
        encoder_in_pos_embed = torch.cat(encoder_in_pos_embed_list, dim=0)  # (S, D)
        encoder_in_cls_pos_embed = torch.cat(list(self.cls_encoder_pos_enc))  # (S_cls, D)

        # expand pos embeds to (S, B, D)
        encoder_in_pos_embed = encoder_in_pos_embed.unsqueeze(1).expand(-1, encoder_in_tokens.size(0), -1)
        encoder_in_cls_pos_embed = encoder_in_cls_pos_embed.unsqueeze(1).expand(-1, encoder_in_tokens.size(0), -1)

        # ----- encode -----
        encoder_out = self.encoder(
            encoder_in_tokens, pos_embed=encoder_in_pos_embed, pos_embed_cls=encoder_in_cls_pos_embed
        )

        # ----- slice segments back -----
        arm = self.num_cls_tokens_arm
        img = self.num_cls_tokens_image
        idx = 0
        encoder_out_arm1 = encoder_out[:, idx : idx + arm + 7]
        idx += arm + 7
        encoder_pos_arm1 = encoder_in_pos_embed[idx - (arm + 7) : idx]
        encoder_out_arm2 = encoder_out[:, idx : idx + arm + 7]
        idx += arm + 7
        encoder_pos_arm2 = encoder_in_pos_embed[idx - (arm + 7) : idx]
        if self.use_av_arm:
            encoder_out_av = encoder_out[:, idx : idx + arm + 7]
            idx += arm + 7
            encoder_pos_av = encoder_in_pos_embed[idx - (arm + 7) : idx]
        encoder_out_img = encoder_out[:, idx:]
        encoder_pos_img = encoder_in_pos_embed[idx:]

        if self.use_av_arm:
            encoder_out_real = torch.cat(
                [
                    encoder_out_arm1[:, :arm],
                    encoder_out_arm2[:, :arm],
                    encoder_out_av[:, :arm],
                    encoder_out_img[:, img:],
                ],
                dim=1,
            )
            encoder_pos_real = torch.cat(
                [encoder_pos_arm1[:arm], encoder_pos_arm2[:arm], encoder_pos_av[:arm], encoder_pos_img[img:]],
                dim=0,
            )
        else:
            encoder_out_real = torch.cat(
                [encoder_out_arm1[:, :arm], encoder_out_arm2[:, :arm], encoder_out_img[:, img:]],
                dim=1,
            )
            encoder_pos_real = torch.cat(
                [encoder_pos_arm1[:arm], encoder_pos_arm2[:arm], encoder_pos_img[img:]],
                dim=0,
            )

        # ----- decode -----
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_pos_real.dtype,
            device=encoder_pos_real.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out_real.transpose(0, 1),  # (S, B, D)
            encoder_pos_embed=encoder_pos_real,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        actions = self.action_head(decoder_out.transpose(0, 1))  # (B, S, action_dim)
        return actions


class InterACTEncoder(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.num_blocks = config.num_blocks
        self.segment_wise_encoder = nn.ModuleList([ACTEncoderLayer(config) for _ in range(config.num_blocks)])
        self.cross_segment_encoder = nn.ModuleList([ACTEncoderLayer(config) for _ in range(config.num_blocks)])
        self.arm_cls = config.num_cls_tokens_arm
        self.cam_cls = config.num_cls_tokens_image
        self.use_av_arm = config.use_av_arm

    def forward(self, segments: Tensor, pos_embed: Tensor, pos_embed_cls: Tensor) -> Tensor:
        # 输入 (B, S, D)
        segments = einops.rearrange(segments, "b s d -> s b d")

        seg_arm1 = segments[: self.arm_cls + 7]
        seg_arm2 = segments[self.arm_cls + 7 : 2 * self.arm_cls + 14]
        pos_arm1 = pos_embed[: self.arm_cls + 7]
        pos_arm2 = pos_embed[self.arm_cls + 7 : 2 * self.arm_cls + 14]
        if self.use_av_arm:
            seg_av = segments[2 * self.arm_cls + 14 : 3 * self.arm_cls + 21]
            seg_img = segments[3 * self.arm_cls + 21 :]
            pos_av = pos_embed[2 * self.arm_cls + 14 : 3 * self.arm_cls + 21]
            pos_img = pos_embed[3 * self.arm_cls + 21 :]
        else:
            seg_img = segments[2 * self.arm_cls + 14 :]
            pos_img = pos_embed[2 * self.arm_cls + 14 :]

        for i in range(self.num_blocks):
            upd_arm1 = self.segment_wise_encoder[i](seg_arm1, pos_arm1)
            upd_arm2 = self.segment_wise_encoder[i](seg_arm2, pos_arm2)
            if self.use_av_arm:
                upd_av = self.segment_wise_encoder[i](seg_av, pos_av)
            upd_img = self.segment_wise_encoder[i](seg_img, pos_img)

            if self.use_av_arm:
                upd_cls = self.cross_segment_encoder[i](
                    torch.cat(
                        [
                            upd_arm1[: self.arm_cls],
                            upd_arm2[: self.arm_cls],
                            upd_av[: self.arm_cls],
                            upd_img[: self.cam_cls],
                        ],
                        dim=0,
                    ),
                    pos_embed_cls,
                )
            else:
                upd_cls = self.cross_segment_encoder[i](
                    torch.cat([upd_arm1[: self.arm_cls], upd_arm2[: self.arm_cls], upd_img[: self.cam_cls]], dim=0),
                    pos_embed_cls,
                )

            seg_arm1 = torch.cat([upd_cls[: self.arm_cls], upd_arm1[self.arm_cls :]], dim=0)
            seg_arm2 = torch.cat([upd_cls[self.arm_cls : 2 * self.arm_cls], upd_arm2[self.arm_cls :]], dim=0)
            if self.use_av_arm:
                seg_av = torch.cat([upd_cls[2 * self.arm_cls : 3 * self.arm_cls], upd_av[self.arm_cls :]], dim=0)
                seg_img = torch.cat([upd_cls[3 * self.arm_cls :], upd_img[self.cam_cls :]], dim=0)
            else:
                seg_img = torch.cat([upd_cls[2 * self.arm_cls :], upd_img[self.cam_cls :]], dim=0)

        if self.use_av_arm:
            segments = torch.cat([seg_arm1, seg_arm2, seg_av, seg_img], dim=0)
        else:
            segments = torch.cat([seg_arm1, seg_arm2, seg_img], dim=0)

        return segments.transpose(0, 1)  # (B, S, D)


class ACTDecoder(nn.Module):
    def __init__(self, config: InterACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


"""
Use separate encoder for vision
Use only cls tokens in decoder
use only image features in decoder
"""
