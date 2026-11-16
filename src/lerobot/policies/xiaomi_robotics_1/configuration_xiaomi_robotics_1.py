#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Configuration for Xiaomi-Robotics-1 (XR-1).

XR-1 is a Qwen3-VL-4B backbone plus a 36-layer action expert (a DiT) that attends to the backbone's key/value cache
and integrates a flow-matching velocity field. Original implementation and checkpoints (Apache-2.0):
https://github.com/XiaomiRobotics/Xiaomi-Robotics-1 and https://huggingface.co/XiaomiRobotics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig
else:
    Qwen3VLConfig = None
    Qwen3VLTextConfig = None

# Official repository the released tokenizer, chat template and image-processor settings are read from. The XR-1
# tokenizer is NOT the stock Qwen3-VL one (different `tokenizer.json`), so the backbone's name alone is not enough.
XR1_PROCESSOR_REPO = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa"
# Snapshot the 52.1 % reproduction and the parity script were run against.
XR1_PROCESSOR_REVISION = "82b7c2210c1a39fca4a87e681cffbbed134ccde0"


def _qwen3_vl_4b_config() -> dict:
    """Qwen3-VL-4B-Instruct as the released XR-1 checkpoints use it (their `config.json`, `vlm_config`).

    Embedded so that building the config never touches the network; a test checks that it reproduces the released
    parameter layout exactly.
    """
    return {
        "model_type": "qwen3_vl",
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
            "rope_theta": 5000000,
            "attention_bias": False,
            "hidden_act": "silu",
            "tie_word_embeddings": True,
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 2560,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": [5, 11, 17],
            "hidden_act": "gelu_pytorch_tanh",
            "in_channels": 3,
        },
    }


@PreTrainedConfig.register_subclass("xiaomi_robotics_1")
@dataclass
class XiaomiRobotics1Config(PreTrainedConfig):
    """Configuration for the Xiaomi-Robotics-1 policy.

    Args:
        vlm_config: Serialized `Qwen3VLConfig` of the backbone; the default is Qwen3-VL-4B-Instruct as the released
            checkpoints use it (no network access is needed to build it).
        processor_name: Hub id (or local path) of the tokenizer, chat template and image processor.
        processor_revision: Revision of `processor_name`.
        base_view_keys, wrist_view_keys: Camera features listed under each view header of the prompt.
        dit_hidden_size, dit_num_layers, dit_head_dim, dit_num_key_value_heads: Action-expert shape. The expert
            attends to the backbone's cached keys and values, so `dit_head_dim` must equal the backbone head
            dimension and `dit_hidden_size // dit_head_dim` the backbone's key/value head count.
        state_length: Number of state vectors fed to the action expert (the released checkpoints use 1).
        max_state_dim, max_action_dim: Width the state and the action are zero-padded to (60 in the checkpoints).
        chunk_size: Action horizon predicted per call (10 in the RoboCasa checkpoints, 30 at pre-training).
        n_action_steps: Steps of a chunk executed before predicting again.
        num_denoise_steps: Euler steps of the flow-matching sampler (5 in the checkpoints).
        noise_seed: When set, the sampling noise is drawn from `torch.manual_seed(noise_seed)` on every call. The
            authors' evaluation harness passes `seed=42` on every call, so set this to 42 to reproduce it.
            None draws fresh noise.
        image_crop_ratio: Centre-crop ratio applied to every camera image and resized back (the authors evaluate
            with 0.95). 1.0 disables it.
        image_min_pixels, image_max_pixels: Bounds of the Qwen3-VL image resize (the checkpoints' values).
        training_repeat: Number of (time, noise) samples drawn per observation in training; the backbone runs once.
        freq_coefficient, freq_excluded_dims: Weight of the frequency-domain loss term and the action dimensions
            left out of it.
    """

    vlm_config: dict = field(default_factory=_qwen3_vl_4b_config)
    processor_name: str = XR1_PROCESSOR_REPO
    processor_revision: str | None = XR1_PROCESSOR_REVISION
    use_fast_processor: bool = False
    # Cameras shown under "# Base View" and under "# Left-Wrist View" in the prompt, by feature name. When both are
    # empty every visual feature is a base view. The RoboCasa checkpoints use two agent-view cameras and the wrist
    # camera, in that order.
    base_view_keys: list[str] = field(default_factory=list)
    wrist_view_keys: list[str] = field(default_factory=list)

    # Action expert.
    dit_hidden_size: int = 1024
    dit_num_layers: int = 36
    dit_head_dim: int = 128
    dit_num_key_value_heads: int = 8

    # Inputs and outputs.
    n_obs_steps: int = 1
    state_length: int = 1
    max_state_dim: int = 60
    max_action_dim: int = 60
    chunk_size: int = 10
    n_action_steps: int = 10

    # Sampling.
    num_denoise_steps: int = 5
    noise_seed: int | None = None

    # Image preprocessing.
    image_crop_ratio: float = 0.95
    image_min_pixels: int | None = 1024
    image_max_pixels: int | None = 90000
    tokenizer_max_length: int = 2048

    # Training.
    training_repeat: int = 4
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    mse_loss_weight: float = 0.5
    freq_coefficient: float = 1.0
    freq_excluded_dims: tuple[int, ...] = (17, 18, 19)
    gradient_checkpointing: bool = False
    train_vlm: bool = True

    dtype: torch.dtype | None = torch.bfloat16
    # None lets transformers pick (sdpa). The authors run the backbone with flash_attention_2; the action expert is
    # always `scaled_dot_product_attention`.
    attn_implementation: str | None = None

    # The RoboCasa checkpoints feed the raw (unnormalized) joint state and normalize only the action.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.num_denoise_steps < 1:
            raise ValueError("num_denoise_steps must be at least 1.")
        if self.state_length < 1 or self.training_repeat < 1:
            raise ValueError("state_length and training_repeat must be at least 1.")
        if not 0.0 < self.image_crop_ratio <= 1.0:
            raise ValueError("image_crop_ratio must be in (0, 1].")
        if self.dit_hidden_size % self.dit_head_dim != 0:
            raise ValueError("dit_hidden_size must be a multiple of dit_head_dim.")
        num_heads = self.dit_hidden_size // self.dit_head_dim
        if num_heads % self.dit_num_key_value_heads != 0:
            raise ValueError("dit_hidden_size // dit_head_dim must be a multiple of dit_num_key_value_heads.")
        self.freq_excluded_dims = tuple(self.freq_excluded_dims)

    @property
    def vlm_backbone_config(self) -> Qwen3VLConfig:
        require_package("transformers", extra="xiaomi_robotics_1")
        config = Qwen3VLConfig(**deepcopy(self.vlm_config))
        if self.attn_implementation is not None:
            config._attn_implementation = self.attn_implementation
        return config

    @property
    def text_config(self) -> Qwen3VLTextConfig:
        return self.vlm_backbone_config.text_config

    def validate_features(self) -> None:
        """Validate the features and fill in the padded state and action entries."""
        if self.input_features is None or self.output_features is None:
            raise ValueError(
                "`input_features` and `output_features` must be resolved before `validate_features()` is called."
            )
        if not any(feat.type == FeatureType.VISUAL for feat in self.input_features.values()):
            raise ValueError("Xiaomi-Robotics-1 requires at least one visual input feature.")
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.max_state_dim,)
            )
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.max_action_dim,)
            )
        if self.input_features[OBS_STATE].shape[0] > self.max_state_dim:
            raise ValueError(f"State is wider than max_state_dim={self.max_state_dim}.")
        if self.output_features[ACTION].shape[0] > self.max_action_dim:
            raise ValueError(f"Action is wider than max_action_dim={self.max_action_dim}.")

        text = self.text_config
        if text.head_dim != self.dit_head_dim:
            raise ValueError(
                f"dit_head_dim ({self.dit_head_dim}) must equal the backbone head_dim ({text.head_dim}): the "
                "action expert attends to the backbone's cached keys and values."
            )
        dit_heads = self.dit_hidden_size // self.dit_head_dim
        if text.num_key_value_heads * (dit_heads // self.dit_num_key_value_heads) != dit_heads:
            raise ValueError(
                f"The backbone's {text.num_key_value_heads} key/value heads repeated "
                f"{dit_heads // self.dit_num_key_value_heads}x must give the action expert's {dit_heads} heads."
            )
        if self.dit_num_layers > text.num_hidden_layers:
            raise ValueError("dit_num_layers cannot exceed the backbone's num_hidden_layers.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
