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

"""Tiny random Xiaomi-Robotics-1 configuration and batch builders shared by the tests (CPU, no network)."""

from __future__ import annotations

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.xiaomi_robotics_1.configuration_xiaomi_robotics_1 import XiaomiRobotics1Config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

STATE_DIM = 5
ACTION_DIM = 4
MAX_DIM = 8
CHUNK = 6
VOCAB = 64
IMAGE_TOKEN_ID = 40
VISION_START_ID = 41
VISION_END_ID = 42
IMAGE_SIZE = 32  # one 32x32 image = 2x2 patches of 16 = one merged token... see GRID

# patch 16, spatial merge 2: a 32x32 image is a 2x2 patch grid -> 1 merged token.
GRID = (1, 2, 2)
TOKENS_PER_IMAGE = 1

VLM_CONFIG = {
    "model_type": "qwen3_vl",
    "image_token_id": IMAGE_TOKEN_ID,
    "video_token_id": 43,
    "vision_start_token_id": VISION_START_ID,
    "vision_end_token_id": VISION_END_ID,
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "qwen3_vl_text",
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "rope_scaling": {"rope_type": "default", "mrope_interleaved": True, "mrope_section": [2, 3, 3]},
        "rope_theta": 5000000,
    },
    "vision_config": {
        "model_type": "qwen3_vl",
        "depth": 2,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 32,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0, 1],
    },
}


def make_config(**overrides) -> XiaomiRobotics1Config:
    kwargs = {
        "vlm_config": VLM_CONFIG,
        "dit_hidden_size": 32,
        "dit_num_layers": 2,
        "dit_head_dim": 16,
        "dit_num_key_value_heads": 2,
        "max_state_dim": MAX_DIM,
        "max_action_dim": MAX_DIM,
        "chunk_size": CHUNK,
        "n_action_steps": 3,
        "num_denoise_steps": 3,
        "training_repeat": 2,
        "dtype": torch.float32,
        "device": "cpu",
        "input_features": {
            f"{OBS_IMAGES}.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
            f"{OBS_IMAGES}.cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    }
    kwargs.update(overrides)
    return XiaomiRobotics1Config(**kwargs)


def make_prefix(
    batch_size: int = 2, n_images: int = 2, pad: int = 0, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Token ids shaped like the processor's output: text, then `n_images` image blocks, then text."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(batch_size):
        ids: list[int] = [1, 2, 3]
        for _ in range(n_images):
            ids += [VISION_START_ID, *([IMAGE_TOKEN_ID] * TOKENS_PER_IMAGE), VISION_END_ID, 4]
        ids += torch.randint(5, 30, (3,), generator=generator).tolist()
        rows.append(ids)
    length = len(rows[0]) + pad
    input_ids = torch.zeros(batch_size, length, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, length, dtype=torch.long)
    for i, ids in enumerate(rows):
        n = len(ids)
        input_ids[i, length - n :] = torch.tensor(ids)  # left padding
        attention_mask[i, length - n :] = 1
    mm = (input_ids == IMAGE_TOKEN_ID).long()
    pixel_values = torch.randn(batch_size * n_images * 4, 3 * 2 * 16 * 16, generator=generator)
    image_grid_thw = torch.tensor([GRID] * (batch_size * n_images))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm,
    }


def make_batch(batch_size: int = 2, with_action: bool = False, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    batch = make_prefix(batch_size, seed=seed)
    batch[OBS_STATE] = torch.randn(batch_size, STATE_DIM, generator=generator)
    if with_action:
        batch[ACTION] = torch.randn(batch_size, CHUNK, ACTION_DIM, generator=generator)
        batch["action_is_pad"] = torch.zeros(batch_size, CHUNK, dtype=torch.bool)
    return batch
