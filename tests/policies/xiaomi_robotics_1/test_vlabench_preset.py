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

"""The converter's VLABench preset and its processor revision rule (tools branch, not for upstream)."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from lerobot.policies.xiaomi_robotics_1.configuration_xiaomi_robotics_1 import XiaomiRobotics1Config
from lerobot.policies.xiaomi_robotics_1.convert_xiaomi_robotics_1_to_lerobot import (
    VLABENCH_ROBOT_TYPE,
    build_config,
)
from lerobot.utils.constants import OBS_IMAGES
from tests.policies.xiaomi_robotics_1.tiny import MAX_DIM, VLM_CONFIG

MEAN = [0.1, -0.2, 0.3, 0.0, 0.5, -0.5, 0.25]
STD = [2.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.1]


def test_the_vlabench_preset_of_the_converter():
    chunk = 10
    row_mean = [0.0] * 60
    row_std = [0.0] * 60
    for i in range(7):
        row_mean[i], row_std[i] = MEAN[i], STD[i]
    preprocessor_config = {
        "action_config": {VLABENCH_ROBOT_TYPE: {"mean": [row_mean] * chunk, "std": [row_std] * chunk}}
    }
    source_config = {
        "action_dim": 60,
        "state_dim": 60,
        "state_length": 1,
        "dit_config": {"hidden_size": 32, "head_dim": 16, "num_hidden_layers": 2, "num_key_value_heads": 2},
        "vlm_config": VLM_CONFIG,
    }
    config, stats = build_config(source_config, preprocessor_config, robot_type=VLABENCH_ROBOT_TYPE)
    assert isinstance(config, XiaomiRobotics1Config)
    assert config.eef_delta_actions and config.noise_seed == 42
    assert config.image_crop_ratio == 1.0  # the authors' VLABench client does not crop
    assert config.image_max_pixels > 480 * 480  # and does not downscale its 480-pixel cameras
    assert config.chunk_size == 10 and config.n_action_steps == 5
    assert config.eef_action_mean == MEAN and config.eef_action_std == STD
    assert [g[0] for g in config.view_groups] == ["Ego View", "Base View", "Left-Wrist View"]
    assert config.input_features["observation.state"].shape == (7,)
    assert config.input_features[f"{OBS_IMAGES}.ego"].shape == (3, 480, 480)
    assert config.output_features["action"].shape == (7,)
    assert config.normalization_mapping["ACTION"].value == "IDENTITY"
    assert stats["action"]["mean"].shape == (7,)
    assert MAX_DIM  # tiny helpers are only used for the backbone config here


def test_the_processor_revision_follows_the_checkpoint_only_for_the_processor_repository():
    from lerobot.policies.xiaomi_robotics_1.convert_xiaomi_robotics_1_to_lerobot import (
        XR1_PROCESSOR_REPO,
        XR1_PROCESSOR_REVISION,
        default_processor_revision,
    )

    assert default_processor_revision(XR1_PROCESSOR_REPO, "abc") == "abc"
    assert (
        default_processor_revision("XiaomiRobotics/Xiaomi-Robotics-1-VLABench", "f498")
        == XR1_PROCESSOR_REVISION
    )
