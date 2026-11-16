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

"""The converter must turn a released-layout checkpoint into one the policy loads and runs identically."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.xiaomi_robotics_1.convert_xiaomi_robotics_1_to_lerobot import (
    action_statistics,
    convert,
    convert_key,
)
from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Policy
from lerobot.policies.xiaomi_robotics_1.processor_xiaomi_robotics_1 import XiaomiRobotics1QwenProcessorStep
from tests.policies.xiaomi_robotics_1.tiny import CHUNK, MAX_DIM, VLM_CONFIG, make_batch, make_config
from tests.policies.xiaomi_robotics_1.tiny_processor import make_tiny_processor

ACTIVE = 4


def _action_config(chunk=CHUNK, dim=MAX_DIM, active=ACTIVE, mean=0.5, std=0.25):
    std_row = [std] * active + [1e-6] * (dim - active)
    return {"mean": [[mean] * dim] * chunk, "std": [std_row] * chunk}


@pytest.fixture(autouse=True)
def tiny_processor(monkeypatch):
    # The converter builds the processors, which would read the tokenizer from the Hub.
    monkeypatch.setattr(
        XiaomiRobotics1QwenProcessorStep, "_load_processor", lambda self: make_tiny_processor()
    )


@pytest.fixture
def released(tmp_path):
    """A checkpoint laid out like the authors' release, built from a tiny random policy."""
    torch.manual_seed(0)
    policy = XiaomiRobotics1Policy(make_config(num_denoise_steps=5)).eval()  # the checkpoint default
    state = {k.removeprefix("model."): v.contiguous() for k, v in policy.state_dict().items()}
    state.pop("vlm.lm_head.weight")  # tied; the release carries only the embedding
    keys = sorted(state)
    shards = {"model-00001-of-00002.safetensors": keys[::2], "model-00002-of-00002.safetensors": keys[1::2]}
    (tmp_path / "src").mkdir()
    for name, names in shards.items():
        save_file({k: state[k] for k in names}, tmp_path / "src" / name)
    index = {"weight_map": {k: name for name, names in shards.items() for k in names}}
    (tmp_path / "src" / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "src" / "config.json").write_text(
        json.dumps(
            {
                "action_dim": MAX_DIM,
                "state_dim": MAX_DIM,
                "state_length": 1,
                "dit_config": {
                    "hidden_size": 32,
                    "head_dim": 16,
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                },
                "vlm_config": VLM_CONFIG,
            }
        )
    )
    (tmp_path / "src" / "preprocessor_config.json").write_text(
        json.dumps({"action_config": {"robocasa_mg": _action_config()}})
    )
    return tmp_path / "src", policy


def test_converted_checkpoint_loads_and_matches_the_source_model(released, tmp_path):
    source, policy = released
    out = convert(
        str(source),
        tmp_path / "out",
        revision="abc",
        base_views=["a"],
        wrist_views=["b"],
        state_dim=5,
        dtype=torch.float32,
    )
    loaded = XiaomiRobotics1Policy.from_pretrained(out).eval()
    batch = make_batch()
    noise = torch.randn(2, CHUNK, MAX_DIM, generator=torch.Generator().manual_seed(4))
    assert loaded.config.chunk_size == CHUNK and loaded.config.processor_revision == "abc"
    assert loaded.config.base_view_keys == ["observation.images.a"]
    assert loaded.config.output_features["action"].shape == (ACTIVE,)
    torch.testing.assert_close(
        loaded.predict_action_chunk(batch, noise=noise),
        policy.predict_action_chunk(batch, noise=noise)[..., :ACTIVE],
        atol=1e-6,
        rtol=1e-6,
    )


def test_converted_processors_carry_the_action_statistics(released, tmp_path):
    source, _ = released
    out = convert(str(source), tmp_path / "out", base_views=["a"], wrist_views=[], state_dim=5)
    config = PreTrainedConfig.from_pretrained(out)
    _, post = make_pre_post_processors(config, pretrained_path=out)
    chunk = torch.zeros(1, CHUNK, ACTIVE)
    torch.testing.assert_close(post(chunk), torch.full((1, CHUNK, ACTIVE), 0.5))  # 0 * std + mean


def test_every_tensor_is_written_with_the_model_prefix_and_dtype(released, tmp_path):
    from safetensors import safe_open

    source, _ = released
    out = convert(str(source), tmp_path / "out", base_views=["a"], wrist_views=[], state_dim=5)
    with (
        safe_open(out / "model.safetensors", framework="pt") as written,
        safe_open(source / "model-00001-of-00002.safetensors", framework="pt") as original,
    ):
        for key in original.keys():  # noqa: SIM118
            assert torch.equal(written.get_tensor(convert_key(key)), original.get_tensor(key))


def test_statistics_that_differ_across_the_chunk_are_refused():
    per_step = _action_config()
    per_step["std"][1] = [9.0] * MAX_DIM
    with pytest.raises(ValueError, match="differ across the chunk"):
        action_statistics({"action_config": {"r": per_step}}, "r")
    with pytest.raises(KeyError, match="available"):
        action_statistics({"action_config": {"r": _action_config()}}, "other")


def test_a_gap_in_the_active_dimensions_is_refused():
    config = _action_config()
    config["std"] = [[0.2, 1e-6, 0.2, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]] * CHUNK
    with pytest.raises(ValueError, match="leading block"):
        action_statistics({"action_config": {"r": config}}, "r")
