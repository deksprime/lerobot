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

"""Xiaomi-Robotics-1 processor tests: prompt layout, image preprocessing and the full pre/post pipeline (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("transformers")

from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Policy
from lerobot.policies.xiaomi_robotics_1.processor_xiaomi_robotics_1 import (
    ASSISTANT_TURN,
    PROMPT_HEADER,
    WRIST_HEADER,
    XiaomiRobotics1PrepareMessagesStep,
    XiaomiRobotics1QwenProcessorStep,
    center_crop_resize,
    make_xiaomi_robotics_1_pre_post_processors,
    tensor_to_pil,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from tests.policies.xiaomi_robotics_1.tiny import ACTION_DIM, CHUNK, IMAGE_TOKEN_ID, STATE_DIM, make_config
from tests.policies.xiaomi_robotics_1.tiny_processor import make_tiny_processor

CAM0, CAM1 = f"{OBS_IMAGES}.cam0", f"{OBS_IMAGES}.cam1"


@pytest.fixture
def tiny_processor(monkeypatch):
    monkeypatch.setattr(
        XiaomiRobotics1QwenProcessorStep, "_load_processor", lambda self: make_tiny_processor()
    )


def _raw_batch(seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    uint8 = torch.randint(0, 256, (3, 32, 32), generator=generator, dtype=torch.uint8)
    return {
        CAM0: uint8.float() / 255.0,
        CAM1: torch.flip(uint8, dims=[2]).float() / 255.0,
        OBS_STATE: torch.randn(STATE_DIM, generator=generator),
        "task": "pick up the cube",
    }


# ---- image helpers -----------------------------------------------------------------------------------------


def test_tensor_to_pil_is_exact_for_values_that_came_from_uint8():
    uint8 = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    image = tensor_to_pil(uint8.float() / 255.0)
    assert image.size == (8, 8)
    assert np.array_equal(np.asarray(image), uint8.permute(1, 2, 0).numpy())


def test_center_crop_resize_matches_the_authors_harness():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))
    got = center_crop_resize(image, 0.95)
    # The authors' `center_crop_pil`, written out.
    w, h = image.size
    new_w, new_h = int(w * 0.95), int(h * 0.95)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    want = image.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BILINEAR)
    assert got.size == (60, 40) and np.array_equal(np.asarray(got), np.asarray(want))
    assert center_crop_resize(image, 1.0) is image


# ---- prompt ------------------------------------------------------------------------------------------------


def _messages(step, observation, tasks):
    from lerobot.lerobot_types import TransitionKey

    step._current_transition = {TransitionKey.OBSERVATION: observation}
    return step.complementary_data({"task": tasks})["messages"]


def _layout(conversation) -> list[str]:
    return [part["text"] if part["type"] == "text" else "<image>" for part in conversation[0]["content"]]


def test_prompt_groups_cameras_under_view_headers():
    step = XiaomiRobotics1PrepareMessagesStep(
        base_view_keys=[CAM0, CAM1], wrist_view_keys=["observation.images.wrist"], image_crop_ratio=1.0
    )
    observation = {k: torch.rand(1, 3, 32, 32) for k in (CAM0, CAM1, "observation.images.wrist")}
    conversation = _messages(step, observation, ["  pick up the cube \n"])[0]
    assert _layout(conversation) == [
        PROMPT_HEADER,
        "<image>",
        "<image>",
        WRIST_HEADER,
        "<image>",
        "\nGenerate robot actions for the task:\npick up the cube /no_cot",
    ]
    assert conversation[1] == {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TURN}]}


def test_prompt_without_groups_uses_every_camera_as_base_view():
    step = XiaomiRobotics1PrepareMessagesStep(image_keys=[CAM0, CAM1], image_crop_ratio=1.0)
    observation = {CAM0: torch.rand(2, 3, 32, 32), CAM1: torch.rand(2, 3, 32, 32)}
    conversations = _messages(step, observation, ["a", "b"])
    assert len(conversations) == 2
    assert _layout(conversations[1])[:3] == [PROMPT_HEADER, "<image>", "<image>"]
    assert WRIST_HEADER not in _layout(conversations[1])


def test_prompt_rejects_missing_camera_and_missing_task():
    step = XiaomiRobotics1PrepareMessagesStep(base_view_keys=[CAM0, "observation.images.nope"])
    with pytest.raises(KeyError, match="nope"):
        _messages(step, {CAM0: torch.rand(1, 3, 32, 32)}, ["x"])
    with pytest.raises(ValueError, match="Task"):
        from lerobot.lerobot_types import TransitionKey

        step._current_transition = {TransitionKey.OBSERVATION: {}}
        step.complementary_data({})


# ---- full pipeline -----------------------------------------------------------------------------------------


def _pipeline(**config_overrides):
    config = make_config(**config_overrides)
    stats = {
        OBS_STATE: {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
        ACTION: {"mean": torch.full((ACTION_DIM,), 2.0), "std": torch.full((ACTION_DIM,), 3.0)},
    }
    return config, *make_xiaomi_robotics_1_pre_post_processors(config, dataset_stats=stats)


def test_preprocessor_emits_backbone_inputs(tiny_processor):
    _, preprocessor, _ = _pipeline(base_view_keys=[CAM0], wrist_view_keys=[CAM1])
    processed = preprocessor(_raw_batch())
    assert {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"} <= set(
        processed
    )
    assert int((processed["input_ids"] == IMAGE_TOKEN_ID).sum()) == 2  # one merged token per 32x32 image
    assert processed["image_grid_thw"].shape == (2, 3)
    assert processed["mm_token_type_ids"].sum() == 2
    assert processed[OBS_STATE].shape == (1, STATE_DIM)
    assert torch.equal(processed[OBS_STATE][0], _raw_batch()[OBS_STATE])  # state is not normalized
    assert "messages" not in processed


def test_pipeline_end_to_end_unnormalizes_the_chunk(tiny_processor):
    torch.manual_seed(0)
    config, preprocessor, postprocessor = _pipeline()
    policy = XiaomiRobotics1Policy(config).eval()
    batch = preprocessor(_raw_batch())
    noise = torch.randn(1, CHUNK, config.max_action_dim, generator=torch.Generator().manual_seed(1))
    chunk = policy.predict_action_chunk(batch, noise=noise)
    actions = postprocessor(chunk)
    torch.testing.assert_close(actions, chunk * 3.0 + 2.0)
    assert actions.shape == (1, CHUNK, ACTION_DIM)


def test_pipelines_serialize_and_reload(tiny_processor, tmp_path):
    _, preprocessor, postprocessor = _pipeline(
        base_view_keys=[CAM0], wrist_view_keys=[CAM1], image_crop_ratio=0.9, processor_revision="abc"
    )
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    reloaded = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    by_name = {type(step).__name__: step for step in reloaded.steps}
    prepare = by_name["XiaomiRobotics1PrepareMessagesStep"]
    assert (prepare.base_view_keys, prepare.wrist_view_keys, prepare.image_crop_ratio) == (
        [CAM0],
        [CAM1],
        0.9,
    )
    qwen = by_name["XiaomiRobotics1QwenProcessorStep"]
    assert qwen.processor_revision == "abc" and qwen.image_max_pixels == 90000
    again = reloaded(_raw_batch())
    assert torch.equal(again["input_ids"], preprocessor(_raw_batch())["input_ids"])


def test_training_batches_are_right_padded_and_rollout_batches_left_padded(tiny_processor):
    _, preprocessor, _ = _pipeline()
    long_task, short_task = "pick up the cube place it", "pick"
    batch = {
        CAM0: torch.rand(2, 3, 32, 32),
        CAM1: torch.rand(2, 3, 32, 32),
        OBS_STATE: torch.randn(2, STATE_DIM),
        "task": [long_task, short_task],
    }
    rollout = preprocessor(dict(batch))
    assert rollout["attention_mask"][1, 0] == 0 and rollout["attention_mask"][1, -1] == 1
    training = preprocessor({**batch, ACTION: torch.randn(2, CHUNK, ACTION_DIM)})
    assert training["attention_mask"][1, 0] == 1 and training["attention_mask"][1, -1] == 0
