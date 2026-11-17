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

"""End-effector delta checkpoints (VLABench): the policy hands the simulator absolute targets; explicit prompt views."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Policy
from lerobot.policies.xiaomi_robotics_1.processor_xiaomi_robotics_1 import (
    PROMPT_INTRO,
    XiaomiRobotics1PrepareMessagesStep,
)
from lerobot.utils.constants import OBS_IMAGES
from tests.policies.xiaomi_robotics_1.tiny import CHUNK, make_config

MEAN = [0.1, -0.2, 0.3, 0.0, 0.5, -0.5, 0.25]
STD = [2.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.1]


def _policy(**overrides) -> XiaomiRobotics1Policy:
    config = make_config(
        eef_delta_actions=True,
        eef_action_mean=MEAN,
        eef_action_std=STD,
        eef_gripper_threshold=0.2,
        **overrides,
    )
    return XiaomiRobotics1Policy(config).eval()


def test_eef_targets_sum_the_denormalized_deltas_on_top_of_the_state():
    policy = _policy()
    normalized = torch.zeros(1, 3, 7)  # normalized zero = the mean delta
    state = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0]])
    targets = policy.eef_targets(normalized, state)
    assert targets.shape == (1, 3, 7)
    expected = torch.tensor(MEAN[:6]).cumsum(0)  # not cumulative across steps: add the mean every step
    for step in range(3):
        want = state[0, :6] + torch.tensor(MEAN[:6]) * (step + 1)
        assert torch.allclose(targets[0, step, :6], want, atol=1e-6), expected
    # the normalized gripper output 0 is the mean 0.25, which is at least the 0.2 threshold: open
    assert torch.all(targets[0, :, 6] == 1.0)


def test_eef_targets_denormalize_with_std_and_threshold_the_gripper():
    policy = _policy()
    normalized = torch.zeros(1, 2, 7)
    normalized[0, 0, 0] = 1.0  # +1 std on x
    normalized[0, 1, 6] = -3.0  # gripper 0.25 - 0.3 = -0.05: closed
    targets = policy.eef_targets(normalized, torch.zeros(1, 7))
    assert torch.isclose(targets[0, 0, 0], torch.tensor(MEAN[0] + STD[0]))
    assert targets[0, 0, 6] == 1.0
    assert targets[0, 1, 6] == 0.0


def test_eef_targets_wrap_euler_angles_after_every_step():
    policy = _policy()
    state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0]])
    normalized = torch.zeros(1, 4, 7)
    normalized[:, :, 5] = (1.0 - MEAN[5]) / STD[5]  # +1.0 rad of rz every step
    targets = policy.eef_targets(normalized, state)
    rz = targets[0, :, 5]
    assert torch.all(rz <= math.pi) and torch.all(rz >= -math.pi)
    assert torch.isclose(rz[0], torch.tensor(4.0 - 2 * math.pi), atol=1e-5)
    assert torch.isclose(rz[1], torch.tensor(5.0 - 2 * math.pi), atol=1e-5)


def test_select_action_returns_targets_one_step_at_a_time_and_replans_after_n_action_steps():
    policy = _policy(n_action_steps=3)
    calls = []

    def fake_chunk(batch, noise=None):
        calls.append(1)
        return torch.zeros(batch["observation.state"].shape[0], CHUNK, 7)

    policy.predict_action_chunk = fake_chunk
    batch = {"observation.state": torch.zeros(1, 7)}
    first = [policy.select_action(batch) for _ in range(3)]
    assert len(calls) == 1
    assert [a.shape for a in first] == [(1, 7)] * 3
    # a later chunk is anchored on the state at the time of the new call, not on the first one
    batch2 = {"observation.state": torch.ones(1, 7)}
    policy.select_action(batch2)
    assert len(calls) == 2


def test_the_adapter_needs_its_statistics():
    with pytest.raises(ValueError, match="eef_action_mean"):
        make_config(eef_delta_actions=True)
    with pytest.raises(ValueError, match="at least 7"):
        make_config(eef_delta_actions=True, eef_action_mean=[0.0] * 3, eef_action_std=[1.0] * 3)


def test_view_groups_become_the_ego_base_wrist_prompt_of_the_authors_vlabench_client():
    cams = [f"{OBS_IMAGES}.{n}" for n in ("ego", "base", "wrist")]
    step = XiaomiRobotics1PrepareMessagesStep(
        view_groups=[["Ego View", cams[0]], ["Base View", cams[1]], ["Left-Wrist View", cams[2]]],
        image_keys=cams,
        image_crop_ratio=1.0,
    )
    from lerobot.lerobot_types import TransitionKey

    step._current_transition = {TransitionKey.OBSERVATION: {k: torch.rand(1, 3, 32, 32) for k in cams}}
    content = step.complementary_data({"task": ["pick the apple"]})["messages"][0][0]["content"]
    layout = [p["text"] if p["type"] == "text" else "<image>" for p in content]
    assert layout == [
        PROMPT_INTRO + "# Ego View\n",
        "<image>",
        "\n# Base View\n",
        "<image>",
        "\n# Left-Wrist View\n",
        "<image>",
        "\nGenerate robot actions for the task:\npick the apple /no_cot",
    ]


def test_view_groups_need_a_header_and_a_key():
    with pytest.raises(ValueError, match="at least one camera key"):
        make_config(view_groups=[["Ego View"]])
