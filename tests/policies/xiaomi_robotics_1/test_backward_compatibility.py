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

"""Saved Xiaomi-Robotics-1 checkpoints must keep loading and keep producing the same actions."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("transformers")

from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Policy
from tests.policies.xiaomi_robotics_1.make_artifact import ARTIFACT_DIR, reference_inputs

# Resolve against the repository root so the test does not depend on the working directory.
ARTIFACT = Path(__file__).resolve().parents[3] / ARTIFACT_DIR


def test_committed_checkpoint_still_loads_and_reproduces_its_chunk():
    policy = XiaomiRobotics1Policy.from_pretrained(ARTIFACT).eval()
    batch, noise = reference_inputs()
    expected = load_file(ARTIFACT / "expected_chunk.safetensors")["chunk"]
    got = policy.predict_action_chunk(batch, noise=noise)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
