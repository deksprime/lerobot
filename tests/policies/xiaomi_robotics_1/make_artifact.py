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

"""Regenerate the Xiaomi-Robotics-1 backward-compatibility artifact.

    uv run python -m tests.policies.xiaomi_robotics_1.make_artifact

Writes `tests/artifacts/policies/xiaomi_robotics_1_tiny/`: a tiny random checkpoint saved with `save_pretrained`
and the action chunk it produces for fixed inputs and fixed noise. `test_backward_compatibility.py` loads the
committed checkpoint with the current code and requires the same chunk, so a change that renames a parameter,
reorders the expert, or alters the sampler fails loudly instead of silently breaking saved checkpoints.
Regenerate only for an intentional, documented break, and say so in the PR.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Policy
from tests.policies.xiaomi_robotics_1.tiny import CHUNK, MAX_DIM, make_batch, make_config

ARTIFACT_DIR = Path("tests/artifacts/policies/xiaomi_robotics_1_tiny")


def reference_inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch = make_batch(batch_size=2, seed=123)
    noise = torch.randn(2, CHUNK, MAX_DIM, generator=torch.Generator().manual_seed(7))
    return batch, noise


def main() -> None:
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    torch.manual_seed(1337)
    policy = XiaomiRobotics1Policy(make_config()).eval()
    policy.save_pretrained(ARTIFACT_DIR)
    batch, noise = reference_inputs()
    chunk = policy.predict_action_chunk(batch, noise=noise)
    save_file({"chunk": chunk.contiguous()}, ARTIFACT_DIR / "expected_chunk.safetensors")
    print(f"wrote {ARTIFACT_DIR} ({sum(p.stat().st_size for p in ARTIFACT_DIR.iterdir()) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
