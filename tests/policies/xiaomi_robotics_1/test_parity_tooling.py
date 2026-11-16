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
"""The parity tooling itself: verdict thresholds and the comparison loop, on a tiny model (CPU, no original code)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Policy
from lerobot.utils.constants import OBS_STATE
from tests.policies.xiaomi_robotics_1.tiny import CHUNK, make_batch, make_config
from tests.policies.xiaomi_robotics_1.utils.compare_with_original import (
    chunk_metrics,
    stage_model,
    verdict,
)


def test_verdict_thresholds_follow_phase_35():
    assert verdict(0.0, 1.0) == "EXACT"
    assert verdict(1e-6, 0.5) == "EXACT"
    assert verdict(2e-3, 0.9995) == "NUM"
    assert verdict(2e-3, 0.99) == "DIFFERS"


def test_chunk_metrics_handles_identical_and_zero_chunks():
    a = torch.randn(1, 4, 3)
    assert chunk_metrics(a, a) == {"max_abs": 0.0, "cosine": pytest.approx(1.0)}
    assert chunk_metrics(torch.zeros(2), torch.zeros(2))["cosine"] == 1.0


def _records(policy: XiaomiRobotics1Policy, n: int = 3) -> list[dict]:
    """What `dump_original_xr1.py` would store, with this model standing in for the original."""
    records = []
    for i in range(n):
        batch = make_batch(batch_size=1, seed=i)
        state = policy._state(batch)
        mask = policy._action_mask(state, CHUNK)
        noise = policy.model.sample_noise(mask.shape, mask.device, mask.dtype)
        chunk = policy.model.sample_actions(state, mask, noise=noise, **policy._prefix_inputs(batch))
        processor = {k: v for k, v in batch.items() if k != OBS_STATE}
        records.append(
            {"processor": processor, "state": batch[OBS_STATE][0], "chunk_normalized": chunk.float().cpu()}
        )
    return records


def test_model_stage_reports_exact_against_itself_and_differs_when_perturbed():
    torch.manual_seed(0)
    policy = XiaomiRobotics1Policy(make_config(noise_seed=42)).eval()
    records = _records(policy)
    result = stage_model(records, policy, end_to_end=False, device="cpu")
    assert result["all_dims"]["verdict"] == "EXACT" and result["all_dims"]["n"] == 3
    assert result["action_dims"]["verdict"] == "EXACT"

    for record in records:
        record["chunk_normalized"] = -record["chunk_normalized"]
    perturbed = stage_model(records, policy, end_to_end=False, device="cpu")
    assert perturbed["all_dims"]["verdict"] == "DIFFERS"
