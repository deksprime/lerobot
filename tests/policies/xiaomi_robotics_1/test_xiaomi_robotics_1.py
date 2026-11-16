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

"""Xiaomi-Robotics-1 policy tests: tiny random models on CPU, no network, no checkpoint download."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xiaomi_robotics_1.configuration_xiaomi_robotics_1 import XiaomiRobotics1Config
from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import (
    XiaomiRobotics1Policy,
    XR1DiTAttention,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.xiaomi_robotics_1.tiny import ACTION_DIM, CHUNK, MAX_DIM, make_batch, make_config

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def policy() -> XiaomiRobotics1Policy:
    torch.manual_seed(0)
    return XiaomiRobotics1Policy(make_config()).eval()


# ---- configuration -----------------------------------------------------------------------------------------


def test_defaults_match_released_robocasa_checkpoint():
    config = XiaomiRobotics1Config()
    assert config.type == "xiaomi_robotics_1"
    assert (config.max_state_dim, config.max_action_dim) == (60, 60)
    assert (config.chunk_size, config.num_denoise_steps, config.state_length) == (10, 5, 1)
    assert (config.dit_num_layers, config.dit_hidden_size, config.dit_head_dim) == (36, 1024, 128)
    assert config.freq_excluded_dims == (17, 18, 19)
    assert config.image_crop_ratio == 0.95


def test_config_is_registered_and_round_trips(tmp_path):
    config = make_config(noise_seed=42, base_view_keys=["observation.images.cam0"])
    config.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, XiaomiRobotics1Config)
    assert loaded.noise_seed == 42
    assert loaded.base_view_keys == ["observation.images.cam0"]
    assert loaded.vlm_config == config.vlm_config
    assert loaded.freq_excluded_dims == config.freq_excluded_dims


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_action_steps": CHUNK + 1},
        {"num_denoise_steps": 0},
        {"image_crop_ratio": 0.0},
        {"dit_hidden_size": 33},
    ],
)
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        make_config(**overrides)


def test_expert_must_match_backbone_head_shape():
    with pytest.raises(ValueError, match="head_dim"):
        XiaomiRobotics1Policy(make_config(dit_head_dim=8, dit_hidden_size=16, dit_num_key_value_heads=2))
    with pytest.raises(ValueError, match="key/value heads"):
        XiaomiRobotics1Policy(make_config(dit_num_key_value_heads=1))


# ---- parameter layout (backward compatibility with the released checkpoints) -------------------------------


def _patterns(names) -> set[str]:
    return {re.sub(r"\.\d+(?=\.|$)", ".N", name) for name in names}


def test_state_dict_layout_matches_released_checkpoint(policy):
    released = set(json.loads((HERE / "released_key_patterns.json").read_text())["patterns"])
    ours = _patterns(name.removeprefix("model.") for name in policy.state_dict())
    ours.discard("vlm.lm_head.weight")  # tied to the embedding, absent from the checkpoint
    assert ours == released, (sorted(ours - released), sorted(released - ours))


def test_rotary_buffers_follow_model_dtype():
    """The authors cast the whole model to bf16, which also rounds the (non-persistent) rotary `inv_freq`."""
    policy = XiaomiRobotics1Policy(make_config(dtype=torch.bfloat16))
    assert policy.model.rotary_emb.inv_freq.dtype == torch.bfloat16
    assert policy.model.vlm.model.language_model.rotary_emb.inv_freq.dtype == torch.bfloat16


# ---- training ----------------------------------------------------------------------------------------------


def test_forward_returns_finite_loss_and_gradients_reach_expert_and_backbone():
    torch.manual_seed(0)
    policy = XiaomiRobotics1Policy(make_config()).train()
    loss, info = policy(make_batch(with_action=True))
    assert torch.isfinite(loss) and loss.ndim == 0
    assert {"loss", "loss_mse", "loss_freq"} <= set(info)
    loss.backward()
    grads = {name for name, p in policy.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any(n.startswith("model.dit.layers.0.") for n in grads)
    assert any("language_model.layers.0." in n for n in grads)
    assert any(n.startswith("model.state_projector") for n in grads)


def test_frozen_backbone_gets_no_gradient():
    torch.manual_seed(0)
    policy = XiaomiRobotics1Policy(make_config(train_vlm=False)).train()
    policy(make_batch(with_action=True))[0].backward()
    assert all(p.grad is None for p in policy.model.vlm.parameters())
    assert any(p.grad is not None for p in policy.model.dit.parameters())


def test_padded_actions_and_padding_dims_do_not_contribute():
    torch.manual_seed(0)
    policy = XiaomiRobotics1Policy(make_config()).train()
    batch = make_batch(with_action=True)
    batch["action_is_pad"][:, CHUNK // 2 :] = True
    torch.manual_seed(1)
    loss_a, _ = policy(batch)
    batch[ACTION][:, CHUNK // 2 :] += 100.0  # garbage in the padded steps
    torch.manual_seed(1)
    loss_b, _ = policy(batch)
    assert torch.allclose(loss_a, loss_b, atol=1e-5)


def test_a_few_optimizer_steps_run():
    torch.manual_seed(0)
    policy = XiaomiRobotics1Policy(make_config()).train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    before = policy.model.state_projector.layers[0].weight.detach().clone()
    for _ in range(3):
        loss, _ = policy(make_batch(with_action=True))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
    assert not torch.equal(before, policy.model.state_projector.layers[0].weight)


# ---- inference ---------------------------------------------------------------------------------------------


def test_predict_action_chunk_shape_and_determinism(policy):
    batch = make_batch()
    noise = torch.randn(2, CHUNK, MAX_DIM, generator=torch.Generator().manual_seed(3))
    a = policy.predict_action_chunk(batch, noise=noise)
    b = policy.predict_action_chunk(batch, noise=noise)
    assert a.shape == (2, CHUNK, ACTION_DIM) and a.dtype == torch.float32
    assert torch.equal(a, b)
    assert not torch.equal(a, policy.predict_action_chunk(batch, noise=noise * 2))


def test_noise_seed_fixes_the_sample_without_touching_global_rng():
    policy = XiaomiRobotics1Policy(make_config(noise_seed=42)).eval()
    batch = make_batch()
    state = torch.get_rng_state()
    a = policy.predict_action_chunk(batch)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(a, policy.predict_action_chunk(batch))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_seeded_noise_equals_manual_seed_randn_like(dtype):
    """Reproduces the authors' `torch.manual_seed(seed); torch.randn_like(action_mask)`."""
    policy = XiaomiRobotics1Policy(make_config(noise_seed=42))
    mask = torch.zeros(1, CHUNK, MAX_DIM, dtype=dtype)
    ours = policy.model.sample_noise(mask.shape, mask.device, dtype)
    torch.manual_seed(42)
    assert torch.equal(ours, torch.randn_like(mask))


def test_select_action_pops_one_step_and_refills_per_chunk(policy):
    policy.reset()
    batch = make_batch()
    noise = torch.randn(2, CHUNK, MAX_DIM, generator=torch.Generator().manual_seed(3))
    chunk = policy.predict_action_chunk(batch, noise=noise)
    first = [policy.select_action(batch, noise=noise) for _ in range(policy.config.n_action_steps)]
    assert all(step.shape == (2, ACTION_DIM) for step in first)
    assert torch.equal(torch.stack(first, dim=1), chunk[:, : policy.config.n_action_steps])
    assert len(policy._action_queue) == 0
    policy.select_action(batch, noise=noise)  # refills
    assert len(policy._action_queue) == policy.config.n_action_steps - 1


def test_each_row_of_a_batch_matches_running_it_alone(policy):
    batch = make_batch(batch_size=2, seed=5)
    noise = torch.randn(2, CHUNK, MAX_DIM, generator=torch.Generator().manual_seed(9))
    both = policy.predict_action_chunk(batch, noise=noise)
    for row in range(2):
        single = {
            "input_ids": batch["input_ids"][row : row + 1],
            "attention_mask": batch["attention_mask"][row : row + 1],
            "mm_token_type_ids": batch["mm_token_type_ids"][row : row + 1],
            "image_grid_thw": batch["image_grid_thw"][2 * row : 2 * row + 2],
            "pixel_values": batch["pixel_values"][8 * row : 8 * row + 8],
            OBS_STATE: batch[OBS_STATE][row : row + 1],
        }
        alone = policy.predict_action_chunk(single, noise=noise[row : row + 1])
        torch.testing.assert_close(alone[0], both[row], atol=2e-5, rtol=2e-5)


def test_left_padding_does_not_change_the_chunk(policy):
    """Positions come from the attention mask and the expert masks the padded cache entries."""
    from tests.policies.xiaomi_robotics_1.tiny import make_prefix

    noise = torch.randn(2, CHUNK, MAX_DIM, generator=torch.Generator().manual_seed(11))
    base = make_batch(seed=2)
    padded = {**base, **make_prefix(batch_size=2, pad=7, seed=2)}
    plain = policy.predict_action_chunk(base, noise=noise)
    shifted = policy.predict_action_chunk(padded, noise=noise)
    torch.testing.assert_close(plain, shifted, atol=2e-5, rtol=2e-5)


def test_chunk_depends_on_images_text_and_state(policy):
    batch = make_batch()
    noise = torch.randn(2, CHUNK, MAX_DIM, generator=torch.Generator().manual_seed(3))
    reference = policy.predict_action_chunk(batch, noise=noise)
    images = {**batch, "pixel_values": batch["pixel_values"] + 1.0}
    words = {
        **batch,
        "input_ids": torch.where(batch["input_ids"] < 30, (batch["input_ids"] + 1) % 30, batch["input_ids"]),
    }
    state = {**batch, OBS_STATE: batch[OBS_STATE] + 1.0}
    for changed in (images, words, state):
        assert not torch.allclose(reference, policy.predict_action_chunk(changed, noise=noise), atol=1e-6)


def test_missing_prefix_is_a_clear_error(policy):
    with pytest.raises(KeyError, match="preprocessor"):
        policy.predict_action_chunk({OBS_STATE: torch.zeros(1, 5)})


def test_expert_attention_matches_a_naive_reference():
    """`XR1DiTAttention` (sdpa, repeated cache, masks) against explicit softmax attention."""
    torch.manual_seed(0)
    attn = XR1DiTAttention(hidden_size=32, head_dim=16, num_key_value_heads=1).eval()  # 2 heads, group 2
    batch, q_len, cache_len = 2, 4, 5
    x = torch.randn(batch, q_len, 32)
    cache_k, cache_v = torch.randn(batch, 1, cache_len, 16), torch.randn(batch, 1, cache_len, 16)
    cos, sin = torch.ones(batch, q_len, 16), torch.zeros(batch, q_len, 16)  # identity rotation
    mask = torch.ones(batch, 1, q_len, cache_len + q_len, dtype=torch.bool)
    mask[1, :, :, :2] = False
    got = attn(x, cache_k, cache_v, (cos, sin), mask)

    qkv = attn.qkv_proj(x).view(batch, q_len, 3, 2, 16)
    q, k, v = qkv.unbind(2)
    q, k = attn.q_norm(q).transpose(1, 2), attn.k_norm(k).transpose(1, 2)
    v = v.transpose(1, 2)
    k = torch.cat([cache_k.expand(-1, 2, -1, -1), k], dim=2)
    v = torch.cat([cache_v.expand(-1, 2, -1, -1), v], dim=2)
    scores = (q @ k.transpose(-1, -2)) / 16**0.5
    scores = scores.masked_fill(~mask, float("-inf"))
    want = attn.o_proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, q_len, 32))
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


# ---- persistence -------------------------------------------------------------------------------------------


def test_save_and_reload_gives_identical_chunks(tmp_path):
    torch.manual_seed(0)
    policy = XiaomiRobotics1Policy(make_config()).eval()
    policy.save_pretrained(tmp_path)
    reloaded = XiaomiRobotics1Policy.from_pretrained(tmp_path).eval()
    batch = make_batch()
    noise = torch.randn(2, CHUNK, MAX_DIM, generator=torch.Generator().manual_seed(3))
    assert torch.equal(
        policy.predict_action_chunk(batch, noise=noise), reloaded.predict_action_chunk(batch, noise=noise)
    )


def test_token_types_are_derived_when_the_processor_does_not_return_them(policy):
    """The authors' processor (transformers 4.57.1) does not emit `mm_token_type_ids`; 5.x needs them for mrope."""
    batch = make_batch(seed=4)
    noise = torch.randn(2, CHUNK, MAX_DIM, generator=torch.Generator().manual_seed(3))
    with_types = policy.predict_action_chunk(batch, noise=noise)
    without = policy.predict_action_chunk(
        {k: v for k, v in batch.items() if k != "mm_token_type_ids"}, noise=noise
    )
    assert torch.equal(with_types, without)


def test_mrope_positions_need_the_image_token_types(policy):
    """A 4x4 patch grid (2x2 merged tokens) occupies a 2-D block of positions, not a run of text positions."""
    from tests.policies.xiaomi_robotics_1.tiny import IMAGE_TOKEN_ID, VISION_END_ID, VISION_START_ID

    ids = torch.tensor([[1, 2, VISION_START_ID, *[IMAGE_TOKEN_ID] * 4, VISION_END_ID, 5, 6]])
    mask = torch.ones_like(ids)
    grid = torch.tensor([[1, 4, 4]])
    backbone = policy.model.vlm
    derived = backbone.mrope_position_ids(ids, mask, grid, None)
    explicit = backbone.mrope_position_ids(ids, mask, grid, (ids == IMAGE_TOKEN_ID).long())
    text_only = backbone.mrope_position_ids(ids, mask, grid, torch.zeros_like(ids))
    assert torch.equal(derived, explicit)
    assert not torch.equal(derived, text_only)
    assert derived.shape == (3, 1, ids.shape[1])


def test_default_config_reproduces_the_released_architecture_without_downloading_anything():
    """The embedded Qwen3-VL-4B + expert config builds exactly the released tensor set (on the meta device)."""
    from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Model

    config = XiaomiRobotics1Config(device="cpu")
    with torch.device("meta"):
        model = XiaomiRobotics1Model(config)
    released = set(json.loads((HERE / "released_key_patterns.json").read_text())["patterns"])
    state = model.state_dict()
    ours = _patterns(state)
    ours.discard("vlm.lm_head.weight")
    assert ours == released
    # 1120 tensors in the release (the tied lm_head is not one of them) and about 5B parameters.
    assert len([k for k in state if k != "vlm.lm_head.weight"]) == 1120
    total = sum(p.numel() for p in model.parameters())
    assert 4.9e9 < total < 5.3e9, total
