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

"""Xiaomi-Robotics-1 (XR-1) policy.

Architecture (Xiaomi Robotics, Apache-2.0, https://github.com/XiaomiRobotics/Xiaomi-Robotics-1): a Qwen3-VL backbone
reads the camera images and the instruction once. An action expert, a 36-layer DiT, then runs five Euler steps of a
flow-matching sampler; every expert layer attends to its own tokens (sink, state, noisy actions) and to the key/value
cache of the backbone layer of the same index.

The backbone is the stock `transformers` Qwen3-VL (only subclassed to expose the prefix cache and positions); the
action expert is written against the released tensor layout, so a converted checkpoint keeps the authors' parameter
names under a `model.` prefix. Nothing here uses `trust_remote_code`.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from ..common.vla_utils import pad_vector
from ..pretrained import PreTrainedPolicy
from .configuration_xiaomi_robotics_1 import XiaomiRobotics1Config

if TYPE_CHECKING or _transformers_available:
    from transformers.activations import ACT2FN
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLTextRMSNorm,
        Qwen3VLTextRotaryEmbedding,
        apply_rotary_pos_emb,
    )
else:
    ACT2FN = None
    Qwen3VLForConditionalGeneration = nn.Module
    Qwen3VLTextRMSNorm = None
    Qwen3VLTextRotaryEmbedding = None
    apply_rotary_pos_emb = None

ADALN_CHUNKS = 6  # shift/scale/gate for attention and for the MLP


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class XR1Projector(nn.Module):
    """`num_layers` linear layers with tanh-GELU in between. Parameters live under `layers.<i>`."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inter_dim: int | None = None,
        num_layers: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        inter_dim = output_dim if inter_dim is None else inter_dim
        if num_layers == 1:
            layers: list[nn.Module] = [nn.Linear(input_dim, output_dim, bias=bias)]
        else:
            layers = [nn.Linear(input_dim, inter_dim, bias=bias)]
            for _ in range(1, num_layers - 1):
                layers += [nn.GELU(approximate="tanh"), nn.Linear(inter_dim, inter_dim, bias=bias)]
            layers += [nn.GELU(approximate="tanh"), nn.Linear(inter_dim, output_dim, bias=bias)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class XR1TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of the flow time followed by a two-layer SiLU MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, timestep: Tensor) -> Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
        )
        args = timestep[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # The authors hard-code bfloat16 here; follow the module's dtype so fp32 CPU runs work. Identical for bf16.
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))[:, None]


class XR1DiTAttention(nn.Module):
    """Attention over [backbone cache ; own tokens] with a fused qkv projection and per-head q/k RMSNorm."""

    def __init__(self, hidden_size: int, head_dim: int, num_key_value_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.kv_group = self.num_heads // num_key_value_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = Qwen3VLTextRMSNorm(head_dim)
        self.k_norm = Qwen3VLTextRMSNorm(head_dim)

    def forward(
        self,
        hidden_states: Tensor,
        cache_key: Tensor,
        cache_value: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attn_mask: Tensor,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # The cache holds one row per observation; training draws several noise samples per observation.
        if cache_key.shape[0] != batch_size:
            repeats = batch_size // cache_key.shape[0]
            cache_key = cache_key.repeat_interleave(repeats, dim=0)
            cache_value = cache_value.repeat_interleave(repeats, dim=0)
        if self.kv_group > 1:
            cache_key = cache_key.repeat_interleave(self.kv_group, dim=1)
            cache_value = cache_value.repeat_interleave(self.kv_group, dim=1)
        key = torch.cat([cache_key.to(key.dtype), key], dim=-2)
        value = torch.cat([cache_value.to(value.dtype), value], dim=-2)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        return self.o_proj(out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size))


class XR1DiTMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class XR1DiTLayer(nn.Module):
    """Pre-norm block with adaLN-zero style shift/scale/gate modulation from the flow time."""

    def __init__(self, hidden_size: int, head_dim: int, num_key_value_heads: int):
        super().__init__()
        self.attn = XR1DiTAttention(hidden_size, head_dim, num_key_value_heads)
        self.mlp = XR1DiTMLP(hidden_size)
        self.input_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=1e-6)
        self.post_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=1e-6)
        self.adaln_table = nn.Parameter(torch.randn(ADALN_CHUNKS, hidden_size) / hidden_size**0.5)

    def forward(self, hidden_states, cache_key, cache_value, position_embeddings, time_embeds, attn_mask):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_table[None] + time_embeds
        ).chunk(ADALN_CHUNKS, dim=1)
        residual = hidden_states
        hidden_states = _modulate(self.input_layernorm(hidden_states), shift_attn, scale_attn)
        hidden_states = residual + gate_attn * self.attn(
            hidden_states, cache_key, cache_value, position_embeddings, attn_mask
        )
        residual = hidden_states
        hidden_states = _modulate(self.post_layernorm(hidden_states), shift_mlp, scale_mlp)
        return residual + gate_mlp * self.mlp(hidden_states)


class XR1DiT(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, head_dim: int, num_key_value_heads: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [XR1DiTLayer(hidden_size, head_dim, num_key_value_heads) for _ in range(num_layers)]
        )

    def forward(self, hidden_states, prefix_cache, position_embeddings, time_embeds, attn_mask):
        # The expert layer i reads the backbone layer (n_backbone - n_expert + i); both are 36 in the checkpoints.
        start = len(prefix_cache) - len(self.layers)
        for index, layer in enumerate(self.layers):
            cache_key, cache_value = prefix_cache[start + index]
            hidden_states = layer(
                hidden_states, cache_key, cache_value, position_embeddings, time_embeds, attn_mask
            )
        return hidden_states


class XR1Backbone(Qwen3VLForConditionalGeneration):
    """Stock Qwen3-VL that can hand back its key/value cache and the multimodal positions of the prefix.

    The language-model head is never evaluated: the action expert only needs the cache. (The authors' copy of the
    modeling file did the same with a `skip_logits` flag.)
    """

    def _mm_token_type_ids(self, input_ids: Tensor, mm_token_type_ids: Tensor | None) -> Tensor:
        """transformers 5.x needs the token types (text 0, image 1) to place image tokens in mrope space; 4.57.1
        derived them from `input_ids`, and the authors' processor does not return them, so derive them here."""
        if mm_token_type_ids is not None:
            return mm_token_type_ids
        return (input_ids == self.config.image_token_id).long()

    @torch.no_grad()
    def mrope_position_ids(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        image_grid_thw: Tensor | None,
        mm_token_type_ids: Tensor | None,
    ) -> Tensor:
        mm_token_type_ids = self._mm_token_type_ids(input_ids, mm_token_type_ids)
        position_ids, _ = self.model.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return position_ids

    def encode_prefix(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        mm_token_type_ids: Tensor | None = None,
    ) -> tuple[list[tuple[Tensor, Tensor]], Tensor]:
        """Run the prompt once. Returns the per-layer (key, value) cache and the (3, B, L) positions."""
        mm_token_type_ids = self._mm_token_type_ids(input_ids, mm_token_type_ids)
        position_ids = self.mrope_position_ids(input_ids, attention_mask, image_grid_thw, mm_token_type_ids)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=True,
        )
        cache = outputs.past_key_values
        return [(layer.keys, layer.values) for layer in cache.layers], position_ids


class XiaomiRobotics1Model(nn.Module):
    """Backbone, action expert and projectors. State-dict keys match the released checkpoints."""

    def __init__(self, config: XiaomiRobotics1Config):
        super().__init__()
        self.config = config
        text_config = config.text_config
        self.vlm = XR1Backbone._from_config(config.vlm_backbone_config, dtype=config.dtype)
        hidden = config.dit_hidden_size
        self.dit = XR1DiT(config.dit_num_layers, hidden, config.dit_head_dim, config.dit_num_key_value_heads)
        self.state_projector = XR1Projector(config.max_state_dim, hidden, inter_dim=hidden, num_layers=2)
        self.action_projector = XR1Projector(config.max_action_dim, hidden, inter_dim=hidden, num_layers=2)
        self.action_output_layer = XR1Projector(hidden, config.max_action_dim, inter_dim=hidden, num_layers=2)
        self.t_embedder = XR1TimestepEmbedder(hidden)
        self.t_projector = XR1Projector(hidden, ADALN_CHUNKS * hidden, bias=True)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(text_config)
        self.sink = nn.Embedding(1, hidden)
        if config.dtype is not None:
            self.to(config.dtype)
        if not config.train_vlm:
            self.vlm.requires_grad_(False)
        if config.gradient_checkpointing:
            self.vlm.gradient_checkpointing_enable()

    # ---- flow-matching pieces -------------------------------------------------------------------------------

    def _dit_forward(
        self, noisy_action, timestep, action_mask, state_embed, position_embeddings, prefix_cache, attn_mask
    ):
        time_embeds = self.t_embedder(timestep[:, 0, 0] * 1000)
        time_embeds = self.t_projector(time_embeds).view(time_embeds.shape[0], ADALN_CHUNKS, -1)
        action_embed = self.action_projector(noisy_action * action_mask)
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_embed, action_embed], dim=1).contiguous()
        hidden_states = self.dit(hidden_states, prefix_cache, position_embeddings, time_embeds, attn_mask)
        return self.action_output_layer(hidden_states[:, -action_embed.shape[1] :])

    def _expert_inputs(self, positions: Tensor, attention_mask: Tensor, state: Tensor, action: Tensor):
        """Positions, attention mask and state tokens shared by every denoising step."""
        batch_size, action_length, _ = action.shape
        query_length = 1 + state.shape[1] + action_length
        dit_positions = (
            torch.arange(query_length, device=action.device).view(1, 1, -1).repeat(3, batch_size, 1)
            + positions.max(dim=-1).values[..., None]
            + 1
        )
        position_embeddings = self.rotary_emb(action, dit_positions)
        prefix_mask = attention_mask.bool()[:, None, :].expand(-1, query_length, -1)
        causal = torch.tril(
            torch.ones(batch_size, query_length, query_length, dtype=torch.bool, device=action.device)
        )
        attn_mask = torch.cat([prefix_mask, causal], dim=-1)[:, None]
        return position_embeddings, attn_mask, self.state_projector(state)

    def sample_noise(self, shape: tuple[int, ...], device, dtype) -> Tensor:
        seed = self.config.noise_seed
        if seed is None:
            return torch.randn(shape, device=device, dtype=dtype)
        # A private generator reproduces `torch.manual_seed(seed); torch.randn_like(x)` without touching global RNG.
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    @torch.no_grad()
    def sample_actions(self, state, action_mask, noise=None, **prefix_inputs) -> Tensor:
        prefix_cache, positions = self.vlm.encode_prefix(**prefix_inputs)
        if noise is None:
            noise = self.sample_noise(action_mask.shape, action_mask.device, action_mask.dtype)
        position_embeddings, attn_mask, state_embed = self._expert_inputs(
            positions, prefix_inputs["attention_mask"], state, noise
        )
        x = noise.clone()
        steps = self.config.num_denoise_steps
        dt = 1.0 / steps
        for step in range(steps):
            # Same expression, same dtype rounding as the reference sampler.
            t = torch.ones((x.shape[0], 1, 1), device=x.device, dtype=x.dtype) * step / steps
            velocity = self._dit_forward(
                x, t, action_mask, state_embed, position_embeddings, prefix_cache, attn_mask
            )
            x = x + velocity * dt
        return x

    def compute_loss(self, state, action, action_mask, **prefix_inputs) -> tuple[Tensor, dict[str, Tensor]]:
        with torch.set_grad_enabled(self.config.train_vlm and torch.is_grad_enabled()):
            prefix_cache, positions = self.vlm.encode_prefix(**prefix_inputs)
        repeats = self.config.training_repeat
        position_embeddings, attn_mask, state_embed = self._expert_inputs(
            positions, prefix_inputs["attention_mask"], state, action
        )

        def expand(x: Tensor, dim: int = 0) -> Tensor:
            return x.repeat_interleave(repeats, dim=dim) if repeats > 1 else x

        action, action_mask = expand(action), expand(action_mask)
        state_embed, attn_mask = expand(state_embed), expand(attn_mask)
        position_embeddings = tuple(expand(x) for x in position_embeddings)

        noise = torch.randn_like(action)
        beta = torch.distributions.Beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta
        )
        timestep = (1 - beta.sample((action.shape[0],)).to(action.device)) * self.config.time_sampling_scale
        timestep = timestep.to(action.dtype)[:, None, None]
        noisy_action = (1 - timestep) * noise + timestep * action
        target = action - noise
        pred = self._dit_forward(
            noisy_action, timestep, action_mask, state_embed, position_embeddings, prefix_cache, attn_mask
        )
        return self._flow_loss(pred, target, action_mask)

    def _flow_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Masked MSE plus a frequency-domain L1 term along the chunk, as in the authors' training code."""
        pred, target, mask = pred.float(), target.float(), mask.bool()
        if not mask.any():
            zero = (pred.sum() + target.sum()) * 0.0
            return zero, {"loss_mse": zero.detach(), "loss_freq": zero.detach()}
        loss_mse = F.mse_loss(pred, target, reduction="none")[mask].mean()
        freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
        valid = mask[:, -1].any(dim=-1)
        if valid.any():
            freq_mask = mask[valid, : freq.shape[1]].clone()
            dims = [d for d in self.config.freq_excluded_dims if d < freq_mask.shape[-1]]
            freq_mask[:, :, dims] = False
            loss_freq = freq[valid][freq_mask].mean() if freq_mask.any() else freq.sum() * 0.0
        else:
            loss_freq = freq.sum() * 0.0
        loss = self.config.mse_loss_weight * loss_mse + self.config.freq_coefficient * loss_freq
        return loss, {"loss_mse": loss_mse.detach(), "loss_freq": loss_freq.detach()}


_PREFIX_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")


class XiaomiRobotics1Policy(PreTrainedPolicy):
    """LeRobot wrapper around [`XiaomiRobotics1Model`]."""

    config_class = XiaomiRobotics1Config
    name = "xiaomi_robotics_1"
    _fsdp_wrap_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock", "XR1DiTLayer"]

    def __init__(self, config: XiaomiRobotics1Config, **kwargs):
        require_package("transformers", extra="xiaomi_robotics_1")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = XiaomiRobotics1Model(config)
        self.model.to(config.device)
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    @property
    def _action_dim(self) -> int:
        if self.config.output_features is None:
            raise ValueError("`output_features` must be resolved before the policy can size its actions.")
        return self.config.output_features[ACTION].shape[0]

    def _prefix_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        missing = [key for key in ("input_ids", "attention_mask") if key not in batch]
        if missing:
            raise KeyError(
                f"Missing {missing}: run the batch through the Xiaomi-Robotics-1 preprocessor first "
                "(`make_xiaomi_robotics_1_pre_post_processors`)."
            )
        return {key: batch[key] for key in _PREFIX_KEYS if key in batch}

    def _state(self, batch: dict[str, Tensor]) -> Tensor:
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        if state.ndim == 2:
            state = state[:, None]
        if state.shape[1] != self.config.state_length:
            raise ValueError(f"Expected {self.config.state_length} state vector(s), got {state.shape[1]}.")
        return state

    def _action_mask(self, like: Tensor, length: int) -> Tensor:
        mask = torch.zeros(
            like.shape[0], length, self.config.max_action_dim, device=like.device, dtype=like.dtype
        )
        mask[..., : self._action_dim] = 1
        return mask

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if reduction != "mean":
            raise NotImplementedError("Xiaomi-Robotics-1 only supports reduction='mean'.")
        dtype = next(self.model.dit.parameters()).dtype
        state = self._state(batch).to(dtype)
        action = pad_vector(batch[ACTION], self.config.max_action_dim).to(dtype)
        mask = self._action_mask(action, action.shape[1])
        if "action_is_pad" in batch:
            mask = mask * (~batch["action_is_pad"].bool())[:, :, None].to(mask.dtype)
        loss, parts = self.model.compute_loss(state, action, mask, **self._prefix_inputs(batch))
        return loss, {"loss": float(loss.detach()), **{k: float(v) for k, v in parts.items()}}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        """Return `(B, chunk_size, action_dim)` normalized actions.

        `noise` is the optional starting sample, `(B, chunk_size, max_action_dim)`; it overrides `noise_seed`.
        """
        self.eval()
        dtype = next(self.model.dit.parameters()).dtype
        state = self._state(batch).to(dtype)
        mask = self._action_mask(state, self.config.chunk_size)
        if noise is not None:
            noise = noise.to(device=mask.device, dtype=mask.dtype)
        actions = self.model.sample_actions(state, mask, noise=noise, **self._prefix_inputs(batch))
        return actions[:, :, : self._action_dim].to(torch.float32)

    def eef_targets(self, chunk: Tensor, state: Tensor) -> Tensor:
        """Turn normalized end-effector deltas into absolute targets, as the authors' VLABench client does.

        `chunk` is `(B, T, >=7)` normalized, `state` `(B, >=7)` with `[x, y, z, rx, ry, rz, gripper]` first. Returns
        `(B, T, 7)`: the position and Euler targets `state + cumulative sum of the de-normalized deltas` (Euler
        angles wrapped to [-pi, pi] after every step) and a gripper value of 1.0 when the de-normalized gripper
        output is at least `eef_gripper_threshold`, else 0.0.
        """
        mean = torch.as_tensor(self.config.eef_action_mean[:7], dtype=torch.float32, device=chunk.device)
        std = torch.as_tensor(self.config.eef_action_std[:7], dtype=torch.float32, device=chunk.device)
        raw = chunk[:, :, :7].to(torch.float32) * std + mean
        current = state[:, :7].to(device=chunk.device, dtype=torch.float32).clone()
        targets = []
        for step in range(raw.shape[1]):
            current[:, :6] = current[:, :6] + raw[:, step, :6]
            current[:, 3:6] = (current[:, 3:6] + torch.pi) % (2.0 * torch.pi) - torch.pi
            gripper = (raw[:, step, 6] >= self.config.eef_gripper_threshold).to(current.dtype)
            targets.append(torch.cat([current[:, :6], gripper[:, None]], dim=1))
        return torch.stack(targets, dim=1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)[:, : self.config.n_action_steps]
            if self.config.eef_delta_actions:
                # The state is read before it is padded: the first 7 values are the end-effector pose and gripper.
                actions = self.eef_targets(actions, batch[OBS_STATE])
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def get_optim_params(self) -> list[dict[str, Any]]:
        """One parameter group with every trainable parameter (no backbone parameters when `train_vlm=False`)."""
        return [{"params": [param for param in self.parameters() if param.requires_grad]}]
