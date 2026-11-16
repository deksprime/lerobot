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

"""Pre- and post-processing for Xiaomi-Robotics-1.

The prompt is the one the authors' evaluation harness builds: the camera images grouped under view headers, the
instruction, `/no_cot`, and an empty `<cot></cot>` assistant turn. Tokenization uses the stock `transformers`
Qwen3-VL classes with the tokenizer, chat template and image-processor settings read from the official checkpoint
repository (the XR-1 tokenizer differs from the stock Qwen3-VL one). No remote code is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_xiaomi_robotics_1 import XiaomiRobotics1Config

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor
    from transformers.models.qwen3_vl import Qwen3VLProcessor
else:
    AutoImageProcessor = None
    AutoTokenizer = None
    AutoVideoProcessor = None
    Qwen3VLProcessor = None

PROMPT_HEADER = "The following observations are captured from multiple views.\n# Base View\n"
WRIST_HEADER = "\n# Left-Wrist View\n"
PROMPT_TASK = "\nGenerate robot actions for the task:\n{task} /no_cot"
ASSISTANT_TURN = "<cot></cot>"


def center_crop_resize(image: Any, crop_ratio: float) -> Any:
    """Centre-crop a PIL image to `crop_ratio` of each side and resize back (bilinear), as the authors' harness."""
    if crop_ratio >= 1.0:
        return image
    from PIL import Image

    width, height = image.size
    new_width, new_height = int(width * crop_ratio), int(height * crop_ratio)
    left, top = (width - new_width) // 2, (height - new_height) // 2
    return image.crop((left, top, left + new_width, top + new_height)).resize((width, height), Image.BILINEAR)


def tensor_to_pil(image: torch.Tensor) -> Any:
    """`(C, H, W)` float in [0, 1] -> uint8 PIL image (exact for values that came from uint8 / 255)."""
    from PIL import Image

    array = (
        image.detach().float().clamp(0, 1).mul(255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(np.ascontiguousarray(array))


@ProcessorStepRegistry.register(name="xiaomi_robotics_1_prepare_messages")
@dataclass
class XiaomiRobotics1PrepareMessagesStep(ComplementaryDataProcessorStep):
    """Turn the camera images and the task string into chat messages (one conversation per batch row)."""

    base_view_keys: list[str] = field(default_factory=list)
    wrist_view_keys: list[str] = field(default_factory=list)
    image_crop_ratio: float = 0.95
    # Visual keys of the policy, in feature order. Used when `base_view_keys` is empty (every image is a base view).
    image_keys: list[str] = field(default_factory=list)

    def _groups(self) -> tuple[list[str], list[str]]:
        if self.base_view_keys or self.wrist_view_keys:
            return list(self.base_view_keys), list(self.wrist_view_keys)
        return list(self.image_keys), []

    def _images(self, observation: dict[str, Any], keys: list[str], row: int) -> list[dict[str, Any]]:
        return [
            {
                "type": "image",
                "image": center_crop_resize(tensor_to_pil(observation[key][row]), self.image_crop_ratio),
            }
            for key in keys
        ]

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        tasks = complementary_data.get("task")
        if tasks is None:
            raise ValueError("Task is required for Xiaomi-Robotics-1.")
        if isinstance(tasks, str):
            tasks = [tasks]
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for Xiaomi-Robotics-1.")
        base_keys, wrist_keys = self._groups()
        missing = [key for key in (*base_keys, *wrist_keys) if key not in observation]
        if missing:
            raise KeyError(f"Camera feature(s) {missing} missing from the observation.")
        if not base_keys and not wrist_keys:
            raise ValueError("Xiaomi-Robotics-1 needs at least one camera image.")

        messages = []
        for row, task in enumerate(tasks):
            content: list[dict[str, Any]] = [
                {"type": "text", "text": PROMPT_HEADER},
                *self._images(observation, base_keys, row),
            ]
            if wrist_keys:
                content += [
                    {"type": "text", "text": WRIST_HEADER},
                    *self._images(observation, wrist_keys, row),
                ]
            content.append({"type": "text", "text": PROMPT_TASK.format(task=str(task).strip())})
            messages.append(
                [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TURN}]},
                ]
            )
        complementary_data["messages"] = messages
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "base_view_keys": self.base_view_keys,
            "wrist_view_keys": self.wrist_view_keys,
            "image_crop_ratio": self.image_crop_ratio,
            "image_keys": self.image_keys,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def load_qwen3_vl_processor(name: str, revision: str | None, use_fast: bool) -> Qwen3VLProcessor:
    """Assemble the stock Qwen3-VL processor from the pieces of an XR-1 repository (no remote code)."""
    require_package("transformers", extra="xiaomi_robotics_1")
    kwargs = {"revision": revision, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    image_processor = AutoImageProcessor.from_pretrained(name, use_fast=use_fast, **kwargs)
    video_processor = AutoVideoProcessor.from_pretrained(name, **kwargs)
    return Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )


@ProcessorStepRegistry.register(name="xiaomi_robotics_1_qwen_processor")
@dataclass
class XiaomiRobotics1QwenProcessorStep(ComplementaryDataProcessorStep):
    """Apply the chat template and the image processor; emit the backbone's inputs."""

    processor_name: str = ""
    processor_revision: str | None = None
    use_fast_processor: bool = False
    image_min_pixels: int | None = 1024
    image_max_pixels: int | None = 90000
    tokenizer_max_length: int = 2048

    _processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self._processor is None:
            self._processor = self._load_processor()

    def _load_processor(self):
        return load_qwen3_vl_processor(self.processor_name, self.processor_revision, self.use_fast_processor)

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        messages = complementary_data.pop("messages", None)
        if messages is None:
            raise ValueError("Messages are required for XiaomiRobotics1QwenProcessorStep.")
        # Training batches are right-padded; rollout batches left-padded so every row ends at the same position.
        padding_side = "right" if self.transition.get(TransitionKey.ACTION) is not None else "left"
        size = {}
        if self.image_min_pixels is not None:
            size["shortest_edge"] = self.image_min_pixels
        if self.image_max_pixels is not None:
            size["longest_edge"] = self.image_max_pixels
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "padding_side": padding_side,
                "truncation": True,
                "max_length": self.tokenizer_max_length,
                "return_mm_token_type_ids": True,
                **({"size": size} if size else {}),
            },
        )
        for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if key in inputs:
                complementary_data[key] = inputs[key]
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_name": self.processor_name,
            "processor_revision": self.processor_revision,
            "use_fast_processor": self.use_fast_processor,
            "image_min_pixels": self.image_min_pixels,
            "image_max_pixels": self.image_max_pixels,
            "tokenizer_max_length": self.tokenizer_max_length,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_xiaomi_robotics_1_pre_post_processors(
    config: XiaomiRobotics1Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the pre- and post-processing pipelines.

    Pre: rename, add batch dimension, normalize (state raw, action mean/std), build the multi-view prompt, tokenize
    and preprocess the images, move to device. Post: unnormalize the action chunk and move it to the CPU.
    """
    if config.input_features is None:
        raise ValueError("`input_features` must be resolved before building the processors.")

    steps = make_default_policy_processor_steps(config, dataset_stats)
    image_keys = [key for key, feature in config.input_features.items() if feature.type == FeatureType.VISUAL]

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.normalize,
        XiaomiRobotics1PrepareMessagesStep(
            base_view_keys=list(config.base_view_keys),
            wrist_view_keys=list(config.wrist_view_keys),
            image_crop_ratio=config.image_crop_ratio,
            image_keys=image_keys,
        ),
        XiaomiRobotics1QwenProcessorStep(
            processor_name=config.processor_name,
            processor_revision=config.processor_revision,
            use_fast_processor=config.use_fast_processor,
            image_min_pixels=config.image_min_pixels,
            image_max_pixels=config.image_max_pixels,
            tokenizer_max_length=config.tokenizer_max_length,
        ),
        steps.to_device,
    ]
    output_steps: list[ProcessorStep] = [steps.unnormalize, steps.to_cpu]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
