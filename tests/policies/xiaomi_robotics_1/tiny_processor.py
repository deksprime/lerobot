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

"""A real `Qwen3VLProcessor` built from tiny local pieces (no Hub access)."""

from __future__ import annotations

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from tests.policies.xiaomi_robotics_1.tiny import IMAGE_TOKEN_ID, VISION_END_ID, VISION_START_ID

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% for c in m.content %}{% if c.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif c.type == 'text' %}{{ c.text }}{% endif %}{% endfor %}<|im_end|>\n{% endfor %}"
)


def make_tiny_processor() -> Qwen3VLProcessor:
    specials = {
        "<|pad|>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<unk>": 3,
        "<|image_pad|>": IMAGE_TOKEN_ID,
        "<|vision_start|>": VISION_START_ID,
        "<|vision_end|>": VISION_END_ID,
        "<|video_pad|>": 43,
    }
    words = [
        "The",
        "following",
        "observations",
        "are",
        "captured",
        "from",
        "multiple",
        "views",
        ".",
        "#",
        "Base",
        "View",
        "Left-Wrist",
        "Generate",
        "robot",
        "actions",
        "for",
        "the",
        "task",
        ":",
        "/no_cot",
        "<cot></cot>",
        "user",
        "assistant",
        "pick",
        "up",
        "the",
        "cube",
        "place",
        "it",
    ]
    # keep ids unique and below the tiny vocabulary
    used, next_id = set(specials.values()), 4
    vocab = dict(specials)
    for word in words:
        if word in vocab:
            continue
        while next_id in used:
            next_id += 1
        vocab[word] = next_id
        used.add(next_id)
    tokenizer_object = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_object.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(r"<\|[a-z_]+\|>", behavior="isolated"), pre_tokenizers.WhitespaceSplit()]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        pad_token="<|pad|>",
        unk_token="<unk>",
        additional_special_tokens=[t for t in specials if t.startswith("<|")],
    )
    tokenizer.image_token = "<|image_pad|>"
    tokenizer.video_token = "<|video_pad|>"
    tokenizer.image_token_id = IMAGE_TOKEN_ID
    tokenizer.video_token_id = 43
    tokenizer.vision_start_token = "<|vision_start|>"
    tokenizer.vision_end_token = "<|vision_end|>"
    tokenizer.vision_start_token_id = VISION_START_ID
    tokenizer.vision_end_token_id = VISION_END_ID
    image_processor = Qwen2VLImageProcessor(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        size={"shortest_edge": 1024, "longest_edge": 90000},
    )
    video_processor = Qwen3VLVideoProcessor()
    return Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=CHAT_TEMPLATE,
    )
