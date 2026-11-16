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
"""Producer (run in the ORIGINAL Xiaomi-Robotics-1 environment): record inputs and the original model's chunks.

The authors' reference stack pins `transformers==4.57.1` and loads the checkpoint with `trust_remote_code=True`.
That cannot share a process with LeRobot v0.7.0 (transformers 5.22), so, as for GR00T, the original runs in its own
environment and writes what the LeRobot side needs to replay:

  * the observations (uint8 camera frames, joint state, task text),
  * the exact tensors the authors' processor produced for them (input ids, pixels, grid, state, action mask),
  * the chunk `outputs.actions` the authors' model returned with `seed=42`, as their evaluation harness calls it,
  * the rotary `inv_freq` buffers of both the backbone and the action expert (they are rounded to bfloat16 when the
    authors cast the model, which affects every position embedding).

Stages:
  `--skip-model`  processor only. Needs no GPU and no checkpoint weights, only the small config/tokenizer files.
  default         also loads the 5B model (bfloat16, flash-attention 2, one GPU of >= 16 GB) and answers every
                  observation. THIS STAGE COSTS GPU TIME. Do not run it without an owner-ok-benchmark line.

Usage (original env: python 3.12, torch 2.8, transformers 4.57.1, flash-attn 2.8.3; the authors' repo is on
`PYTHONPATH` only for `deploy/`-style imports, which this script does not need):

    python dump_original_xr1.py --repo XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa \
        --revision 82b7c2210c1a39fca4a87e681cffbbed134ccde0 --out-dir ./xr1_original --n 24 [--skip-model]
    # recorded frames instead of the synthetic set: --frames frames.npz  (images [N,3,H,W,3] uint8, state [N,8],
    # task [N] str; for example saved from the authors' eval_robocasa loop)

Original code revision this was written against: Xiaomi-Robotics-1 0dd7aef (reproduced 52.1 % vs 56.5 % published).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROBOT_TYPE = "robocasa_mg"
STATE_DIM = 60
CROP_RATIO = 0.95
SEED = 42  # the authors' harness passes seed=42 on every call
TASKS = [
    "pick the apple from the counter and place it in the sink",
    "pick the mug from the cabinet and place it on the counter",
    "pick the can from the counter and place it in the microwave",
    "pick the bottle from the sink and place it on the counter",
    "open the cabinet door",
    "close the drawer",
    "turn on the stove",
    "press the coffee machine button",
]


def synthetic_observations(n: int, seed: int = 0) -> dict[str, np.ndarray | list[str]]:
    """Smooth, structured frames (not white noise) so activations are closer to real scenes."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 3, 256, 256, 3), dtype=np.uint8)
    for i in range(n):
        for cam in range(3):
            low = rng.uniform(0, 255, (8, 8, 3)).astype(np.float32)
            frame = np.asarray(
                Image.fromarray(low.astype(np.uint8)).resize((256, 256), Image.BICUBIC), dtype=np.float32
            )
            yy, xx = np.mgrid[0:256, 0:256].astype(np.float32) / 255.0
            frame += 30.0 * np.sin(6.28 * (xx * (1 + cam) + yy * (1 + i % 3)))[..., None]
            images[i, cam] = np.clip(frame, 0, 255).astype(np.uint8)
    state = rng.uniform(-1.0, 1.0, (n, 8)).astype(np.float32)
    return {"images": images, "state": state, "task": [TASKS[i % len(TASKS)] for i in range(n)]}


def center_crop_pil(images: list[Image.Image], crop_ratio: float) -> list[Image.Image]:
    """The authors' `eval_robocasa/main.py::center_crop_pil`, verbatim in behaviour."""
    if crop_ratio >= 1.0:
        return images
    cropped = []
    for img in images:
        w, h = img.size
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        left, top = (w - new_w) // 2, (h - new_h) // 2
        cropped.append(img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BILINEAR))
    return cropped


def build_messages(images: list[Image.Image], instruction: str) -> list[dict]:
    """The authors' `EvalClient.__call__` prompt."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "The following observations are captured from multiple views.\n# Base View\n",
                },
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": "\n# Left-Wrist View\n"},
                {"type": "image", "image": images[2]},
                {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default="XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa")
    parser.add_argument(
        "--revision", required=True, help="Pin the official repo snapshot (full commit hash)."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--frames", default=None, help="npz of recorded frames; default is a synthetic set.")
    parser.add_argument(
        "--skip-model", action="store_true", help="processor stage only (no GPU, no weights)."
    )
    args = parser.parse_args()

    import transformers
    from transformers import AutoProcessor

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.frames:
        loaded = np.load(args.frames, allow_pickle=False)
        obs = {"images": loaded["images"], "state": loaded["state"], "task": [str(t) for t in loaded["task"]]}
    else:
        obs = synthetic_observations(args.n)

    # The authors' own classes from the checkpoint repo (remote code), exactly as `deploy/client.py` loads them.
    processor = AutoProcessor.from_pretrained(
        args.repo, revision=args.revision, trust_remote_code=True, use_fast=False
    )
    model = None
    if not args.skip_model:
        from transformers import AutoModel

        model = (
            AutoModel.from_pretrained(
                args.repo,
                revision=args.revision,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
            )
            .cuda()
            .to(torch.bfloat16)
            .eval()
        )

    records = []
    for i in range(len(obs["task"])):
        images = center_crop_pil([Image.fromarray(obs["images"][i, c]) for c in range(3)], CROP_RATIO)
        state_padded = np.zeros(STATE_DIM, dtype=np.float32)
        state_padded[: obs["state"].shape[1]] = obs["state"][i]
        data = processor.apply_chat_template(
            build_messages(images, obs["task"][i]),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=state_padded.reshape(1, 1, STATE_DIM),
            robot_type=ROBOT_TYPE,
        )
        record = {
            "task": obs["task"][i],
            "images": torch.from_numpy(obs["images"][i]),
            "state": torch.from_numpy(obs["state"][i]),
            "processor": {k: v.cpu() for k, v in dict(data).items() if isinstance(v, torch.Tensor)},
        }
        if model is not None:
            request = {**dict(data), "task_id": ROBOT_TYPE, "seed": SEED}
            on_device = {
                k: (
                    v.to(device=model.device, dtype=model.dtype)
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else v.to(device=model.device)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in request.items()
            }
            with torch.no_grad():
                actions = model(**on_device).actions
            record["chunk_normalized"] = actions.float().cpu()  # (1, 10, 60), normalized space
            record["chunk_decoded"] = processor.decode_action(actions.cpu(), robot_type=ROBOT_TYPE)
        records.append(record)
        print(f"recorded {i + 1}/{len(obs['task'])}", flush=True)

    meta = {
        "repo": args.repo,
        "revision": args.revision,
        "seed": SEED,
        "crop_ratio": CROP_RATIO,
        "n": len(records),
        "frames": args.frames or "synthetic (seed 0)",
        "model_stage": model is not None,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and model is not None else None,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if model is not None:
        meta["rotary_inv_freq_dtype"] = {
            "expert": str(model.rotary_emb.inv_freq.dtype),
            "backbone": str(model.vlm.model.language_model.rotary_emb.inv_freq.dtype),
        }
        torch.save(
            {
                "expert": model.rotary_emb.inv_freq.detach().cpu(),
                "backbone": model.vlm.model.language_model.rotary_emb.inv_freq.detach().cpu(),
            },
            out_dir / "rotary_inv_freq.pt",
        )
    torch.save(records, out_dir / "records.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
