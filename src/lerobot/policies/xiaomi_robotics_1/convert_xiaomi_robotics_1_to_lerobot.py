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

"""Convert a released Xiaomi-Robotics-1 checkpoint to the LeRobot layout.

    python -m lerobot.policies.xiaomi_robotics_1.convert_xiaomi_robotics_1_to_lerobot \
        --source XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa --revision 82b7c2210c1a39fca4a87e681cffbbed134ccde0 \
        --output_dir ./xr1_robocasa_lerobot

Reads the authors' `config.json`, `preprocessor_config.json` (which carries the per-robot action statistics) and
the safetensors shards. Writes a LeRobot policy directory: `config.json`, `model.safetensors` (every tensor under
a `model.` prefix, streamed one at a time, so the 10 GB checkpoint never has to fit in memory), and the saved
pre- and post-processor pipelines. The authors' custom modeling and processing files are never imported or run.
The tokenizer, chat template and image-processor settings stay in the official repository and are read from it
(at the pinned revision) when the processor is built.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_xiaomi_robotics_1 import XR1_PROCESSOR_REPO, XR1_PROCESSOR_REVISION, XiaomiRobotics1Config
from .processor_xiaomi_robotics_1 import make_xiaomi_robotics_1_pre_post_processors

ACTIVE_STD = 1e-5  # the authors' action mask: a dimension is active when its std exceeds this
DEFAULT_ROBOT_TYPE = "robocasa_mg"
# The RoboCasa checkpoints: two agent-view cameras (base views) and the in-hand camera (left-wrist view).
ROBOCASA_BASE_VIEWS = ["robot0_agentview_left", "robot0_agentview_right"]
ROBOCASA_WRIST_VIEWS = ["robot0_eye_in_hand"]
ROBOCASA_STATE_DIM = 8  # 7 joint positions + gripper; the rest of the 60 is zero padding

# The VLABench checkpoint (`Xiaomi-Robotics-1-VLABench`): the authors' client sends three 480x480 cameras under the
# headers Ego / Base / Left-Wrist, the 7 values `[x, y, z (robot frame), rx, ry, rz, gripper]` as state, and executes
# the first 5 steps of every 10-step chunk of end-effector deltas as absolute targets (see `eval_vlabench/main.py`).
VLABENCH_ROBOT_TYPE = "vlabench_choice"
VLABENCH_STATE_DIM = 7
VLABENCH_IMAGE_SIZE = 480
VLABENCH_REPLAN_STEPS = 5
VLABENCH_MAX_PIXELS = 16_777_216  # no downscaling of the 480-pixel cameras
VLABENCH_VIEW_GROUPS = [
    ["Ego View", f"{OBS_IMAGES}.ego"],
    ["Base View", f"{OBS_IMAGES}.base"],
    ["Left-Wrist View", f"{OBS_IMAGES}.wrist"],
]

_ITEMSIZE = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
_TORCH_BYTES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}


def convert_key(key: str) -> str:
    """Authors' parameter name -> LeRobot parameter name (the policy holds the model as `self.model`)."""
    return f"model.{key}"


def action_statistics(preprocessor_config: dict, robot_type: str) -> tuple[list[float], list[float], int]:
    """Per-dimension action mean/std of `robot_type`, trimmed to its active dimensions."""
    try:
        stats = preprocessor_config["action_config"][robot_type]
    except KeyError as error:
        available = list(preprocessor_config.get("action_config", {}))
        raise KeyError(f"robot type {robot_type!r} not found; available: {available}") from error
    mean, std = np.asarray(stats["mean"], dtype=np.float64), np.asarray(stats["std"], dtype=np.float64)
    if mean.ndim != 2 or mean.shape != std.shape:
        raise ValueError(f"expected (chunk, dim) statistics, got mean {mean.shape} and std {std.shape}")
    if not (np.all(mean == mean[0]) and np.all(std == std[0])):
        # LeRobot normalizes with one statistic per dimension; per-step statistics would need a different step.
        raise ValueError("action statistics differ across the chunk; per-step normalization is not supported")
    active = np.flatnonzero(std[0] > ACTIVE_STD)
    action_dim = int(active.max()) + 1 if active.size else 0
    if action_dim == 0 or active.tolist() != list(range(action_dim)):
        raise ValueError(f"active action dimensions {active.tolist()} are not a leading block")
    return mean[0, :action_dim].tolist(), std[0, :action_dim].tolist(), mean.shape[0]


def build_config(
    source_config: dict,
    preprocessor_config: dict,
    robot_type: str = DEFAULT_ROBOT_TYPE,
    state_dim: int | None = None,
    base_views: list[str] | None = None,
    wrist_views: list[str] | None = None,
    image_size: int | None = None,
    processor_name: str = XR1_PROCESSOR_REPO,
    processor_revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[XiaomiRobotics1Config, dict[str, dict[str, torch.Tensor]]]:
    require_package("transformers", extra="xiaomi_robotics_1")
    from transformers import Qwen3VLConfig

    mean, std, chunk = action_statistics(preprocessor_config, robot_type)
    vlabench = robot_type == VLABENCH_ROBOT_TYPE
    if vlabench:
        state_dim = VLABENCH_STATE_DIM if state_dim is None else state_dim
        image_size = VLABENCH_IMAGE_SIZE if image_size is None else image_size
        view_groups = [list(group) for group in VLABENCH_VIEW_GROUPS]
        base_views = [] if base_views is None else base_views
        wrist_views = [] if wrist_views is None else wrist_views
        camera_names = [group[1].removeprefix(f"{OBS_IMAGES}.") for group in VLABENCH_VIEW_GROUPS]
    else:
        state_dim = ROBOCASA_STATE_DIM if state_dim is None else state_dim
        image_size = 256 if image_size is None else image_size
        view_groups = []
        base_views = ROBOCASA_BASE_VIEWS if base_views is None else base_views
        wrist_views = ROBOCASA_WRIST_VIEWS if wrist_views is None else wrist_views
        camera_names = [*base_views, *wrist_views]
    dit = source_config["dit_config"]
    camera = lambda name: f"{OBS_IMAGES}.{name}"  # noqa: E731
    vlm_config = Qwen3VLConfig(**source_config["vlm_config"]).to_dict()
    config = XiaomiRobotics1Config(
        vlm_config=vlm_config,
        processor_name=processor_name,
        processor_revision=processor_revision,
        dit_hidden_size=dit["hidden_size"],
        dit_num_layers=dit["num_hidden_layers"],
        dit_head_dim=dit["head_dim"],
        dit_num_key_value_heads=dit["num_key_value_heads"],
        state_length=source_config.get("state_length", 1),
        max_state_dim=source_config["state_dim"],
        max_action_dim=source_config["action_dim"],
        chunk_size=chunk,
        n_action_steps=VLABENCH_REPLAN_STEPS if vlabench else chunk,
        base_view_keys=[camera(n) for n in base_views],
        wrist_view_keys=[camera(n) for n in wrist_views],
        view_groups=view_groups,
        # The authors' harness passes seed=42 on every call; the VLABench client also sends robot_type and
        # converts the deltas to targets itself, which the policy does when `eef_delta_actions` is set.
        noise_seed=42 if vlabench else None,
        # The authors' RoboCasa harness centre-crops every image to 95 %; their VLABench client only resizes to 480.
        image_crop_ratio=1.0 if vlabench else 0.95,
        # The authors' client loads the slow processor, which does not apply the repository's min/max pixel bounds: a
        # 480x480 image stays 480x480 (900 patches, 225 tokens). The bounds (90000 pixels) belong to the 256-pixel
        # RoboCasa cameras, where they do nothing.
        image_max_pixels=VLABENCH_MAX_PIXELS if vlabench else 90000,
        eef_delta_actions=vlabench,
        eef_action_mean=mean[:7] if vlabench else None,
        eef_action_std=std[:7] if vlabench else None,
        input_features={
            **{
                camera(n): PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size))
                for n in camera_names
            },
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7 if vlabench else len(mean),))
        },
        dtype=dtype,
    )
    if vlabench:
        # The policy de-normalizes and integrates the deltas itself (`eef_delta_actions`): the processors must not
        # normalize the action a second time.
        config.normalization_mapping = {**config.normalization_mapping, "ACTION": NormalizationMode.IDENTITY}
    stats = {ACTION: {"mean": torch.tensor(mean), "std": torch.tensor(std)}}
    return config, stats


# ---- streaming safetensors writer -----------------------------------------------------------------------------


def write_safetensors_streaming(
    path: Path, entries: list[tuple[str, str, list[int], Callable[[], torch.Tensor]]]
) -> None:
    """Write `(name, dtype, shape, loader)` entries one tensor at a time (peak memory = the largest tensor)."""
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, dtype, shape, _ in entries:
        size = int(np.prod(shape, dtype=np.int64)) * _ITEMSIZE[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for name, _dtype, shape, load in entries:
            tensor = load().contiguous()
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"{name}: shape changed while writing")
            raw = (
                tensor.reshape(-1).view(torch.uint8) if tensor.numel() else tensor.reshape(-1).to(torch.uint8)
            )
            out.write(memoryview(raw.numpy()))


def _shards(source_dir: Path) -> Iterator[Path]:
    index = source_dir / "model.safetensors.index.json"
    if index.exists():
        yield from (source_dir / n for n in sorted(set(json.loads(index.read_text())["weight_map"].values())))
    else:
        yield source_dir / "model.safetensors"


def _tensor_loader(shard: Path, key: str) -> Callable[[], torch.Tensor]:
    from safetensors import safe_open

    def load() -> torch.Tensor:
        with safe_open(shard, framework="pt") as handle:
            return handle.get_tensor(key)

    return load


def default_processor_revision(source: str | Path, revision: str | None) -> str | None:
    """Revision of the official processor repository to read the tokenizer, chat template and image settings from.

    A checkpoint from another Hub repository (the VLABench one) has its own revision, which does not exist in the
    processor repository: the pinned processor revision is used. The RoboCasa repository, or a local directory,
    keeps `revision` (the processors it is converted with).
    """
    other_hub_repo = not Path(source).is_dir() and str(source) != XR1_PROCESSOR_REPO
    return XR1_PROCESSOR_REVISION if other_hub_repo else revision


def convert(
    source: str,
    output_dir: str | Path,
    revision: str | None = None,
    robot_type: str = DEFAULT_ROBOT_TYPE,
    processor_revision: str | None = None,
    **config_kwargs,
) -> Path:
    from safetensors import safe_open

    source_dir = Path(source)
    if processor_revision is None:
        processor_revision = default_processor_revision(source, revision)
    if not source_dir.is_dir():
        from huggingface_hub import snapshot_download

        source_dir = Path(
            snapshot_download(
                source,
                revision=revision,
                allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors", "*.index.json"],
            )
        )
    source_config = json.loads((source_dir / "config.json").read_text())
    preprocessor_config = json.loads((source_dir / "preprocessor_config.json").read_text())
    config, stats = build_config(
        source_config,
        preprocessor_config,
        robot_type=robot_type,
        processor_revision=processor_revision,
        **config_kwargs,
    )

    entries: list[tuple[str, str, list[int], Callable[[], torch.Tensor]]] = []
    for shard in _shards(source_dir):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118
                piece = handle.get_slice(key)
                entries.append(
                    (convert_key(key), piece.get_dtype(), piece.get_shape(), _tensor_loader(shard, key))
                )
    entries.sort(key=lambda entry: entry[0])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_safetensors_streaming(output_dir / "model.safetensors", entries)
    config.save_pretrained(output_dir)
    preprocessor, postprocessor = make_xiaomi_robotics_1_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", required=True, help="Hub repo id or local directory of the released checkpoint."
    )
    parser.add_argument("--revision", default=None, help="Revision to pin (recommended).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--robot_type", default=DEFAULT_ROBOT_TYPE)
    parser.add_argument(
        "--processor_revision",
        default=None,
        help="Revision of the official processor repository (tokenizer, chat template, image-processor settings). "
        f"Default: `--revision` for {XR1_PROCESSOR_REPO} or a local directory, else the pinned {XR1_PROCESSOR_REVISION[:8]}.",
    )
    args = parser.parse_args()
    print(
        convert(
            args.source,
            args.output_dir,
            revision=args.revision,
            robot_type=args.robot_type,
            processor_revision=args.processor_revision,
        )
    )


if __name__ == "__main__":
    main()
