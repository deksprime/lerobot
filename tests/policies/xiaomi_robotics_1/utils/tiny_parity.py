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
"""Tiny-model parity across the transformers 4.57.1 -> 5.22 boundary, on CPU, with no checkpoint download.

Both sides get the same random weights and the same REAL processed observations (`records.pt` from
`dump_original_xr1.py --skip-model`: 248 tokens, three 256x256 images, 16x16 patch grids). What differs is the code:
the authors' remote-code model under transformers 4.57.1 against this port under 5.22. It isolates our expert and
prefix logic and the transformers changes from the 5B weights, the GPU and flash-attention.

    # 1. in the ORIGINAL env (transformers 4.57.1):
    python tiny_parity.py original --snapshot <dir with the repo's small files + remote code> \
        --records ./xr1_original --out ./tiny_parity
    # 2. in the LeRobot env:
    python tiny_parity.py port --records ./xr1_original --out ./tiny_parity

bfloat16 on CPU: the authors' timestep embedder hard-codes a bfloat16 cast, so their model cannot run in float32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

TINY = {
    "text": {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
    },
    "vision": {
        "depth": 2,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "out_hidden_size": 64,
        "num_position_embeddings": 256,
        "deepstack_visual_indexes": [0, 1],
    },
    "dit": {"hidden_size": 32, "head_dim": 16, "num_hidden_layers": 2, "num_key_value_heads": 2},
    "mrope_section": [2, 3, 3],
}
SEED = 42


def run_original(args: argparse.Namespace) -> None:
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(args.snapshot, trust_remote_code=True)
    for key, value in TINY["text"].items():
        setattr(config.vlm_config.text_config, key, value)
    config.vlm_config.text_config.rope_scaling = {
        **config.vlm_config.text_config.rope_scaling,
        "mrope_section": TINY["mrope_section"],
    }
    for key, value in TINY["vision"].items():
        setattr(config.vlm_config.vision_config, key, value)
    for key, value in TINY["dit"].items():
        setattr(config.dit_config, key, value)
    torch.manual_seed(0)
    model = AutoModel.from_config(
        config, trust_remote_code=True, attn_implementation="eager", dtype=torch.bfloat16
    )
    model = model.to(torch.bfloat16).eval()

    records = torch.load(Path(args.records) / "records.pt", weights_only=True)
    chunks = []
    for record in records[: args.n]:
        data = {**record["processor"], "task_id": "robocasa_mg", "seed": SEED}
        data = {
            k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
            for k, v in data.items()
        }
        with torch.no_grad():
            chunks.append(model(**data).actions.float())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # lm_head is tied to the embedding (and absent from the released checkpoints).
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items() if k != "vlm.lm_head.weight"},
        out / "tiny_original.safetensors",
    )
    torch.save(chunks, out / "chunks_original.pt")
    (out / "tiny_config.json").write_text(json.dumps(TINY))
    print(
        f"original: {len(chunks)} chunks, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters"
    )


def run_port(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.xiaomi_robotics_1.configuration_xiaomi_robotics_1 import XiaomiRobotics1Config
    from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Policy
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
    from tests.policies.xiaomi_robotics_1.utils.compare_with_original import chunk_metrics, summarize

    out = Path(args.out)
    real = json.loads((Path(args.snapshot) / "config.json").read_text())["vlm_config"]
    real["text_config"].update(TINY["text"])
    real["text_config"]["rope_scaling"] = {
        **real["text_config"]["rope_scaling"],
        "mrope_section": TINY["mrope_section"],
    }
    real["vision_config"].update(TINY["vision"])
    config = XiaomiRobotics1Config(
        vlm_config=real,
        dit_hidden_size=TINY["dit"]["hidden_size"],
        dit_num_layers=TINY["dit"]["num_hidden_layers"],
        dit_head_dim=TINY["dit"]["head_dim"],
        dit_num_key_value_heads=TINY["dit"]["num_key_value_heads"],
        chunk_size=10,
        n_action_steps=10,
        noise_seed=SEED,
        dtype=torch.bfloat16,
        device="cpu",
        input_features={
            f"{OBS_IMAGES}.a": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = XiaomiRobotics1Policy(config).eval()
    state = {f"model.{k}": v for k, v in load_file(out / "tiny_original.safetensors").items()}
    missing, unexpected = policy.load_state_dict(state, strict=False)
    missing = [k for k in missing if "lm_head" not in k]  # tied to the embedding
    print(json.dumps({"missing": missing, "unexpected": unexpected}))
    assert not missing and not unexpected, "parameter names do not line up"

    records = torch.load(Path(args.records) / "records.pt", weights_only=True)
    originals = torch.load(out / "chunks_original.pt", weights_only=True)
    rows = []
    for record, original in zip(records, originals, strict=False):
        batch = {k: v for k, v in record["processor"].items() if k not in ("state", "action_mask")}
        state_vec = policy._state({OBS_STATE: record["state"].float()[None]}).to(torch.bfloat16)
        mask = torch.zeros(1, 10, 60, dtype=torch.bfloat16)
        mask[..., :7] = 1
        with torch.no_grad():
            chunk = policy.model.sample_actions(state_vec, mask, **policy._prefix_inputs(batch))
        rows.append(chunk_metrics(chunk.float(), original))
    report = summarize("tiny-model parity (port 5.22 vs original 4.57.1, bf16 CPU)", rows)
    (out / "report.json").write_text(json.dumps({"rows": rows, **report}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="side", required=True)
    original = sub.add_parser("original")
    original.add_argument("--snapshot", required=True)
    original.add_argument("--records", required=True)
    original.add_argument("--out", required=True)
    original.add_argument("--n", type=int, default=4)
    port = sub.add_parser("port")
    port.add_argument(
        "--snapshot", required=True, help="Same directory as for `original` (read for config.json)."
    )
    port.add_argument("--records", required=True)
    port.add_argument("--out", required=True)
    args = parser.parse_args()
    run_original(args) if args.side == "original" else run_port(args)


if __name__ == "__main__":
    main()
