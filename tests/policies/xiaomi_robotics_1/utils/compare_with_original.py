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
"""Consumer (run in the LeRobot v0.7.0 env): compare the port with what `dump_original_xr1.py` recorded.

    python compare_with_original.py --records ./xr1_original --stage processor \
        --source XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa --revision <same revision as the dump>
    python compare_with_original.py --records ./xr1_original --stage model --checkpoint ./xr1_lerobot --device cuda
    python compare_with_original.py --records ./xr1_original --stage end-to-end --checkpoint ./xr1_lerobot --device cuda

Stages
  processor   CPU. The port's prompt, tokenization and image preprocessing against the authors' processor output:
              input ids, attention mask and grid must be identical; pixel values are reported as max abs difference.
              Needs only the small tokenizer/config files of the official repo (no weights, no GPU).
  model       GPU. The authors' own processed tensors go straight into the port, noise from `seed=42` as in their
              harness. Isolates the model: transformers 4.57.1 -> 5.22, flash-attention 2 -> sdpa, our expert code.
  end-to-end  GPU. Raw observations through the port's own preprocessor, then the model.

Verdicts follow Phase 35 / Phase 21: **EXACT** is max abs difference <= 1e-6 (bitwise equal counts), **NUM** is cosine
similarity >= 0.999 on every full chunk, otherwise **DIFFERS**. A bfloat16 model on different attention kernels is
expected to land on NUM, not EXACT; EXACT on the first stage and NUM on the others is a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

EXACT_MAX_ABS = 1e-6
NUM_MIN_COSINE = 0.999
ROBOT_TYPE = "robocasa_mg"


def verdict(max_abs: float, min_cosine: float) -> str:
    if max_abs <= EXACT_MAX_ABS:
        return "EXACT"
    return "NUM" if min_cosine >= NUM_MIN_COSINE else "DIFFERS"


def chunk_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = (
        a.double().flatten(),
        b.double().flatten(),
    )  # float64: cosine of ~1e6 float32 values is only good to ~1e-4
    cosine = (
        torch.nn.functional.cosine_similarity(a, b, dim=0).item() if a.norm() > 0 and b.norm() > 0 else 1.0
    )
    return {"max_abs": (a - b).abs().max().item(), "cosine": cosine}


def summarize(name: str, rows: list[dict[str, float]]) -> dict:
    worst_abs = max(r["max_abs"] for r in rows)
    worst_cos = min(r["cosine"] for r in rows)
    result = {
        "stage": name,
        "n": len(rows),
        "max_abs": worst_abs,
        "min_cosine": worst_cos,
        "verdict": verdict(worst_abs, worst_cos),
    }
    print(json.dumps(result))
    return result


def load_records(path: Path) -> tuple[list[dict], dict]:
    records = torch.load(path / "records.pt", weights_only=True)  # tensors, strings and dicts only
    return records, json.loads((path / "meta.json").read_text())


# ---- stage: processor -----------------------------------------------------------------------------------------


def stage_processor(records: list[dict], source: str, revision: str) -> dict:
    """Port prompt/tokenizer/image preprocessing vs the authors' processor outputs (CPU)."""
    from huggingface_hub import hf_hub_download

    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.xiaomi_robotics_1.convert_xiaomi_robotics_1_to_lerobot import build_config
    from lerobot.policies.xiaomi_robotics_1.processor_xiaomi_robotics_1 import (
        XiaomiRobotics1PrepareMessagesStep,
        XiaomiRobotics1QwenProcessorStep,
    )

    def small(name: str) -> dict:
        return json.loads(Path(hf_hub_download(source, name, revision=revision)).read_text())

    config, _ = build_config(
        small("config.json"), small("preprocessor_config.json"), processor_revision=revision
    )
    prepare = XiaomiRobotics1PrepareMessagesStep(
        base_view_keys=config.base_view_keys,
        wrist_view_keys=config.wrist_view_keys,
        image_crop_ratio=config.image_crop_ratio,
    )
    qwen = XiaomiRobotics1QwenProcessorStep(
        processor_name=config.processor_name,
        processor_revision=config.processor_revision,
        use_fast_processor=config.use_fast_processor,
        image_min_pixels=config.image_min_pixels,
        image_max_pixels=config.image_max_pixels,
        tokenizer_max_length=config.tokenizer_max_length,
    )
    cams = [*config.base_view_keys, *config.wrist_view_keys]
    rows, mismatches = [], []
    for index, record in enumerate(records):
        observation = {
            cam: (record["images"][c].permute(2, 0, 1).float() / 255.0)[None] for c, cam in enumerate(cams)
        }
        transition = {
            TransitionKey.OBSERVATION: observation,
            TransitionKey.COMPLEMENTARY_DATA: {"task": [record["task"]]},
        }
        transition = prepare(transition)
        transition = qwen(transition)
        ours = transition[TransitionKey.COMPLEMENTARY_DATA]
        theirs = record["processor"]
        for key in ("input_ids", "attention_mask", "image_grid_thw"):
            if not torch.equal(ours[key].cpu(), theirs[key].cpu().to(ours[key].dtype)):
                mismatches.append((index, key))
        rows.append(chunk_metrics(ours["pixel_values"].cpu(), theirs["pixel_values"].cpu()))
    result = summarize("processor", rows)
    result["identical_ids_mask_grid"] = not mismatches
    result["mismatches"] = mismatches[:10]
    print(json.dumps({k: result[k] for k in ("identical_ids_mask_grid", "mismatches")}))
    return result


# ---- stages: model and end-to-end -----------------------------------------------------------------------------


def _load_policy(checkpoint: str, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Policy

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.noise_seed = 42  # the authors' harness passes seed=42 on every call
    return XiaomiRobotics1Policy.from_pretrained(checkpoint, config=config).eval()


def stage_rotary(records_dir: Path, policy) -> dict | None:
    path = records_dir / "rotary_inv_freq.pt"
    if not path.exists():
        return None
    original = torch.load(path, weights_only=True)
    ours = {
        "expert": policy.model.rotary_emb.inv_freq.detach().cpu(),
        "backbone": policy.model.vlm.model.language_model.rotary_emb.inv_freq.detach().cpu(),
    }
    out = {
        k: {
            "dtype_original": str(original[k].dtype),
            "dtype_port": str(ours[k].dtype),
            **chunk_metrics(original[k].float(), ours[k].float()),
        }
        for k in original
    }
    print(json.dumps({"stage": "rotary", **out}))
    return out


def stage_model(records: list[dict], policy, end_to_end: bool, device: str) -> dict:
    from lerobot.utils.constants import OBS_STATE

    rows_all, rows_decoded = [], []
    cams = [*policy.config.base_view_keys, *policy.config.wrist_view_keys]
    if end_to_end:
        from lerobot.policies.xiaomi_robotics_1.processor_xiaomi_robotics_1 import (
            make_xiaomi_robotics_1_pre_post_processors,
        )

        stats = {"action": {"mean": torch.zeros(policy._action_dim), "std": torch.ones(policy._action_dim)}}
        pre, _ = make_xiaomi_robotics_1_pre_post_processors(policy.config, dataset_stats=stats)
    for record in records:
        if end_to_end:
            raw = {cam: (record["images"][c].permute(2, 0, 1).float() / 255.0) for c, cam in enumerate(cams)}
            raw[OBS_STATE] = record["state"].float()
            raw["task"] = record["task"]
            batch = pre(raw)
        else:
            batch = {
                k: v.to(device) for k, v in record["processor"].items() if k not in ("state", "action_mask")
            }
            batch[OBS_STATE] = record["state"].float()[None].to(device)
        with torch.no_grad():
            state = policy._state(batch).to(next(policy.model.dit.parameters()).dtype)
            mask = torch.zeros(
                1, policy.config.chunk_size, policy.config.max_action_dim, device=device, dtype=state.dtype
            )
            mask[..., : policy._action_dim] = 1
            prefix = policy._prefix_inputs(batch)
            full = policy.model.sample_actions(
                state, mask, **prefix
            )  # (1, chunk, 60) normalized, padding included
        rows_all.append(chunk_metrics(full.cpu(), record["chunk_normalized"]))
        rows_decoded.append(
            chunk_metrics(
                full.cpu()[..., : policy._action_dim], record["chunk_normalized"][..., : policy._action_dim]
            )
        )
    return {
        "all_dims": summarize("end-to-end" if end_to_end else "model (all 60 dims, normalized)", rows_all),
        "action_dims": summarize("action dims only, normalized", rows_decoded),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--records", required=True, help="Output directory of dump_original_xr1.py")
    parser.add_argument("--stage", choices=["processor", "model", "end-to-end"], required=True)
    parser.add_argument("--source", default="XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--checkpoint", default=None, help="LeRobot checkpoint made by the converter.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", default=None, help="Write the JSON report here.")
    args = parser.parse_args()

    records_dir = Path(args.records)
    records, meta = load_records(records_dir)
    revision = args.revision or meta["revision"]
    if revision != meta["revision"]:
        raise SystemExit(f"records were made at {meta['revision']}, not {revision}")
    report: dict = {"meta": meta, "stage": args.stage}
    if args.stage == "processor":
        report["result"] = stage_processor(records, args.source, revision)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for the model stages")
        if not meta["model_stage"]:
            raise SystemExit("these records were made with --skip-model: no original chunks to compare")
        policy = _load_policy(args.checkpoint, args.device)
        report["rotary"] = stage_rotary(records_dir, policy)
        report["result"] = stage_model(records, policy, args.stage == "end-to-end", args.device)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
