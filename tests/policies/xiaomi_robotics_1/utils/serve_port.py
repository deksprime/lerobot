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
"""Serve the LeRobot port behind the authors' evaluation harness wire format (for the reproduction eval).

The authors' RoboCasa evaluation (`eval_robocasa/main.py`) talks to `deploy/server.py` over a socket: a 4-byte
big-endian length followed by a pickled dict (the authors' processor output plus `task_id` and `seed`), answered with
a pickled `(1, chunk, 60)` tensor of normalized actions that the client decodes. This script answers the same
requests with the port, so the authors' client, simulator pins and task protocol stay byte-for-byte unchanged and
only the model behind the socket changes.

    python serve_port.py --checkpoint ./xr1_robocasa_lerobot --device cuda --port 10086

SECURITY: the wire format is `pickle`, which executes code on load. Bind to loopback only (the default) and run it
inside a throwaway container with no secrets, as the reference reproduction did. Never expose the port.
THIS NEEDS A GPU. Do not run it without an owner-ok-benchmark line.
"""

from __future__ import annotations

import argparse
import pickle  # nosec B403 - the authors' wire format; loopback only, see the module docstring
import socket
import struct
import time

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Policy
from lerobot.utils.constants import OBS_STATE

MAX_REQUEST_BYTES = 512 * 1024 * 1024  # a 3-image request is ~7 MB; refuse absurd length prefixes


def handle_request(policy: XiaomiRobotics1Policy, request: dict) -> torch.Tensor:
    """One authors'-format request -> the normalized chunk `(1, chunk, max_action_dim)` as they return it."""
    device = next(policy.parameters()).device
    dtype = next(policy.model.dit.parameters()).dtype
    if "seed" in request:
        policy.config.noise_seed = int(request["seed"])  # their harness passes seed=42 on every call
    state = request["state"].to(device=device, dtype=dtype)
    batch = {
        key: request[key].to(device)
        for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        if key in request
    }
    batch[OBS_STATE] = state.reshape(state.shape[0], -1)[:, : policy.config.max_state_dim]
    with torch.no_grad():
        prefix = policy._prefix_inputs(batch)
        mask = torch.zeros(
            state.shape[0], policy.config.chunk_size, policy.config.max_action_dim, device=device, dtype=dtype
        )
        mask[..., : policy._action_dim] = 1
        return policy.model.sample_actions(policy._state(batch).to(dtype), mask, **prefix).cpu()


def _recv_exact(connection: socket.socket, length: int) -> bytes | None:
    data = b""
    while len(data) < length:
        packet = connection.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data


def serve(policy: XiaomiRobotics1Policy, host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"serving on {host}:{port}", flush=True)
        while True:
            connection, _ = server.accept()
            with connection:
                while True:
                    header = _recv_exact(connection, 4)
                    if header is None:
                        break
                    (length,) = struct.unpack(">I", header)
                    if length > MAX_REQUEST_BYTES:
                        break
                    payload = _recv_exact(connection, length)
                    if payload is None:
                        break
                    started = time.time()
                    response = pickle.dumps(handle_request(policy, pickle.loads(payload)))  # nosec B301
                    connection.sendall(struct.pack(">I", len(response)) + response)
                    print(f"{(time.time() - started) * 1000:.0f} ms", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="LeRobot checkpoint made by the converter.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    args = parser.parse_args()
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = args.device
    policy = XiaomiRobotics1Policy.from_pretrained(args.checkpoint, config=config).eval()
    serve(policy, args.host, args.port)


if __name__ == "__main__":
    main()
