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
"""`serve_port.py` answers the authors' wire format with the port (tiny model, loopback socket, CPU)."""

from __future__ import annotations

import pickle  # nosec B403 - the authors' wire format, exercised on loopback with our own payloads
import socket
import struct
import threading

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.xiaomi_robotics_1.modeling_xiaomi_robotics_1 import XiaomiRobotics1Policy
from lerobot.utils.constants import OBS_STATE
from tests.policies.xiaomi_robotics_1.tiny import CHUNK, MAX_DIM, make_batch, make_config
from tests.policies.xiaomi_robotics_1.utils.serve_port import handle_request, serve


def _request(seed: int = 42) -> dict:
    batch = make_batch(batch_size=1, seed=1)
    state = torch.zeros(1, 1, MAX_DIM)
    state[0, 0, :5] = batch[OBS_STATE][0]
    return {
        **{
            k: v for k, v in batch.items() if k not in (OBS_STATE, "mm_token_type_ids")
        },  # as the authors' processor
        "state": state,
        "action_mask": torch.ones(1, CHUNK, MAX_DIM),
        "task_id": "robocasa_mg",
        "seed": seed,
    }


def test_request_in_the_authors_format_gives_the_full_normalized_chunk():
    torch.manual_seed(0)
    policy = XiaomiRobotics1Policy(make_config()).eval()
    chunk = handle_request(policy, _request())
    assert chunk.shape == (1, CHUNK, MAX_DIM)  # all padded dimensions, as the authors' server returns
    assert torch.equal(chunk, handle_request(policy, _request()))  # seed=42 -> the same noise on every call
    expected = policy.predict_action_chunk(
        make_batch(batch_size=1, seed=1),
        noise=policy.model.sample_noise((1, CHUNK, MAX_DIM), "cpu", torch.float32),  # noise_seed is 42 now
    )
    torch.testing.assert_close(chunk[..., :4], expected, atol=1e-5, rtol=1e-5)
    assert not torch.equal(chunk, handle_request(policy, _request(seed=7)))


def test_socket_round_trip_on_loopback():
    torch.manual_seed(0)
    policy = XiaomiRobotics1Policy(make_config()).eval()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=serve, args=(policy, "127.0.0.1", port), daemon=True).start()
    for _ in range(100):
        try:
            client = socket.create_connection(("127.0.0.1", port), timeout=5)
            break
        except OSError:
            threading.Event().wait(0.05)
    with client:
        payload = pickle.dumps(_request())
        client.sendall(struct.pack(">I", len(payload)) + payload)
        (length,) = struct.unpack(">I", client.recv(4))
        data = b""
        while len(data) < length:
            data += client.recv(length - len(data))
    chunk = pickle.loads(data)  # nosec B301
    assert chunk.shape == (1, CHUNK, MAX_DIM)
    assert torch.equal(chunk, handle_request(policy, _request()))
