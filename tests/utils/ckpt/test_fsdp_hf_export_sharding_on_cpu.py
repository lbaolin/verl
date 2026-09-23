# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import json

import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from verl.trainer.config import CheckpointConfig
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


def _tiny_qwen3_model():
    config = Qwen3Config(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=16,
        intermediate_size=32,
        vocab_size=64,
    )
    config.architectures = ["Qwen3ForCausalLM"]
    return AutoModelForCausalLM.from_config(config)


def test_configured_max_shard_size_produces_sharded_hf_export(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    checkpoint_config = CheckpointConfig(
        save_contents=["hf_model"],
        load_contents=[],
        hf_model_max_shard_size="1KB",
    )
    manager = FSDPCheckpointManager(
        model=_tiny_qwen3_model(),
        optimizer=None,
        lr_scheduler=None,
        processing_class=None,
        checkpoint_config=checkpoint_config,
    )

    manager.save_checkpoint(local_path=str(tmp_path), global_step=1)

    hf_path = tmp_path / "huggingface"
    index_path = hf_path / "model.safetensors.index.json"
    assert index_path.is_file()

    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = set(index["weight_map"].values())
    assert len(shard_names) > 1
    assert all((hf_path / shard_name).is_file() for shard_name in shard_names)
