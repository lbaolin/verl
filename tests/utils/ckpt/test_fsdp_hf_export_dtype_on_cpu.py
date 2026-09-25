# Copyright 2025 Individual Contributor: Baolin Li
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

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, Qwen3Config

from verl.trainer.config import CheckpointConfig
from verl.utils.checkpoint import fsdp_checkpoint_manager
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


def test_checkpoint_manager_rejects_unsupported_hf_export_dtype(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    checkpoint_config = CheckpointConfig(hf_export_dtype="float16")

    with pytest.raises(ValueError, match="hf_export_dtype must be null or bfloat16"):
        FSDPCheckpointManager(model=object(), checkpoint_config=checkpoint_config)


@pytest.mark.parametrize(
    ("export_dtype", "expected_dtype"),
    [(None, torch.float32), ("bfloat16", torch.bfloat16)],
)
def test_checkpoint_manager_exports_configured_hf_dtype(monkeypatch, tmp_path, export_dtype, expected_dtype):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    model = AutoModelForCausalLM.from_config(
        Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=16,
            intermediate_size=32,
            vocab_size=32,
        ),
        dtype=torch.float32,
    )
    state_dict = model.state_dict()
    state_dict["integer_buffer"] = torch.tensor([1, 2], dtype=torch.int64)
    monkeypatch.setattr(fsdp_checkpoint_manager, "get_fsdp_full_state_dict", lambda *args, **kwargs: state_dict)
    checkpoint_config = CheckpointConfig(
        save_contents=["hf_model"],
        load_contents=["model"],
        hf_export_dtype=export_dtype,
    )
    manager = FSDPCheckpointManager(model=model, checkpoint_config=checkpoint_config)

    manager.save_checkpoint(local_path=str(tmp_path), global_step=0)

    exported_state_dict = load_file(tmp_path / "huggingface" / "model.safetensors")
    floating_dtypes = {tensor.dtype for tensor in exported_state_dict.values() if torch.is_floating_point(tensor)}
    assert floating_dtypes == {expected_dtype}
    assert exported_state_dict["integer_buffer"].dtype == torch.int64
