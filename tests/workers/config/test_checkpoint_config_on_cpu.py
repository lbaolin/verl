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

import os

import pytest
from hydra import compose, initialize_config_dir

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import FSDPCheckpointConfig


@pytest.mark.parametrize(
    ("config_subdir", "config_name"),
    [
        ("actor", "dp_actor"),
        ("actor", "veomni_actor"),
        ("critic", "dp_critic"),
        ("critic", "veomni_critic"),
        (None, "sft_trainer_engine"),
    ],
)
def test_fsdp_checkpoint_manager_consumers_use_fsdp_checkpoint_config(config_subdir, config_name):
    config_dir = os.path.abspath("verl/trainer/config")
    if config_subdir is not None:
        config_dir = os.path.join(config_dir, config_subdir)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)

    checkpoint_config = omega_conf_to_dataclass(cfg.checkpoint)
    assert isinstance(checkpoint_config, FSDPCheckpointConfig)
    assert checkpoint_config.hf_model_max_shard_size is None
