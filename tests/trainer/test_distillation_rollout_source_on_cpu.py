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

import random
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from tensordict import NonTensorData, TensorDict

from verl.trainer.ppo.v1 import agent_loop_tq, trainer_base
from verl.utils import tensordict_utils as tu


def test_source_endpoints_and_validation():
    from verl.trainer.distillation.rollout import select_rollout_source

    for step in range(10):
        assert select_rollout_source(0.0, 42, step) == "student"
        assert select_rollout_source(1.0, 42, step) == "teacher"
        assert select_rollout_source(1.0, 42, step, validate=True) == "student"


def test_repeat_resume_and_rng_isolation():
    from verl.trainer.distillation.rollout import select_rollout_source

    state = random.getstate()
    first = [select_rollout_source(0.5, 42, step) for step in range(64)]
    assert set(first) == {"student", "teacher"}
    assert first == [select_rollout_source(0.5, 42, step) for step in range(64)]
    assert first[30:] == [select_rollout_source(0.5, 42, step) for step in range(30, 64)]
    assert random.getstate() == state


@pytest.mark.parametrize("num_workers", [1, 2, 4])
@pytest.mark.parametrize("probability,expected", [(0.0, None), (1.0, "teacher"), (0.5, "teacher")])
def test_driver_decides_before_real_manager_fanout(monkeypatch, num_workers, probability, expected):
    chunks = []
    manager = SimpleNamespace(
        agent_loop_workers=[
            SimpleNamespace(generate_sequences=SimpleNamespace(remote=chunks.append)) for _ in range(num_workers)
        ]
    )
    manager.generate_sequences = lambda batch: agent_loop_tq.AgentLoopManagerTQ.generate_sequences(manager, batch)
    monkeypatch.setattr(agent_loop_tq.ray, "get", lambda refs: refs)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: None)
    trainer = SimpleNamespace(
        trainer_mode="sync",
        global_steps=1,
        config=OmegaConf.create(
            {
                "distillation": {"teacher_rollout_probability": probability},
                "actor_rollout_ref": {"rollout": {"seed": 42}},
            }
        ),
        agent_loop_manager=manager,
    )
    batch = TensorDict({"uid": NonTensorData("prompt")}, batch_size=[8])
    for _ in range(2):
        assert trainer_base.PPOTrainer._submit_batch_to_rollout(trainer, batch.clone()) == 8
    assert len(chunks) == 2 * num_workers
    assert sum(len(chunk) for chunk in chunks) == 16
    assert all(tu.get_non_tensor_data(chunk, "rollout_source", None) == expected for chunk in chunks)
