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

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.utils import config as config_utils
from verl.workers.config.distillation import DistillationConfig


@pytest.fixture
def supported_config(request):
    config_dir = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        overrides = [f"model_engine={request.param}"] if hasattr(request, "param") else []
        config = compose(config_name="ppo_trainer", overrides=overrides)
    for path, value in {
        "distillation.teacher_rollout_probability": 0.5,
        "distillation.enabled": True,
        "distillation.n_gpus_per_node": 1,
        "distillation.nnodes": 1,
        "distillation.teacher_models.teacher_model.model_path": "Qwen/Qwen3-1.7B",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size": 1,
        "distillation.teacher_models.teacher_model.inference.name": "vllm",
        "distillation.distillation_loss.loss_mode": "forward_kl_topk",
        "distillation.distillation_loss.use_policy_gradient": False,
        "distillation.distillation_loss.use_task_rewards": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.calculate_log_probs": False,
        "algorithm.adv_estimator": "grpo",
    }.items():
        OmegaConf.update(config, path, value, force_add=True)
    return config


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf")])
def test_probability_range_is_validated_even_when_disabled(probability):
    with pytest.raises(ValueError, match="teacher_rollout_probability"):
        DistillationConfig(enabled=False, teacher_rollout_probability=probability)


def test_default_is_student_only():
    assert DistillationConfig().teacher_rollout_probability == 0.0


def test_supported_config(supported_config):
    config_utils._validate_teacher_rollout_config(supported_config)


@pytest.mark.parametrize(
    "path,value",
    [
        ("distillation.enabled", False),
        ("trainer.use_v1", False),
        ("trainer.v1.trainer_mode", "colocate_async"),
        ("trainer.v1.trainer_mode", "separate_async"),
        ("actor_rollout_ref.rollout.name", "sglang"),
        ("distillation.teacher_models.teacher_model.inference.name", "sglang"),
        ("distillation.distillation_loss.use_task_rewards", True),
        ("distillation.distillation_loss.use_policy_gradient", True),
        ("distillation.distillation_loss.loss_mode", "k1"),
        ("distillation.distillation_loss.loss_mode", "k3"),
        ("distillation.distillation_loss.topk", 0),
        ("algorithm.rollout_correction.bypass_mode", True),
        ("algorithm.rollout_correction.rollout_is", "token"),
        ("algorithm.rollout_correction.rollout_rs", "token_k1"),
        ("algorithm.rollout_correction.score_centering", True),
        ("actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering", True),
        ("actor_rollout_ref.rollout.calculate_log_probs", True),
        ("actor_rollout_ref.rollout.topk_log_probs", 2),
        ("actor_rollout_ref.actor.engine.router_replay.mode", "R3"),
        ("actor_rollout_ref.rollout.enable_rollout_routing_replay", True),
        ("distillation.teacher_models.teacher_model.inference.enable_rollout_routing_replay", True),
        ("actor_rollout_ref.rollout.multi_turn.enable", True),
        ("actor_rollout_ref.rollout.agent.default_agent_loop", "tool_agent"),
        ("actor_rollout_ref.rollout.agent.agent_loop_manager_class", "custom.Manager"),
        ("actor_rollout_ref.rollout.agent.agent_loop_config_path", "/tmp/custom.yaml"),
        ("trainer.v1.sampler.custom_sampler.name", "CustomSampler"),
        ("algorithm.filter_groups.enable", True),
        ("trainer.v1.sampler.sync_refill_failed_groups", True),
        ("algorithm.adv_estimator", "gae"),
        ("critic.enable", True),
        ("algorithm.use_kl_in_reward", True),
        ("actor_rollout_ref.actor.use_kl_loss", True),
        ("actor_rollout_ref.actor.entropy_coeff", 0.01),
        ("actor_rollout_ref.rollout.temperature", 0.7),
        ("distillation.teacher_models.teacher_model.inference.temperature", 0.7),
        ("actor_rollout_ref.model.mtp.enable_rollout", True),
        ("distillation.teacher_models.teacher_model.inference.mtp.enable", True),
        ("distillation.teacher_models.teacher_model.inference.mtp.enable_train", True),
        ("distillation.teacher_models.teacher_model.inference.mtp.enable_rollout", True),
    ],
)
def test_opt_in_rejects_unsupported_config(supported_config, path, value):
    OmegaConf.update(supported_config, path, value, force_add=True)
    with pytest.raises(ValueError, match=path.split(".")[-1]):
        config_utils._validate_teacher_rollout_config(supported_config)


def test_zero_preserves_existing_config_freedom(supported_config):
    supported_config.distillation.teacher_rollout_probability = 0.0
    supported_config.trainer.v1.trainer_mode = "separate_async"
    supported_config.distillation.distillation_loss.use_policy_gradient = True
    config_utils._validate_teacher_rollout_config(supported_config)


@pytest.mark.parametrize("supported_config", ["megatron", "veomni"], indirect=True)
@pytest.mark.parametrize("mode", ["disabled", "R2", "R3"])
def test_backend_router_replay_config(supported_config, mode):
    strategy = supported_config.actor_rollout_ref.actor.strategy
    path = f"actor_rollout_ref.actor.{strategy}.router_replay.mode"
    # Raw Hydra configs do not expose the dataclass's synthesized actor.engine alias.
    assert OmegaConf.select(supported_config, "actor_rollout_ref.actor.engine") is None
    OmegaConf.update(supported_config, path, mode)
    if mode == "disabled":
        config_utils._validate_teacher_rollout_config(supported_config)
    else:
        with pytest.raises(ValueError, match="router_replay"):
            config_utils._validate_teacher_rollout_config(supported_config)


def test_counts_resolved_teachers(supported_config):
    teachers = supported_config.distillation.teacher_models
    OmegaConf.update(teachers, "other", OmegaConf.to_container(teachers.teacher_model, resolve=True), force_add=True)
    teachers.other.key = "other"
    config_utils._validate_teacher_rollout_config(supported_config)
    OmegaConf.update(teachers, "third", OmegaConf.to_container(teachers.other, resolve=True), force_add=True)
    teachers.third.key = "third"
    with pytest.raises(ValueError, match="teacher_models"):
        config_utils._validate_teacher_rollout_config(supported_config)
