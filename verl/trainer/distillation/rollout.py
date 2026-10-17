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
from typing import Literal


def select_rollout_source(
    probability: float, seed: int, global_step: int, *, validate: bool = False
) -> Literal["student", "teacher"]:
    """Choose one source per global step without consuming checkpointed or global RNG state."""
    if validate or probability == 0.0:
        return "student"
    if probability == 1.0:
        return "teacher"
    draw = random.Random(f"teacher-rollout:{seed}:{global_step}").random()
    return "teacher" if draw < probability else "student"
