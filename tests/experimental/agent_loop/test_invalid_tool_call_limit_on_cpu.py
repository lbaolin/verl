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

from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.schemas import ToolResponse


def _run_tracking_scenario(limit: int | None, invalid_markers: list[bool | None]) -> SimpleNamespace:
    state = SimpleNamespace(
        consecutive_invalid_tool_calls=0,
        invalid_tool_call_limit_reached=False,
        extra_fields={},
    )
    loop = SimpleNamespace(max_consecutive_invalid_tool_calls=limit)
    results = [
        (ToolResponse(text="tool result"), 0.0, {} if invalid is None else {"invalid_tool_call": invalid})
        for invalid in invalid_markers
    ]
    ToolAgentLoop._update_invalid_tool_call_tracking(loop, state, results)
    ToolAgentLoop._write_invalid_tool_call_diagnostics(loop, state)
    return state


def test_disabled_limit_does_not_change_output_metadata() -> None:
    state = _run_tracking_scenario(None, [True, True])

    assert state.extra_fields == {}
    assert state.invalid_tool_call_limit_reached is False


@pytest.mark.parametrize(
    ("results", "expected_streak", "expected_reached"),
    [
        pytest.param([True, True], 2, True, id="reaches-limit"),
        pytest.param([True, None, True], 1, False, id="missing-marker-resets-streak"),
        pytest.param([True, True, False], 0, False, id="final-valid-cancels-limit"),
    ],
)
def test_tracking_uses_model_order_and_reports_termination_reason(
    results: list[bool | None], expected_streak: int, expected_reached: bool
) -> None:
    state = _run_tracking_scenario(2, results)

    assert state.consecutive_invalid_tool_calls == expected_streak
    assert state.invalid_tool_call_limit_reached is expected_reached
    assert state.extra_fields == ({"termination_reason": "invalid_tool_call_limit"} if expected_reached else {})
