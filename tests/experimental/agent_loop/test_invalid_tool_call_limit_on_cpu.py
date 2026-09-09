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


def _result(invalid: bool) -> tuple[ToolResponse, float, dict]:
    return ToolResponse(text="tool result"), 0.0, {"invalid_tool_call": invalid}


def _tracking_state() -> SimpleNamespace:
    return SimpleNamespace(
        consecutive_invalid_tool_calls=0,
        max_consecutive_invalid_tool_calls_observed=0,
        invalid_tool_call_limit_reached=False,
        extra_fields={},
    )


def _track(limit: int | None, state: SimpleNamespace, results: list[tuple]) -> None:
    loop = SimpleNamespace(max_consecutive_invalid_tool_calls=limit)
    ToolAgentLoop._update_invalid_tool_call_tracking(loop, state, results)


def _write_diagnostics(limit: int | None, state: SimpleNamespace) -> None:
    loop = SimpleNamespace(max_consecutive_invalid_tool_calls=limit)
    ToolAgentLoop._write_invalid_tool_call_diagnostics(loop, state)


def test_disabled_limit_does_not_change_output_metadata() -> None:
    state = _tracking_state()

    _track(None, state, [_result(True), _result(True)])
    _write_diagnostics(None, state)

    assert state.extra_fields == {}
    assert state.invalid_tool_call_limit_reached is False


@pytest.mark.parametrize(
    ("results", "expected_streak", "expected_max", "expected_reached"),
    [
        pytest.param([True, True], 2, 2, True, id="reaches-limit"),
        pytest.param([True, False, True], 1, 1, False, id="valid-resets-streak"),
        pytest.param([True, True, False], 0, 2, False, id="final-valid-cancels-limit"),
    ],
)
def test_tracking_uses_model_order_and_reports_limit(
    results: list[bool], expected_streak: int, expected_max: int, expected_reached: bool
) -> None:
    state = _tracking_state()

    _track(2, state, [_result(invalid) for invalid in results])
    _write_diagnostics(2, state)

    assert state.consecutive_invalid_tool_calls == expected_streak
    assert state.max_consecutive_invalid_tool_calls_observed == expected_max
    assert state.invalid_tool_call_limit_reached is expected_reached
    assert state.extra_fields == {
        "max_consecutive_invalid_tool_calls_observed": expected_max,
        "invalid_tool_call_limit_reached": expected_reached,
        **({"termination_reason": "invalid_tool_call_limit"} if expected_reached else {}),
    }
