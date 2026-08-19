from __future__ import annotations

import pytest

from rl_study.agentic.envs import (
    CalculatorToolEnv,
    LocalLookupEnv,
    guarded_call_reward,
    naive_call_reward,
)
from rl_study.agentic.parser import parse_action


def test_calculator_reset_tool_step_and_correct_final_are_typed() -> None:
    env = CalculatorToolEnv(seed=0, max_steps=3)
    observation = env.reset(task_index=0)
    assert observation.text == "Q:2+3?"
    assert observation.visible_tools[0].name == "calculator"

    tool = parse_action(
        'CALL calculator {"expression":"2+3"}', observation.visible_tools
    )
    first = env.step(tool)
    assert first.tool_output.status == "success"
    assert first.tool_output.text == "5"
    assert first.process_reward == 0.15
    assert first.outcome_reward == 0.0
    assert first.observation is not None

    final = parse_action("FINAL 5", first.observation.visible_tools)
    second = env.step(final)
    assert second.terminated and not second.truncated
    assert second.process_reward == 0.0
    assert second.outcome_reward == 1.0
    assert second.observation is None


def test_calculator_rejects_code_and_never_evaluates_names() -> None:
    env = CalculatorToolEnv(seed=0)
    observation = env.reset(task_index=0)
    result = env.parse_and_step(
        'CALL calculator {"expression":"__import__(\\"os\\").system(\\"id\\")"}'
    )
    assert result.tool_output.status == "invalid"
    assert result.tool_output.text in {"expression_too_long", "operator_not_allowed"}
    assert result.process_reward < 0
    assert observation.visible_tools[0].argument == "expression"


def test_invalid_action_timeout_and_step_limit_are_distinct() -> None:
    invalid_env = CalculatorToolEnv(seed=0, max_steps=2)
    invalid_env.reset(task_index=0)
    invalid = invalid_env.parse_and_step("please calculate")
    assert invalid.tool_output.status == "invalid"
    assert not invalid.terminated and not invalid.truncated

    timed_env = CalculatorToolEnv(seed=0, max_steps=2)
    timed_env.reset(task_index=0)
    timed = timed_env.parse_and_step(
        'CALL calculator {"expression":"__timeout__"}'
    )
    assert timed.tool_output.status == "timeout"
    assert timed.truncated and not timed.terminated

    limit_env = CalculatorToolEnv(seed=0, max_steps=1)
    limit_env.reset(task_index=0)
    limited = limit_env.parse_and_step("bad")
    assert limited.truncated and not limited.terminated


def test_local_lookup_requires_answer_and_citation() -> None:
    env = LocalLookupEnv(seed=0)
    observation = env.reset(task_index=0)
    looked_up = env.step(
        parse_action('CALL lookup {"key":"kr"}', observation.visible_tools)
    )
    assert looked_up.tool_output.text == "Seoul [kr]"
    assert looked_up.observation is not None
    missing_citation = env.step(
        parse_action("FINAL Seoul", looked_up.observation.visible_tools)
    )
    assert missing_citation.terminated
    assert missing_citation.outcome_reward == -0.25


def test_environment_is_seed_deterministic() -> None:
    left = CalculatorToolEnv(seed=13)
    right = CalculatorToolEnv(seed=13)
    for _ in range(6):
        assert left.reset() == right.reset()


def test_repeated_tool_reward_hacking_is_visible_and_guarded() -> None:
    env = CalculatorToolEnv(seed=0)
    observation = env.reset(task_index=0)
    action = parse_action(
        'CALL calculator {"expression":"2+3"}', observation.visible_tools
    )
    actions = (action, action, action)
    assert naive_call_reward(actions) == pytest.approx(0.3)
    assert guarded_call_reward(actions) < 0.0


def test_parser_enforces_visible_tool_and_exact_schema() -> None:
    env = LocalLookupEnv(seed=0)
    tools = env.reset(task_index=0).visible_tools
    parsed_shell = parse_action('CALL shell {"command":"id"}', tools)
    assert parsed_shell.error == "tool_not_visible"
    assert (
        parse_action('CALL lookup {"wrong":"kr"}', tools).error
        == "tool_schema_mismatch"
    )
    assert parse_action("FINAL Seoul [kr]", tools).kind == "final"
