from __future__ import annotations

import pytest

from rl_study.data import build_tiny_reasoning, verify_response
from rl_study.envs import BernoulliBandit, GridAction, TinyGridWorld


def test_tiny_reasoning_sizes_hash_and_split_isolation() -> None:
    first = build_tiny_reasoning(seed=42)
    second = build_tiny_reasoning(seed=42)
    assert (len(first.train), len(first.validation), len(first.test)) == (256, 64, 128)
    assert len(first.preferences) == 512
    assert first.split_hash == second.split_hash
    train_ids = {example.uid for example in first.train}
    validation_ids = {example.uid for example in first.validation}
    test_ids = {example.uid for example in first.test}
    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


def test_verifier_requires_correct_number_and_format() -> None:
    dataset = build_tiny_reasoning(seed=42)
    example = dataset.train[0]
    assert verify_response(example, example.target_response)
    assert not verify_response(example, f"The answer is {example.answer}.")
    assert not verify_response(example, f"<answer>{example.answer + 1}</answer>")
    numeric_pair, format_pair = dataset.preferences[:2]
    assert verify_response(example, numeric_pair.chosen)
    assert not verify_response(example, numeric_pair.rejected)
    assert not verify_response(example, format_pair.rejected)


def test_bandit_seed_and_regret_are_deterministic() -> None:
    first = BernoulliBandit(horizon=4)
    second = BernoulliBandit(horizon=4)
    first.reset(seed=3)
    second.reset(seed=3)
    first_results = [first.step(0) for _ in range(4)]
    second_results = [second.step(0) for _ in range(4)]
    assert [result.reward for result in first_results] == [
        result.reward for result in second_results
    ]
    assert first_results[-1].truncated
    assert first_results[0].info["expected_regret"] == pytest.approx(0.7)


def test_gridworld_goal_terminates_without_truncating() -> None:
    environment = TinyGridWorld(size=4, max_steps=32)
    state, _ = environment.reset(seed=42)
    assert state == 0
    result = None
    for action in [GridAction.RIGHT] * 3 + [GridAction.DOWN] * 3:
        result = environment.step(action)
    assert result is not None
    assert result.observation == environment.goal_state
    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False


def test_gridworld_time_limit_truncates_and_terminal_does_not_bootstrap() -> None:
    environment = TinyGridWorld(size=4, max_steps=2)
    environment.reset(seed=42)
    first = environment.step(GridAction.LEFT)
    second = environment.step(GridAction.LEFT)
    assert not first.terminated and not first.truncated
    assert not second.terminated and second.truncated
    next_state, reward, terminated = environment.transition(
        environment.goal_state, GridAction.LEFT
    )
    assert (next_state, reward, terminated) == (environment.goal_state, 0.0, True)
