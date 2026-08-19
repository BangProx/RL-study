from __future__ import annotations

import torch

from rl_study.algorithms.tabular import (
    gridworld_model,
    mc_value_update,
    monte_carlo_returns,
    policy_iteration,
    q_learning_target,
    td0_update,
    td_target,
    train_mc_prediction,
    train_q_learning,
    train_td0_prediction,
    value_iteration,
)
from rl_study.envs import TinyGridWorld


def test_value_and_policy_iteration_agree() -> None:
    environment = TinyGridWorld()
    transitions, rewards, terminal = gridworld_model(environment)
    value_result = value_iteration(transitions, rewards, terminal, gamma=0.99)
    policy_result = policy_iteration(transitions, rewards, terminal, gamma=0.99)
    assert value_result.converged and policy_result.converged
    torch.testing.assert_close(value_result.values, policy_result.values)
    assert value_result.values[environment.goal_state].item() == 0.0
    assert value_result.values[environment.start_state].item() > 0.9


def test_mc_first_visit_update_and_returns() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0])
    returns = monte_carlo_returns(rewards, gamma=0.5)
    torch.testing.assert_close(returns, torch.tensor([2.75, 3.5, 3.0]))
    values, counts = mc_value_update(
        torch.zeros(3),
        torch.tensor([0, 1, 0]),
        returns,
        torch.zeros(3),
    )
    torch.testing.assert_close(values, torch.tensor([2.75, 3.5, 0.0]))
    torch.testing.assert_close(counts, torch.tensor([1.0, 1.0, 0.0]))


def test_td_and_q_targets_disable_only_terminal_bootstrap() -> None:
    reward = torch.tensor([1.0, 1.0])
    terminated = torch.tensor([True, False])
    td = td_target(reward, torch.tensor([10.0, 10.0]), terminated, gamma=0.9)
    torch.testing.assert_close(td, torch.tensor([1.0, 10.0]))
    q = q_learning_target(
        reward,
        torch.tensor([[100.0, 1.0], [2.0, 3.0]]),
        terminated,
        gamma=0.9,
    )
    torch.testing.assert_close(q, torch.tensor([1.0, 3.7]))
    assert td0_update(torch.tensor(2.0), torch.tensor(4.0), learning_rate=0.5) == 3.0


def test_q_learning_short_train_learns_grid() -> None:
    result = train_q_learning(episodes=180, seed=42)
    assert len(result.episode_returns) == 180
    assert torch.isfinite(result.q_values).all()
    assert result.success_rate >= 0.9


def test_mc_and_td_prediction_train_and_resume() -> None:
    mc_first = train_mc_prediction(episodes=5)
    mc_resumed = train_mc_prediction(
        episodes=5,
        episode_offset=5,
        initial_values=mc_first.values,
        initial_counts=mc_first.counts,
    )
    mc_full = train_mc_prediction(episodes=10)
    torch.testing.assert_close(mc_resumed.values, mc_full.values)
    torch.testing.assert_close(mc_resumed.counts, mc_full.counts)
    assert mc_full.rmse_to_dynamic_programming < 1e-6

    td_first = train_td0_prediction(episodes=50)
    td_resumed = train_td0_prediction(
        episodes=50, episode_offset=50, initial_values=td_first.values
    )
    td_full = train_td0_prediction(episodes=100)
    torch.testing.assert_close(td_resumed.values, td_full.values)
    assert td_full.rmse_to_dynamic_programming < 1e-3


def test_q_learning_resume_from_table() -> None:
    first = train_q_learning(episodes=20, seed=42)
    second = train_q_learning(episodes=20, seed=62, initial_q_values=first.q_values)
    assert not torch.equal(first.q_values, second.q_values)
