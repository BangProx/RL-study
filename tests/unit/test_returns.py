from __future__ import annotations

import pytest
import torch

from rl_study.math import discounted_returns, generalized_advantage_estimate


def test_discounted_returns_hand_calculation() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0])
    returns = discounted_returns(rewards, gamma=0.5)
    torch.testing.assert_close(returns, torch.tensor([2.75, 3.5, 3.0]))


def test_discounted_returns_stop_between_packed_episodes() -> None:
    rewards = torch.tensor([1.0, 2.0, 10.0])
    ends = torch.tensor([False, True, True])
    returns = discounted_returns(rewards, gamma=1.0, episode_ends=ends)
    torch.testing.assert_close(returns, torch.tensor([3.0, 2.0, 10.0]))


def test_gae_matches_analytic_trajectory() -> None:
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.4, 0.0])
    terminated = torch.tensor([False, True])
    truncated = torch.tensor([False, False])
    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=0.8,
    )
    torch.testing.assert_close(advantages, torch.tensor([1.292, 0.6]))
    torch.testing.assert_close(returns, torch.tensor([1.792, 1.0]))


def test_truncation_bootstraps_but_stops_gae_recurrence() -> None:
    rewards = torch.tensor([0.0])
    values = torch.tensor([0.0, 2.0])
    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        terminated=torch.tensor([False]),
        truncated=torch.tensor([True]),
        gamma=0.9,
        gae_lambda=1.0,
    )
    assert advantages.item() == pytest.approx(1.8)
    assert returns.item() == pytest.approx(1.8)


def test_gae_rejects_done_overlap() -> None:
    with pytest.raises(ValueError, match="both terminated and truncated"):
        generalized_advantage_estimate(
            torch.tensor([1.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([True]),
            torch.tensor([True]),
            gamma=0.9,
            gae_lambda=0.95,
        )
