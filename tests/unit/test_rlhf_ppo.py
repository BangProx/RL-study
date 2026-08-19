from __future__ import annotations

import pytest
import torch

from rl_study.algorithms.rlhf_ppo import (
    compose_rlhf_rewards,
    masked_token_returns,
    rlhf_ppo_loss,
)


def test_rlhf_reward_plus_token_kl_decomposition() -> None:
    output = compose_rlhf_rewards(
        torch.tensor([1.0]),
        torch.tensor([[0.0, 0.2, 0.4]]),
        torch.zeros(1, 3),
        torch.tensor([[False, True, True]]),
        kl_coefficient=0.1,
    )
    torch.testing.assert_close(output.sampled_kl, torch.tensor([[0.0, 0.2, 0.4]]))
    torch.testing.assert_close(output.token_rewards, torch.tensor([[0.0, -0.02, 0.96]]))
    assert output.non_score_rewards.item() == pytest.approx(-0.06)
    assert output.total_rewards.item() == pytest.approx(0.94)


def test_masked_token_returns_keep_prompt_zero() -> None:
    rewards = torch.tensor([[0.0, -0.02, 0.96]])
    mask = torch.tensor([[False, True, True]])
    returns = masked_token_returns(rewards, mask)
    torch.testing.assert_close(returns, torch.tensor([[0.0, 0.94, 0.96]]))


def test_rlhf_ppo_ratio_one_and_gradient_ownership() -> None:
    current = torch.tensor([[0.0, -0.2, -0.3]], requires_grad=True)
    old = current.detach().clone().requires_grad_(True)
    advantages = torch.tensor([[100.0, 1.0, -1.0]], requires_grad=True)
    values = torch.tensor([[0.0, 0.1, 0.2]], requires_grad=True)
    returns = torch.tensor([[0.0, 0.5, -0.1]], requires_grad=True)
    entropy = torch.ones_like(current, requires_grad=True)
    mask = torch.tensor([[False, True, True]])
    output = rlhf_ppo_loss(current, old, advantages, values, returns, entropy, mask)
    torch.testing.assert_close(output.ratio, torch.ones_like(current))
    assert output.approximate_kl.item() == pytest.approx(0.0)
    output.loss.backward()
    assert current.grad is not None
    assert current.grad[0, 0].item() == 0.0
    assert old.grad is None
    assert advantages.grad is None
    assert values.grad is not None and values.grad[0, 0].item() == 0.0
    assert returns.grad is None


def test_zero_action_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty mask"):
        rlhf_ppo_loss(
            *[torch.zeros(1, 2) for _ in range(6)],
            torch.zeros(1, 2, dtype=torch.bool),
        )
