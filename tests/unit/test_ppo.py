from __future__ import annotations

import pytest
import torch

from rl_study.algorithms.ppo import ppo_policy_loss, ppo_value_loss


def test_ppo_ratio_one_matches_unclipped_objective() -> None:
    log_probs = torch.tensor([[-0.2, -1.0]])
    advantages = torch.tensor([[2.0, -3.0]])
    output = ppo_policy_loss(log_probs, log_probs, advantages)
    torch.testing.assert_close(output.ratio, torch.ones_like(log_probs))
    torch.testing.assert_close(output.unclipped_objective, advantages)
    assert output.clip_fraction.item() == 0.0
    assert output.approximate_kl.item() == pytest.approx(0.0)


def test_ppo_clip_handles_positive_and_negative_advantages() -> None:
    old = torch.zeros(2)
    current = torch.log(torch.tensor([2.0, 0.2]))
    advantages = torch.tensor([1.0, -1.0])
    output = ppo_policy_loss(current, old, advantages, clip_low=0.2, clip_high=0.2)
    torch.testing.assert_close(output.clipped_objective, torch.tensor([1.2, -0.8]))
    torch.testing.assert_close(
        torch.minimum(output.unclipped_objective, output.clipped_objective),
        torch.tensor([1.2, -0.8]),
    )
    assert output.clip_fraction.item() == 1.0


def test_value_loss_does_not_backpropagate_into_returns() -> None:
    values = torch.tensor([1.0], requires_grad=True)
    returns = torch.tensor([3.0], requires_grad=True)
    loss = ppo_value_loss(values, returns)
    loss.backward()
    assert values.grad is not None
    assert returns.grad is None
