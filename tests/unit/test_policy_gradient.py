from __future__ import annotations

import pytest
import torch

from rl_study.algorithms.policy_gradient import actor_critic_loss, reinforce_loss


def test_reinforce_sign_increases_rewarded_action_log_prob() -> None:
    log_prob = torch.tensor(-0.7, requires_grad=True)
    loss = reinforce_loss(log_prob.unsqueeze(0), torch.tensor([2.0]))
    loss.backward()
    assert log_prob.grad is not None
    assert log_prob.grad.item() < 0


def test_baseline_changes_variance_weight_not_return_target() -> None:
    log_probs = torch.tensor([-0.2, -0.3])
    returns = torch.tensor([2.0, 4.0])
    without = reinforce_loss(log_probs, returns)
    with_baseline = reinforce_loss(log_probs, returns, baseline=3.0)
    assert without.item() != pytest.approx(with_baseline.item())


def test_actor_advantage_is_detached_from_actor_loss() -> None:
    logits = torch.tensor([[0.2, -0.1]], requires_grad=True)
    values = torch.tensor([0.5], requires_grad=True)
    next_values = torch.tensor([0.7], requires_grad=True)
    output = actor_critic_loss(
        logits,
        torch.tensor([0]),
        values,
        torch.tensor([1.0]),
        next_values,
        torch.tensor([False]),
        gamma=0.9,
        value_coefficient=0.0,
        entropy_coefficient=0.0,
    )
    output.loss.backward()
    assert logits.grad is not None
    assert values.grad is None or torch.equal(values.grad, torch.zeros_like(values))
    assert next_values.grad is None
