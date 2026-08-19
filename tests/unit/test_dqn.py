from __future__ import annotations

import pytest
import torch

from rl_study.algorithms.dqn import DQNBatch, DQNNetwork, dqn_loss, hard_update


def _batch(*, terminated: bool, truncated: bool = False) -> DQNBatch:
    return DQNBatch(
        states=torch.tensor([0]),
        actions=torch.tensor([1]),
        rewards=torch.tensor([1.0]),
        next_states=torch.tensor([2]),
        terminated=torch.tensor([terminated]),
        truncated=torch.tensor([truncated]),
    )


def test_dqn_target_network_is_detached() -> None:
    torch.manual_seed(1)
    policy = DQNNetwork(4, 2)
    target = DQNNetwork(4, 2)
    hard_update(target, policy)
    output = dqn_loss(policy, target, _batch(terminated=False), gamma=0.9)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())
    assert output.targets.requires_grad is False


def test_dqn_terminal_removes_bootstrap_but_truncation_keeps_it() -> None:
    policy = DQNNetwork(4, 2)
    target = DQNNetwork(4, 2)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
        target.network[-1].bias.fill_(2.0)
    terminal = dqn_loss(policy, target, _batch(terminated=True), gamma=0.9)
    truncated = dqn_loss(
        policy, target, _batch(terminated=False, truncated=True), gamma=0.9
    )
    assert terminal.targets.item() == 1.0
    assert truncated.targets.item() == pytest.approx(2.8)
