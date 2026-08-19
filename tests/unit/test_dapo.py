from __future__ import annotations

import pytest
import torch

from rl_study.algorithms.dapo import (
    dapo_loss,
    dynamic_sampling_filter,
    overlong_reward_shaping,
)


def test_dynamic_sampling_filters_all_correct_and_all_wrong_groups() -> None:
    output = dynamic_sampling_filter(
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
        required_groups=2,
    )
    assert output.selected_group_indices.tolist() == [1, 3]
    assert output.rejected_groups == 2
    assert not output.exhausted


def test_dynamic_sampling_reports_budget_exhaustion() -> None:
    output = dynamic_sampling_filter(torch.ones(3, 2), required_groups=1)
    assert output.exhausted
    assert output.selected_group_indices.numel() == 0


def test_overlong_reward_shaping_boundaries() -> None:
    penalties = overlong_reward_shaping(
        torch.tensor([5, 6, 8, 10, 12]),
        max_response_length=10,
        buffer_length=4,
    )
    torch.testing.assert_close(penalties, torch.tensor([0.0, 0.0, -0.5, -1.0, -1.0]))


def test_clip_higher_changes_positive_but_not_negative_advantage() -> None:
    current = torch.log(torch.tensor([[1.25], [1.25]]))
    old = torch.zeros_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)
    symmetric = dapo_loss(
        current,
        old,
        old,
        torch.tensor([1.0, -1.0]),
        mask,
        clip_low=0.2,
        clip_high=0.3,
        use_clip_higher=False,
    )
    higher = dapo_loss(
        current,
        old,
        old,
        torch.tensor([1.0, -1.0]),
        mask,
        clip_low=0.2,
        clip_high=0.3,
        use_clip_higher=True,
    )
    assert higher.policy_loss.item() < symmetric.policy_loss.item()
    assert higher.ratio[1].item() == pytest.approx(symmetric.ratio[1].item())


def test_token_level_toggle_changes_unequal_length_reduction() -> None:
    current = torch.zeros(2, 2)
    old = torch.zeros_like(current)
    mask = torch.tensor([[True, True], [True, False]])
    advantages = torch.tensor([1.0, 3.0])
    token = dapo_loss(
        current,
        old,
        old,
        advantages,
        mask,
        use_token_level_loss=True,
    )
    sequence = dapo_loss(
        current,
        old,
        old,
        advantages,
        mask,
        use_token_level_loss=False,
    )
    assert token.policy_loss.item() == pytest.approx(-5.0 / 3.0)
    assert sequence.policy_loss.item() == pytest.approx(-2.0)
