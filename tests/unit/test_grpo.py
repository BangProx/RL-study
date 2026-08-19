from __future__ import annotations

import pytest
import torch

from rl_study.algorithms.grpo import (
    dr_grpo_advantages,
    group_relative_advantages,
    grpo_loss,
    gspo_sequence_loss,
    per_token_reference_kl,
    reduce_group_tokens,
    rloo_advantages,
    rloo_sequence_loss,
)


def test_group_relative_advantages_and_zero_variance_group() -> None:
    output = group_relative_advantages(torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]]))
    assert output.informative_groups.tolist() == [True, False]
    assert output.advantages[0].mean().item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(output.advantages[1], torch.zeros(3))
    assert torch.isfinite(output.advantages).all()


def test_group_advantage_handles_minimum_group_and_duplicate_samples() -> None:
    minimum = group_relative_advantages(torch.tensor([[0.0, 1.0]]))
    assert minimum.informative_groups.tolist() == [True]
    assert minimum.advantages[0, 0].item() < 0
    assert minimum.advantages[0, 1].item() > 0

    duplicated = group_relative_advantages(torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    assert duplicated.advantages[0, 0].item() == pytest.approx(
        duplicated.advantages[0, 1].item()
    )
    assert duplicated.advantages[0, 2].item() == pytest.approx(
        duplicated.advantages[0, 3].item()
    )

    with pytest.raises(ValueError, match="group>=2"):
        group_relative_advantages(torch.tensor([[1.0]]))


def test_rloo_and_dr_grpo_hand_values() -> None:
    rewards = torch.tensor([[1.0, 2.0, 4.0]])
    torch.testing.assert_close(
        rloo_advantages(rewards), torch.tensor([[-2.0, -0.5, 2.5]])
    )
    torch.testing.assert_close(
        dr_grpo_advantages(rewards),
        torch.tensor([[-4.0 / 3.0, -1.0 / 3.0, 5.0 / 3.0]]),
    )


def test_rloo_uses_full_sequence_as_one_action() -> None:
    token_log_probs = torch.tensor([[-1.0, -2.0], [-3.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True], [True, False]])
    output = rloo_sequence_loss(token_log_probs, torch.tensor([2.0, -1.0]), mask)
    assert output.loss.item() == pytest.approx(1.5)
    output.loss.backward()
    torch.testing.assert_close(
        token_log_probs.grad, torch.tensor([[-1.0, -1.0], [0.5, 0.0]])
    )


def test_grpo_ratio_one_and_reference_gradient_contract() -> None:
    current = torch.tensor([[0.0, -0.2]], requires_grad=True)
    old = current.detach().clone().requires_grad_(True)
    reference = torch.tensor([[0.1, -0.3]], requires_grad=True)
    mask = torch.tensor([[True, True]])
    output = grpo_loss(
        current,
        old,
        reference,
        torch.tensor([1.0], requires_grad=True),
        mask,
        kl_coefficient=0.1,
    )
    torch.testing.assert_close(output.ratio, torch.ones_like(current))
    output.loss.backward()
    assert current.grad is not None
    assert old.grad is None and reference.grad is None


def test_reference_kl_clone_tolerance_has_zero_value_and_gradient() -> None:
    current = torch.tensor([[1.0 + 1e-6]], requires_grad=True)
    reference = torch.tensor([[1.0]])
    kl = per_token_reference_kl(current, reference)
    assert kl.item() == 0.0
    kl.sum().backward()
    assert current.grad is not None and current.grad.item() == 0.0


def test_reduction_variants_expose_length_weighting() -> None:
    values = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])
    sequence_mean = reduce_group_tokens(values, mask, reduction="sequence_mean")
    token_mean = reduce_group_tokens(values, mask, reduction="token_mean")
    fixed = reduce_group_tokens(
        values, mask, reduction="dr_grpo", fixed_response_length=2
    )
    assert sequence_mean.item() == pytest.approx(2.0)
    assert token_mean.item() == pytest.approx(5.0 / 3.0)
    assert fixed.item() == pytest.approx(1.25)


def test_gspo_uses_one_geometric_mean_ratio_per_sequence() -> None:
    current = torch.log(torch.tensor([[2.0, 8.0], [1.0, 1.0]]))
    old = torch.zeros_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)
    output = gspo_sequence_loss(
        current,
        old,
        old,
        torch.tensor([1.0, -1.0]),
        mask,
        clip_low=10.0,
        clip_high=10.0,
    )
    torch.testing.assert_close(output.ratio, torch.tensor([4.0, 1.0]))
