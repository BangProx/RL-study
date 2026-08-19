from __future__ import annotations

import math

import pytest
import torch

from rl_study.math import (
    categorical_cross_entropy,
    categorical_entropy,
    categorical_kl,
    masked_mean,
    masked_sequence_mean,
    masked_sum,
    selected_log_probs,
)


def test_entropy_cross_entropy_and_kl_match_hand_values() -> None:
    p = torch.tensor([0.25, 0.75], dtype=torch.float64)
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    p_logits = p.log()
    q_logits = q.log()

    expected_entropy = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    expected_cross_entropy = -math.log(0.5)
    expected_kl = expected_cross_entropy - expected_entropy

    assert categorical_entropy(p_logits).item() == pytest.approx(expected_entropy)
    assert categorical_cross_entropy(p_logits, q_logits).item() == pytest.approx(
        expected_cross_entropy
    )
    assert categorical_kl(p_logits, q_logits).item() == pytest.approx(expected_kl)
    assert categorical_kl(p_logits, p_logits).item() == pytest.approx(0.0, abs=1e-12)


def test_kl_direction_is_not_silently_reversed() -> None:
    p_logits = torch.tensor([0.0, 1.0], dtype=torch.float64)
    q_logits = torch.tensor([2.0, -1.0], dtype=torch.float64)
    forward = categorical_kl(p_logits, q_logits)
    reverse = categorical_kl(q_logits, p_logits)
    assert forward.item() != pytest.approx(reverse.item())


def test_selected_log_probs_shape_value_and_gradient() -> None:
    logits = torch.tensor([[[0.0, 0.0], [math.log(3.0), 0.0]]], requires_grad=True)
    token_ids = torch.tensor([[1, 0]], dtype=torch.int64)
    selected = selected_log_probs(logits, token_ids)
    assert selected.shape == (1, 2)
    assert selected[0, 0].item() == pytest.approx(math.log(0.5))
    assert selected[0, 1].item() == pytest.approx(math.log(0.75))
    (-selected.mean()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_masked_reductions_ignore_only_false_positions() -> None:
    values = torch.tensor([[1.0, 100.0], [3.0, 5.0]])
    mask = torch.tensor([[True, False], [True, True]])
    assert masked_sum(values, mask).item() == pytest.approx(9.0)
    assert masked_mean(values, mask).item() == pytest.approx(3.0)
    torch.testing.assert_close(
        masked_sequence_mean(values, mask), torch.tensor([1.0, 4.0])
    )


def test_empty_mask_requires_an_explicit_policy() -> None:
    values = torch.tensor([1.0, 2.0])
    mask = torch.tensor([False, False])
    with pytest.raises(ValueError, match="empty mask"):
        masked_mean(values, mask)
    assert masked_mean(values, mask, zero_policy="zero").item() == 0.0


def test_probability_inputs_reject_wrong_shape_or_dtype() -> None:
    with pytest.raises(TypeError, match="floating"):
        categorical_entropy(torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="same shape"):
        categorical_kl(torch.zeros(2), torch.zeros(3))
    with pytest.raises(TypeError, match="int64"):
        selected_log_probs(torch.zeros(1, 2), torch.tensor([0], dtype=torch.int32))
