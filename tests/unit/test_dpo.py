from __future__ import annotations

import copy
import math

import pytest
import torch

from rl_study.algorithms.dpo import dpo_loss, dpo_sequence_loss
from rl_study.data import build_tiny_reasoning
from rl_study.models import TinyCausalLM, TinyTokenizer
from rl_study.models.roles import freeze_module, parameter_sha256


def test_dpo_loss_matches_hand_calculation() -> None:
    policy_chosen = torch.tensor([-1.0], requires_grad=True)
    policy_rejected = torch.tensor([-2.0], requires_grad=True)
    reference_chosen = torch.tensor([-1.5], requires_grad=True)
    reference_rejected = torch.tensor([-1.5], requires_grad=True)
    output = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=1.0,
    )
    assert output.logits.item() == pytest.approx(1.0)
    assert output.loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-1))))
    output.loss.backward()
    assert policy_chosen.grad is not None and policy_rejected.grad is not None
    assert reference_chosen.grad is None and reference_rejected.grad is None


def test_dpo_pair_order_and_label_smoothing() -> None:
    correct = dpo_loss(
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=1.0,
    )
    reversed_pair = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=1.0,
    )
    smoothed = dpo_loss(
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=1.0,
        label_smoothing=0.1,
    )
    assert correct.loss < reversed_pair.loss
    assert smoothed.loss > correct.loss


def test_dpo_sequence_loss_keeps_reference_frozen() -> None:
    torch.manual_seed(2)
    policy = TinyCausalLM()
    reference = copy.deepcopy(policy)
    freeze_module(reference)
    before = parameter_sha256(reference)
    preferences = build_tiny_reasoning().preferences[:2]
    output = dpo_sequence_loss(
        policy,
        reference,
        preferences,
        tokenizer=TinyTokenizer(),
        beta=0.1,
    )
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert all(parameter.grad is None for parameter in reference.parameters())
    assert parameter_sha256(reference) == before
