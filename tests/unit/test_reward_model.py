from __future__ import annotations

import copy

import pytest
import torch

from rl_study.algorithms.reward_model import pairwise_reward_loss
from rl_study.data import (
    build_tiny_reasoning,
    has_valid_format,
    verifier_reward,
    verify_response,
)
from rl_study.diagnostics.reward import pearson_correlation
from rl_study.models import (
    TinyCausalLM,
    TinyLMConfig,
    TinyRewardModel,
    TinyTokenizer,
    TinyValueModel,
    build_sequence_batch,
)
from rl_study.models.roles import assert_frozen, freeze_module, parameter_sha256
from rl_study.models.sequence import response_sequence_log_probs


def test_verifier_separates_correct_format_and_answer() -> None:
    example = build_tiny_reasoning().train[0]
    assert verify_response(example, example.target_response)
    assert verifier_reward(example, example.target_response) == 1.0
    wrong = f"<answer>{example.answer + 1}</answer>"
    assert has_valid_format(wrong)
    assert verifier_reward(example, wrong) == 0.1
    assert verifier_reward(example, f"answer={example.answer}") == 0.0


def test_pairwise_loss_matches_bradley_terry_hand_value() -> None:
    chosen = torch.tensor([2.0, 0.0], requires_grad=True)
    rejected = torch.tensor([0.0, 1.0], requires_grad=True)
    output = pairwise_reward_loss(chosen, rejected)
    expected = (
        -(
            torch.nn.functional.logsigmoid(torch.tensor(2.0))
            + torch.nn.functional.logsigmoid(torch.tensor(-1.0))
        )
        / 2
    )
    assert output.loss.item() == pytest.approx(expected.item())
    assert output.accuracy.item() == pytest.approx(0.5)
    output.loss.backward()
    assert chosen.grad is not None and rejected.grad is not None


def test_reward_model_outputs_one_score_per_sequence() -> None:
    tokenizer = TinyTokenizer()
    batch = build_sequence_batch(
        ["1+1?", "2+2?"],
        ["<answer>2</answer>", "<answer>4</answer>"],
        tokenizer=tokenizer,
        max_length=64,
    )
    model = TinyRewardModel()
    scores = model(batch.input_ids, batch.attention_mask)
    assert scores.shape == (2,)


def test_frozen_reference_hash_and_gradient_contract() -> None:
    model = TinyRewardModel()
    before = parameter_sha256(model)
    freeze_module(model)
    assert_frozen(model, role="reward")
    after = parameter_sha256(model)
    assert before == after


def test_policy_update_leaves_reference_and_reward_unchanged() -> None:
    config = TinyLMConfig(
        vocab_size=128,
        max_sequence_length=16,
        hidden_size=16,
        num_heads=4,
        num_layers=1,
        intermediate_size=32,
    )
    policy = TinyCausalLM(config)
    reference = copy.deepcopy(policy)
    reward = TinyRewardModel(TinyCausalLM(config))
    freeze_module(reference)
    freeze_module(reward)
    reference_hash = parameter_sha256(reference)
    reward_hash = parameter_sha256(reward)
    batch = build_sequence_batch(
        ["1+1?"], ["2"], tokenizer=TinyTokenizer(), max_length=16
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    loss = -response_sequence_log_probs(policy, batch).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert parameter_sha256(reference) == reference_hash
    assert parameter_sha256(reward) == reward_hash
    assert_frozen(reference, role="reference")
    assert_frozen(reward, role="reward")


def test_value_model_returns_one_value_per_token() -> None:
    tokenizer = TinyTokenizer()
    batch = build_sequence_batch(["1+1?"], ["2"], tokenizer=tokenizer, max_length=16)
    value_model = TinyValueModel(
        TinyCausalLM(
            TinyLMConfig(
                vocab_size=128,
                max_sequence_length=16,
                hidden_size=16,
                num_heads=4,
                num_layers=1,
                intermediate_size=32,
            )
        )
    )
    values = value_model(batch.input_ids, batch.attention_mask)
    assert values.shape == batch.input_ids.shape
    values.mean().backward()
    assert value_model.value_head.weight.grad is not None


def test_pearson_correlation_detects_length_direction() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0])
    lengths = torch.tensor([10.0, 20.0, 30.0])
    assert pearson_correlation(scores, lengths) == pytest.approx(1.0)
    assert pearson_correlation(scores, -lengths) == pytest.approx(-1.0)
