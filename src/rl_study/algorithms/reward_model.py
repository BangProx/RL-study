"""Bradley-Terry pairwise reward-model loss and tiny trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.data import PreferenceExample, build_preferences, build_tiny_reasoning
from rl_study.diagnostics.reward import RewardDiagnostics, diagnose_reward_model
from rl_study.models import TinyRewardModel, TinyTokenizer, build_sequence_batch


@dataclass(frozen=True, slots=True)
class PairwiseRewardOutput:
    loss: Tensor
    accuracy: Tensor
    margins: Tensor


@dataclass(frozen=True, slots=True)
class RewardTrainResult:
    model: TinyRewardModel
    optimizer: torch.optim.Optimizer
    losses: tuple[float, ...]
    train_accuracy: float
    validation: RewardDiagnostics


def pairwise_reward_loss(
    chosen_scores: Tensor, rejected_scores: Tensor
) -> PairwiseRewardOutput:
    if chosen_scores.shape != rejected_scores.shape or chosen_scores.ndim != 1:
        raise ValueError("chosen and rejected scores must be one-dimensional and equal")
    margins = chosen_scores - rejected_scores
    loss = -torch.nn.functional.logsigmoid(margins).mean()
    accuracy = (margins > 0).to(torch.float32).mean()
    return PairwiseRewardOutput(loss=loss, accuracy=accuracy, margins=margins)


def _score_pairs(
    model: TinyRewardModel,
    preferences: tuple[PreferenceExample, ...],
    tokenizer: TinyTokenizer,
) -> PairwiseRewardOutput:
    chosen = build_sequence_batch(
        (item.prompt for item in preferences),
        (item.chosen for item in preferences),
        tokenizer=tokenizer,
        max_length=64,
    )
    rejected = build_sequence_batch(
        (item.prompt for item in preferences),
        (item.rejected for item in preferences),
        tokenizer=tokenizer,
        max_length=64,
    )
    chosen_scores = model(chosen.input_ids, chosen.attention_mask)
    rejected_scores = model(rejected.input_ids, rejected.attention_mask)
    return pairwise_reward_loss(chosen_scores, rejected_scores)


def train_reward_model(
    *,
    steps: int = 120,
    batch_size: int = 16,
    seed: int = 42,
    learning_rate: float = 2e-3,
    model: TinyRewardModel | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> RewardTrainResult:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    torch.manual_seed(seed)
    dataset = build_tiny_reasoning(seed=seed)
    validation_preferences = build_preferences(dataset.validation)
    tokenizer = TinyTokenizer()
    if model is None:
        model = TinyRewardModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer is None:
        raise ValueError("resumed reward-model training requires an optimizer")
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    train_accuracy = 0.0
    for _ in range(steps):
        indices = torch.randint(
            len(dataset.preferences), (batch_size,), generator=generator
        )
        preferences = tuple(dataset.preferences[int(index)] for index in indices)
        output = _score_pairs(model, preferences, tokenizer)
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(output.loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(output.loss.detach()))
        train_accuracy = float(output.accuracy)
    model.eval()
    validation = diagnose_reward_model(
        model, validation_preferences, tokenizer=tokenizer
    )
    return RewardTrainResult(
        model=model,
        optimizer=optimizer,
        losses=tuple(losses),
        train_accuracy=train_accuracy,
        validation=validation,
    )
