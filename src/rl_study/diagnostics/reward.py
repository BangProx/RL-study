"""Held-out preference, length, and format-shortcut diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.data import PreferenceExample
from rl_study.models import TinyRewardModel, TinyTokenizer, build_sequence_batch


@dataclass(frozen=True, slots=True)
class RewardDiagnostics:
    preference_accuracy: float
    numeric_accuracy: float
    format_accuracy: float
    score_length_correlation: float
    examples: int


def pearson_correlation(first: Tensor, second: Tensor) -> float:
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Pearson inputs must be one-dimensional with equal shape")
    if first.numel() < 2:
        return 0.0
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    denominator = torch.sqrt(
        centered_first.square().sum() * centered_second.square().sum()
    )
    if denominator.item() == 0.0:
        return 0.0
    return float((centered_first * centered_second).sum() / denominator)


@torch.no_grad()
def diagnose_reward_model(
    model: TinyRewardModel,
    preferences: tuple[PreferenceExample, ...],
    *,
    tokenizer: TinyTokenizer | None = None,
) -> RewardDiagnostics:
    if not preferences:
        raise ValueError("reward diagnostics require held-out preferences")
    tokenizer = tokenizer or TinyTokenizer()
    chosen_batch = build_sequence_batch(
        (item.prompt for item in preferences),
        (item.chosen for item in preferences),
        tokenizer=tokenizer,
        max_length=64,
    )
    rejected_batch = build_sequence_batch(
        (item.prompt for item in preferences),
        (item.rejected for item in preferences),
        tokenizer=tokenizer,
        max_length=64,
    )
    chosen_scores = model(chosen_batch.input_ids, chosen_batch.attention_mask)
    rejected_scores = model(rejected_batch.input_ids, rejected_batch.attention_mask)
    correct = chosen_scores > rejected_scores
    reasons = [item.reason for item in preferences]
    numeric_mask = torch.tensor(
        [reason == "incorrect_numeric_answer" for reason in reasons]
    )
    format_mask = torch.tensor(
        [reason == "invalid_required_format" for reason in reasons]
    )
    all_scores = torch.cat((chosen_scores, rejected_scores))
    lengths = torch.tensor(
        [len(item.chosen) for item in preferences]
        + [len(item.rejected) for item in preferences],
        dtype=all_scores.dtype,
    )

    def accuracy(mask: Tensor) -> float:
        return float(correct[mask].to(torch.float32).mean()) if torch.any(mask) else 0.0

    return RewardDiagnostics(
        preference_accuracy=float(correct.to(torch.float32).mean()),
        numeric_accuracy=accuracy(numeric_mask),
        format_accuracy=accuracy(format_mask),
        score_length_correlation=pearson_correlation(all_scores.cpu(), lengths),
        examples=len(preferences),
    )
