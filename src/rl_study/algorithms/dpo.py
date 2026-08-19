"""Direct Preference Optimization with response-only sequence log-ratios."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.algorithms.sft import train_sft
from rl_study.data import PreferenceExample, build_preferences, build_tiny_reasoning
from rl_study.models import (
    SequenceBatch,
    TinyCausalLM,
    TinyTokenizer,
    build_sequence_batch,
)
from rl_study.models.roles import assert_frozen, freeze_module, parameter_sha256
from rl_study.models.sequence import response_sequence_log_probs


@dataclass(frozen=True, slots=True)
class DPOLossOutput:
    loss: Tensor
    logits: Tensor
    chosen_rewards: Tensor
    rejected_rewards: Tensor
    preference_accuracy: Tensor


@dataclass(frozen=True, slots=True)
class DPOTrainResult:
    policy: TinyCausalLM
    reference: TinyCausalLM
    optimizer: torch.optim.Optimizer
    losses: tuple[float, ...]
    train_accuracy: float
    validation_accuracy: float
    reference_hash: str
    prompt_uids: tuple[str, ...]
    processed_response_tokens: int
    model_forwards: int


def dpo_loss(
    policy_chosen_log_probs: Tensor,
    policy_rejected_log_probs: Tensor,
    reference_chosen_log_probs: Tensor,
    reference_rejected_log_probs: Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> DPOLossOutput:
    shapes = {
        policy_chosen_log_probs.shape,
        policy_rejected_log_probs.shape,
        reference_chosen_log_probs.shape,
        reference_rejected_log_probs.shape,
    }
    if len(shapes) != 1 or policy_chosen_log_probs.ndim != 1:
        raise ValueError("all DPO log-probs must be one-dimensional with equal shape")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    policy_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs
    reference_log_ratios = (
        reference_chosen_log_probs - reference_rejected_log_probs
    ).detach()
    logits = beta * (policy_log_ratios - reference_log_ratios)
    positive_loss = -torch.nn.functional.logsigmoid(logits)
    negative_loss = -torch.nn.functional.logsigmoid(-logits)
    loss = (
        (1.0 - label_smoothing) * positive_loss + label_smoothing * negative_loss
    ).mean()
    chosen_rewards = beta * (
        policy_chosen_log_probs - reference_chosen_log_probs.detach()
    )
    rejected_rewards = beta * (
        policy_rejected_log_probs - reference_rejected_log_probs.detach()
    )
    return DPOLossOutput(
        loss=loss,
        logits=logits,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        preference_accuracy=(chosen_rewards > rejected_rewards)
        .to(torch.float32)
        .mean(),
    )


def _pair_batches(
    preferences: tuple[PreferenceExample, ...], tokenizer: TinyTokenizer
) -> tuple[SequenceBatch, SequenceBatch]:
    chosen = build_sequence_batch(
        (pair.prompt for pair in preferences),
        (pair.chosen for pair in preferences),
        tokenizer=tokenizer,
        max_length=64,
    )
    rejected = build_sequence_batch(
        (pair.prompt for pair in preferences),
        (pair.rejected for pair in preferences),
        tokenizer=tokenizer,
        max_length=64,
    )
    return chosen, rejected


def dpo_sequence_loss(
    policy: TinyCausalLM,
    reference: TinyCausalLM,
    preferences: tuple[PreferenceExample, ...],
    *,
    tokenizer: TinyTokenizer,
    beta: float,
    label_smoothing: float = 0.0,
) -> DPOLossOutput:
    chosen, rejected = _pair_batches(preferences, tokenizer)
    policy_chosen = response_sequence_log_probs(policy, chosen)
    policy_rejected = response_sequence_log_probs(policy, rejected)
    with torch.no_grad():
        reference_chosen = response_sequence_log_probs(reference, chosen)
        reference_rejected = response_sequence_log_probs(reference, rejected)
    return dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
        label_smoothing=label_smoothing,
    )


@torch.no_grad()
def evaluate_dpo_preferences(
    policy: TinyCausalLM,
    reference: TinyCausalLM,
    preferences: tuple[PreferenceExample, ...],
    *,
    tokenizer: TinyTokenizer,
    beta: float,
) -> float:
    return float(
        dpo_sequence_loss(
            policy,
            reference,
            preferences,
            tokenizer=tokenizer,
            beta=beta,
        ).preference_accuracy
    )


def train_dpo(
    *,
    steps: int = 80,
    batch_size: int = 16,
    seed: int = 42,
    beta: float = 0.1,
    learning_rate: float = 5e-4,
    label_smoothing: float = 0.0,
    policy: TinyCausalLM | None = None,
    reference: TinyCausalLM | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    step_offset: int = 0,
    prompt_batches: tuple[tuple[int, ...], ...] | None = None,
) -> DPOTrainResult:
    if steps < 1 or batch_size < 1 or step_offset < 0:
        raise ValueError("steps/batch must be positive and offset non-negative")
    if prompt_batches is not None and len(prompt_batches) != steps:
        raise ValueError("prompt_batches must contain one batch per DPO step")
    dataset = build_tiny_reasoning(seed=seed)
    tokenizer = TinyTokenizer()
    if policy is None:
        policy = train_sft(seed=seed).model
    if reference is None:
        reference = copy.deepcopy(policy)
    freeze_module(reference)
    reference_hash = parameter_sha256(reference)
    if optimizer is None:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    losses: list[float] = []
    prompt_uids: list[str] = []
    processed_response_tokens = 0
    train_accuracy = 0.0
    for local_step in range(steps):
        global_step = step_offset + local_step
        if prompt_batches is None:
            generator = torch.Generator().manual_seed(seed + global_step * 1_000_003)
            indices = torch.randint(
                len(dataset.preferences), (batch_size,), generator=generator
            )
            preferences = tuple(dataset.preferences[int(index)] for index in indices)
        else:
            prompt_indices = prompt_batches[local_step]
            if len(prompt_indices) != batch_size or any(
                index < 0 or index >= len(dataset.train) for index in prompt_indices
            ):
                raise ValueError(
                    "each DPO prompt batch must contain valid train indices"
                )
            preferences = tuple(
                dataset.preferences[2 * index + (global_step + position) % 2]
                for position, index in enumerate(prompt_indices)
            )
        prompt_uids.extend(pair.prompt_uid for pair in preferences)
        processed_response_tokens += sum(
            len(tokenizer.encode(response, add_bos=False, add_eos=True))
            for pair in preferences
            for response in (pair.chosen, pair.rejected)
        )
        output = dpo_sequence_loss(
            policy,
            reference,
            preferences,
            tokenizer=tokenizer,
            beta=beta,
            label_smoothing=label_smoothing,
        )
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(output.loss)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append(float(output.loss.detach()))
        train_accuracy = float(output.preference_accuracy)
    validation = build_preferences(dataset.validation)
    validation_accuracy = evaluate_dpo_preferences(
        policy,
        reference,
        validation,
        tokenizer=tokenizer,
        beta=beta,
    )
    assert_frozen(reference, role="DPO reference")
    if parameter_sha256(reference) != reference_hash:
        raise RuntimeError("DPO reference parameters changed")
    return DPOTrainResult(
        policy=policy,
        reference=reference,
        optimizer=optimizer,
        losses=tuple(losses),
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        reference_hash=reference_hash,
        prompt_uids=tuple(prompt_uids),
        processed_response_tokens=processed_response_tokens,
        model_forwards=4 * (steps + 1),
    )
