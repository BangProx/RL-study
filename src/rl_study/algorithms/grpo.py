"""GRPO-family objectives with explicit variant and reduction choices."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.math import masked_mean


@dataclass(frozen=True, slots=True)
class GroupAdvantageOutput:
    advantages: Tensor
    means: Tensor
    standard_deviations: Tensor
    informative_groups: Tensor


@dataclass(frozen=True, slots=True)
class GroupPolicyLossOutput:
    loss: Tensor
    policy_loss: Tensor
    reference_kl: Tensor
    ratio: Tensor
    clip_fraction: Tensor


@dataclass(frozen=True, slots=True)
class RLOOLossOutput:
    loss: Tensor
    sequence_log_probs: Tensor


def group_relative_advantages(
    rewards: Tensor, *, epsilon: float = 1e-4
) -> GroupAdvantageOutput:
    """Mean/std normalize each prompt group; constant groups become exact zeros."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must be [prompts, group>=2]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    means = rewards.mean(dim=1, keepdim=True)
    standard_deviations = rewards.std(dim=1, keepdim=True, unbiased=False)
    informative = standard_deviations.squeeze(1) > 0
    normalized = (rewards - means) / (standard_deviations + epsilon)
    advantages = torch.where(
        informative.unsqueeze(1), normalized, torch.zeros_like(normalized)
    )
    return GroupAdvantageOutput(
        advantages=advantages,
        means=means.squeeze(1),
        standard_deviations=standard_deviations.squeeze(1),
        informative_groups=informative,
    )


def rloo_advantages(rewards: Tensor) -> Tensor:
    """Sequence reward minus the mean reward of all other samples in its group."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("RLOO rewards must be [prompts, group>=2]")
    group_size = rewards.shape[1]
    leave_one_out_mean = (rewards.sum(dim=1, keepdim=True) - rewards) / (group_size - 1)
    return rewards - leave_one_out_mean


def rloo_sequence_loss(
    current_log_probs: Tensor,
    sequence_advantages: Tensor,
    action_mask: Tensor,
) -> RLOOLossOutput:
    """Sequence-action REINFORCE loss with externally computed LOO advantages."""
    if current_log_probs.shape != action_mask.shape:
        raise ValueError("current_log_probs and action_mask must have equal shapes")
    if action_mask.dtype != torch.bool:
        raise TypeError("action_mask must be bool")
    if sequence_advantages.shape != (current_log_probs.shape[0],):
        raise ValueError("sequence_advantages must be [sequences]")
    if torch.any(action_mask.sum(dim=1) == 0):
        raise ValueError("every RLOO sequence needs an action token")
    sequence_log_probs = (current_log_probs * action_mask).sum(dim=1)
    loss = -(sequence_log_probs * sequence_advantages.detach()).mean()
    return RLOOLossOutput(loss=loss, sequence_log_probs=sequence_log_probs)


def dr_grpo_advantages(rewards: Tensor) -> Tensor:
    """Mean-center without reward-std normalization for the Dr. GRPO variant."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Dr. GRPO rewards must be [prompts, group>=2]")
    return rewards - rewards.mean(dim=1, keepdim=True)


def per_token_reference_kl(
    current_log_probs: Tensor,
    reference_log_probs: Tensor,
    *,
    zero_tolerance: float = 1e-5,
) -> Tensor:
    """DeepSeekMath k3 estimator with a clone-equality numerical guard."""
    if current_log_probs.shape != reference_log_probs.shape:
        raise ValueError("current/reference log-probs must have equal shapes")
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative")
    log_reference_ratio = reference_log_probs.detach() - current_log_probs
    estimate = torch.expm1(log_reference_ratio) - log_reference_ratio
    return torch.where(
        log_reference_ratio.abs() <= zero_tolerance,
        torch.zeros_like(estimate),
        estimate,
    )


def reduce_group_tokens(
    values: Tensor,
    action_mask: Tensor,
    *,
    reduction: str,
    fixed_response_length: int | None = None,
) -> Tensor:
    if values.shape != action_mask.shape or action_mask.dtype != torch.bool:
        raise ValueError("values and bool action_mask must have equal shapes")
    lengths = action_mask.sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError("every sequence needs at least one action token")
    masked = values * action_mask
    if reduction == "sequence_mean":
        return (masked.sum(dim=1) / lengths).mean()
    if reduction == "token_mean":
        return masked.sum() / lengths.sum()
    if reduction == "dr_grpo":
        if fixed_response_length is None or fixed_response_length < int(lengths.max()):
            raise ValueError(
                "dr_grpo requires fixed_response_length >= observed response length"
            )
        return masked.sum() / (values.shape[0] * fixed_response_length)
    raise ValueError("reduction must be sequence_mean, token_mean, or dr_grpo")


def grpo_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    sequence_advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    kl_coefficient: float = 0.0,
    reduction: str = "sequence_mean",
    fixed_response_length: int | None = None,
) -> GroupPolicyLossOutput:
    if not (
        current_log_probs.shape
        == old_log_probs.shape
        == reference_log_probs.shape
        == action_mask.shape
    ):
        raise ValueError("GRPO token tensors must have equal shapes")
    if sequence_advantages.shape != (current_log_probs.shape[0],):
        raise ValueError("sequence_advantages must be [sequences]")
    if action_mask.dtype != torch.bool:
        raise TypeError("action_mask must be bool")
    if clip_low < 0 or clip_high < 0 or kl_coefficient < 0:
        raise ValueError("clip bounds and KL coefficient must be non-negative")

    log_ratio = current_log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    advantages = sequence_advantages.detach().unsqueeze(1)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    reference_kl_tokens = per_token_reference_kl(current_log_probs, reference_log_probs)
    policy_objective = reduce_group_tokens(
        surrogate,
        action_mask,
        reduction=reduction,
        fixed_response_length=fixed_response_length,
    )
    reference_kl = masked_mean(reference_kl_tokens, action_mask)
    clipped = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    clip_fraction = masked_mean(clipped.to(ratio.dtype), action_mask)
    policy_loss = -policy_objective
    return GroupPolicyLossOutput(
        loss=policy_loss + kl_coefficient * reference_kl,
        policy_loss=policy_loss,
        reference_kl=reference_kl,
        ratio=ratio,
        clip_fraction=clip_fraction,
    )


def gspo_sequence_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    sequence_advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    kl_coefficient: float = 0.0,
) -> GroupPolicyLossOutput:
    """GSPO uses one geometric-mean importance ratio per sequence."""
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current/old log-probs must have equal shapes")
    token_log_ratio = current_log_probs - old_log_probs.detach()
    sequence_log_ratio = masked_mean(token_log_ratio, action_mask, dim=1)
    sequence_ratio = sequence_log_ratio.exp()
    clipped_ratio = sequence_ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    advantages = sequence_advantages.detach()
    objective = torch.minimum(sequence_ratio * advantages, clipped_ratio * advantages)
    reference_kl = masked_mean(
        per_token_reference_kl(current_log_probs, reference_log_probs),
        action_mask,
    )
    clipped = (sequence_ratio < 1.0 - clip_low) | (sequence_ratio > 1.0 + clip_high)
    policy_loss = -objective.mean()
    return GroupPolicyLossOutput(
        loss=policy_loss + kl_coefficient * reference_kl,
        policy_loss=policy_loss,
        reference_kl=reference_kl,
        ratio=sequence_ratio,
        clip_fraction=clipped.to(sequence_ratio.dtype).mean(),
    )
