"""Paper-based clean-room implementations of DAPO's four components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.algorithms.grpo import GroupPolicyLossOutput, grpo_loss


@dataclass(frozen=True, slots=True)
class DynamicSamplingOutput:
    selected_group_indices: Tensor
    informative_groups: Tensor
    rejected_groups: int
    exhausted: bool


def dynamic_sampling_filter(
    group_rewards: Tensor, *, required_groups: int
) -> DynamicSamplingOutput:
    """Keep groups with non-constant reward, preserving deterministic input order."""
    if group_rewards.ndim != 2 or group_rewards.shape[1] < 2:
        raise ValueError("group_rewards must be [candidate_groups, group>=2]")
    if required_groups < 1:
        raise ValueError("required_groups must be positive")
    informative = group_rewards.max(dim=1).values > group_rewards.min(dim=1).values
    candidates = torch.where(informative)[0]
    selected = candidates[:required_groups]
    return DynamicSamplingOutput(
        selected_group_indices=selected,
        informative_groups=informative,
        rejected_groups=int((~informative).sum()),
        exhausted=selected.numel() < required_groups,
    )


def overlong_reward_shaping(
    response_lengths: Tensor,
    *,
    max_response_length: int,
    buffer_length: int,
    penalty_scale: float = 1.0,
) -> Tensor:
    """Return DAPO's linear soft overlong penalty in [-penalty_scale, 0]."""
    if response_lengths.ndim != 1 or response_lengths.dtype == torch.bool:
        raise ValueError("response_lengths must be one-dimensional")
    if max_response_length < 1 or not 1 <= buffer_length <= max_response_length:
        raise ValueError("buffer_length must be in [1, max_response_length]")
    if penalty_scale < 0:
        raise ValueError("penalty_scale must be non-negative")
    start = max_response_length - buffer_length
    excess = (response_lengths.to(torch.float32) - start).clamp(
        min=0, max=buffer_length
    )
    return -penalty_scale * excess / buffer_length


def dapo_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    sequence_advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    kl_coefficient: float = 0.0,
    use_clip_higher: bool = True,
    use_token_level_loss: bool = True,
) -> GroupPolicyLossOutput:
    """Compose Clip-Higher and token-level PG; other two components precede loss."""
    effective_high = clip_high if use_clip_higher else clip_low
    reduction = "token_mean" if use_token_level_loss else "sequence_mean"
    return grpo_loss(
        current_log_probs,
        old_log_probs,
        reference_log_probs,
        sequence_advantages,
        action_mask,
        clip_low=clip_low,
        clip_high=effective_high,
        kl_coefficient=kl_coefficient,
        reduction=reduction,
    )
