"""Return and generalized-advantage calculations with explicit done semantics."""

from __future__ import annotations

import torch
from torch import Tensor


def discounted_returns(
    rewards: Tensor,
    *,
    gamma: float,
    bootstrap_value: Tensor | float = 0.0,
    episode_ends: Tensor | None = None,
) -> Tensor:
    """Compute reward-to-go along dimension 0.

    ``episode_ends[t]`` stops the recurrence after step ``t``. Use it when a tensor
    packs more than one episode. A truncation also ends recurrence even when its TD
    target is allowed to bootstrap.
    """

    if not rewards.is_floating_point() or rewards.ndim < 1:
        raise ValueError("rewards must be a floating tensor with a time dimension")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if episode_ends is None:
        episode_ends = torch.zeros_like(rewards, dtype=torch.bool)
    if episode_ends.shape != rewards.shape or episode_ends.dtype != torch.bool:
        raise ValueError("episode_ends must be bool with the same shape as rewards")
    running = torch.as_tensor(
        bootstrap_value, dtype=rewards.dtype, device=rewards.device
    )
    if running.shape not in (torch.Size(), rewards.shape[1:]):
        raise ValueError("bootstrap_value must be scalar or match rewards without time")
    returns = torch.empty_like(rewards)
    for time_index in range(rewards.shape[0] - 1, -1, -1):
        continuation = (~episode_ends[time_index]).to(rewards.dtype)
        running = rewards[time_index] + gamma * continuation * running
        returns[time_index] = running
    return returns


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE and returns over time dimension 0.

    ``values`` has one more time step than ``rewards``. Termination disables the
    one-step bootstrap. Truncation permits that bootstrap but stops the GAE
    recurrence so a following reset episode cannot leak backward.
    """

    if not rewards.is_floating_point() or not values.is_floating_point():
        raise TypeError("rewards and values must have floating dtypes")
    if values.shape != (rewards.shape[0] + 1, *rewards.shape[1:]):
        raise ValueError("values must have one more time step than rewards")
    if terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError("terminated and truncated must match rewards")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise TypeError("terminated and truncated must be bool")
    if torch.any(terminated & truncated):
        raise ValueError("a step cannot be both terminated and truncated")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be between 0 and 1")

    bootstrap_mask = (~terminated).to(rewards.dtype)
    deltas = rewards + gamma * bootstrap_mask * values[1:] - values[:-1]
    recurrence_mask = (~(terminated | truncated)).to(rewards.dtype)
    advantages = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[0])
    for time_index in range(rewards.shape[0] - 1, -1, -1):
        running = (
            deltas[time_index]
            + gamma * gae_lambda * recurrence_mask[time_index] * running
        )
        advantages[time_index] = running
    returns = advantages + values[:-1]
    return advantages, returns
