"""Classic discrete PPO with explicit old/current policy separation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.algorithms.policy_gradient import (
    DiscreteActorCritic,
    collect_episode,
    evaluate_policy,
)
from rl_study.envs import TinyGridWorld
from rl_study.math.returns import generalized_advantage_estimate


@dataclass(frozen=True, slots=True)
class PPOLossOutput:
    loss: Tensor
    unclipped_objective: Tensor
    clipped_objective: Tensor
    ratio: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


@dataclass(frozen=True, slots=True)
class PPOTrainResult:
    model: DiscreteActorCritic
    optimizer: torch.optim.Optimizer
    episode_returns: tuple[float, ...]
    policy_losses: tuple[float, ...]
    success_rate: float


def ppo_policy_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> PPOLossOutput:
    if not (current_log_probs.shape == old_log_probs.shape == advantages.shape):
        raise ValueError("current, old log_probs, and advantages must share a shape")
    if clip_low < 0 or clip_high < 0:
        raise ValueError("clip bounds must be non-negative")
    ratio = torch.exp(current_log_probs - old_log_probs.detach())
    clipped_ratio = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    detached_advantages = advantages.detach()
    unclipped = ratio * detached_advantages
    clipped = clipped_ratio * detached_advantages
    objective = torch.minimum(unclipped, clipped)
    log_ratio = current_log_probs - old_log_probs.detach()
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    outside = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    return PPOLossOutput(
        loss=-objective.mean(),
        unclipped_objective=unclipped,
        clipped_objective=clipped,
        ratio=ratio,
        approximate_kl=approximate_kl,
        clip_fraction=outside.to(ratio.dtype).mean(),
    )


def ppo_value_loss(values: Tensor, returns: Tensor) -> Tensor:
    if values.shape != returns.shape:
        raise ValueError("values and returns must have the same shape")
    return 0.5 * (values - returns.detach()).square().mean()


def train_ppo(
    *,
    episodes: int = 180,
    seed: int = 42,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    learning_rate: float = 0.02,
    update_epochs: int = 4,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    model: DiscreteActorCritic | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    episode_offset: int = 0,
) -> PPOTrainResult:
    if episodes < 1 or episode_offset < 0:
        raise ValueError("episodes must be positive and episode_offset non-negative")
    environment = TinyGridWorld()
    if model is None:
        torch.manual_seed(seed)
        model = DiscreteActorCritic(environment.num_states, environment.num_actions)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer is None:
        raise ValueError("resumed PPO requires an optimizer")
    episode_returns: list[float] = []
    policy_losses: list[float] = []
    for episode in range(episodes):
        global_episode = episode_offset + episode
        generator = torch.Generator().manual_seed(seed + global_episode * 1_000_003)
        batch = collect_episode(
            model.actor,
            environment,
            seed=seed + global_episode,
            generator=generator,
        )
        with torch.no_grad():
            old_logits, old_values = model(batch.states)
            _, final_next_value = model(batch.next_states[-1:])
            old_all_log_probs = torch.log_softmax(old_logits, dim=-1)
            old_log_probs = old_all_log_probs.gather(
                1, batch.actions.unsqueeze(1)
            ).squeeze(1)
            values_with_bootstrap = torch.cat((old_values, final_next_value))
            advantages, returns = generalized_advantage_estimate(
                batch.rewards,
                values_with_bootstrap,
                batch.terminated,
                batch.truncated,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )
        for _ in range(update_epochs):
            logits, values = model(batch.states)
            all_log_probs = torch.log_softmax(logits, dim=-1)
            current_log_probs = all_log_probs.gather(
                1, batch.actions.unsqueeze(1)
            ).squeeze(1)
            policy_output = ppo_policy_loss(
                current_log_probs,
                old_log_probs,
                advantages,
                clip_low=clip_low,
                clip_high=clip_high,
            )
            value_loss = ppo_value_loss(values, returns)
            entropy = -(all_log_probs.exp() * all_log_probs).sum(dim=-1).mean()
            total_loss = policy_output.loss + 0.5 * value_loss - 0.01 * entropy
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(total_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            policy_losses.append(float(policy_output.loss.detach()))
        episode_returns.append(float(batch.rewards.sum()))
    return PPOTrainResult(
        model=model,
        optimizer=optimizer,
        episode_returns=tuple(episode_returns),
        policy_losses=tuple(policy_losses),
        success_rate=evaluate_policy(model.actor),
    )
