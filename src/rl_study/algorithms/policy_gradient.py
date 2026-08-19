"""REINFORCE and actor-critic equations plus short GridWorld trainers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rl_study.algorithms.tabular import monte_carlo_returns
from rl_study.envs import TinyGridWorld


class DiscretePolicy(nn.Module):
    def __init__(self, num_states: int, num_actions: int) -> None:
        super().__init__()
        self.logits = nn.Embedding(num_states, num_actions)
        nn.init.zeros_(self.logits.weight)

    def forward(self, states: Tensor) -> Tensor:
        if states.dtype != torch.int64:
            raise TypeError("states must be int64")
        logits: Tensor = self.logits(states)
        return logits


class DiscreteActorCritic(nn.Module):
    def __init__(self, num_states: int, num_actions: int) -> None:
        super().__init__()
        self.actor = DiscretePolicy(num_states, num_actions)
        self.critic = nn.Embedding(num_states, 1)
        nn.init.zeros_(self.critic.weight)

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        return self.actor(states), self.critic(states).squeeze(-1)


@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    terminated: Tensor
    truncated: Tensor


@dataclass(frozen=True, slots=True)
class ActorCriticLossOutput:
    loss: Tensor
    actor_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    advantages: Tensor
    targets: Tensor


@dataclass(frozen=True, slots=True)
class PolicyTrainResult:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    episode_returns: tuple[float, ...]
    losses: tuple[float, ...]
    success_rate: float
    running_baseline: float | None = None


def collect_episode(
    policy: DiscretePolicy,
    environment: TinyGridWorld,
    *,
    seed: int,
    generator: torch.Generator,
) -> EpisodeBatch:
    state, _ = environment.reset(seed=seed)
    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[int] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    while True:
        with torch.no_grad():
            logits = policy(torch.tensor([state])).squeeze(0)
            action = int(
                torch.multinomial(
                    torch.softmax(logits, dim=-1), 1, generator=generator
                ).item()
            )
        result = environment.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(result.reward)
        next_states.append(result.observation)
        terminated.append(result.terminated)
        truncated.append(result.truncated)
        state = result.observation
        if result.terminated or result.truncated:
            break
    return EpisodeBatch(
        states=torch.tensor(states, dtype=torch.int64),
        actions=torch.tensor(actions, dtype=torch.int64),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        next_states=torch.tensor(next_states, dtype=torch.int64),
        terminated=torch.tensor(terminated, dtype=torch.bool),
        truncated=torch.tensor(truncated, dtype=torch.bool),
    )


def reinforce_loss(
    log_probs: Tensor, returns: Tensor, *, baseline: Tensor | float = 0.0
) -> Tensor:
    if log_probs.shape != returns.shape:
        raise ValueError("log_probs and returns must have the same shape")
    baseline_tensor = torch.as_tensor(
        baseline, dtype=returns.dtype, device=returns.device
    )
    advantages = (returns - baseline_tensor).detach()
    return -(log_probs * advantages).mean()


def actor_critic_loss(
    logits: Tensor,
    actions: Tensor,
    values: Tensor,
    rewards: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    *,
    gamma: float,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> ActorCriticLossOutput:
    expected_shape = actions.shape
    for name, value in (
        ("values", values),
        ("rewards", rewards),
        ("next_values", next_values),
        ("terminated", terminated),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have the same shape as actions")
    all_log_probs = torch.log_softmax(logits, dim=-1)
    log_probs = all_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    entropy = -(all_log_probs.exp() * all_log_probs).sum(dim=-1).mean()
    targets = rewards + gamma * (~terminated).to(rewards.dtype) * next_values.detach()
    advantages = targets - values
    actor_loss = -(log_probs * advantages.detach()).mean()
    value_loss = 0.5 * advantages.square().mean()
    loss = actor_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    return ActorCriticLossOutput(
        loss=loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        entropy=entropy,
        advantages=advantages,
        targets=targets,
    )


@torch.no_grad()
def evaluate_policy(
    policy: DiscretePolicy, *, episodes: int = 50, seed: int = 1_000
) -> float:
    successes = 0
    for episode in range(episodes):
        environment = TinyGridWorld()
        state, _ = environment.reset(seed=seed + episode)
        while True:
            action = int(policy(torch.tensor([state])).argmax(dim=-1).item())
            result = environment.step(action)
            state = result.observation
            if result.terminated or result.truncated:
                successes += int(result.terminated)
                break
    return successes / episodes


def train_reinforce(
    *,
    episodes: int = 350,
    seed: int = 42,
    gamma: float = 0.99,
    learning_rate: float = 0.03,
    use_running_baseline: bool = True,
    policy: DiscretePolicy | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    episode_offset: int = 0,
    running_baseline: float = 0.0,
) -> PolicyTrainResult:
    if episodes < 1 or episode_offset < 0:
        raise ValueError("episodes must be positive and episode_offset non-negative")
    environment = TinyGridWorld()
    if policy is None:
        torch.manual_seed(seed)
        policy = DiscretePolicy(environment.num_states, environment.num_actions)
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    elif optimizer is None:
        raise ValueError("resumed REINFORCE requires an optimizer")
    returns_history: list[float] = []
    losses: list[float] = []
    for episode in range(episodes):
        global_episode = episode_offset + episode
        generator = torch.Generator().manual_seed(seed + global_episode * 1_000_003)
        batch = collect_episode(
            policy,
            environment,
            seed=seed + global_episode,
            generator=generator,
        )
        returns = monte_carlo_returns(batch.rewards, gamma=gamma)
        logits = policy(batch.states)
        log_probs = (
            torch.log_softmax(logits, dim=-1)
            .gather(1, batch.actions.unsqueeze(1))
            .squeeze(1)
        )
        baseline = running_baseline if use_running_baseline else 0.0
        loss = reinforce_loss(log_probs, returns, baseline=baseline)
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        optimizer.step()
        episode_return = float(batch.rewards.sum())
        returns_history.append(episode_return)
        losses.append(float(loss.detach()))
        running_baseline = 0.9 * running_baseline + 0.1 * float(returns.mean())
    return PolicyTrainResult(
        model=policy,
        optimizer=optimizer,
        episode_returns=tuple(returns_history),
        losses=tuple(losses),
        success_rate=evaluate_policy(policy),
        running_baseline=running_baseline,
    )


def train_actor_critic(
    *,
    episodes: int = 250,
    seed: int = 42,
    gamma: float = 0.99,
    learning_rate: float = 0.03,
    model: DiscreteActorCritic | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    episode_offset: int = 0,
) -> PolicyTrainResult:
    if episodes < 1 or episode_offset < 0:
        raise ValueError("episodes must be positive and episode_offset non-negative")
    environment = TinyGridWorld()
    if model is None:
        torch.manual_seed(seed)
        model = DiscreteActorCritic(environment.num_states, environment.num_actions)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer is None:
        raise ValueError("resumed actor-critic requires an optimizer")
    returns_history: list[float] = []
    losses: list[float] = []
    for episode in range(episodes):
        global_episode = episode_offset + episode
        generator = torch.Generator().manual_seed(seed + global_episode * 1_000_003)
        batch = collect_episode(
            model.actor,
            environment,
            seed=seed + global_episode,
            generator=generator,
        )
        logits, values = model(batch.states)
        _, next_values = model(batch.next_states)
        output = actor_critic_loss(
            logits,
            batch.actions,
            values,
            batch.rewards,
            next_values,
            batch.terminated,
            gamma=gamma,
        )
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(output.loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        returns_history.append(float(batch.rewards.sum()))
        losses.append(float(output.loss.detach()))
    return PolicyTrainResult(
        model=model,
        optimizer=optimizer,
        episode_returns=tuple(returns_history),
        losses=tuple(losses),
        success_rate=evaluate_policy(model.actor),
        running_baseline=None,
    )
