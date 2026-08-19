"""Small DQN with replay and a detached target network."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rl_study.envs import TinyGridWorld


@dataclass(frozen=True, slots=True)
class DQNBatch:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    terminated: Tensor
    truncated: Tensor

    def __post_init__(self) -> None:
        batch = self.states.shape[0]
        for name in ("actions", "rewards", "next_states", "terminated", "truncated"):
            if getattr(self, name).shape != (batch,):
                raise ValueError(f"{name} must be [B]")
        if self.states.dtype != torch.int64 or self.next_states.dtype != torch.int64:
            raise TypeError("states and next_states must be int64")
        if self.actions.dtype != torch.int64:
            raise TypeError("actions must be int64")
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must be bool")


@dataclass(frozen=True, slots=True)
class DQNLossOutput:
    loss: Tensor
    predictions: Tensor
    targets: Tensor


@dataclass(frozen=True, slots=True)
class DQNTrainResult:
    policy: DQNNetwork
    target: DQNNetwork
    optimizer: torch.optim.Optimizer
    replay_items: tuple[tuple[int, int, float, int, bool, bool], ...]
    environment_steps: int
    episode_returns: tuple[float, ...]
    losses: tuple[float, ...]
    success_rate: float


class DQNNetwork(nn.Module):
    def __init__(
        self, num_states: int, num_actions: int, *, hidden_size: int = 32
    ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.network = nn.Sequential(
            nn.Linear(num_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, states: Tensor) -> Tensor:
        if states.dtype != torch.int64:
            raise TypeError("states must be int64")
        dtype = next(self.parameters()).dtype
        one_hot = torch.nn.functional.one_hot(states, num_classes=self.num_states).to(
            dtype=dtype
        )
        output: Tensor = self.network(one_hot)
        return output


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        items: tuple[tuple[int, int, float, int, bool, bool], ...] = (),
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._items: deque[tuple[int, int, float, int, bool, bool]] = deque(
            items, maxlen=capacity
        )

    def __len__(self) -> int:
        return len(self._items)

    def append(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._items.append((state, action, reward, next_state, terminated, truncated))

    @property
    def items(self) -> tuple[tuple[int, int, float, int, bool, bool], ...]:
        return tuple(self._items)

    def sample(self, batch_size: int, *, seed: int) -> DQNBatch:
        if batch_size < 1 or batch_size > len(self._items):
            raise ValueError("batch_size must be between 1 and current buffer size")
        rows = random.Random(seed).sample(list(self._items), batch_size)
        states, actions, rewards, next_states, terminated, truncated = zip(
            *rows, strict=True
        )
        return DQNBatch(
            states=torch.tensor(states, dtype=torch.int64),
            actions=torch.tensor(actions, dtype=torch.int64),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            next_states=torch.tensor(next_states, dtype=torch.int64),
            terminated=torch.tensor(terminated, dtype=torch.bool),
            truncated=torch.tensor(truncated, dtype=torch.bool),
        )


def dqn_loss(
    policy: DQNNetwork,
    target: DQNNetwork,
    batch: DQNBatch,
    *,
    gamma: float,
    double_dqn: bool = False,
) -> DQNLossOutput:
    """Compute TD loss; only MDP termination disables bootstrap."""

    predictions = policy(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target_q_values = target(batch.next_states)
        if double_dqn:
            best_actions = policy(batch.next_states).argmax(dim=-1)
            next_values = target_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(
                1
            )
        else:
            next_values = target_q_values.max(dim=-1).values
        targets = (
            batch.rewards
            + gamma * (~batch.terminated).to(batch.rewards.dtype) * next_values
        )
    loss = torch.nn.functional.smooth_l1_loss(predictions, targets)
    return DQNLossOutput(loss=loss, predictions=predictions, targets=targets)


def hard_update(target: nn.Module, policy: nn.Module) -> None:
    target.load_state_dict(policy.state_dict())
    target.requires_grad_(False)


@torch.no_grad()
def soft_update(target: nn.Module, policy: nn.Module, *, tau: float) -> None:
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    for target_parameter, policy_parameter in zip(
        target.parameters(), policy.parameters(), strict=True
    ):
        target_parameter.lerp_(policy_parameter, tau)


@torch.no_grad()
def evaluate_dqn(policy: DQNNetwork, *, episodes: int = 50, seed: int = 1_000) -> float:
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


def train_dqn(
    *,
    episodes: int = 250,
    seed: int = 42,
    gamma: float = 0.99,
    learning_rate: float = 3e-3,
    replay_capacity: int = 1_024,
    batch_size: int = 32,
    target_sync_steps: int | None = 40,
    policy: DQNNetwork | None = None,
    target: DQNNetwork | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    replay_items: tuple[tuple[int, int, float, int, bool, bool], ...] = (),
    environment_steps: int = 0,
    episode_offset: int = 0,
    schedule_episodes: int | None = None,
) -> DQNTrainResult:
    if episodes < 1 or batch_size < 1 or episode_offset < 0:
        raise ValueError("episodes/batch must be positive and offset non-negative")
    schedule_episodes = schedule_episodes or episodes
    if schedule_episodes < episode_offset + episodes:
        raise ValueError("schedule_episodes must cover offset plus requested episodes")
    environment = TinyGridWorld()
    if policy is None:
        torch.manual_seed(seed)
        policy = DQNNetwork(environment.num_states, environment.num_actions)
        target = DQNNetwork(environment.num_states, environment.num_actions)
        hard_update(target, policy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    elif target is None or optimizer is None:
        raise ValueError("resumed DQN requires target and optimizer")
    replay = ReplayBuffer(replay_capacity, items=replay_items)
    episode_returns: list[float] = []
    losses: list[float] = []
    for episode in range(episodes):
        global_episode = episode_offset + episode
        rng = random.Random(seed + global_episode * 1_000_003)
        state, _ = environment.reset(seed=seed + global_episode)
        total_reward = 0.0
        epsilon = 1.0 + (global_episode / max(schedule_episodes - 1, 1)) * (0.05 - 1.0)
        while True:
            if rng.random() < epsilon:
                action = rng.randrange(environment.num_actions)
            else:
                with torch.no_grad():
                    action = int(policy(torch.tensor([state])).argmax(dim=-1).item())
            result = environment.step(action)
            replay.append(
                state,
                action,
                result.reward,
                result.observation,
                result.terminated,
                result.truncated,
            )
            state = result.observation
            total_reward += result.reward
            environment_steps += 1
            if len(replay) >= batch_size:
                output = dqn_loss(
                    policy,
                    target,
                    replay.sample(
                        batch_size, seed=seed + environment_steps * 1_000_033
                    ),
                    gamma=gamma,
                )
                optimizer.zero_grad(set_to_none=True)
                torch.autograd.backward(output.loss)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
                optimizer.step()
                losses.append(float(output.loss.detach()))
                if (
                    target_sync_steps is not None
                    and environment_steps % target_sync_steps == 0
                ):
                    hard_update(target, policy)
            if result.terminated or result.truncated:
                episode_returns.append(total_reward)
                break
    return DQNTrainResult(
        policy=policy,
        target=target,
        optimizer=optimizer,
        replay_items=replay.items,
        environment_steps=environment_steps,
        episode_returns=tuple(episode_returns),
        losses=tuple(losses),
        success_rate=evaluate_dqn(policy),
    )
