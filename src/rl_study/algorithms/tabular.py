"""Dynamic programming, Monte Carlo, TD, and Q-learning reference code."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.envs import TinyGridWorld


@dataclass(frozen=True, slots=True)
class DynamicProgrammingResult:
    values: Tensor
    policy: Tensor
    iterations: int
    converged: bool


@dataclass(frozen=True, slots=True)
class TabularTrainResult:
    q_values: Tensor
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    success_rate: float


@dataclass(frozen=True, slots=True)
class PredictionTrainResult:
    values: Tensor
    counts: Tensor
    rmse_to_dynamic_programming: float


def _validate_model(
    transition_probabilities: Tensor, rewards: Tensor, terminal_states: Tensor
) -> tuple[int, int]:
    if transition_probabilities.ndim != 3:
        raise ValueError("transition_probabilities must be [S, A, S]")
    states, actions, next_states = transition_probabilities.shape
    if states != next_states or rewards.shape != transition_probabilities.shape:
        raise ValueError("transition and reward tensors must both be [S, A, S]")
    if terminal_states.shape != (states,) or terminal_states.dtype != torch.bool:
        raise ValueError("terminal_states must be bool [S]")
    if (
        not transition_probabilities.is_floating_point()
        or not rewards.is_floating_point()
    ):
        raise TypeError("transition probabilities and rewards must be floating")
    row_sums = transition_probabilities.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums))
    return states, actions


def value_iteration(
    transition_probabilities: Tensor,
    rewards: Tensor,
    terminal_states: Tensor,
    *,
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> DynamicProgrammingResult:
    """Apply the Bellman optimality backup until values converge."""

    states, _ = _validate_model(transition_probabilities, rewards, terminal_states)
    if not 0.0 <= gamma <= 1.0 or tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid gamma, tolerance, or max_iterations")
    values = torch.zeros(states, dtype=rewards.dtype, device=rewards.device)
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        bootstrap = (~terminal_states).to(rewards.dtype) * values
        q_values = (
            transition_probabilities * (rewards + gamma * bootstrap.view(1, 1, -1))
        ).sum(dim=-1)
        updated_values = q_values.max(dim=-1).values
        updated_values = torch.where(
            terminal_states, torch.zeros_like(updated_values), updated_values
        )
        if torch.max(torch.abs(updated_values - values)).item() < tolerance:
            converged = True
            values = updated_values
            break
        values = updated_values
    bootstrap = (~terminal_states).to(rewards.dtype) * values
    final_q = (
        transition_probabilities * (rewards + gamma * bootstrap.view(1, 1, -1))
    ).sum(dim=-1)
    policy = final_q.argmax(dim=-1)
    return DynamicProgrammingResult(values, policy, iterations, converged)


def policy_evaluation(
    transition_probabilities: Tensor,
    rewards: Tensor,
    terminal_states: Tensor,
    policy: Tensor,
    *,
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> Tensor:
    states, actions = _validate_model(
        transition_probabilities, rewards, terminal_states
    )
    if policy.shape != (states,) or policy.dtype != torch.int64:
        raise ValueError("policy must be int64 [S]")
    if torch.any(policy < 0) or torch.any(policy >= actions):
        raise ValueError("policy contains an invalid action")
    values = torch.zeros(states, dtype=rewards.dtype, device=rewards.device)
    state_indices = torch.arange(states, device=rewards.device)
    chosen_probabilities = transition_probabilities[state_indices, policy]
    chosen_rewards = rewards[state_indices, policy]
    for _ in range(max_iterations):
        bootstrap = (~terminal_states).to(rewards.dtype) * values
        updated = (chosen_probabilities * (chosen_rewards + gamma * bootstrap)).sum(
            dim=-1
        )
        updated = torch.where(terminal_states, torch.zeros_like(updated), updated)
        if torch.max(torch.abs(updated - values)).item() < tolerance:
            return updated
        values = updated
    raise RuntimeError("policy evaluation did not converge")


def policy_iteration(
    transition_probabilities: Tensor,
    rewards: Tensor,
    terminal_states: Tensor,
    *,
    gamma: float = 0.99,
    max_iterations: int = 1_000,
) -> DynamicProgrammingResult:
    states, _ = _validate_model(transition_probabilities, rewards, terminal_states)
    policy = torch.zeros(states, dtype=torch.int64, device=rewards.device)
    for iteration in range(1, max_iterations + 1):
        values = policy_evaluation(
            transition_probabilities,
            rewards,
            terminal_states,
            policy,
            gamma=gamma,
        )
        bootstrap = (~terminal_states).to(rewards.dtype) * values
        q_values = (
            transition_probabilities * (rewards + gamma * bootstrap.view(1, 1, -1))
        ).sum(dim=-1)
        improved = q_values.argmax(dim=-1)
        if torch.equal(improved, policy):
            return DynamicProgrammingResult(values, improved, iteration, True)
        policy = improved
    return DynamicProgrammingResult(values, policy, max_iterations, False)


def gridworld_model(environment: TinyGridWorld) -> tuple[Tensor, Tensor, Tensor]:
    probabilities = torch.zeros(
        environment.num_states, environment.num_actions, environment.num_states
    )
    rewards = torch.zeros_like(probabilities)
    terminal = torch.zeros(environment.num_states, dtype=torch.bool)
    terminal[environment.goal_state] = True
    for state in range(environment.num_states):
        for action in range(environment.num_actions):
            next_state, reward, _ = environment.transition(state, action)
            probabilities[state, action, next_state] = 1.0
            rewards[state, action, next_state] = reward
    return probabilities, rewards, terminal


def monte_carlo_returns(rewards: Tensor, *, gamma: float) -> Tensor:
    if rewards.ndim != 1 or not rewards.is_floating_point():
        raise ValueError("rewards must be floating [T]")
    result = torch.empty_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for time_index in range(len(rewards) - 1, -1, -1):
        running = rewards[time_index] + gamma * running
        result[time_index] = running
    return result


def mc_value_update(
    values: Tensor, states: Tensor, returns: Tensor, counts: Tensor
) -> tuple[Tensor, Tensor]:
    """First-visit incremental Monte Carlo state-value update."""

    if states.dtype != torch.int64:
        raise TypeError("states must be int64")
    if states.ndim != 1 or returns.ndim != 1:
        raise ValueError("states and returns must be one-dimensional")
    if states.shape != returns.shape:
        raise ValueError("states and returns must have the same shape")
    updated_values = values.clone()
    updated_counts = counts.clone()
    visited: set[int] = set()
    for state_tensor, return_tensor in zip(states, returns, strict=True):
        state = int(state_tensor)
        if state in visited:
            continue
        visited.add(state)
        updated_counts[state] += 1
        updated_values[state] += (
            return_tensor - updated_values[state]
        ) / updated_counts[state]
    return updated_values, updated_counts


def td_target(
    reward: Tensor, next_value: Tensor, terminated: Tensor, *, gamma: float
) -> Tensor:
    if reward.shape != next_value.shape or reward.shape != terminated.shape:
        raise ValueError("reward, next_value, and terminated must have the same shape")
    if terminated.dtype != torch.bool:
        raise TypeError("terminated must be bool")
    return reward + gamma * (~terminated).to(reward.dtype) * next_value


def td0_update(value: Tensor, target: Tensor, *, learning_rate: float) -> Tensor:
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    return value + learning_rate * (target - value)


def q_learning_target(
    reward: Tensor, next_q_values: Tensor, terminated: Tensor, *, gamma: float
) -> Tensor:
    if next_q_values.shape[:-1] != reward.shape or reward.shape != terminated.shape:
        raise ValueError("next_q_values must be reward shape plus action dimension")
    if terminated.dtype != torch.bool:
        raise TypeError("terminated must be bool")
    best_next = next_q_values.max(dim=-1).values
    return reward + gamma * (~terminated).to(reward.dtype) * best_next


def q_learning_update(
    q_value: Tensor, target: Tensor, *, learning_rate: float
) -> Tensor:
    return td0_update(q_value, target, learning_rate=learning_rate)


def _shortest_path_policy(environment: TinyGridWorld) -> Tensor:
    policy = torch.zeros(environment.num_states, dtype=torch.int64)
    for state in range(environment.num_states):
        _, column = divmod(state, environment.size)
        policy[state] = 1 if column < environment.size - 1 else 2
    return policy


def _prediction_reference(environment: TinyGridWorld, *, gamma: float) -> Tensor:
    transitions, rewards, terminal = gridworld_model(environment)
    return policy_evaluation(
        transitions,
        rewards,
        terminal,
        _shortest_path_policy(environment),
        gamma=gamma,
    )


def _prediction_rmse(values: Tensor, reference: Tensor, counts: Tensor) -> float:
    visited = counts > 0
    if not torch.any(visited):
        raise ValueError("cannot evaluate prediction before a state was visited")
    return float(torch.sqrt((values[visited] - reference[visited]).square().mean()))


def train_mc_prediction(
    *,
    episodes: int = 20,
    seed: int = 42,
    gamma: float = 0.99,
    initial_values: Tensor | None = None,
    initial_counts: Tensor | None = None,
    episode_offset: int = 0,
) -> PredictionTrainResult:
    if episodes < 1 or episode_offset < 0:
        raise ValueError("episodes must be positive and episode_offset non-negative")
    environment = TinyGridWorld()
    policy = _shortest_path_policy(environment)
    values = (
        torch.zeros(environment.num_states)
        if initial_values is None
        else initial_values.clone()
    )
    counts = (
        torch.zeros(environment.num_states)
        if initial_counts is None
        else initial_counts.clone()
    )
    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode_offset + episode)
        states: list[int] = []
        rewards: list[float] = []
        while True:
            states.append(state)
            result = environment.step(int(policy[state]))
            rewards.append(result.reward)
            state = result.observation
            if result.terminated or result.truncated:
                break
        returns = monte_carlo_returns(torch.tensor(rewards), gamma=gamma)
        values, counts = mc_value_update(
            values, torch.tensor(states, dtype=torch.int64), returns, counts
        )
    reference = _prediction_reference(environment, gamma=gamma)
    return PredictionTrainResult(
        values, counts, _prediction_rmse(values, reference, counts)
    )


def train_td0_prediction(
    *,
    episodes: int = 100,
    seed: int = 42,
    gamma: float = 0.99,
    learning_rate: float = 0.2,
    initial_values: Tensor | None = None,
    episode_offset: int = 0,
) -> PredictionTrainResult:
    if episodes < 1 or episode_offset < 0:
        raise ValueError("episodes must be positive and episode_offset non-negative")
    environment = TinyGridWorld()
    policy = _shortest_path_policy(environment)
    values = (
        torch.zeros(environment.num_states)
        if initial_values is None
        else initial_values.clone()
    )
    counts = torch.zeros(environment.num_states)
    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode_offset + episode)
        while True:
            result = environment.step(int(policy[state]))
            target = td_target(
                torch.tensor(result.reward),
                values[result.observation],
                torch.tensor(result.terminated),
                gamma=gamma,
            )
            values[state] = td0_update(
                values[state], target, learning_rate=learning_rate
            )
            counts[state] += 1
            state = result.observation
            if result.terminated or result.truncated:
                break
    reference = _prediction_reference(environment, gamma=gamma)
    return PredictionTrainResult(
        values, counts, _prediction_rmse(values, reference, counts)
    )


def evaluate_q_policy(
    q_values: Tensor, *, episodes: int = 50, seed: int = 1_000
) -> float:
    successes = 0
    for episode in range(episodes):
        environment = TinyGridWorld()
        state, _ = environment.reset(seed=seed + episode)
        while True:
            action = int(q_values[state].argmax())
            result = environment.step(action)
            state = result.observation
            if result.terminated or result.truncated:
                successes += int(result.terminated)
                break
    return successes / episodes


def train_q_learning(
    *,
    episodes: int = 300,
    seed: int = 42,
    gamma: float = 0.99,
    learning_rate: float = 0.2,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    initial_q_values: Tensor | None = None,
    episode_offset: int = 0,
    schedule_episodes: int | None = None,
) -> TabularTrainResult:
    if episodes < 1 or episode_offset < 0:
        raise ValueError("episodes must be positive and episode_offset non-negative")
    schedule_episodes = schedule_episodes or episodes
    if schedule_episodes < episode_offset + episodes:
        raise ValueError("schedule_episodes must cover offset plus requested episodes")
    environment = TinyGridWorld()
    q_values = (
        torch.zeros(environment.num_states, environment.num_actions)
        if initial_q_values is None
        else initial_q_values.detach().clone()
    )
    if q_values.shape != (environment.num_states, environment.num_actions):
        raise ValueError("initial_q_values has the wrong shape")
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    for episode in range(episodes):
        global_episode = episode_offset + episode
        rng = random.Random(seed + global_episode * 1_000_003)
        state, _ = environment.reset(seed=seed + global_episode)
        total_reward = 0.0
        epsilon_fraction = global_episode / max(schedule_episodes - 1, 1)
        epsilon = epsilon_start + epsilon_fraction * (epsilon_end - epsilon_start)
        while True:
            action = (
                rng.randrange(environment.num_actions)
                if rng.random() < epsilon
                else int(q_values[state].argmax())
            )
            result = environment.step(action)
            reward = torch.tensor(result.reward)
            terminated = torch.tensor(result.terminated)
            target = q_learning_target(
                reward,
                q_values[result.observation],
                terminated,
                gamma=gamma,
            )
            q_values[state, action] = q_learning_update(
                q_values[state, action], target, learning_rate=learning_rate
            )
            total_reward += result.reward
            state = result.observation
            if result.terminated or result.truncated:
                episode_returns.append(total_reward)
                step_count = result.info.get("steps")
                if not isinstance(step_count, int):
                    raise RuntimeError(
                        "GridWorld step info is missing an integer step count"
                    )
                episode_lengths.append(step_count)
                break
    success_rate = evaluate_q_policy(q_values)
    return TabularTrainResult(
        q_values=q_values,
        episode_returns=tuple(episode_returns),
        episode_lengths=tuple(episode_lengths),
        success_rate=success_rate,
    )
