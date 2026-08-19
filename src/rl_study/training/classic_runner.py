"""Train/evaluate/checkpoint lifecycle for the five classic learning agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from rl_study.algorithms.dqn import (
    DQNNetwork,
    evaluate_dqn,
    hard_update,
    train_dqn,
)
from rl_study.algorithms.policy_gradient import (
    DiscreteActorCritic,
    DiscretePolicy,
    evaluate_policy,
    train_actor_critic,
    train_reinforce,
)
from rl_study.algorithms.ppo import train_ppo
from rl_study.algorithms.tabular import evaluate_q_policy, train_q_learning
from rl_study.config import ExperimentConfig
from rl_study.errors import ConfigError
from rl_study.training.checkpoint import load_checkpoint, save_checkpoint

CLASSIC_ALGORITHMS = frozenset(
    {"q_learning", "dqn", "reinforce", "actor_critic", "ppo"}
)


class TabularQState(nn.Module):
    q_values: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("q_values", torch.zeros(16, 4))


class DQNState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = DQNNetwork(16, 4)
        self.target = DQNNetwork(16, 4)
        hard_update(self.target, self.policy)


@dataclass(frozen=True, slots=True)
class ClassicRunResult:
    algorithm: str
    step: int
    success_rate: float
    episode_returns: tuple[float, ...]
    losses: tuple[float, ...]
    checkpoint: Path
    model: nn.Module


def _new_state(
    algorithm: str, *, seed: int
) -> tuple[nn.Module, torch.optim.Optimizer | None]:
    torch.manual_seed(seed)
    if algorithm == "q_learning":
        return TabularQState(), None
    if algorithm == "dqn":
        state = DQNState()
        return state, torch.optim.AdamW(state.policy.parameters(), lr=3e-3)
    if algorithm == "reinforce":
        policy = DiscretePolicy(16, 4)
        return policy, torch.optim.Adam(policy.parameters(), lr=0.03)
    if algorithm in {"actor_critic", "ppo"}:
        actor_critic = DiscreteActorCritic(16, 4)
        return actor_critic, torch.optim.Adam(
            actor_critic.parameters(), lr=0.03 if algorithm == "actor_critic" else 0.02
        )
    raise ConfigError(f"{algorithm!r} is not a classic trainer")


def _dqn_replay_items(
    value: object,
) -> tuple[tuple[int, int, float, int, bool, bool], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError("DQN checkpoint replay_items must be a list")
    parsed: list[tuple[int, int, float, int, bool, bool]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 6:
            raise ConfigError("DQN replay row must contain six fields")
        state, action, reward, next_state, terminated, truncated = row
        if not (
            isinstance(state, int)
            and isinstance(action, int)
            and isinstance(reward, (int, float))
            and isinstance(next_state, int)
            and isinstance(terminated, bool)
            and isinstance(truncated, bool)
        ):
            raise ConfigError("DQN replay row has invalid field types")
        parsed.append((state, action, float(reward), next_state, terminated, truncated))
    return tuple(parsed)


def _cursor_int(value: object, *, name: str, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"checkpoint {name} must be an integer")
    return value


def _cursor_float(value: object, *, name: str, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"checkpoint {name} must be numeric")
    return float(value)


def train_classic(
    config: ExperimentConfig,
    *,
    output_root: str | Path | None = None,
    resume: str | Path | None = None,
    stop_after: int | None = None,
) -> ClassicRunResult:
    algorithm = config.algorithm.name
    if config.profile != "toy" or algorithm not in CLASSIC_ALGORITHMS:
        raise ConfigError("classic runner requires toy profile and a classic algorithm")
    if config.data.id != "tiny_grid_world":
        raise ConfigError("classic runner requires data.id=tiny_grid_world")
    target_step = config.training.steps if stop_after is None else stop_after
    if not 1 <= target_step <= config.training.steps:
        raise ConfigError("stop_after must be between 1 and configured training.steps")

    model, optimizer = _new_state(algorithm, seed=config.training.seed)
    episode_offset = 0
    data_cursor: dict[str, object] = {}
    prior_metrics: list[dict[str, object]] = []
    if resume is not None:
        loaded = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            config=config,
        )
        episode_offset = loaded.step
        data_cursor = loaded.data_cursor
        prior_metrics = loaded.metrics
    if episode_offset >= target_step:
        raise ConfigError(
            f"checkpoint step {episode_offset} is not before target step {target_step}"
        )
    episodes = target_step - episode_offset
    losses: tuple[float, ...] = ()

    if algorithm == "q_learning":
        q_state = cast(TabularQState, model)
        q_result = train_q_learning(
            episodes=episodes,
            seed=config.training.seed,
            gamma=config.algorithm.gamma,
            initial_q_values=q_state.q_values,
            episode_offset=episode_offset,
            schedule_episodes=config.training.steps,
        )
        q_state.q_values.copy_(q_result.q_values)
        success_rate = q_result.success_rate
        episode_returns = q_result.episode_returns
    elif algorithm == "dqn":
        dqn_state = cast(DQNState, model)
        if optimizer is None:
            raise RuntimeError("DQN optimizer is missing")
        dqn_result = train_dqn(
            episodes=episodes,
            seed=config.training.seed,
            gamma=config.algorithm.gamma,
            policy=dqn_state.policy,
            target=dqn_state.target,
            optimizer=optimizer,
            replay_items=_dqn_replay_items(data_cursor.get("replay_items")),
            environment_steps=_cursor_int(
                data_cursor.get("environment_steps"), name="environment_steps"
            ),
            episode_offset=episode_offset,
            schedule_episodes=config.training.steps,
        )
        data_cursor["replay_items"] = dqn_result.replay_items
        data_cursor["environment_steps"] = dqn_result.environment_steps
        success_rate = dqn_result.success_rate
        episode_returns = dqn_result.episode_returns
        losses = dqn_result.losses
    elif algorithm == "reinforce":
        policy = cast(DiscretePolicy, model)
        if optimizer is None:
            raise RuntimeError("REINFORCE optimizer is missing")
        reinforce_result = train_reinforce(
            episodes=episodes,
            seed=config.training.seed,
            gamma=config.algorithm.gamma,
            policy=policy,
            optimizer=optimizer,
            episode_offset=episode_offset,
            running_baseline=_cursor_float(
                data_cursor.get("running_baseline"), name="running_baseline"
            ),
        )
        data_cursor["running_baseline"] = reinforce_result.running_baseline
        success_rate = reinforce_result.success_rate
        episode_returns = reinforce_result.episode_returns
        losses = reinforce_result.losses
    elif algorithm == "actor_critic":
        actor_critic = cast(DiscreteActorCritic, model)
        if optimizer is None:
            raise RuntimeError("actor-critic optimizer is missing")
        actor_critic_result = train_actor_critic(
            episodes=episodes,
            seed=config.training.seed,
            gamma=config.algorithm.gamma,
            model=actor_critic,
            optimizer=optimizer,
            episode_offset=episode_offset,
        )
        success_rate = actor_critic_result.success_rate
        episode_returns = actor_critic_result.episode_returns
        losses = actor_critic_result.losses
    else:
        actor_critic = cast(DiscreteActorCritic, model)
        if optimizer is None:
            raise RuntimeError("PPO optimizer is missing")
        ppo_result = train_ppo(
            episodes=episodes,
            seed=config.training.seed,
            gamma=config.algorithm.gamma,
            clip_low=config.algorithm.clip_low,
            clip_high=config.algorithm.clip_high,
            model=actor_critic,
            optimizer=optimizer,
            episode_offset=episode_offset,
        )
        success_rate = ppo_result.success_rate
        episode_returns = ppo_result.episode_returns
        losses = ppo_result.policy_losses

    metrics = list(prior_metrics)
    metrics.extend(
        {
            "episode": episode_offset + index + 1,
            "return": value,
            "algorithm": algorithm,
        }
        for index, value in enumerate(episode_returns)
    )
    data_cursor["episode"] = target_step
    root = Path(output_root or config.output.root)
    checkpoint = (
        root
        / f"{algorithm}-seed{config.training.seed}"
        / f"checkpoint-{target_step:06d}"
    )
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        step=target_step,
        data_cursor=data_cursor,
        metrics=metrics,
    )
    return ClassicRunResult(
        algorithm=algorithm,
        step=target_step,
        success_rate=success_rate,
        episode_returns=episode_returns,
        losses=losses,
        checkpoint=checkpoint,
        model=model,
    )


def _config_from_checkpoint(checkpoint: Path) -> ExperimentConfig:
    try:
        payload = json.loads(
            (checkpoint / "config.resolved.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot read checkpoint config: {error}") from error
    return ExperimentConfig.from_mapping(payload)


def evaluate_classic_checkpoint(
    checkpoint: str | Path,
) -> tuple[str, int, float]:
    source = Path(checkpoint)
    config = _config_from_checkpoint(source)
    algorithm = config.algorithm.name
    model, optimizer = _new_state(algorithm, seed=config.training.seed)
    loaded = load_checkpoint(
        source,
        model=model,
        optimizer=optimizer,
        config=config,
        restore_rng=False,
    )
    if algorithm == "q_learning":
        success_rate = evaluate_q_policy(cast(TabularQState, model).q_values)
    elif algorithm == "dqn":
        success_rate = evaluate_dqn(cast(DQNState, model).policy)
    elif algorithm == "reinforce":
        success_rate = evaluate_policy(cast(DiscretePolicy, model))
    else:
        success_rate = evaluate_policy(cast(DiscreteActorCritic, model).actor)
    return algorithm, loaded.step, success_rate
