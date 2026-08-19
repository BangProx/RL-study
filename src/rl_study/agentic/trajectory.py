"""Rollout, validation, and masked policy updates for offline tool agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

from rl_study.agentic.credit import CreditMode, assign_step_credit
from rl_study.agentic.envs import OfflineAgentEnv
from rl_study.agentic.parser import parse_action
from rl_study.agentic.policy import candidate_log_probs, select_action
from rl_study.agentic.types import AgentStep, AgentTrajectory
from rl_study.errors import RetokenizationDriftError, StaleTrajectoryError
from rl_study.models.tiny_lm import TinyCausalLM
from rl_study.models.tiny_tokenizer import TinyTokenizer


@dataclass(frozen=True, slots=True)
class PolicyUpdate:
    loss: float
    step_credits: tuple[float, ...]
    model_forwards: int


def rollout_episode(
    model: TinyCausalLM,
    tokenizer: TinyTokenizer,
    env: OfflineAgentEnv,
    *,
    generator: torch.Generator,
    policy_version: int,
    task_index: int | None = None,
    greedy: bool = False,
) -> tuple[AgentTrajectory, int, int]:
    observation = env.reset(task_index=task_index)
    steps: list[AgentStep] = []
    generated_tokens = 0
    model_forwards = 0
    while True:
        selection = select_action(
            model,
            tokenizer,
            observation_text=observation.text,
            candidates=env.candidate_actions(),
            generator=generator,
            greedy=greedy,
        )
        model_forwards += 1
        generated_tokens += len(selection.action_token_ids)
        action = parse_action(selection.raw_text, observation.visible_tools)
        result = env.step(action)
        steps.append(
            AgentStep(
                observation=observation,
                context_token_ids=selection.context_token_ids,
                action=action,
                action_token_ids=selection.action_token_ids,
                candidate_action_token_ids=selection.candidate_action_token_ids,
                chosen_candidate_index=selection.chosen_index,
                behavior_logprob=selection.behavior_logprob,
                tool_output=result.tool_output,
                process_reward=result.process_reward,
                outcome_reward=result.outcome_reward,
                next_observation=result.observation,
                terminated=result.terminated,
                truncated=result.truncated,
                policy_version=policy_version,
            )
        )
        if result.terminated or result.truncated:
            break
        if result.observation is None:  # pragma: no cover - protected by invariant
            raise RuntimeError("continuing environment result needs an observation")
        observation = result.observation
    trajectory = AgentTrajectory(episode_id=env.episode_id, steps=tuple(steps))
    return trajectory, generated_tokens, model_forwards


def validate_trajectory(
    trajectory: AgentTrajectory,
    tokenizer: TinyTokenizer,
    *,
    current_policy_version: int,
    maximum_policy_lag: int = 0,
) -> None:
    lag = current_policy_version - trajectory.policy_version
    if lag < 0 or lag > maximum_policy_lag:
        raise StaleTrajectoryError(
            "trajectory policy version is stale or from the future: "
            f"rollout={trajectory.policy_version}, current={current_policy_version}, "
            f"maximum_lag={maximum_policy_lag}"
        )
    for step in trajectory.steps:
        encoded = tuple(
            tokenizer.encode(step.action.raw_text, add_bos=False, add_eos=True)
        )
        if encoded != step.action_token_ids:
            raise RetokenizationDriftError(
                f"action token drift in {trajectory.episode_id} step "
                f"{step.observation.step_id}"
            )


def update_policy(
    model: TinyCausalLM,
    tokenizer: TinyTokenizer,
    optimizer: torch.optim.Optimizer,
    trajectory: AgentTrajectory,
    *,
    current_policy_version: int,
    credit_mode: CreditMode,
    gamma: float,
) -> PolicyUpdate:
    validate_trajectory(
        trajectory,
        tokenizer,
        current_policy_version=current_policy_version,
    )
    credits = assign_step_credit(trajectory, mode=credit_mode, gamma=gamma)
    selected_log_probs = []
    for step in trajectory.steps:
        log_probs = candidate_log_probs(
            model,
            context_token_ids=step.context_token_ids,
            candidate_action_token_ids=step.candidate_action_token_ids,
        )
        selected_log_probs.append(log_probs[step.chosen_candidate_index])
    stacked = torch.stack(selected_log_probs)
    credit_tensor = torch.tensor(credits, dtype=stacked.dtype, device=stacked.device)
    loss = -(stacked * credit_tensor).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("agent policy loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return PolicyUpdate(
        loss=float(loss.detach()),
        step_credits=credits,
        model_forwards=len(trajectory.steps),
    )


T = TypeVar("T")


def tool_is_deterministic(call: Callable[[], T], *, repetitions: int = 3) -> bool:
    """Detect equal-input tool drift before accepting it into an offline benchmark."""

    if repetitions < 2:
        raise ValueError("determinism check requires at least two repetitions")
    values = [call() for _ in range(repetitions)]
    return all(value == values[0] for value in values[1:])
