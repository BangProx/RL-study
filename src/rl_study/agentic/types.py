"""Immutable contracts for step-level language-agent trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    argument: str
    description: str

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or not self.argument.isidentifier():
            raise ValueError("tool and argument names must be identifiers")


@dataclass(frozen=True, slots=True)
class AgentObservation:
    episode_id: str
    step_id: int
    text: str
    visible_tools: tuple[ToolSchema, ...]

    def __post_init__(self) -> None:
        if not self.episode_id or self.step_id < 0 or not self.text:
            raise ValueError(
                "observation requires episode_id, non-negative step, and text"
            )


@dataclass(frozen=True, slots=True)
class AgentAction:
    kind: Literal["tool", "final", "invalid"]
    raw_text: str
    tool_name: str | None = None
    argument_name: str | None = None
    argument_value: str | None = None
    final_answer: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.raw_text:
            raise ValueError("action text cannot be empty")
        if self.kind == "tool":
            if not all((self.tool_name, self.argument_name, self.argument_value)):
                raise ValueError("tool action requires a tool and one string argument")
        elif self.kind == "final":
            if not self.final_answer:
                raise ValueError("final action requires a non-empty answer")
        elif not self.error:
            raise ValueError("invalid action requires an error")


@dataclass(frozen=True, slots=True)
class ToolOutput:
    status: Literal["not_called", "success", "invalid", "timeout"]
    text: str
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class AgentEnvResult:
    observation: AgentObservation | None
    tool_output: ToolOutput
    process_reward: float
    outcome_reward: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if self.terminated and self.truncated:
            raise ValueError("an environment result cannot terminate and truncate")
        if (self.terminated or self.truncated) != (self.observation is None):
            raise ValueError("only terminal/truncated results omit an observation")
        if self.outcome_reward != 0.0 and not self.terminated:
            raise ValueError("outcome reward is only defined on termination")


@dataclass(frozen=True, slots=True)
class AgentStep:
    observation: AgentObservation
    context_token_ids: tuple[int, ...]
    action: AgentAction
    action_token_ids: tuple[int, ...]
    candidate_action_token_ids: tuple[tuple[int, ...], ...]
    chosen_candidate_index: int
    behavior_logprob: float
    tool_output: ToolOutput
    process_reward: float
    outcome_reward: float
    next_observation: AgentObservation | None
    terminated: bool
    truncated: bool
    policy_version: int

    def __post_init__(self) -> None:
        if not self.context_token_ids or not self.action_token_ids:
            raise ValueError("agent steps must preserve original context/action tokens")
        if not self.candidate_action_token_ids:
            raise ValueError("agent step requires its rollout action space")
        if not 0 <= self.chosen_candidate_index < len(self.candidate_action_token_ids):
            raise ValueError("chosen candidate index is outside the action space")
        if self.action_token_ids != self.candidate_action_token_ids[
            self.chosen_candidate_index
        ]:
            raise ValueError("chosen action tokens must match the selected candidate")
        if not math.isfinite(self.behavior_logprob):
            raise ValueError("behavior_logprob must be finite")
        if self.terminated and self.truncated:
            raise ValueError("a step cannot be both terminated and truncated")
        if (self.terminated or self.truncated) != (self.next_observation is None):
            raise ValueError("only terminal/truncated steps omit next_observation")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if self.outcome_reward != 0.0 and not self.terminated:
            raise ValueError("outcome reward is only defined on termination")

    @property
    def reward(self) -> float:
        """The unique sum used by return-based credit (no reward is counted twice)."""

        return self.process_reward + self.outcome_reward


@dataclass(frozen=True, slots=True)
class AgentTrajectory:
    episode_id: str
    steps: tuple[AgentStep, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("trajectory requires at least one step")
        for index, step in enumerate(self.steps):
            if step.observation.episode_id != self.episode_id:
                raise ValueError("trajectory episode IDs must agree")
            if step.observation.step_id != index:
                raise ValueError("trajectory step IDs must be contiguous from zero")
            if index < len(self.steps) - 1 and (step.terminated or step.truncated):
                raise ValueError("terminal step must be last")
        if not (self.steps[-1].terminated or self.steps[-1].truncated):
            raise ValueError("trajectory must end by termination or truncation")

    @property
    def policy_version(self) -> int:
        versions = {step.policy_version for step in self.steps}
        if len(versions) != 1:
            raise ValueError("one trajectory cannot mix policy versions")
        return next(iter(versions))

    @property
    def succeeded(self) -> bool:
        return self.steps[-1].outcome_reward > 0.0
