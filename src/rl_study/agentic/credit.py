"""Explicit step-credit baselines for multi-turn trajectories."""

from __future__ import annotations

from typing import Literal

from rl_study.agentic.types import AgentTrajectory

CreditMode = Literal["broadcast_outcome", "discounted_returns"]


def assign_step_credit(
    trajectory: AgentTrajectory,
    *,
    mode: CreditMode,
    gamma: float,
) -> tuple[float, ...]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if mode == "broadcast_outcome":
        terminal_outcome = trajectory.steps[-1].outcome_reward
        return (terminal_outcome,) * len(trajectory.steps)
    if mode != "discounted_returns":
        raise ValueError(f"unknown credit mode: {mode}")
    running = 0.0
    returns: list[float] = []
    for step in reversed(trajectory.steps):
        running = step.reward + gamma * running
        returns.append(running)
    return tuple(reversed(returns))
