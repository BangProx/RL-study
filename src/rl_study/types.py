"""Immutable tensor and environment contracts shared by algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

import torch
from torch import Tensor

ObservationT = TypeVar("ObservationT")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class StepResult(Generic[ObservationT]):
    observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(
            not (self.terminated and self.truncated),
            "step cannot be both terminated and truncated",
        )


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    terminated: Tensor
    truncated: Tensor
    behavior_logprobs: Tensor | None = None

    def __post_init__(self) -> None:
        batch = self.rewards.shape[0]
        for name in (
            "observations",
            "actions",
            "next_observations",
            "terminated",
            "truncated",
        ):
            value = getattr(self, name)
            _require(
                value.shape[0] == batch,
                f"{name} batch dimension does not match rewards",
            )
        _require(self.terminated.dtype == torch.bool, "terminated must have bool dtype")
        _require(self.truncated.dtype == torch.bool, "truncated must have bool dtype")
        _require(
            not torch.any(self.terminated & self.truncated).item(),
            "a transition cannot be both terminated and truncated",
        )
        if self.behavior_logprobs is not None:
            _require(
                self.behavior_logprobs.shape[0] == batch,
                "behavior_logprobs batch mismatch",
            )


@dataclass(frozen=True, slots=True)
class PreferenceBatch:
    prompt_ids: Tensor
    chosen_ids: Tensor
    rejected_ids: Tensor
    chosen_mask: Tensor
    rejected_mask: Tensor
    prompt_uids: tuple[str, ...]

    def __post_init__(self) -> None:
        batch = self.prompt_ids.shape[0]
        _require(
            len(self.prompt_uids) == batch, "prompt_uids length must equal batch size"
        )
        _require(
            self.chosen_ids.shape == self.chosen_mask.shape,
            "chosen_ids/mask shape mismatch",
        )
        _require(
            self.rejected_ids.shape == self.rejected_mask.shape,
            "rejected_ids/mask shape mismatch",
        )
        _require(self.chosen_ids.shape[0] == batch, "chosen batch mismatch")
        _require(self.rejected_ids.shape[0] == batch, "rejected batch mismatch")
        for name in ("prompt_ids", "chosen_ids", "rejected_ids"):
            _require(getattr(self, name).dtype == torch.int64, f"{name} must be int64")
        for name in ("chosen_mask", "rejected_mask"):
            _require(getattr(self, name).dtype == torch.bool, f"{name} must be bool")


@dataclass(frozen=True, slots=True)
class TokenTrajectoryBatch:
    prompt_ids: Tensor
    response_ids: Tensor
    attention_mask: Tensor
    action_mask: Tensor
    old_logprobs: Tensor | None
    reference_logprobs: Tensor | None
    values: Tensor | None
    rewards: Tensor
    advantages: Tensor | None
    returns: Tensor | None
    episode_ids: Tensor
    step_ids: Tensor | None
    terminated: Tensor
    truncated: Tensor

    def __post_init__(self) -> None:
        batch, response_length = self.response_ids.shape
        _require(self.prompt_ids.shape[0] == batch, "prompt batch mismatch")
        _require(
            self.action_mask.shape == (batch, response_length),
            "action_mask must match response_ids",
        )
        _require(self.attention_mask.shape[0] == batch, "attention_mask batch mismatch")
        _require(self.attention_mask.dtype == torch.bool, "attention_mask must be bool")
        _require(self.action_mask.dtype == torch.bool, "action_mask must be bool")
        _require(self.prompt_ids.dtype == torch.int64, "prompt_ids must be int64")
        _require(self.response_ids.dtype == torch.int64, "response_ids must be int64")
        _require(self.terminated.dtype == torch.bool, "terminated must be bool")
        _require(self.truncated.dtype == torch.bool, "truncated must be bool")
        for name in (
            "old_logprobs",
            "reference_logprobs",
            "values",
            "advantages",
            "returns",
        ):
            value = getattr(self, name)
            if value is not None:
                _require(
                    value.shape == (batch, response_length), f"{name} must be [B, T]"
                )
