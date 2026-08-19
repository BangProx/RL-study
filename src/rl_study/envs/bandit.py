"""A minimal Bernoulli multi-armed bandit with explicit regret."""

from __future__ import annotations

import random

from rl_study.types import StepResult


class BernoulliBandit:
    def __init__(
        self,
        probabilities: tuple[float, ...] = (0.1, 0.25, 0.4, 0.6, 0.8),
        *,
        horizon: int = 500,
    ) -> None:
        if len(probabilities) < 2:
            raise ValueError("a bandit requires at least two arms")
        if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
            raise ValueError("Bernoulli probabilities must be between 0 and 1")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.probabilities = probabilities
        self.horizon = horizon
        self._rng = random.Random()
        self._step = 0
        self._done = True

    @property
    def num_actions(self) -> int:
        return len(self.probabilities)

    @property
    def optimal_action(self) -> int:
        return max(range(self.num_actions), key=self.probabilities.__getitem__)

    def reset(self, *, seed: int) -> tuple[int, dict[str, object]]:
        self._rng.seed(seed)
        self._step = 0
        self._done = False
        return self._step, {"num_actions": self.num_actions, "horizon": self.horizon}

    def step(self, action: int) -> StepResult[int]:
        if self._done:
            raise RuntimeError("reset must be called before step or after truncation")
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("bandit action must be an integer")
        if not 0 <= action < self.num_actions:
            raise ValueError(f"action must be in [0, {self.num_actions})")
        reward = float(self._rng.random() < self.probabilities[action])
        expected_regret = max(self.probabilities) - self.probabilities[action]
        self._step += 1
        truncated = self._step >= self.horizon
        self._done = truncated
        return StepResult(
            observation=self._step,
            reward=reward,
            terminated=False,
            truncated=truncated,
            info={
                "action_probability": self.probabilities[action],
                "expected_regret": expected_regret,
                "optimal_action": self.optimal_action,
            },
        )
