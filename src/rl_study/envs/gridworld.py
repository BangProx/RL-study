"""Small deterministic GridWorld that keeps termination and truncation separate."""

from __future__ import annotations

from enum import IntEnum

from rl_study.types import StepResult


class GridAction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


_DELTAS = {
    GridAction.UP: (-1, 0),
    GridAction.RIGHT: (0, 1),
    GridAction.DOWN: (1, 0),
    GridAction.LEFT: (0, -1),
}


class TinyGridWorld:
    def __init__(self, *, size: int = 4, max_steps: int = 32) -> None:
        if size < 2:
            raise ValueError("size must be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.size = size
        self.max_steps = max_steps
        self.start_state = 0
        self.goal_state = size * size - 1
        self._state = self.start_state
        self._steps = 0
        self._done = True

    @property
    def num_states(self) -> int:
        return self.size * self.size

    @property
    def num_actions(self) -> int:
        return len(GridAction)

    @property
    def state(self) -> int:
        return self._state

    def _position(self, state: int) -> tuple[int, int]:
        return divmod(state, self.size)

    def reset(self, *, seed: int) -> tuple[int, dict[str, object]]:
        del seed  # The environment is deterministic; seed remains part of the contract.
        self._state = self.start_state
        self._steps = 0
        self._done = False
        return self._state, {"goal_state": self.goal_state, "max_steps": self.max_steps}

    def transition(
        self, state: int, action: int | GridAction
    ) -> tuple[int, float, bool]:
        if not 0 <= state < self.num_states:
            raise ValueError("state is outside the grid")
        try:
            parsed_action = GridAction(action)
        except ValueError as error:
            raise ValueError(
                f"action must be one of {[int(item) for item in GridAction]}"
            ) from error
        if state == self.goal_state:
            return state, 0.0, True
        row, column = self._position(state)
        row_delta, column_delta = _DELTAS[parsed_action]
        new_row = min(max(row + row_delta, 0), self.size - 1)
        new_column = min(max(column + column_delta, 0), self.size - 1)
        next_state = new_row * self.size + new_column
        terminated = next_state == self.goal_state
        reward = 1.0 if terminated else -0.01
        return next_state, reward, terminated

    def step(self, action: int | GridAction) -> StepResult[int]:
        if self._done:
            raise RuntimeError("reset must be called before step or after episode end")
        next_state, reward, terminated = self.transition(self._state, action)
        self._state = next_state
        self._steps += 1
        truncated = self._steps >= self.max_steps and not terminated
        self._done = terminated or truncated
        return StepResult(
            observation=next_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"steps": self._steps, "success": terminated},
        )

    def render_ascii(self) -> str:
        cells: list[str] = []
        for state in range(self.num_states):
            if state == self._state:
                cells.append("A")
            elif state == self.goal_state:
                cells.append("G")
            else:
                cells.append(".")
        rows = [
            " ".join(cells[index : index + self.size])
            for index in range(0, len(cells), self.size)
        ]
        return "\n".join(rows)
