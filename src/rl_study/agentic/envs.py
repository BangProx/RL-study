"""Deterministic tool environments that never execute shell or network commands."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from typing import ClassVar, Protocol

from rl_study.agentic.parser import parse_action
from rl_study.agentic.types import (
    AgentAction,
    AgentEnvResult,
    AgentObservation,
    ToolOutput,
    ToolSchema,
)


class OfflineAgentEnv(Protocol):
    @property
    def episode_id(self) -> str: ...

    def reset(self, *, task_index: int | None = None) -> AgentObservation: ...

    def step(self, action: AgentAction) -> AgentEnvResult: ...

    def candidate_actions(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class _Task:
    uid: str
    question: str
    answer: str
    query: str


def _calculate(expression: str) -> str:
    """Evaluate arithmetic through an AST allowlist; Python ``eval`` is never used."""

    if len(expression) > 32:
        raise ValueError("expression_too_long")
    try:
        root = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid_expression") from error

    def visit(node: ast.AST, *, depth: int = 0) -> float:
        if depth > 8:
            raise ValueError("expression_too_deep")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth=depth + 1)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            value = float(node.value)
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = visit(node.left, depth=depth + 1)
            right = visit(node.right, depth=depth + 1)
            try:
                if isinstance(node.op, ast.Add):
                    value = left + right
                elif isinstance(node.op, ast.Sub):
                    value = left - right
                elif isinstance(node.op, ast.Mult):
                    value = left * right
                else:
                    value = left / right
            except ZeroDivisionError as error:
                raise ValueError("division_by_zero") from error
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            operand = visit(node.operand, depth=depth + 1)
            value = operand if isinstance(node.op, ast.UAdd) else -operand
        else:
            raise ValueError("operator_not_allowed")
        if not math.isfinite(value) or abs(value) > 1e9:
            raise ValueError("numeric_limit_exceeded")
        return value

    result = visit(root)
    return str(int(result)) if result.is_integer() else f"{result:.8g}"


class _BaseToolEnv:
    schema: ToolSchema
    tasks: tuple[_Task, ...]
    name: str

    def __init__(
        self, *, seed: int = 42, max_steps: int = 3, episode_offset: int = 0
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if episode_offset < 0:
            raise ValueError("episode_offset must be non-negative")
        self._seed = seed
        self._max_steps = max_steps
        self._reset_count = episode_offset
        self._task: _Task | None = None
        self._episode_id = "not-reset"
        self._step_id = 0
        self._transcript = ""
        self._queries: list[str] = []
        self._done = True

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def task(self) -> _Task:
        if self._task is None:
            raise RuntimeError("reset must be called before using the environment")
        return self._task

    def reset(self, *, task_index: int | None = None) -> AgentObservation:
        index = (
            (self._seed + self._reset_count) % len(self.tasks)
            if task_index is None
            else task_index
        )
        if not 0 <= index < len(self.tasks):
            raise IndexError("task_index is outside the task set")
        self._task = self.tasks[index]
        self._episode_id = f"{self.name}-{self._reset_count:06d}-{self.task.uid}"
        self._reset_count += 1
        self._step_id = 0
        self._queries = []
        self._done = False
        self._transcript = f"Q:{self.task.question}"
        return self._observation()

    def _observation(self) -> AgentObservation:
        return AgentObservation(
            episode_id=self.episode_id,
            step_id=self._step_id,
            text=self._transcript,
            visible_tools=(self.schema,),
        )

    def parse_and_step(self, raw_text: str) -> AgentEnvResult:
        return self.step(parse_action(raw_text, (self.schema,)))

    def step(self, action: AgentAction) -> AgentEnvResult:
        if self._done:
            raise RuntimeError("reset is required after termination or truncation")
        tool_output = ToolOutput(status="not_called", text="")
        process_reward = 0.0
        outcome_reward = 0.0
        terminated = False
        force_truncated = False
        if action.kind == "invalid":
            process_reward = -0.2
            tool_output = ToolOutput(status="invalid", text=action.error or "invalid")
        elif action.kind == "final":
            terminated = True
            outcome_reward = self._score_final(action.final_answer or "")
        else:
            query = action.argument_value or ""
            if query == "__timeout__":
                tool_output = ToolOutput(
                    status="timeout", text="tool_timeout", tool_name=self.schema.name
                )
                process_reward = -0.2
                force_truncated = True
            else:
                tool_output, process_reward = self._call_tool(query)
                if query in self._queries:
                    process_reward = -0.25
                self._queries.append(query)

        self._transcript += f"\nA:{action.raw_text}"
        if tool_output.status != "not_called":
            self._transcript += f"\nO:{tool_output.text}"
        self._step_id += 1
        truncated = force_truncated or (
            not terminated and self._step_id >= self._max_steps
        )
        self._done = terminated or truncated
        return AgentEnvResult(
            observation=None if self._done else self._observation(),
            tool_output=tool_output,
            process_reward=process_reward,
            outcome_reward=outcome_reward,
            terminated=terminated,
            truncated=truncated,
        )

    def candidate_actions(self) -> tuple[str, ...]:
        calls = tuple(
            f"CALL {self.schema.name} "
            + json.dumps(
                {self.schema.argument: task.query}, separators=(",", ":")
            )
            for task in self.tasks
        )
        finals = tuple(f"FINAL {task.answer}" for task in self.tasks)
        return (*calls, *finals, "FINAL unknown")

    def _call_tool(self, query: str) -> tuple[ToolOutput, float]:
        raise NotImplementedError

    def _score_final(self, answer: str) -> float:
        raise NotImplementedError


class CalculatorToolEnv(_BaseToolEnv):
    """Small arithmetic tasks with a pure AST calculator tool."""

    name = "calculator"
    schema = ToolSchema(
        name="calculator",
        argument="expression",
        description="+, -, *, / arithmetic only",
    )
    tasks = (
        _Task("c0", "2+3?", "5", "2+3"),
        _Task("c1", "7*4?", "28", "7*4"),
        _Task("c2", "18/3?", "6", "18/3"),
        _Task("c3", "11-9?", "2", "11-9"),
    )

    def _call_tool(self, query: str) -> tuple[ToolOutput, float]:
        try:
            value = _calculate(query)
        except ValueError as error:
            return (
                ToolOutput(
                    status="invalid", text=str(error), tool_name=self.schema.name
                ),
                -0.1,
            )
        reward = 0.15 if query == self.task.query else -0.05
        return (
            ToolOutput(status="success", text=value, tool_name=self.schema.name),
            reward,
        )

    def _score_final(self, answer: str) -> float:
        return 1.0 if answer.strip() == self.task.answer else -0.25


class LocalLookupEnv(_BaseToolEnv):
    """Offline key-value retrieval with citation-aware final answers."""

    name = "lookup"
    schema = ToolSchema(
        name="lookup",
        argument="key",
        description="read one key from the immutable local corpus",
    )
    tasks = (
        _Task("l0", "Korea capital? Cite.", "Seoul [kr]", "kr"),
        _Task("l1", "Japan capital? Cite.", "Tokyo [jp]", "jp"),
        _Task("l2", "France capital? Cite.", "Paris [fr]", "fr"),
        _Task("l3", "Italy capital? Cite.", "Rome [it]", "it"),
    )
    _corpus: ClassVar[dict[str, str]] = {
        "kr": "Seoul [kr]",
        "jp": "Tokyo [jp]",
        "fr": "Paris [fr]",
        "it": "Rome [it]",
    }

    def _call_tool(self, query: str) -> tuple[ToolOutput, float]:
        value = self._corpus.get(query)
        if value is None:
            return (
                ToolOutput(
                    status="invalid", text="key_not_found", tool_name=self.schema.name
                ),
                -0.1,
            )
        reward = 0.15 if query == self.task.query else -0.05
        return (
            ToolOutput(status="success", text=value, tool_name=self.schema.name),
            reward,
        )

    def _score_final(self, answer: str) -> float:
        return 1.0 if answer.strip() == self.task.answer else -0.25


def naive_call_reward(actions: tuple[AgentAction, ...]) -> float:
    """Intentionally vulnerable reward used only in the reward-hacking lesson."""

    return 0.1 * sum(action.kind == "tool" for action in actions)


def guarded_call_reward(actions: tuple[AgentAction, ...]) -> float:
    """One useful call may earn credit; repeated identical calls are penalized."""

    seen: set[tuple[str | None, str | None]] = set()
    reward = 0.0
    for action in actions:
        if action.kind != "tool":
            continue
        key = (action.tool_name, action.argument_value)
        reward += -0.25 if key in seen else 0.1
        seen.add(key)
    return reward


def agentic_task_split_hash() -> str:
    payload = [
        (environment, task.uid, task.question, task.answer, task.query)
        for environment, tasks in (
            (CalculatorToolEnv.name, CalculatorToolEnv.tasks),
            (LocalLookupEnv.name, LocalLookupEnv.tasks),
        )
        for task in tasks
    ]
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
