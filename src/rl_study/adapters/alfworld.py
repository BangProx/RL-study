"""Optional ALFWorld boundary; installation and game assets stay external-manual."""

from __future__ import annotations

import importlib.metadata
import platform
from typing import Protocol

from rl_study.agentic.types import (
    AgentAction,
    AgentEnvResult,
    AgentObservation,
    ToolOutput,
    ToolSchema,
)

ALFWORLD_VERSION = "0.4.2"
ALFWORLD_REVISION = "1558ba46d078279ecb4c5d33a6cdffc96714a2d2"


class AlfWorldBackend(Protocol):
    """Normalized single-environment subset of the optional runtime API."""

    def reset(self) -> tuple[str, dict[str, object]]: ...

    def step(
        self, command: str
    ) -> tuple[str, float, bool, bool, dict[str, object]]: ...


def alfworld_preflight() -> dict[str, object]:
    try:
        installed_version: str | None = importlib.metadata.version("alfworld")
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    machine = platform.machine().lower()
    mac_arm = platform.system() == "Darwin" and machine in {"arm64", "aarch64"}
    return {
        "status": "external-manual",
        "package": "alfworld",
        "required_version": ALFWORLD_VERSION,
        "revision": ALFWORLD_REVISION,
        "installed_version": installed_version,
        "version_matches": installed_version == ALFWORLD_VERSION,
        "assets_downloaded": None,
        "network_used": False,
        "license": "MIT",
        "optional_solver_license": "GPL-3.0",
        "platform_note": (
            "macOS arm64 needs the upstream x86_64 Conda compatibility path"
            if mac_arm
            else "follow the pinned upstream 0.4.2 installation instructions"
        ),
    }


class AlfWorldAdapter:
    """Allowlisted wrapper around an injected, already-configured ALFWorld backend."""

    schema = ToolSchema(
        name="alfworld_action",
        argument="command",
        description="one command from the latest admissible_commands list",
    )

    def __init__(self, backend: AlfWorldBackend, *, max_steps: int = 50) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._backend = backend
        self._max_steps = max_steps
        self._step_id = 0
        self._episode_count = 0
        self._episode_id = "not-reset"
        self._admissible: frozenset[str] = frozenset()
        self._done = True

    @staticmethod
    def _commands(info: dict[str, object]) -> frozenset[str]:
        commands = info.get("admissible_commands", [])
        if not isinstance(commands, list) or not all(
            isinstance(command, str) for command in commands
        ):
            raise ValueError("ALFWorld info must contain string admissible_commands")
        return frozenset(commands)

    def reset(self) -> AgentObservation:
        text, info = self._backend.reset()
        self._admissible = self._commands(info)
        self._episode_id = f"alfworld-{self._episode_count:06d}"
        self._episode_count += 1
        self._step_id = 0
        self._done = False
        return AgentObservation(
            episode_id=self._episode_id,
            step_id=0,
            text=text,
            visible_tools=(self.schema,),
        )

    def step(self, action: AgentAction) -> AgentEnvResult:
        if self._done:
            raise RuntimeError("reset is required after termination or truncation")
        if (
            action.kind != "tool"
            or action.tool_name != self.schema.name
            or action.argument_value not in self._admissible
        ):
            self._step_id += 1
            truncated = self._step_id >= self._max_steps
            self._done = truncated
            return AgentEnvResult(
                observation=(
                    None
                    if truncated
                    else AgentObservation(
                        self._episode_id,
                        self._step_id,
                        "invalid_or_non_admissible_action",
                        (self.schema,),
                    )
                ),
                tool_output=ToolOutput("invalid", "action_not_admissible"),
                process_reward=-0.1,
                outcome_reward=0.0,
                terminated=False,
                truncated=truncated,
            )
        text, reward, terminated, backend_truncated, info = self._backend.step(
            action.argument_value
        )
        self._step_id += 1
        truncated = backend_truncated or (
            not terminated and self._step_id >= self._max_steps
        )
        if terminated and truncated:
            truncated = False
        self._done = terminated or truncated
        self._admissible = frozenset() if self._done else self._commands(info)
        observation = (
            None
            if self._done
            else AgentObservation(
                self._episode_id,
                self._step_id,
                text,
                (self.schema,),
            )
        )
        return AgentEnvResult(
            observation=observation,
            tool_output=ToolOutput("success", text, self.schema.name),
            process_reward=0.0 if terminated else float(reward),
            outcome_reward=float(reward) if terminated else 0.0,
            terminated=terminated,
            truncated=truncated,
        )
