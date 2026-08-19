from __future__ import annotations

from rl_study.adapters.alfworld import AlfWorldAdapter, alfworld_preflight
from rl_study.agentic.parser import parse_action


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def reset(self) -> tuple[str, dict[str, object]]:
        return "room", {"admissible_commands": ["look", "finish"]}

    def step(
        self, command: str
    ) -> tuple[str, float, bool, bool, dict[str, object]]:
        self.calls.append(command)
        return (
            "done" if command == "finish" else "room seen",
            1.0 if command == "finish" else 0.1,
            command == "finish",
            False,
            {"admissible_commands": ["finish"]},
        )


def test_alfworld_adapter_never_calls_backend_for_non_admissible_action() -> None:
    backend = _FakeBackend()
    adapter = AlfWorldAdapter(backend)
    observation = adapter.reset()
    invalid = parse_action(
        'CALL alfworld_action {"command":"shell id"}',
        observation.visible_tools,
    )
    result = adapter.step(invalid)
    assert result.tool_output.status == "invalid"
    assert backend.calls == []


def test_alfworld_adapter_separates_process_and_terminal_outcome_reward() -> None:
    backend = _FakeBackend()
    adapter = AlfWorldAdapter(backend)
    first_observation = adapter.reset()
    look = parse_action(
        'CALL alfworld_action {"command":"look"}',
        first_observation.visible_tools,
    )
    first = adapter.step(look)
    assert first.process_reward == 0.1 and first.outcome_reward == 0.0
    assert first.observation is not None
    finish = parse_action(
        'CALL alfworld_action {"command":"finish"}',
        first.observation.visible_tools,
    )
    final = adapter.step(finish)
    assert final.terminated
    assert final.process_reward == 0.0 and final.outcome_reward == 1.0


def test_alfworld_preflight_is_read_only_and_external_manual() -> None:
    report = alfworld_preflight()
    assert report["status"] == "external-manual"
    assert report["network_used"] is False
    assert report["required_version"] == "0.4.2"
