from __future__ import annotations

import json
import socket
from pathlib import Path

import torch

from rl_study.config import ExperimentConfig
from rl_study.training.agentic_runner import (
    evaluate_agentic_checkpoint,
    train_agentic,
)


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_agentic_train_checkpoint_resume_eval_and_exact_model_parity(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig.load("configs/toy/agentic_returns.yaml")
    continuous = train_agentic(
        config, output_root=tmp_path / "continuous", stop_after=4
    )
    interrupted = train_agentic(config, output_root=tmp_path / "resumed", stop_after=2)
    resumed = train_agentic(
        config,
        output_root=tmp_path / "resumed",
        resume=interrupted.checkpoint,
        stop_after=4,
    )

    continuous_state = _state(continuous.model)
    resumed_state = _state(resumed.model)
    assert continuous_state.keys() == resumed_state.keys()
    assert all(
        torch.equal(continuous_state[name], resumed_state[name])
        for name in continuous_state
    )

    algorithm, step, metrics = evaluate_agentic_checkpoint(resumed.checkpoint)
    assert algorithm == "agentic_reinforce"
    assert step == 4
    assert 0.0 <= metrics["validation_success_rate"] <= 1.0

    cursor = json.loads(
        (resumed.checkpoint / "data_cursor.json").read_text(encoding="utf-8")
    )
    records = json.loads(
        (resumed.checkpoint / "metrics.json").read_text(encoding="utf-8")
    )
    assert cursor["optimizer_steps"] == 4
    assert cursor["environment_steps"] >= 4
    assert len(cursor["episode_ids"]) == 4
    assert {record["environment"] for record in records} == {
        "calculator",
        "lookup",
    }
    assert any(record["episode_steps"] > 1 for record in records)


def test_broadcast_and_return_configs_share_initialization_and_budget(
    tmp_path: Path,
) -> None:
    broadcast_config = ExperimentConfig.load("configs/toy/agentic_broadcast.yaml")
    returns_config = ExperimentConfig.load("configs/toy/agentic_returns.yaml")
    broadcast = train_agentic(
        broadcast_config, output_root=tmp_path / "broadcast", stop_after=2
    )
    returns = train_agentic(
        returns_config, output_root=tmp_path / "returns", stop_after=2
    )
    broadcast_cursor = json.loads(
        (broadcast.checkpoint / "data_cursor.json").read_text(encoding="utf-8")
    )
    returns_cursor = json.loads(
        (returns.checkpoint / "data_cursor.json").read_text(encoding="utf-8")
    )
    assert broadcast_cursor["episode_ids"] == returns_cursor["episode_ids"]
    assert broadcast_cursor["optimizer_steps"] == returns_cursor["optimizer_steps"]
    assert (
        broadcast_cursor["initial_policy_hash"]
        == returns_cursor["initial_policy_hash"]
    )
    assert broadcast_cursor["split_hash"] == returns_cursor["split_hash"]
    assert broadcast_cursor["generated_tokens"] <= 4096
    assert returns_cursor["generated_tokens"] <= 4096


def test_agentic_training_works_when_network_sockets_are_denied(
    tmp_path: Path, monkeypatch,
) -> None:
    def deny_socket(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("toy Agentic RL attempted network access")

    monkeypatch.setattr(socket, "socket", deny_socket)
    config = ExperimentConfig.load("configs/toy/agentic_returns.yaml")
    result = train_agentic(config, output_root=tmp_path, stop_after=1)
    assert result.step == 1
