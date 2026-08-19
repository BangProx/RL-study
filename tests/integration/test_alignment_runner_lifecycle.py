from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rl_study.config import ExperimentConfig
from rl_study.training.alignment_runner import (
    evaluate_alignment_checkpoint,
    train_alignment,
)


def _short_config(path: str, *, steps: int) -> ExperimentConfig:
    mapping = ExperimentConfig.load(path).to_dict()
    training = mapping["training"]
    assert isinstance(training, dict)
    training["steps"] = steps
    training["batch_size"] = 4
    algorithm = mapping["algorithm"]
    assert isinstance(algorithm, dict)
    algorithm["update_epochs"] = 1
    return ExperimentConfig.from_mapping(mapping)


def _assert_state_equal(expected: torch.nn.Module, actual: torch.nn.Module) -> None:
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    assert expected_state.keys() == actual_state.keys()
    for name in expected_state:
        torch.testing.assert_close(
            expected_state[name], actual_state[name], rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize(
    ("path", "steps", "split_step"),
    [
        ("configs/toy/dpo.yaml", 4, 2),
        ("configs/toy/rlhf_ppo.yaml", 2, 1),
    ],
)
def test_alignment_interrupt_resume_eval_parity_and_card(
    path: str, steps: int, split_step: int, tmp_path: Path
) -> None:
    config = _short_config(path, steps=steps)
    full = train_alignment(config, output_root=tmp_path / "full")
    interrupted = train_alignment(
        config, output_root=tmp_path / "resumed", stop_after=split_step
    )
    resumed = train_alignment(
        config,
        output_root=tmp_path / "resumed",
        resume=interrupted.checkpoint,
    )

    assert full.step == resumed.step == steps
    _assert_state_equal(full.model, resumed.model)
    assert full.metrics == resumed.metrics

    algorithm, evaluated_step, metrics = evaluate_alignment_checkpoint(
        resumed.checkpoint
    )
    assert algorithm == config.algorithm.name
    assert evaluated_step == steps
    assert metrics
    assert all(0.0 <= value <= 1.0 for value in metrics.values())

    card = json.loads(resumed.experiment_card.read_text(encoding="utf-8"))
    assert card["schema_version"] == 1
    assert card["config_hash"] == config.sha256
    assert card["data"]["split_hash"].startswith("sha256:")
    assert card["local_executed"] == resumed.metrics
    assert card["budgets"]["model_forwards"] > 0


def test_rlhf_reward_model_runner_smoke(tmp_path: Path) -> None:
    config = _short_config("configs/toy/rlhf_ppo_reward_model.yaml", steps=1)
    result = train_alignment(config, output_root=tmp_path)
    assert result.algorithm == "rlhf_ppo"
    assert result.step == 1
    assert result.experiment_card.is_file()
