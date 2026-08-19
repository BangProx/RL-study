from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rl_study.config import ExperimentConfig
from rl_study.training.classic_runner import (
    evaluate_classic_checkpoint,
    train_classic,
)

CONFIGS = {
    "q_learning": "configs/toy/q_learning.yaml",
    "dqn": "configs/toy/dqn.yaml",
    "reinforce": "configs/toy/reinforce.yaml",
    "actor_critic": "configs/toy/actor_critic.yaml",
    "ppo": "configs/toy/ppo.yaml",
}


def _short_config(path: str, *, steps: int = 20) -> ExperimentConfig:
    mapping = ExperimentConfig.load(path).to_dict()
    mapping["training"]["steps"] = steps
    return ExperimentConfig.from_mapping(mapping)


def _assert_state_equal(expected: torch.nn.Module, actual: torch.nn.Module) -> None:
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    assert expected_state.keys() == actual_state.keys()
    for name in expected_state:
        torch.testing.assert_close(
            expected_state[name], actual_state[name], rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize(("algorithm", "path"), CONFIGS.items())
def test_train_interrupt_resume_eval_parity(
    algorithm: str, path: str, tmp_path: Path
) -> None:
    config = _short_config(path)
    full = train_classic(config, output_root=tmp_path / "full")
    interrupted = train_classic(config, output_root=tmp_path / "resumed", stop_after=10)
    resumed = train_classic(
        config,
        output_root=tmp_path / "resumed",
        resume=interrupted.checkpoint,
    )
    assert full.step == resumed.step == 20
    _assert_state_equal(full.model, resumed.model)

    evaluated_algorithm, evaluated_step, success_rate = evaluate_classic_checkpoint(
        resumed.checkpoint
    )
    assert evaluated_algorithm == algorithm
    assert evaluated_step == 20
    assert 0.0 <= success_rate <= 1.0
