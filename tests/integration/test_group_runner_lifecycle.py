from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from rl_study.cli import main
from rl_study.config import ExperimentConfig
from rl_study.training.group_runner import (
    evaluate_group_checkpoint,
    train_group,
)

CONFIGS = [
    "configs/toy/grpo.yaml",
    "configs/toy/dapo.yaml",
    "configs/toy/rloo.yaml",
    "configs/toy/dr_grpo.yaml",
    "configs/toy/gspo.yaml",
]


def _short_config(path: str) -> ExperimentConfig:
    mapping = ExperimentConfig.load(path).to_dict()
    training = mapping["training"]
    assert isinstance(training, dict)
    training["steps"] = 2
    training["batch_size"] = 1
    training["group_size"] = 2
    algorithm = mapping["algorithm"]
    assert isinstance(algorithm, dict)
    algorithm["dynamic_sampling_multiplier"] = 2
    return ExperimentConfig.from_mapping(mapping)


def _assert_state_equal(expected: torch.nn.Module, actual: torch.nn.Module) -> None:
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    assert expected_state.keys() == actual_state.keys()
    for name in expected_state:
        torch.testing.assert_close(
            expected_state[name], actual_state[name], rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("path", CONFIGS)
def test_group_runner_interrupt_resume_eval_and_card(path: str, tmp_path: Path) -> None:
    config = _short_config(path)
    full = train_group(config, output_root=tmp_path / "full")
    interrupted = train_group(config, output_root=tmp_path / "resumed", stop_after=1)
    resumed = train_group(
        config,
        output_root=tmp_path / "resumed",
        resume=interrupted.checkpoint,
    )
    assert full.step == resumed.step == 2
    _assert_state_equal(full.model, resumed.model)
    assert full.metrics == resumed.metrics

    algorithm, step, metrics = evaluate_group_checkpoint(resumed.checkpoint)
    assert algorithm == config.algorithm.name
    assert step == 2
    assert all(0.0 <= value <= 1.0 for value in metrics.values())

    card = json.loads(resumed.experiment_card.read_text(encoding="utf-8"))
    assert card["budgets"]["generated_tokens"] > 0
    assert card["budgets"]["model_forwards"] > 0
    assert card["data"]["ordered_prompt_ids_sha256"].startswith("sha256:")


def test_group_cli_train_resume_eval_and_inspect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mapping = ExperimentConfig.load("configs/toy/grpo.yaml").to_dict()
    training = mapping["training"]
    assert isinstance(training, dict)
    training.update({"steps": 2, "batch_size": 1, "group_size": 2})
    config_path = tmp_path / "grpo-cli.yaml"
    config_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    output_root = tmp_path / "cli-artifacts"

    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "train",
                "--config",
                str(config_path),
                "--stop-after",
                "1",
                "--output-root",
                str(output_root),
                "--json",
            ]
        )
    assert stopped.value.code == 0
    first = json.loads(capsys.readouterr().out)

    with pytest.raises(SystemExit) as resumed:
        main(
            [
                "train",
                "--config",
                str(config_path),
                "--resume",
                first["checkpoint"],
                "--output-root",
                str(output_root),
                "--json",
            ]
        )
    assert resumed.value.code == 0
    second = json.loads(capsys.readouterr().out)
    assert second["step"] == 2

    with pytest.raises(SystemExit) as evaluated:
        main(["eval", "--checkpoint", second["checkpoint"], "--json"])
    assert evaluated.value.code == 0
    evaluation = json.loads(capsys.readouterr().out)
    assert evaluation["algorithm"] == "grpo"
    assert evaluation["step"] == 2

    with pytest.raises(SystemExit) as inspected:
        main(["inspect-run", second["checkpoint"], "--json"])
    assert inspected.value.code == 0
    card = json.loads(capsys.readouterr().out)
    assert card["algorithm"]["name"] == "grpo"
    assert card["budgets"]["generated_tokens"] > 0
