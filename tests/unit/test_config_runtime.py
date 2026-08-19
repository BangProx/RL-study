from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from rl_study.config import ConfigError, ExperimentConfig
from rl_study.errors import PreflightError
from rl_study.runtime import resolve_device, seed_everything
from rl_study.types import StepResult, TransitionBatch


def test_example_config_loads_and_hash_is_stable() -> None:
    first = ExperimentConfig.load("configs/toy/grpo.yaml")
    second = ExperimentConfig.from_mapping(first.to_dict())
    assert first == second
    assert first.sha256 == second.sha256
    assert first.sha256.startswith("sha256:")


def test_unknown_config_key_fails_before_training() -> None:
    mapping = yaml.safe_load(Path("configs/toy/grpo.yaml").read_text(encoding="utf-8"))
    mapping = deepcopy(mapping)
    mapping["training"]["silent_typo"] = True
    with pytest.raises(ConfigError, match=r"unknown keys.*silent_typo"):
        ExperimentConfig.from_mapping(mapping)


def test_profile_and_test_split_guards() -> None:
    mapping = yaml.safe_load(Path("configs/toy/grpo.yaml").read_text(encoding="utf-8"))
    laptop_mapping = deepcopy(mapping)
    laptop_mapping["profile"] = "laptop"
    laptop_mapping["model"]["trust_remote_code"] = True
    with pytest.raises(ConfigError, match="separate explicit approval"):
        ExperimentConfig.from_mapping(laptop_mapping)

    train_eval_mapping = deepcopy(mapping)
    train_eval_mapping["evaluation"]["split"] = "train"
    with pytest.raises(ConfigError, match="must not be train"):
        ExperimentConfig.from_mapping(train_eval_mapping)


def test_group_variant_configs_and_dapo_cross_field_guards() -> None:
    for name in ("grpo", "dapo", "rloo", "dr_grpo", "gspo"):
        config = ExperimentConfig.load(f"configs/toy/{name}.yaml")
        assert config.algorithm.name == name
        assert config.training.group_size >= 2

    mapping = yaml.safe_load(Path("configs/toy/grpo.yaml").read_text(encoding="utf-8"))
    foreign_toggle = deepcopy(mapping)
    foreign_toggle["algorithm"]["dynamic_sampling"] = True
    with pytest.raises(ConfigError, match="DAPO component toggles"):
        ExperimentConfig.from_mapping(foreign_toggle)

    invalid_buffer = yaml.safe_load(
        Path("configs/toy/dapo.yaml").read_text(encoding="utf-8")
    )
    invalid_buffer = deepcopy(invalid_buffer)
    invalid_buffer["algorithm"]["overlong_buffer_length"] = 23
    with pytest.raises(ConfigError, match="cannot exceed"):
        ExperimentConfig.from_mapping(invalid_buffer)


def test_cpu_resolution_and_explicit_unavailable_device() -> None:
    resolution = resolve_device("cpu")
    assert resolution.resolved == torch.device("cpu")
    assert resolution.fallback_used is False
    with pytest.raises(PreflightError):
        resolve_device("cuda:9999")


def test_seed_is_repeatable() -> None:
    seed_everything(123)
    first = torch.rand(4)
    seed_everything(123)
    second = torch.rand(4)
    torch.testing.assert_close(first, second)


def test_step_result_and_transition_batch_keep_done_signals_separate() -> None:
    with pytest.raises(ValueError, match="both terminated and truncated"):
        StepResult(observation=0, reward=0.0, terminated=True, truncated=True)
    with pytest.raises(ValueError, match="both terminated and truncated"):
        TransitionBatch(
            observations=torch.zeros(1, 2),
            actions=torch.zeros(1, dtype=torch.int64),
            rewards=torch.zeros(1),
            next_observations=torch.zeros(1, 2),
            terminated=torch.tensor([True]),
            truncated=torch.tensor([True]),
        )
