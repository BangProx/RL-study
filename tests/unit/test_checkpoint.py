from __future__ import annotations

import copy
import random

import pytest
import torch
from torch import nn

from rl_study.config import ExperimentConfig
from rl_study.errors import CheckpointError
from rl_study.training import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_integrity_and_rng(tmp_path) -> None:
    config = ExperimentConfig.load("configs/toy/ppo.yaml")
    torch.manual_seed(1)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    extra_model = nn.Linear(2, 1)
    extra_optimizer = torch.optim.SGD(extra_model.parameters(), lr=1e-2)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    expected_state = copy.deepcopy(model.state_dict())

    random.seed(9)
    torch.manual_seed(9)
    destination = tmp_path / "checkpoint-1"
    save_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        extra_optimizers={"value": extra_optimizer},
        config=config,
        step=1,
        data_cursor={"index": 16},
        metrics=[{"loss": float(loss.detach())}],
    )
    expected_random = random.random()
    expected_tensor = torch.rand(2)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    random.seed(99)
    torch.manual_seed(99)

    result = load_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        extra_optimizers={"value": extra_optimizer},
        config=config,
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_state[name])
    assert result.step == 1
    assert result.data_cursor == {"index": 16}
    assert random.random() == pytest.approx(expected_random)
    torch.testing.assert_close(torch.rand(2), expected_tensor)


def test_checkpoint_refuses_overwrite_and_detects_tampering(tmp_path) -> None:
    config = ExperimentConfig.load("configs/toy/ppo.yaml")
    model = nn.Linear(2, 1)
    destination = tmp_path / "checkpoint"
    save_checkpoint(destination, model=model, config=config, step=0)
    with pytest.raises(CheckpointError, match="overwrite"):
        save_checkpoint(destination, model=model, config=config, step=0)

    with (destination / "model.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(CheckpointError, match="integrity failure"):
        load_checkpoint(destination, model=model, config=config)


def test_checkpoint_allows_more_steps_but_rejects_immutable_change(tmp_path) -> None:
    config = ExperimentConfig.load("configs/toy/ppo.yaml")
    changed_mapping = config.to_dict()
    changed_mapping["training"]["steps"] = 101
    extended = ExperimentConfig.from_mapping(changed_mapping)
    model = nn.Linear(2, 1)
    destination = tmp_path / "checkpoint"
    save_checkpoint(destination, model=model, config=config, step=0)
    load_checkpoint(destination, model=model, config=extended)

    immutable_mapping = config.to_dict()
    immutable_mapping["model"]["policy"] = "tiny-micro-v1"
    incompatible = ExperimentConfig.from_mapping(immutable_mapping)
    with pytest.raises(CheckpointError, match="immutable config hash mismatch"):
        load_checkpoint(destination, model=model, config=incompatible)
