from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rl_study.config import ExperimentConfig
from rl_study.reporting import build_experiment_card, write_experiment_card

ROOT = Path(__file__).resolve().parents[2]


def test_toy_experiment_card_records_environment_and_base_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    config = ExperimentConfig.load(ROOT / "configs/toy/dpo.yaml")
    card = build_experiment_card(
        config,
        run_id="report-test",
        run_status="completed",
        step=1,
        wall_seconds=0.1,
        metrics={"finite_loss": 1.0},
        optimizer_steps=1,
    )
    environment = card["environment"]
    assert isinstance(environment, dict)
    assert isinstance(environment["ram_bytes"], int)
    assert environment["ram_bytes"] > 0
    assert "vram_bytes" in environment
    assert card["result_origin"] == "local_executed"
    expected = "sha256:" + hashlib.sha256(
        (ROOT / "pyproject.toml").read_bytes()
    ).hexdigest()
    assert card["dependency_lock_hash"] == expected


def test_experiment_card_write_is_json_and_refuses_overwrite(tmp_path: Path) -> None:
    path = write_experiment_card(tmp_path, {"schema_version": 1, "value": "실행"})
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == "실행"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_experiment_card(tmp_path, {"schema_version": 1})
