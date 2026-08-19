from __future__ import annotations

import json
import subprocess
import sys

import pytest

from rl_study.cli import main


def test_preflight_json(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["preflight", "--profile", "toy", "--device", "cpu", "--json"])
    assert raised.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["resolved_device"] == "cpu"


def test_cli_module_entrypoint() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_study.cli",
            "preflight",
            "--profile",
            "toy",
            "--device",
            "cpu",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout)["status"] == "passed"


def test_train_dry_run_validates_config(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "train",
                "--config",
                "configs/toy/grpo.yaml",
                "--dry-run",
                "--json",
            ]
        )
    assert raised.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["algorithm"] == "grpo"
    assert payload["config_hash"].startswith("sha256:")


def test_demo_parser_exposes_artifact_directory() -> None:
    from rl_study.cli import _parser

    args = _parser().parse_args(
        ["demo", "--profile", "toy", "--output-dir", "custom-artifacts", "--json"]
    )
    assert args.output_dir.name == "custom-artifacts"


def test_demo_rejects_unavailable_device_before_writing(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "demo",
                "--profile",
                "toy",
                "--device",
                "cuda",
                "--output-dir",
                str(tmp_path),
                "--json",
            ]
        )
    assert raised.value.code != 0
    assert "not available" in capsys.readouterr().err.lower()
    assert not list(tmp_path.iterdir())
