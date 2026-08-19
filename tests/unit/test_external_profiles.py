from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from rl_study.adapters.manifest import (
    MODEL_PRESETS,
    enforce_download_guard,
    estimate_training_memory,
    model_cache_status,
)
from rl_study.adapters.preflight import (
    build_profile_preflight,
    qlora_capability,
    resolve_model_manifest,
)
from rl_study.adapters.trl_adapter import TRLAdapterSpec
from rl_study.adapters.verl_recipe import render_verl_recipe, validate_verl_recipe
from rl_study.cli import main
from rl_study.config import ConfigError, ExperimentConfig
from rl_study.errors import DownloadApprovalRequired, PreflightError
from rl_study.training.laptop_runner import train_laptop_sft


def test_large_download_guard_fails_before_any_loader(tmp_path: Path) -> None:
    manifest = MODEL_PRESETS["laptop-smoke"]
    cached, hint = model_cache_status(manifest, cache_dir=tmp_path)
    assert not cached
    assert str(tmp_path) not in hint
    with pytest.raises(DownloadApprovalRequired, match="269,060,552"):
        enforce_download_guard(manifest, cached=False, accept_download=False)
    enforce_download_guard(manifest, cached=False, accept_download=True)


def test_laptop_train_guard_precedes_optional_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_import() -> None:
        raise AssertionError("optional stack must not be imported before approval")

    monkeypatch.setattr(
        "rl_study.training.laptop_runner._optional_stack", forbidden_import
    )
    config = ExperimentConfig.load("configs/laptop/smollm2_lora_sft.yaml")
    with pytest.raises(DownloadApprovalRequired):
        train_laptop_sft(
            config,
            accept_download=False,
            cache_dir=str(tmp_path),
            output_root=tmp_path / "artifacts",
        )


def test_memory_estimate_exposes_every_term() -> None:
    manifest = MODEL_PRESETS["laptop-smoke"]
    estimate = estimate_training_memory(
        manifest,
        adapter="lora",
        dtype="float32",
        batch_size=1,
        sequence_length=128,
    )
    assert estimate.base_parameters_bytes == manifest.parameters * 4
    assert estimate.adapter_parameters_bytes == manifest.expected_lora_parameters * 4
    assert estimate.recommended_bytes > estimate.subtotal_bytes
    assert len(estimate.assumptions) == 4


def test_preflight_is_no_network_and_qlora_never_falls_back(tmp_path: Path) -> None:
    payload = build_profile_preflight(
        profile="laptop",
        model="laptop-smoke",
        device="cpu",
        cache_dir=str(tmp_path),
    )
    assert payload["network_attempted"] is False
    model = payload["model"]
    assert isinstance(model, dict)
    assert model["requires_accept_download"] is True
    assert model["revision"] == MODEL_PRESETS["laptop-smoke"].revision

    supported, reason = qlora_capability("cpu")
    assert not supported and "fallback is forbidden" in reason


def test_arbitrary_model_requires_local_audit_fields() -> None:
    with pytest.raises(PreflightError, match="--revision"):
        resolve_model_manifest("someone/custom-model")
    custom = resolve_model_manifest(
        "someone/custom-model",
        revision="a" * 40,
        license_id="Apache-2.0",
        expected_weight_bytes=123,
    )
    assert custom.source_id == "user-supplied-audit"


def test_laptop_and_server_configs_keep_strict_profile_contract() -> None:
    laptop = ExperimentConfig.load("configs/laptop/smollm2_lora_sft.yaml")
    assert laptop.profile == "laptop"
    assert laptop.model.adapter == "lora"
    assert laptop.model.revision == MODEL_PRESETS["laptop-smoke"].revision

    server = ExperimentConfig.load("configs/server/qwen3_4b_dapo_verl.yaml")
    assert server.profile == "server"
    recipe = render_verl_recipe(server)
    validate_verl_recipe(recipe)
    assert recipe["run_status"] == "external-manual"
    assert recipe["local_executed"] is None
    overrides = recipe["overrides"]
    assert isinstance(overrides, dict)
    assert overrides["actor_rollout_ref.actor.loss_agg_mode"] == "token-mean"
    assert overrides["algorithm.filter_groups.enable"] is True

    invalid = deepcopy(laptop.to_dict())
    model = invalid["model"]
    assert isinstance(model, dict)
    model["revision"] = None
    with pytest.raises(ConfigError, match="audited provenance"):
        ExperimentConfig.from_mapping(invalid)


def test_verl_gspo_and_trl_variant_names_are_not_collapsed() -> None:
    gspo = ExperimentConfig.load("configs/server/qwen3_4b_gspo_verl.yaml")
    recipe = render_verl_recipe(gspo)
    overrides = recipe["overrides"]
    assert isinstance(overrides, dict)
    assert overrides["actor_rollout_ref.actor.policy_loss.loss_mode"] == "gspo"
    assert overrides["actor_rollout_ref.actor.loss_agg_mode"] == "seq-mean-token-mean"

    assert TRLAdapterSpec.for_algorithm("grpo").trainer_class == "GRPOTrainer"
    assert TRLAdapterSpec.for_algorithm("rloo").trainer_class == "RLOOTrainer"
    with pytest.raises(PreflightError, match="does not map"):
        TRLAdapterSpec.for_algorithm("dapo")


def test_cli_preflight_render_and_download_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as preflight:
        main(
            [
                "preflight",
                "--profile",
                "laptop",
                "--model",
                "laptop-smoke",
                "--device",
                "cpu",
                "--cache-dir",
                str(tmp_path),
                "--json",
            ]
        )
    assert preflight.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["network_attempted"] is False

    with pytest.raises(SystemExit) as blocked:
        main(
            [
                "train",
                "--config",
                "configs/laptop/smollm2_lora_sft.yaml",
                "--cache-dir",
                str(tmp_path),
                "--json",
            ]
        )
    assert blocked.value.code == 4
    error = capsys.readouterr().err
    assert "--accept-download" in error

    with pytest.raises(SystemExit) as rendered:
        main(
            [
                "render-server",
                "--config",
                "configs/server/qwen3_4b_dapo_verl.yaml",
                "--json",
            ]
        )
    assert rendered.value.code == 0
    recipe = json.loads(capsys.readouterr().out)
    assert recipe["framework"]["version"] == "0.9.0"
    assert recipe["result_origin"] == "not_executed"
