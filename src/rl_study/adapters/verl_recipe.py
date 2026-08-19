"""Pure-data renderer for the pinned verl server boundary."""

from __future__ import annotations

from collections.abc import Mapping

from rl_study.config import ExperimentConfig
from rl_study.errors import ConfigError

VERL_VERSION = "0.9.0"
VERL_REVISION = "483b8a009ba3a97563edee3a19887e4862b8094a"


def _variant_overrides(config: ExperimentConfig) -> dict[str, object]:
    algorithm = config.algorithm.name
    common: dict[str, object] = {
        "algorithm.adv_estimator": "grpo",
        "actor_rollout_ref.actor.clip_ratio_low": config.algorithm.clip_low,
        "actor_rollout_ref.actor.clip_ratio_high": config.algorithm.clip_high,
        "actor_rollout_ref.actor.optim.lr": config.training.learning_rate,
    }
    if algorithm == "grpo":
        common.update(
            {
                "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
                "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
                "algorithm.use_kl_in_reward": config.algorithm.kl_coefficient > 0,
                "algorithm.kl_ctrl.kl_coef": config.algorithm.kl_coefficient,
            }
        )
    elif algorithm == "rloo":
        common.update(
            {
                "algorithm.adv_estimator": "rloo",
                "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
                "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum",
            }
        )
    elif algorithm == "dr_grpo":
        common.update(
            {
                "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
                "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum-norm",
                "algorithm.norm_adv_by_std_in_grpo": False,
            }
        )
    elif algorithm == "dapo":
        common.update(
            {
                "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
                "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
                "algorithm.filter_groups.enable": config.algorithm.dynamic_sampling,
                "algorithm.filter_groups.max_num_gen_batches": (
                    config.algorithm.dynamic_sampling_multiplier
                ),
                "algorithm.filter_groups.metric": "acc",
                "reward_model.overlong_buffer.enable": (
                    config.algorithm.overlong_reward_shaping
                ),
                "reward_model.overlong_buffer.len": (
                    config.algorithm.overlong_buffer_length
                ),
                "reward_model.overlong_buffer.penalty_factor": (
                    config.algorithm.overlong_penalty_scale
                ),
            }
        )
    elif algorithm == "gspo":
        common.update(
            {
                "actor_rollout_ref.actor.policy_loss.loss_mode": "gspo",
                "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
            }
        )
    else:
        raise ConfigError(
            "pinned verl recipe supports grpo, rloo, dr_grpo, dapo, or gspo"
        )
    return common


def render_verl_recipe(config: ExperimentConfig) -> dict[str, object]:
    if config.profile != "server":
        raise ConfigError("verl recipe requires profile=server")
    if config.model.revision is None:
        raise ConfigError("server model revision is required")
    overrides: dict[str, object] = {
        "data.train_files": "REPLACE_WITH_TRAIN_PARQUET",
        "data.val_files": "REPLACE_WITH_VALIDATION_PARQUET",
        "data.train_batch_size": config.training.batch_size,
        "data.max_prompt_length": config.training.max_sequence_length,
        "data.max_response_length": config.training.max_new_tokens,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "actor_rollout_ref.model.path": config.model.policy,
        "actor_rollout_ref.model.enable_gradient_checkpointing": (
            config.model.gradient_checkpointing
        ),
        "actor_rollout_ref.rollout.n": config.training.group_size,
        "actor_rollout_ref.rollout.name": "vllm",
        "trainer.n_gpus_per_node": 8,
        "trainer.nnodes": 1,
        "trainer.total_epochs": 1,
        "trainer.save_freq": 20,
        "trainer.test_freq": 20,
        "trainer.default_local_dir": config.output.root,
        **_variant_overrides(config),
    }
    arguments = [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
        "--config-name",
        "ppo_trainer",
        *(f"{key}={value}" for key, value in sorted(overrides.items())),
    ]
    return {
        "schema_version": 1,
        "run_status": "external-manual",
        "result_origin": "not_executed",
        "framework": {
            "name": "verl",
            "version": VERL_VERSION,
            "revision": VERL_REVISION,
            "license_id": "Apache-2.0",
        },
        "model": {
            "id": config.model.policy,
            "revision": config.model.revision,
            "license_id": config.model.license_id,
        },
        "algorithm": {
            "name": config.algorithm.name,
            "variant": config.algorithm.variant,
        },
        "hardware_required": (
            "Linux, CUDA, 1 node x 8 GPUs; adjust only after preflight"
        ),
        "manual_replacements": [
            "REPLACE_WITH_TRAIN_PARQUET",
            "REPLACE_WITH_VALIDATION_PARQUET",
        ],
        "overrides": overrides,
        "command_argv": arguments,
        "success_criteria": [
            "verl reports version 0.9.0 and all requested GPUs",
            "one optimizer step is finite and a checkpoint is written",
            "experiment card records generated tokens, model forwards, and peak memory",
        ],
        "paper_reported": None,
        "local_executed": None,
    }


def validate_verl_recipe(recipe: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "run_status",
        "result_origin",
        "framework",
        "model",
        "algorithm",
        "hardware_required",
        "manual_replacements",
        "overrides",
        "command_argv",
        "success_criteria",
        "paper_reported",
        "local_executed",
    }
    if set(recipe) != required:
        raise ConfigError("verl recipe schema keys do not match version 1")
    framework = recipe["framework"]
    if not isinstance(framework, Mapping):
        raise ConfigError("verl recipe framework must be a mapping")
    if (
        framework.get("version") != VERL_VERSION
        or framework.get("revision") != VERL_REVISION
    ):
        raise ConfigError("verl recipe framework pin does not match the audited source")
    if (
        recipe["run_status"] != "external-manual"
        or recipe["local_executed"] is not None
    ):
        raise ConfigError("unexecuted server recipe must remain external-manual")
    overrides = recipe["overrides"]
    if not isinstance(overrides, Mapping):
        raise ConfigError("verl recipe overrides must be a mapping")
    mandatory = {
        "data.train_files",
        "data.val_files",
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.rollout.n",
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
    }
    if not mandatory.issubset(overrides):
        raise ConfigError("verl recipe is missing mandatory distributed overrides")
