"""Checkpointed toy DPO and RLHF-PPO application runner."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn

from rl_study.algorithms.dpo import evaluate_dpo_preferences, train_dpo
from rl_study.algorithms.reward_model import train_reward_model
from rl_study.algorithms.rlhf_ppo import evaluate_generation, train_rlhf_ppo
from rl_study.algorithms.sft import train_sft
from rl_study.config import ExperimentConfig
from rl_study.data import build_preferences, build_tiny_reasoning
from rl_study.errors import ConfigError
from rl_study.models import TinyCausalLM, TinyRewardModel, TinyTokenizer, TinyValueModel
from rl_study.models.roles import freeze_module
from rl_study.reporting import build_experiment_card, write_experiment_card
from rl_study.training.checkpoint import load_checkpoint, save_checkpoint

ALIGNMENT_ALGORITHMS = frozenset({"dpo", "rlhf_ppo"})
_SFT_DEFAULT_MODEL_FORWARDS = 202
_REWARD_MODEL_DEFAULT_FORWARDS = 242


class DPOState(nn.Module):
    def __init__(self, policy: TinyCausalLM) -> None:
        super().__init__()
        self.policy = policy
        self.reference = copy.deepcopy(policy)
        freeze_module(self.reference)


class RLHFState(nn.Module):
    reward_model: TinyRewardModel | None

    def __init__(
        self,
        policy: TinyCausalLM,
        *,
        reward_model: TinyRewardModel | None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.reference = copy.deepcopy(policy)
        freeze_module(self.reference)
        self.value_model = TinyValueModel(copy.deepcopy(policy))
        self.reward_model = reward_model
        if self.reward_model is not None:
            freeze_module(self.reward_model)


@dataclass(frozen=True, slots=True)
class AlignmentRunResult:
    algorithm: str
    step: int
    checkpoint: Path
    experiment_card: Path
    metrics: dict[str, float]
    model: nn.Module


def _new_state(
    config: ExperimentConfig, *, for_resume: bool
) -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    dict[str, torch.optim.Optimizer],
]:
    torch.manual_seed(config.training.seed)
    policy = (
        TinyCausalLM() if for_resume else train_sft(seed=config.training.seed).model
    )
    if config.algorithm.name == "dpo":
        dpo_state = DPOState(policy)
        optimizer = torch.optim.AdamW(dpo_state.policy.parameters(), lr=5e-4)
        return dpo_state, optimizer, {}
    if config.algorithm.name == "rlhf_ppo":
        reward_model: TinyRewardModel | None = None
        if config.algorithm.reward_source == "reward_model":
            reward_model = (
                TinyRewardModel()
                if for_resume
                else train_reward_model(seed=config.training.seed).model
            )
        rlhf_state = RLHFState(policy, reward_model=reward_model)
        policy_optimizer = torch.optim.AdamW(rlhf_state.policy.parameters(), lr=5e-4)
        value_optimizer = torch.optim.AdamW(
            rlhf_state.value_model.parameters(), lr=1e-3
        )
        return rlhf_state, policy_optimizer, {"value": value_optimizer}
    raise ConfigError(f"{config.algorithm.name!r} is not an alignment runner")


def train_alignment(
    config: ExperimentConfig,
    *,
    output_root: str | Path | None = None,
    resume: str | Path | None = None,
    stop_after: int | None = None,
) -> AlignmentRunResult:
    algorithm = config.algorithm.name
    if config.profile != "toy" or algorithm not in ALIGNMENT_ALGORITHMS:
        raise ConfigError("alignment runner requires toy DPO or RLHF-PPO")
    if config.data.id != "tiny_reasoning":
        raise ConfigError("alignment runner requires data.id=tiny_reasoning")
    target_step = config.training.steps if stop_after is None else stop_after
    if not 1 <= target_step <= config.training.steps:
        raise ConfigError("stop_after must be between 1 and configured training.steps")
    started = time.perf_counter()
    model, optimizer, extra_optimizers = _new_state(
        config, for_resume=resume is not None
    )
    step_offset = 0
    prior_metrics: list[dict[str, object]] = []
    data_cursor: dict[str, object] = {}
    if resume is not None:
        loaded = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            extra_optimizers=extra_optimizers,
            config=config,
        )
        step_offset = loaded.step
        prior_metrics = loaded.metrics
        data_cursor = loaded.data_cursor
    if step_offset >= target_step:
        raise ConfigError(
            f"checkpoint step {step_offset} is not before target step {target_step}"
        )
    steps = target_step - step_offset
    dataset = build_tiny_reasoning(seed=config.training.seed)
    prior_prompt_uids = data_cursor.get("prompt_uids", [])
    if not isinstance(prior_prompt_uids, list) or not all(
        isinstance(value, str) for value in prior_prompt_uids
    ):
        raise ConfigError("checkpoint prompt_uids must be a string list")
    prior_generated_tokens = data_cursor.get("generated_tokens", 0)
    prior_processed_tokens = data_cursor.get("processed_tokens", 0)
    prior_model_forwards = data_cursor.get("model_forwards", 0)
    if (
        isinstance(prior_generated_tokens, bool)
        or not isinstance(prior_generated_tokens, int)
        or isinstance(prior_processed_tokens, bool)
        or not isinstance(prior_processed_tokens, int)
        or isinstance(prior_model_forwards, bool)
        or not isinstance(prior_model_forwards, int)
    ):
        raise ConfigError("checkpoint budget counters must be integers")

    if algorithm == "dpo":
        dpo_state = cast(DPOState, model)
        dpo_result = train_dpo(
            steps=steps,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
            beta=config.algorithm.beta,
            label_smoothing=config.algorithm.label_smoothing,
            policy=dpo_state.policy,
            reference=dpo_state.reference,
            optimizer=optimizer,
            step_offset=step_offset,
        )
        metrics = {
            "train_preference_accuracy": dpo_result.train_accuracy,
            "validation_preference_accuracy": dpo_result.validation_accuracy,
            "last_loss": dpo_result.losses[-1],
        }
        local_prompt_uids = dpo_result.prompt_uids
        generated_tokens = prior_generated_tokens
        processed_tokens = prior_processed_tokens + dpo_result.processed_response_tokens
        model_forwards = prior_model_forwards + dpo_result.model_forwards
        local_records = [
            {
                "step": step_offset + index + 1,
                "loss": value,
                "algorithm": algorithm,
            }
            for index, value in enumerate(dpo_result.losses)
        ]
    else:
        rlhf_state = cast(RLHFState, model)
        value_optimizer = extra_optimizers.get("value")
        if value_optimizer is None:
            raise RuntimeError("RLHF value optimizer is missing")
        rlhf_result = train_rlhf_ppo(
            updates=steps,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
            kl_coefficient=config.algorithm.kl_coefficient,
            reward_source=config.algorithm.reward_source,
            update_epochs=config.algorithm.update_epochs,
            policy=rlhf_state.policy,
            reference=rlhf_state.reference,
            reward_model=rlhf_state.reward_model,
            value_model=rlhf_state.value_model,
            policy_optimizer=optimizer,
            value_optimizer=value_optimizer,
            update_offset=step_offset,
        )
        metrics = {
            "validation_exact_match": rlhf_result.validation_exact_match,
            "validation_format_rate": rlhf_result.validation_format_rate,
            "last_mean_task_reward": rlhf_result.mean_task_rewards[-1],
            "last_mean_sampled_kl": rlhf_result.mean_sampled_kls[-1],
            "last_policy_loss": rlhf_result.policy_losses[-1],
        }
        local_prompt_uids = rlhf_result.prompt_uids
        generated_tokens = prior_generated_tokens + rlhf_result.generated_tokens
        processed_tokens = prior_processed_tokens
        model_forwards = prior_model_forwards + rlhf_result.model_forwards
        local_records = [
            {
                "step": step_offset + index + 1,
                "task_reward": reward,
                "sampled_kl": rlhf_result.mean_sampled_kls[index],
                "exact_match": rlhf_result.exact_match_rates[index],
                "format_rate": rlhf_result.format_rates[index],
                "algorithm": algorithm,
            }
            for index, reward in enumerate(rlhf_result.mean_task_rewards)
        ]

    if max(generated_tokens, processed_tokens) > config.training.response_token_budget:
        raise ConfigError(
            "response_token_budget exceeded: "
            f"generated={generated_tokens}, processed={processed_tokens}, "
            f"budget={config.training.response_token_budget}"
        )
    all_prompt_uids = [*prior_prompt_uids, *local_prompt_uids]
    if resume is None:
        model_forwards += _SFT_DEFAULT_MODEL_FORWARDS
        if algorithm == "rlhf_ppo" and config.algorithm.reward_source == "reward_model":
            model_forwards += _REWARD_MODEL_DEFAULT_FORWARDS
    records = [*prior_metrics, *local_records]
    root = Path(output_root or config.output.root)
    run_name = (
        f"{algorithm}-{config.algorithm.reward_source}-seed{config.training.seed}"
    )
    run_directory = root / run_name
    checkpoint = run_directory / f"checkpoint-{target_step:06d}"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        extra_optimizers=extra_optimizers,
        config=config,
        step=target_step,
        data_cursor={
            "step": target_step,
            "split_hash": dataset.split_hash,
            "prompt_uids": all_prompt_uids,
            "generated_tokens": generated_tokens,
            "processed_tokens": processed_tokens,
            "model_forwards": model_forwards,
        },
        metrics=records,
    )
    card = build_experiment_card(
        config,
        run_id=f"{algorithm}-{config.algorithm.reward_source}-seed{config.training.seed}-step{target_step}",
        run_status="completed",
        step=target_step,
        wall_seconds=time.perf_counter() - started,
        metrics=metrics,
        data_split_hash=dataset.split_hash,
        prompt_uids=all_prompt_uids,
        optimizer_steps=(
            target_step * config.algorithm.update_epochs
            if algorithm == "rlhf_ppo"
            else target_step
        ),
        generated_tokens=generated_tokens,
        processed_tokens=processed_tokens,
        model_forwards=model_forwards,
        known_deviations=(
            "tiny-v1 toy run; not a paper-scale reproduction",
            "teacher-forced SFT and short RL/preference optimization",
        ),
    )
    experiment_card = write_experiment_card(checkpoint, card)
    return AlignmentRunResult(
        algorithm=algorithm,
        step=target_step,
        checkpoint=checkpoint,
        experiment_card=experiment_card,
        metrics=metrics,
        model=model,
    )


def _config_from_checkpoint(checkpoint: Path) -> ExperimentConfig:
    try:
        value = json.loads(
            (checkpoint / "config.resolved.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot read checkpoint config: {error}") from error
    return ExperimentConfig.from_mapping(value)


def evaluate_alignment_checkpoint(
    checkpoint: str | Path,
) -> tuple[str, int, dict[str, float]]:
    source = Path(checkpoint)
    config = _config_from_checkpoint(source)
    model, optimizer, extra_optimizers = _new_state(config, for_resume=True)
    loaded = load_checkpoint(
        source,
        model=model,
        optimizer=optimizer,
        extra_optimizers=extra_optimizers,
        config=config,
        restore_rng=False,
    )
    dataset = build_tiny_reasoning(seed=config.training.seed)
    tokenizer = TinyTokenizer()
    if config.algorithm.name == "dpo":
        dpo_state = cast(DPOState, model)
        accuracy = evaluate_dpo_preferences(
            dpo_state.policy,
            dpo_state.reference,
            build_preferences(dataset.validation),
            tokenizer=tokenizer,
            beta=config.algorithm.beta,
        )
        metrics = {"validation_preference_accuracy": accuracy}
    else:
        rlhf_state = cast(RLHFState, model)
        exact, format_rate = evaluate_generation(
            rlhf_state.policy, dataset.validation[:32], tokenizer=tokenizer
        )
        metrics = {
            "validation_exact_match": exact,
            "validation_format_rate": format_rate,
        }
    return config.algorithm.name, loaded.step, metrics
