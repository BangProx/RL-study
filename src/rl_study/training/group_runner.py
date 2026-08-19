"""Checkpointed CLI lifecycle for GRPO-family toy trainers."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from rl_study.algorithms.group_policy import (
    GROUP_ALGORITHMS,
    train_group_policy,
)
from rl_study.algorithms.rlhf_ppo import evaluate_generation
from rl_study.algorithms.sft import train_sft
from rl_study.config import ExperimentConfig
from rl_study.data import build_tiny_reasoning
from rl_study.errors import ConfigError
from rl_study.models import TinyCausalLM, TinyTokenizer
from rl_study.models.roles import freeze_module
from rl_study.reporting import build_experiment_card, write_experiment_card
from rl_study.training.checkpoint import load_checkpoint, save_checkpoint

_SFT_DEFAULT_MODEL_FORWARDS = 202


class GroupPolicyState(nn.Module):
    def __init__(self, policy: TinyCausalLM) -> None:
        super().__init__()
        self.policy = policy
        self.reference = copy.deepcopy(policy)
        freeze_module(self.reference)


@dataclass(frozen=True, slots=True)
class GroupRunResult:
    algorithm: str
    step: int
    checkpoint: Path
    experiment_card: Path
    metrics: dict[str, float]
    model: GroupPolicyState


def _new_state(
    config: ExperimentConfig, *, for_resume: bool
) -> tuple[GroupPolicyState, torch.optim.Optimizer]:
    torch.manual_seed(config.training.seed)
    policy = (
        TinyCausalLM() if for_resume else train_sft(seed=config.training.seed).model
    )
    state = GroupPolicyState(policy)
    optimizer = torch.optim.AdamW(state.policy.parameters(), lr=5e-4, weight_decay=0.0)
    return state, optimizer


def _string_list(cursor: dict[str, object], name: str) -> list[str]:
    value = cursor.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"checkpoint {name} must be a string list")
    return value


def _counter(cursor: dict[str, object], name: str) -> int:
    value = cursor.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"checkpoint {name} must be a non-negative integer")
    return value


def train_group(
    config: ExperimentConfig,
    *,
    output_root: str | Path | None = None,
    resume: str | Path | None = None,
    stop_after: int | None = None,
) -> GroupRunResult:
    algorithm = config.algorithm.name
    if config.profile != "toy" or algorithm not in GROUP_ALGORITHMS:
        raise ConfigError("group runner requires a toy GRPO-family algorithm")
    if config.data.id != "tiny_reasoning":
        raise ConfigError("group runner requires data.id=tiny_reasoning")
    if config.algorithm.reward_source != "verifier":
        raise ConfigError("C7 group runner supports deterministic verifier reward")
    target_step = config.training.steps if stop_after is None else stop_after
    if not 1 <= target_step <= config.training.steps:
        raise ConfigError("stop_after must be between 1 and configured training.steps")

    started = time.perf_counter()
    state, optimizer = _new_state(config, for_resume=resume is not None)
    step_offset = 0
    prior_metrics: list[dict[str, object]] = []
    cursor: dict[str, object] = {}
    if resume is not None:
        loaded = load_checkpoint(
            resume, model=state, optimizer=optimizer, config=config
        )
        step_offset = loaded.step
        prior_metrics = loaded.metrics
        cursor = loaded.data_cursor
    if step_offset >= target_step:
        raise ConfigError(
            f"checkpoint step {step_offset} is not before target step {target_step}"
        )

    prior_rollout_uids = _string_list(cursor, "rollout_prompt_uids")
    prior_optimized_uids = _string_list(cursor, "optimized_prompt_uids")
    prior_generated = _counter(cursor, "generated_tokens")
    prior_forwards = _counter(cursor, "model_forwards")
    prior_optimizer_steps = _counter(cursor, "optimizer_steps")
    prior_rejected = _counter(cursor, "rejected_dynamic_groups")
    prior_exhausted = _counter(cursor, "exhausted_dynamic_updates")

    result = train_group_policy(
        algorithm=algorithm,
        updates=target_step - step_offset,
        prompt_batch_size=config.training.batch_size,
        group_size=config.training.group_size,
        seed=config.training.seed,
        clip_low=config.algorithm.clip_low,
        clip_high=config.algorithm.clip_high,
        kl_coefficient=config.algorithm.kl_coefficient,
        update_epochs=config.algorithm.update_epochs,
        max_new_tokens=config.training.max_new_tokens,
        dynamic_sampling=config.algorithm.dynamic_sampling,
        dynamic_sampling_multiplier=config.algorithm.dynamic_sampling_multiplier,
        token_level_loss=config.algorithm.token_level_loss,
        clip_higher=config.algorithm.clip_higher,
        overlong_reward=config.algorithm.overlong_reward_shaping,
        overlong_buffer_length=config.algorithm.overlong_buffer_length,
        overlong_penalty_scale=config.algorithm.overlong_penalty_scale,
        policy=state.policy,
        reference=state.reference,
        optimizer=optimizer,
        update_offset=step_offset,
    )
    generated_tokens = prior_generated + result.generated_tokens
    if generated_tokens > config.training.response_token_budget:
        raise ConfigError(
            "response_token_budget exceeded: "
            f"generated={generated_tokens}, "
            f"budget={config.training.response_token_budget}"
        )
    model_forwards = prior_forwards + result.model_forwards
    if resume is None:
        model_forwards += _SFT_DEFAULT_MODEL_FORWARDS
    optimizer_steps = prior_optimizer_steps + result.optimizer_steps
    rejected_groups = prior_rejected + result.rejected_dynamic_groups
    exhausted_updates = prior_exhausted + result.exhausted_dynamic_updates
    rollout_uids = [*prior_rollout_uids, *result.rollout_prompt_uids]
    optimized_uids = [*prior_optimized_uids, *result.optimized_prompt_uids]
    dataset = build_tiny_reasoning(seed=config.training.seed)

    local_records = [
        {
            "step": step_offset + index + 1,
            "algorithm": algorithm,
            "loss": loss,
            "reward_mean": result.mean_rewards[index],
            "exact_match": result.exact_match_rates[index],
            "format_rate": result.format_rates[index],
            "informative_group_rate": result.informative_group_rates[index],
            "response_length_mean": result.mean_response_lengths[index],
        }
        for index, loss in enumerate(result.losses)
    ]
    metrics = {
        "validation_exact_match": result.validation_exact_match,
        "validation_format_rate": result.validation_format_rate,
        "last_mean_reward": result.mean_rewards[-1],
        "last_informative_group_rate": result.informative_group_rates[-1],
        "last_mean_response_length": result.mean_response_lengths[-1],
        "optimizer_steps": float(optimizer_steps),
        "exhausted_dynamic_updates": float(exhausted_updates),
    }
    root = Path(output_root or config.output.root)
    checkpoint = (
        root
        / f"{algorithm}-seed{config.training.seed}"
        / f"checkpoint-{target_step:06d}"
    )
    save_checkpoint(
        checkpoint,
        model=state,
        optimizer=optimizer,
        config=config,
        step=target_step,
        data_cursor={
            "step": target_step,
            "split_hash": dataset.split_hash,
            "rollout_prompt_uids": rollout_uids,
            "optimized_prompt_uids": optimized_uids,
            "generated_tokens": generated_tokens,
            "model_forwards": model_forwards,
            "optimizer_steps": optimizer_steps,
            "rejected_dynamic_groups": rejected_groups,
            "exhausted_dynamic_updates": exhausted_updates,
        },
        metrics=[*prior_metrics, *local_records],
    )
    deviations = [
        "tiny-v1 toy run; not a paper-scale reproduction",
        "fixed 22-token maximum and short optimizer-step budget",
    ]
    if algorithm == "dapo" and config.algorithm.dynamic_sampling:
        deviations.append(
            "dynamic sampling is bounded by dynamic_sampling_multiplier per update"
        )
    card = build_experiment_card(
        config,
        run_id=f"{algorithm}-seed{config.training.seed}-step{target_step}",
        run_status="completed",
        step=target_step,
        wall_seconds=time.perf_counter() - started,
        metrics=metrics,
        data_split_hash=dataset.split_hash,
        prompt_uids=rollout_uids,
        optimized_prompt_uids=optimized_uids,
        optimizer_steps=optimizer_steps,
        generated_tokens=generated_tokens,
        model_forwards=model_forwards,
        known_deviations=tuple(deviations),
        failures=(
            (f"dynamic sampling budget exhausted on {exhausted_updates} updates",)
            if exhausted_updates
            else ()
        ),
    )
    experiment_card = write_experiment_card(checkpoint, card)
    return GroupRunResult(
        algorithm=algorithm,
        step=target_step,
        checkpoint=checkpoint,
        experiment_card=experiment_card,
        metrics=metrics,
        model=state,
    )


def _config_from_checkpoint(checkpoint: Path) -> ExperimentConfig:
    try:
        payload = json.loads(
            (checkpoint / "config.resolved.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot read checkpoint config: {error}") from error
    return ExperimentConfig.from_mapping(payload)


def evaluate_group_checkpoint(
    checkpoint: str | Path,
) -> tuple[str, int, dict[str, float]]:
    source = Path(checkpoint)
    config = _config_from_checkpoint(source)
    state, optimizer = _new_state(config, for_resume=True)
    loaded = load_checkpoint(
        source,
        model=state,
        optimizer=optimizer,
        config=config,
        restore_rng=False,
    )
    dataset = build_tiny_reasoning(seed=config.training.seed)
    exact, format_rate = evaluate_generation(
        state.policy, dataset.validation[:32], tokenizer=TinyTokenizer()
    )
    return (
        config.algorithm.name,
        loaded.step,
        {
            "validation_exact_match": exact,
            "validation_format_rate": format_rate,
        },
    )
