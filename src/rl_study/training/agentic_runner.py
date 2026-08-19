"""Checkpointed Agentic REINFORCE runner for both offline tool environments."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from rl_study.agentic.credit import CreditMode
from rl_study.agentic.envs import (
    CalculatorToolEnv,
    LocalLookupEnv,
    OfflineAgentEnv,
    agentic_task_split_hash,
)
from rl_study.agentic.trajectory import rollout_episode, update_policy
from rl_study.agentic.types import AgentTrajectory
from rl_study.config import ExperimentConfig
from rl_study.errors import ConfigError
from rl_study.models import TinyCausalLM, TinyLMConfig, TinyTokenizer
from rl_study.reporting import build_experiment_card, write_experiment_card
from rl_study.training.checkpoint import load_checkpoint, save_checkpoint

AGENTIC_ALGORITHMS = frozenset({"agentic_reinforce"})


@dataclass(frozen=True, slots=True)
class AgenticRunResult:
    algorithm: str
    step: int
    checkpoint: Path
    experiment_card: Path
    metrics: dict[str, float]
    model: TinyCausalLM


def _model_hash(model: TinyCausalLM) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _new_state(
    config: ExperimentConfig,
) -> tuple[TinyCausalLM, torch.optim.Optimizer]:
    torch.manual_seed(config.training.seed)
    model = TinyCausalLM(
        TinyLMConfig(
            max_sequence_length=config.training.max_sequence_length,
            hidden_size=48,
            num_heads=4,
            num_layers=2,
            intermediate_size=96,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate, weight_decay=0.0
    )
    return model, optimizer


def _counter(cursor: dict[str, object], name: str) -> int:
    value = cursor.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"checkpoint {name} must be a non-negative integer")
    return value


def _strings(cursor: dict[str, object], name: str) -> list[str]:
    value = cursor.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"checkpoint {name} must be a string list")
    return value


def _is_multi_turn(record: dict[str, object]) -> bool:
    steps = record.get("episode_steps")
    return isinstance(steps, int) and not isinstance(steps, bool) and steps > 1


def _scored_tokens(trajectory: AgentTrajectory) -> int:
    return sum(
        sum(
            len(step.context_token_ids) + len(candidate)
            for candidate in step.candidate_action_token_ids
        )
        for step in trajectory.steps
    )


def _generator(
    seed: int, cursor: dict[str, object], *, resumed: bool
) -> torch.Generator:
    generator = torch.Generator().manual_seed(seed)
    if not resumed:
        return generator
    state = cursor.get("rollout_generator_state")
    if not isinstance(state, list) or not all(isinstance(item, int) for item in state):
        raise ConfigError("checkpoint rollout_generator_state must be an integer list")
    generator.set_state(torch.tensor(state, dtype=torch.uint8))
    return generator


def _environment(
    step: int,
    *,
    seed: int,
    subset: str,
    calculator_offset: int,
    lookup_offset: int,
) -> tuple[str, OfflineAgentEnv]:
    if subset == "calculator" or (subset == "mixed" and step % 2 == 0):
        return "calculator", CalculatorToolEnv(
            seed=seed, max_steps=3, episode_offset=calculator_offset
        )
    if subset == "lookup" or subset == "mixed":
        return "lookup", LocalLookupEnv(
            seed=seed, max_steps=3, episode_offset=lookup_offset
        )
    raise ConfigError("agentic data.subset must be calculator, lookup, or mixed")


@torch.no_grad()
def _evaluate(
    model: TinyCausalLM, *, seed: int
) -> tuple[dict[str, float], int, int]:
    tokenizer = TinyTokenizer()
    generator = torch.Generator().manual_seed(seed + 10_000)
    successes: dict[str, int] = {"calculator": 0, "lookup": 0}
    multi_turn = 0
    useful_tool_steps = 0
    total_environment_steps = 0
    process_reward_sum = 0.0
    model_forwards = 0
    processed_tokens = 0
    for name, env_type in (
        ("calculator", CalculatorToolEnv),
        ("lookup", LocalLookupEnv),
    ):
        env = env_type(seed=seed, max_steps=3)
        for task_index in range(4):
            trajectory, _, forwards = rollout_episode(
                model,
                tokenizer,
                env,
                generator=generator,
                policy_version=0,
                task_index=task_index,
                greedy=True,
            )
            successes[name] += int(trajectory.succeeded)
            multi_turn += int(len(trajectory.steps) > 1)
            useful_tool_steps += sum(
                step.action.kind == "tool" and step.process_reward > 0.0
                for step in trajectory.steps
            )
            total_environment_steps += len(trajectory.steps)
            process_reward_sum += sum(
                step.process_reward for step in trajectory.steps
            )
            model_forwards += forwards
            processed_tokens += _scored_tokens(trajectory)
    metrics = {
        "validation_success_rate": sum(successes.values()) / 8.0,
        "validation_calculator_success_rate": successes["calculator"] / 4.0,
        "validation_lookup_success_rate": successes["lookup"] / 4.0,
        "validation_multi_turn_rate": multi_turn / 8.0,
        "validation_useful_tool_step_rate": (
            useful_tool_steps / total_environment_steps
        ),
        "validation_mean_process_reward": process_reward_sum / 8.0,
    }
    return metrics, model_forwards, processed_tokens


def train_agentic(
    config: ExperimentConfig,
    *,
    output_root: str | Path | None = None,
    resume: str | Path | None = None,
    stop_after: int | None = None,
) -> AgenticRunResult:
    if config.profile != "toy" or config.algorithm.name not in AGENTIC_ALGORITHMS:
        raise ConfigError("agentic runner requires toy agentic_reinforce")
    if config.data.id != "agentic_offline":
        raise ConfigError("agentic runner requires data.id=agentic_offline")
    if config.training.batch_size != 1:
        raise ConfigError("C9 agentic runner currently requires batch_size=1")
    subset = config.data.subset or "mixed"
    if subset not in {"calculator", "lookup", "mixed"}:
        raise ConfigError("agentic data.subset must be calculator, lookup, or mixed")
    if config.training.max_sequence_length < 160:
        raise ConfigError("agentic max_sequence_length must be at least 160")
    target_step = config.training.steps if stop_after is None else stop_after
    if not 1 <= target_step <= config.training.steps:
        raise ConfigError("stop_after must be between 1 and configured training.steps")

    started = time.perf_counter()
    model, optimizer = _new_state(config)
    initial_policy_hash = _model_hash(model)
    step_offset = 0
    prior_records: list[dict[str, object]] = []
    cursor: dict[str, object] = {}
    if resume is not None:
        loaded = load_checkpoint(
            resume, model=model, optimizer=optimizer, config=config
        )
        step_offset = loaded.step
        prior_records = loaded.metrics
        cursor = loaded.data_cursor
    if step_offset >= target_step:
        raise ConfigError(
            f"checkpoint step {step_offset} is not before target step {target_step}"
        )

    generated_tokens = _counter(cursor, "generated_tokens")
    model_forwards = _counter(cursor, "model_forwards")
    processed_tokens = _counter(cursor, "processed_tokens")
    environment_steps = _counter(cursor, "environment_steps")
    optimizer_steps = _counter(cursor, "optimizer_steps")
    calculator_episodes = _counter(cursor, "calculator_episodes")
    lookup_episodes = _counter(cursor, "lookup_episodes")
    episode_ids = _strings(cursor, "episode_ids")
    generator = _generator(config.training.seed + 1, cursor, resumed=resume is not None)
    tokenizer = TinyTokenizer()
    records: list[dict[str, object]] = []
    losses: list[float] = []
    credit_mode = cast(CreditMode, config.algorithm.credit_assignment)

    for step in range(step_offset, target_step):
        name, env = _environment(
            step,
            seed=config.training.seed,
            subset=subset,
            calculator_offset=calculator_episodes,
            lookup_offset=lookup_episodes,
        )
        trajectory, episode_tokens, rollout_forwards = rollout_episode(
            model,
            tokenizer,
            env,
            generator=generator,
            policy_version=step,
            task_index=step % 4,
        )
        if generated_tokens + episode_tokens > config.training.response_token_budget:
            raise ConfigError(
                "response_token_budget exceeded before policy update: "
                f"would_generate={generated_tokens + episode_tokens}, "
                f"budget={config.training.response_token_budget}"
            )
        update = update_policy(
            model,
            tokenizer,
            optimizer,
            trajectory,
            current_policy_version=step,
            credit_mode=credit_mode,
            gamma=config.algorithm.gamma,
        )
        generated_tokens += episode_tokens
        model_forwards += rollout_forwards + update.model_forwards
        processed_tokens += 2 * _scored_tokens(trajectory)
        environment_steps += len(trajectory.steps)
        optimizer_steps += 1
        losses.append(update.loss)
        episode_ids.append(trajectory.episode_id)
        calculator_episodes += int(name == "calculator")
        lookup_episodes += int(name == "lookup")
        records.append(
            {
                "step": step + 1,
                "environment": name,
                "episode_id": trajectory.episode_id,
                "episode_steps": len(trajectory.steps),
                "success": trajectory.succeeded,
                "outcome_reward": trajectory.steps[-1].outcome_reward,
                "process_reward_sum": sum(
                    item.process_reward for item in trajectory.steps
                ),
                "useful_tool_steps": sum(
                    item.action.kind == "tool" and item.process_reward > 0.0
                    for item in trajectory.steps
                ),
                "truncated": trajectory.steps[-1].truncated,
                "loss": update.loss,
            }
        )

    evaluation_metrics, evaluation_forwards, evaluation_tokens = _evaluate(
        model, seed=config.training.seed
    )
    model_forwards += evaluation_forwards
    processed_tokens += evaluation_tokens
    all_records = [*prior_records, *records]
    successes = sum(bool(record.get("success")) for record in all_records)
    multi_turn = sum(_is_multi_turn(record) for record in all_records)
    process_rewards = [
        value
        for record in all_records
        if isinstance((value := record.get("process_reward_sum")), (int, float))
        and not isinstance(value, bool)
    ]
    useful_steps = [
        value
        for record in all_records
        if isinstance((value := record.get("useful_tool_steps")), int)
        and not isinstance(value, bool)
    ]
    metrics = {
        **evaluation_metrics,
        "train_success_rate": successes / len(all_records),
        "train_multi_turn_rate": multi_turn / len(all_records),
        "train_mean_process_reward": sum(process_rewards) / len(all_records),
        "train_useful_tool_steps_per_episode": sum(useful_steps) / len(all_records),
        "last_loss": losses[-1],
        "optimizer_steps": float(optimizer_steps),
    }

    root = Path(output_root or config.output.root)
    checkpoint = (
        root
        / f"agentic-{credit_mode}-seed{config.training.seed}"
        / f"checkpoint-{target_step:06d}"
    )
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        step=target_step,
        data_cursor={
            "step": target_step,
            "split_hash": agentic_task_split_hash(),
            "episode_ids": episode_ids,
            "generated_tokens": generated_tokens,
            "model_forwards": model_forwards,
            "processed_tokens": processed_tokens,
            "environment_steps": environment_steps,
            "optimizer_steps": optimizer_steps,
            "calculator_episodes": calculator_episodes,
            "lookup_episodes": lookup_episodes,
            "policy_version": target_step,
            "initial_policy_hash": initial_policy_hash,
            "rollout_generator_state": generator.get_state().tolist(),
        },
        metrics=all_records,
    )
    card = build_experiment_card(
        config,
        run_id=f"agentic-{credit_mode}-seed{config.training.seed}-step{target_step}",
        run_status="completed",
        step=target_step,
        wall_seconds=time.perf_counter() - started,
        metrics=metrics,
        data_split_hash=agentic_task_split_hash(),
        prompt_uids=episode_ids,
        optimized_prompt_uids=episode_ids,
        optimizer_steps=optimizer_steps,
        generated_tokens=generated_tokens,
        processed_tokens=processed_tokens,
        environment_steps=environment_steps,
        model_forwards=model_forwards,
        known_deviations=(
            "finite candidate-action scaffold; not open-vocabulary decoding",
            "repository TinyCausalLM and eight deterministic offline tasks",
            "single-trajectory on-policy REINFORCE; no asynchronous workers",
        ),
    )
    experiment_card = write_experiment_card(checkpoint, card)
    return AgenticRunResult(
        algorithm=config.algorithm.name,
        step=target_step,
        checkpoint=checkpoint,
        experiment_card=experiment_card,
        metrics=metrics,
        model=model,
    )


def _config_from_checkpoint(checkpoint: Path) -> ExperimentConfig:
    try:
        payload = json.loads(
            (checkpoint / "config.resolved.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot read checkpoint config: {error}") from error
    return ExperimentConfig.from_mapping(payload)


def evaluate_agentic_checkpoint(
    checkpoint: str | Path,
) -> tuple[str, int, dict[str, float]]:
    source = Path(checkpoint)
    config = _config_from_checkpoint(source)
    model, _ = _new_state(config)
    loaded = load_checkpoint(source, model=model, config=config, restore_rng=False)
    metrics, _, _ = _evaluate(model, seed=config.training.seed)
    return config.algorithm.name, loaded.step, metrics
