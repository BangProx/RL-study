"""Toy online trainers for GRPO, DAPO, RLOO, Dr. GRPO, and GSPO."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from rl_study.algorithms.dapo import (
    dapo_loss,
    dynamic_sampling_filter,
    overlong_reward_shaping,
)
from rl_study.algorithms.grpo import (
    dr_grpo_advantages,
    group_relative_advantages,
    grpo_loss,
    gspo_sequence_loss,
    rloo_advantages,
    rloo_sequence_loss,
)
from rl_study.algorithms.rlhf_ppo import (
    collect_rollouts,
    evaluate_generation_detailed,
)
from rl_study.algorithms.sft import train_sft
from rl_study.data import build_tiny_reasoning, verify_response
from rl_study.models import SequenceBatch, TinyCausalLM, TinyTokenizer
from rl_study.models.roles import assert_frozen, freeze_module, parameter_sha256
from rl_study.models.sequence import response_token_log_probs

GROUP_ALGORITHMS = frozenset({"grpo", "dapo", "rloo", "dr_grpo", "gspo"})


@dataclass(frozen=True, slots=True)
class GroupPolicyTrainResult:
    algorithm: str
    policy: TinyCausalLM
    reference: TinyCausalLM
    optimizer: torch.optim.Optimizer
    losses: tuple[float, ...]
    mean_rewards: tuple[float, ...]
    exact_match_rates: tuple[float, ...]
    format_rates: tuple[float, ...]
    informative_group_rates: tuple[float, ...]
    mean_response_lengths: tuple[float, ...]
    validation_exact_match: float
    validation_format_rate: float
    generated_tokens: int
    model_forwards: int
    optimizer_steps: int
    rejected_dynamic_groups: int
    exhausted_dynamic_updates: int
    rollout_prompt_uids: tuple[str, ...]
    optimized_prompt_uids: tuple[str, ...]
    reference_hash: str


def _select_batch(batch: SequenceBatch, indices: torch.Tensor) -> SequenceBatch:
    return SequenceBatch(
        input_ids=batch.input_ids.index_select(0, indices),
        attention_mask=batch.attention_mask.index_select(0, indices),
        prompt_target_mask=batch.prompt_target_mask.index_select(0, indices),
        action_mask=batch.action_mask.index_select(0, indices),
        prompt_lengths=batch.prompt_lengths.index_select(0, indices),
        response_lengths=batch.response_lengths.index_select(0, indices),
    )


def _flat_group_rows(group_indices: torch.Tensor, group_size: int) -> torch.Tensor:
    offsets = torch.arange(group_size).unsqueeze(0)
    return (group_indices.unsqueeze(1) * group_size + offsets).reshape(-1)


def train_group_policy(
    *,
    algorithm: str = "grpo",
    updates: int = 8,
    prompt_batch_size: int = 2,
    group_size: int = 4,
    seed: int = 42,
    learning_rate: float = 5e-4,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    kl_coefficient: float = 0.0,
    update_epochs: int = 1,
    max_new_tokens: int = 22,
    dynamic_sampling: bool = False,
    dynamic_sampling_multiplier: int = 4,
    token_level_loss: bool = False,
    clip_higher: bool = False,
    overlong_reward: bool = False,
    overlong_buffer_length: int = 4,
    overlong_penalty_scale: float = 1.0,
    policy: TinyCausalLM | None = None,
    reference: TinyCausalLM | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    update_offset: int = 0,
) -> GroupPolicyTrainResult:
    if algorithm not in GROUP_ALGORITHMS:
        raise ValueError(f"unknown group algorithm: {algorithm}")
    if updates < 1 or prompt_batch_size < 1 or group_size < 2:
        raise ValueError("updates/prompts must be positive and group_size >= 2")
    if update_epochs < 1 or update_offset < 0 or max_new_tokens < 1:
        raise ValueError("epochs/tokens must be positive and offset non-negative")
    if dynamic_sampling_multiplier < 1:
        raise ValueError("dynamic_sampling_multiplier must be positive")
    if algorithm != "dapo" and (
        dynamic_sampling or token_level_loss or clip_higher or overlong_reward
    ):
        raise ValueError("DAPO component toggles require algorithm='dapo'")
    if algorithm == "rloo" and update_epochs != 1:
        raise ValueError("paper RLOO is on-policy and requires update_epochs=1")

    dataset = build_tiny_reasoning(seed=seed)
    tokenizer = TinyTokenizer()
    if policy is None:
        policy = train_sft(seed=seed).model
    if reference is None:
        reference = copy.deepcopy(policy)
    freeze_module(reference)
    reference_hash = parameter_sha256(reference)
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=learning_rate, weight_decay=0.0
        )

    losses: list[float] = []
    mean_rewards: list[float] = []
    exact_match_rates: list[float] = []
    format_rates: list[float] = []
    informative_group_rates: list[float] = []
    mean_response_lengths: list[float] = []
    rollout_prompt_uids: list[str] = []
    optimized_prompt_uids: list[str] = []
    generated_tokens = 0
    model_forwards = 0
    optimizer_steps = 0
    rejected_dynamic_groups = 0
    exhausted_dynamic_updates = 0

    for local_update in range(updates):
        global_update = update_offset + local_update
        old_policy = copy.deepcopy(policy)
        freeze_module(old_policy)
        old_hash = parameter_sha256(old_policy)
        candidate_groups = prompt_batch_size
        if algorithm == "dapo" and dynamic_sampling:
            candidate_groups *= dynamic_sampling_multiplier
        generator = torch.Generator().manual_seed(seed + global_update * 1_000_003)
        prompt_indices = torch.randint(
            len(dataset.train), (candidate_groups,), generator=generator
        ).tolist()
        prompt_examples = tuple(dataset.train[int(index)] for index in prompt_indices)
        repeated_examples = tuple(
            example for example in prompt_examples for _ in range(group_size)
        )
        rollout = collect_rollouts(
            old_policy,
            repeated_examples,
            tokenizer=tokenizer,
            seed=seed + global_update * 1_000_037,
            max_new_tokens=max_new_tokens,
        )
        rollout_prompt_uids.extend(rollout.prompt_uids)
        local_generated = int(rollout.batch.action_mask.sum())
        generated_tokens += local_generated
        model_forwards += local_generated
        group_rewards = rollout.exact_rewards.reshape(candidate_groups, group_size)
        informative = group_rewards.max(dim=1).values > group_rewards.min(dim=1).values
        informative_group_rates.append(float(informative.to(torch.float32).mean()))

        if algorithm == "dapo" and dynamic_sampling:
            selection = dynamic_sampling_filter(
                group_rewards, required_groups=prompt_batch_size
            )
            rejected_dynamic_groups += selection.rejected_groups
            exhausted_dynamic_updates += int(selection.exhausted)
            selected_groups = selection.selected_group_indices
        else:
            selected_groups = torch.arange(candidate_groups)

        if selected_groups.numel() == 0:
            losses.append(0.0)
            mean_rewards.append(float(group_rewards.mean()))
            exact_match_rates.append(0.0)
            format_rates.append(float(rollout.format_valid.to(torch.float32).mean()))
            mean_response_lengths.append(
                float(rollout.batch.response_lengths.to(torch.float32).mean())
            )
            continue

        selected_rows = _flat_group_rows(selected_groups, group_size)
        selected_batch = _select_batch(rollout.batch, selected_rows)
        selected_rewards = group_rewards.index_select(0, selected_groups)
        selected_examples = tuple(
            repeated_examples[int(index)] for index in selected_rows
        )
        selected_responses = tuple(
            rollout.responses[int(index)] for index in selected_rows
        )
        selected_formats = rollout.format_valid.index_select(0, selected_rows)
        selected_prompt_uids = tuple(
            rollout.prompt_uids[int(index)] for index in selected_rows
        )
        optimized_prompt_uids.extend(selected_prompt_uids)

        if algorithm == "dapo" and overlong_reward:
            penalties = overlong_reward_shaping(
                selected_batch.response_lengths,
                max_response_length=max_new_tokens,
                buffer_length=overlong_buffer_length,
                penalty_scale=overlong_penalty_scale,
            ).reshape_as(selected_rewards)
            selected_rewards = selected_rewards + penalties

        with torch.no_grad():
            old_log_probs = response_token_log_probs(old_policy, selected_batch)
            reference_log_probs = response_token_log_probs(reference, selected_batch)
        model_forwards += 2

        if algorithm == "rloo":
            sequence_kl = (
                (old_log_probs - reference_log_probs) * selected_batch.action_mask
            ).sum(dim=1)
            shaped_rewards = selected_rewards.reshape(-1) - kl_coefficient * sequence_kl
            advantages = rloo_advantages(
                shaped_rewards.reshape(selected_groups.numel(), group_size)
            ).reshape(-1)
        elif algorithm == "dr_grpo":
            advantages = dr_grpo_advantages(selected_rewards).reshape(-1)
        else:
            advantages = group_relative_advantages(selected_rewards).advantages.reshape(
                -1
            )

        update_loss = 0.0
        for _ in range(update_epochs):
            current_log_probs = response_token_log_probs(policy, selected_batch)
            model_forwards += 1
            if algorithm == "rloo":
                loss = rloo_sequence_loss(
                    current_log_probs,
                    advantages,
                    selected_batch.action_mask,
                ).loss
            elif algorithm == "dapo":
                loss = dapo_loss(
                    current_log_probs,
                    old_log_probs,
                    reference_log_probs,
                    advantages,
                    selected_batch.action_mask,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    kl_coefficient=kl_coefficient,
                    use_clip_higher=clip_higher,
                    use_token_level_loss=token_level_loss,
                ).loss
            elif algorithm == "gspo":
                loss = gspo_sequence_loss(
                    current_log_probs,
                    old_log_probs,
                    reference_log_probs,
                    advantages,
                    selected_batch.action_mask,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    kl_coefficient=kl_coefficient,
                ).loss
            else:
                reduction = "dr_grpo" if algorithm == "dr_grpo" else "sequence_mean"
                loss = grpo_loss(
                    current_log_probs,
                    old_log_probs,
                    reference_log_probs,
                    advantages,
                    selected_batch.action_mask,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    kl_coefficient=kl_coefficient,
                    reduction=reduction,
                    fixed_response_length=(
                        max_new_tokens if algorithm == "dr_grpo" else None
                    ),
                ).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"{algorithm} produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            update_loss = float(loss.detach())
        losses.append(update_loss)
        mean_rewards.append(float(selected_rewards.mean()))
        exact_match_rates.append(
            sum(
                verify_response(example, response)
                for example, response in zip(
                    selected_examples, selected_responses, strict=True
                )
            )
            / len(selected_examples)
        )
        format_rates.append(float(selected_formats.to(torch.float32).mean()))
        mean_response_lengths.append(
            float(selected_batch.response_lengths.to(torch.float32).mean())
        )
        if parameter_sha256(old_policy) != old_hash:
            raise RuntimeError("old group rollout policy changed during update")

    validation = evaluate_generation_detailed(
        policy, dataset.validation[:32], tokenizer=tokenizer
    )
    model_forwards += validation.generated_tokens
    assert_frozen(reference, role=f"{algorithm} reference")
    if parameter_sha256(reference) != reference_hash:
        raise RuntimeError(f"{algorithm} reference parameters changed")
    return GroupPolicyTrainResult(
        algorithm=algorithm,
        policy=policy,
        reference=reference,
        optimizer=optimizer,
        losses=tuple(losses),
        mean_rewards=tuple(mean_rewards),
        exact_match_rates=tuple(exact_match_rates),
        format_rates=tuple(format_rates),
        informative_group_rates=tuple(informative_group_rates),
        mean_response_lengths=tuple(mean_response_lengths),
        validation_exact_match=validation.exact_match,
        validation_format_rate=validation.format_rate,
        generated_tokens=generated_tokens,
        model_forwards=model_forwards,
        optimizer_steps=optimizer_steps,
        rejected_dynamic_groups=rejected_dynamic_groups,
        exhausted_dynamic_updates=exhausted_dynamic_updates,
        rollout_prompt_uids=tuple(rollout_prompt_uids),
        optimized_prompt_uids=tuple(optimized_prompt_uids),
        reference_hash=reference_hash,
    )
