"""Token-level toy RLHF-PPO with verifier or frozen reward-model feedback."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.algorithms.reward_model import train_reward_model
from rl_study.algorithms.sft import train_sft
from rl_study.data import (
    TinyReasoningExample,
    build_tiny_reasoning,
    has_valid_format,
    verifier_reward,
    verify_response,
)
from rl_study.math import masked_mean
from rl_study.models import (
    SequenceBatch,
    TinyCausalLM,
    TinyRewardModel,
    TinyTokenizer,
    TinyValueModel,
    build_sequence_batch_from_token_ids,
)
from rl_study.models.roles import assert_frozen, freeze_module, parameter_sha256
from rl_study.models.sequence import response_token_log_probs


@dataclass(frozen=True, slots=True)
class GeneratedRollout:
    batch: SequenceBatch
    responses: tuple[str, ...]
    exact_rewards: Tensor
    format_valid: Tensor
    prompt_uids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RLHFRewardOutput:
    token_rewards: Tensor
    sampled_kl: Tensor
    non_score_rewards: Tensor
    total_rewards: Tensor


@dataclass(frozen=True, slots=True)
class RLHFPPOOutput:
    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
    ratio: Tensor


@dataclass(frozen=True, slots=True)
class RLHFTrainResult:
    policy: TinyCausalLM
    reference: TinyCausalLM
    reward_model: TinyRewardModel | None
    value_model: TinyValueModel
    policy_optimizer: torch.optim.Optimizer
    value_optimizer: torch.optim.Optimizer
    policy_losses: tuple[float, ...]
    mean_task_rewards: tuple[float, ...]
    mean_sampled_kls: tuple[float, ...]
    exact_match_rates: tuple[float, ...]
    format_rates: tuple[float, ...]
    validation_exact_match: float
    validation_format_rate: float
    reference_hash: str
    reward_hash: str | None
    prompt_uids: tuple[str, ...]
    generated_tokens: int
    model_forwards: int


@dataclass(frozen=True, slots=True)
class GenerationEvaluation:
    exact_match: float
    format_rate: float
    generated_tokens: int


def compose_rlhf_rewards(
    task_rewards: Tensor,
    policy_log_probs: Tensor,
    reference_log_probs: Tensor,
    action_mask: Tensor,
    *,
    kl_coefficient: float,
) -> RLHFRewardOutput:
    if policy_log_probs.shape != reference_log_probs.shape:
        raise ValueError("policy and reference log-probs must share a shape")
    if action_mask.shape != policy_log_probs.shape or action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be bool and match token log-probs")
    if task_rewards.shape != (policy_log_probs.shape[0],):
        raise ValueError("task_rewards must be [B]")
    if kl_coefficient < 0:
        raise ValueError("kl_coefficient must be non-negative")
    if torch.any(action_mask.sum(dim=-1) == 0):
        raise ValueError("each trajectory needs at least one action token")
    sampled_kl = (policy_log_probs - reference_log_probs) * action_mask
    token_rewards = -kl_coefficient * sampled_kl
    for row in range(action_mask.shape[0]):
        last_action = int(torch.where(action_mask[row])[0][-1])
        token_rewards[row, last_action] += task_rewards[row]
    non_score_rewards = (-kl_coefficient * sampled_kl).sum(dim=-1)
    total_rewards = token_rewards.sum(dim=-1)
    return RLHFRewardOutput(
        token_rewards=token_rewards,
        sampled_kl=sampled_kl,
        non_score_rewards=non_score_rewards,
        total_rewards=total_rewards,
    )


def masked_token_returns(
    token_rewards: Tensor, action_mask: Tensor, *, gamma: float = 1.0
) -> Tensor:
    if token_rewards.shape != action_mask.shape or action_mask.dtype != torch.bool:
        raise ValueError("token_rewards/action_mask shape or dtype mismatch")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    returns = torch.zeros_like(token_rewards)
    running = torch.zeros(
        token_rewards.shape[0],
        dtype=token_rewards.dtype,
        device=token_rewards.device,
    )
    for index in range(token_rewards.shape[1] - 1, -1, -1):
        active = action_mask[:, index]
        updated = token_rewards[:, index] + gamma * running
        running = torch.where(active, updated, running)
        returns[:, index] = torch.where(active, running, torch.zeros_like(running))
    return returns


def rlhf_ppo_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    values: Tensor,
    returns: Tensor,
    entropy: Tensor,
    action_mask: Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> RLHFPPOOutput:
    tensors = (current_log_probs, old_log_probs, advantages, values, returns, entropy)
    if any(tensor.shape != action_mask.shape for tensor in tensors):
        raise ValueError("all RLHF-PPO token tensors must match action_mask")
    if action_mask.dtype != torch.bool:
        raise TypeError("action_mask must be bool")
    log_ratio = current_log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    detached_advantages = advantages.detach()
    objective = torch.minimum(
        ratio * detached_advantages, clipped_ratio * detached_advantages
    )
    policy_loss = -masked_mean(objective, action_mask)
    value_loss = 0.5 * masked_mean((values - returns.detach()).square(), action_mask)
    mean_entropy = masked_mean(entropy, action_mask)
    approximate_kl = masked_mean((ratio - 1.0) - log_ratio, action_mask)
    clipped = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    clip_fraction = masked_mean(clipped.to(ratio.dtype), action_mask)
    return RLHFPPOOutput(
        loss=policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * mean_entropy,
        policy_loss=policy_loss,
        value_loss=value_loss,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        ratio=ratio,
    )


@torch.no_grad()
def collect_rollouts(
    policy: TinyCausalLM,
    examples: tuple[TinyReasoningExample, ...],
    *,
    tokenizer: TinyTokenizer,
    seed: int,
    max_new_tokens: int = 22,
    temperature: float = 0.8,
    do_sample: bool = True,
) -> GeneratedRollout:
    prompt_rows: list[list[int]] = []
    response_rows: list[list[int]] = []
    responses: list[str] = []
    for index, example in enumerate(examples):
        prompt_ids = tokenizer.encode(example.prompt, add_bos=True, add_eos=False)
        available = policy.config.max_sequence_length - len(prompt_ids)
        requested = min(max_new_tokens, available)
        if requested < 1:
            raise ValueError("prompt leaves no context for an action token")
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.int64)
        generator = torch.Generator().manual_seed(seed + index * 1_000_033)
        generated = policy.generate(
            prompt_tensor,
            max_new_tokens=requested,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            generator=generator,
            do_sample=do_sample,
        )
        response_ids = generated[0, len(prompt_ids) :].tolist()
        prompt_rows.append(prompt_ids)
        response_rows.append(response_ids)
        responses.append(tokenizer.decode(response_ids))
    batch = build_sequence_batch_from_token_ids(
        prompt_rows,
        response_rows,
        pad_token_id=tokenizer.pad_token_id,
        max_length=policy.config.max_sequence_length,
    )
    return GeneratedRollout(
        batch=batch,
        responses=tuple(responses),
        exact_rewards=torch.tensor(
            [
                verifier_reward(example, response)
                for example, response in zip(examples, responses, strict=True)
            ],
            dtype=torch.float32,
        ),
        format_valid=torch.tensor(
            [has_valid_format(response) for response in responses], dtype=torch.bool
        ),
        prompt_uids=tuple(example.uid for example in examples),
    )


def _token_entropy(model: TinyCausalLM, batch: SequenceBatch) -> Tensor:
    logits = model(batch.input_ids, batch.attention_mask).logits[:, :-1]
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


@torch.no_grad()
def evaluate_generation_detailed(
    policy: TinyCausalLM,
    examples: tuple[TinyReasoningExample, ...],
    *,
    tokenizer: TinyTokenizer,
) -> GenerationEvaluation:
    rollout = collect_rollouts(
        policy,
        examples,
        tokenizer=tokenizer,
        seed=123_456,
        do_sample=False,
    )
    exact = [
        verify_response(example, response)
        for example, response in zip(examples, rollout.responses, strict=True)
    ]
    return GenerationEvaluation(
        exact_match=sum(exact) / len(exact),
        format_rate=float(rollout.format_valid.float().mean()),
        generated_tokens=int(rollout.batch.action_mask.sum()),
    )


@torch.no_grad()
def evaluate_generation(
    policy: TinyCausalLM,
    examples: tuple[TinyReasoningExample, ...],
    *,
    tokenizer: TinyTokenizer,
) -> tuple[float, float]:
    result = evaluate_generation_detailed(policy, examples, tokenizer=tokenizer)
    return result.exact_match, result.format_rate


def train_rlhf_ppo(
    *,
    updates: int = 20,
    batch_size: int = 8,
    seed: int = 42,
    kl_coefficient: float = 0.02,
    reward_source: str = "verifier",
    update_epochs: int = 2,
    policy: TinyCausalLM | None = None,
    reference: TinyCausalLM | None = None,
    reward_model: TinyRewardModel | None = None,
    value_model: TinyValueModel | None = None,
    policy_optimizer: torch.optim.Optimizer | None = None,
    value_optimizer: torch.optim.Optimizer | None = None,
    update_offset: int = 0,
    prompt_batches: tuple[tuple[int, ...], ...] | None = None,
) -> RLHFTrainResult:
    if updates < 1 or batch_size < 1 or update_offset < 0:
        raise ValueError("updates/batch must be positive and offset non-negative")
    if prompt_batches is not None and len(prompt_batches) != updates:
        raise ValueError("prompt_batches must contain one batch per RLHF update")
    if reward_source not in {"verifier", "reward_model"}:
        raise ValueError("reward_source must be verifier or reward_model")
    dataset = build_tiny_reasoning(seed=seed)
    tokenizer = TinyTokenizer()
    if policy is None:
        policy = train_sft(seed=seed).model
    if reference is None:
        reference = copy.deepcopy(policy)
    freeze_module(reference)
    reference_hash = parameter_sha256(reference)
    if reward_source == "reward_model" and reward_model is None:
        reward_model = train_reward_model(seed=seed).model
    reward_hash: str | None = None
    if reward_model is not None:
        freeze_module(reward_model)
        reward_hash = parameter_sha256(reward_model)
    if value_model is None:
        value_model = TinyValueModel(copy.deepcopy(policy))
    if policy_optimizer is None:
        policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4)
    if value_optimizer is None:
        value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=1e-3)

    policy_losses: list[float] = []
    mean_task_rewards: list[float] = []
    mean_sampled_kls: list[float] = []
    exact_match_rates: list[float] = []
    format_rates: list[float] = []
    prompt_uids: list[str] = []
    generated_tokens = 0
    for local_update in range(updates):
        global_update = update_offset + local_update
        if prompt_batches is None:
            generator = torch.Generator().manual_seed(seed + global_update * 1_000_003)
            indices = torch.randint(
                len(dataset.train), (batch_size,), generator=generator
            ).tolist()
        else:
            indices = list(prompt_batches[local_update])
            if len(indices) != batch_size or any(
                index < 0 or index >= len(dataset.train) for index in indices
            ):
                raise ValueError(
                    "each RLHF prompt batch must contain valid train indices"
                )
        examples = tuple(dataset.train[int(index)] for index in indices)
        old_policy = copy.deepcopy(policy)
        freeze_module(old_policy)
        old_hash = parameter_sha256(old_policy)
        rollout = collect_rollouts(
            old_policy,
            examples,
            tokenizer=tokenizer,
            seed=seed + global_update * 1_000_037,
        )
        prompt_uids.extend(rollout.prompt_uids)
        generated_tokens += int(rollout.batch.action_mask.sum())
        with torch.no_grad():
            old_log_probs = response_token_log_probs(old_policy, rollout.batch)
            reference_log_probs = response_token_log_probs(reference, rollout.batch)
            old_values = value_model(
                rollout.batch.input_ids, rollout.batch.attention_mask
            )[:, :-1]
            if reward_source == "verifier":
                task_rewards = rollout.exact_rewards
            else:
                if reward_model is None:
                    raise RuntimeError("reward model is missing")
                task_rewards = reward_model(
                    rollout.batch.input_ids, rollout.batch.attention_mask
                )
            reward_output = compose_rlhf_rewards(
                task_rewards,
                old_log_probs,
                reference_log_probs,
                rollout.batch.action_mask,
                kl_coefficient=kl_coefficient,
            )
            returns = masked_token_returns(
                reward_output.token_rewards, rollout.batch.action_mask
            )
            advantages = returns - old_values
            valid_advantages = advantages[rollout.batch.action_mask]
            if valid_advantages.numel() > 1:
                normalized = (valid_advantages - valid_advantages.mean()) / (
                    valid_advantages.std(unbiased=False) + 1e-8
                )
                advantages = advantages.clone()
                advantages[rollout.batch.action_mask] = normalized

        for _ in range(update_epochs):
            current_log_probs = response_token_log_probs(policy, rollout.batch)
            current_values = value_model(
                rollout.batch.input_ids, rollout.batch.attention_mask
            )[:, :-1]
            entropy = _token_entropy(policy, rollout.batch)
            loss_output = rlhf_ppo_loss(
                current_log_probs,
                old_log_probs,
                advantages,
                current_values,
                returns,
                entropy,
                rollout.batch.action_mask,
            )
            policy_optimizer.zero_grad(set_to_none=True)
            value_optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss_output.loss)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), 1.0)
            policy_optimizer.step()
            value_optimizer.step()
            policy_losses.append(float(loss_output.policy_loss.detach()))
        if parameter_sha256(old_policy) != old_hash:
            raise RuntimeError("old rollout policy changed during PPO update")
        mean_task_rewards.append(float(task_rewards.mean()))
        mean_sampled_kls.append(
            float(masked_mean(reward_output.sampled_kl, rollout.batch.action_mask))
        )
        exact_match_rates.append(
            sum(
                verify_response(example, response)
                for example, response in zip(examples, rollout.responses, strict=True)
            )
            / len(examples)
        )
        format_rates.append(float(rollout.format_valid.float().mean()))

    assert_frozen(reference, role="RLHF reference")
    if parameter_sha256(reference) != reference_hash:
        raise RuntimeError("RLHF reference parameters changed")
    if reward_model is not None:
        assert_frozen(reward_model, role="RLHF reward model")
        if parameter_sha256(reward_model) != reward_hash:
            raise RuntimeError("RLHF reward model parameters changed")
    validation = evaluate_generation_detailed(
        policy, dataset.validation[:32], tokenizer=tokenizer
    )
    non_generation_forwards_per_update = (
        3 + int(reward_source == "reward_model") + 3 * update_epochs
    )
    return RLHFTrainResult(
        policy=policy,
        reference=reference,
        reward_model=reward_model,
        value_model=value_model,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        policy_losses=tuple(policy_losses),
        mean_task_rewards=tuple(mean_task_rewards),
        mean_sampled_kls=tuple(mean_sampled_kls),
        exact_match_rates=tuple(exact_match_rates),
        format_rates=tuple(format_rates),
        validation_exact_match=validation.exact_match,
        validation_format_rate=validation.format_rate,
        reference_hash=reference_hash,
        reward_hash=reward_hash,
        prompt_uids=tuple(prompt_uids),
        generated_tokens=generated_tokens,
        model_forwards=(
            generated_tokens
            + validation.generated_tokens
            + updates * non_generation_forwards_per_update
        ),
    )
