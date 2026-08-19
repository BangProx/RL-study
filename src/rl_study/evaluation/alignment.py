"""Auditable SFT/RM/DPO/RLHF comparison on the shared toy task."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import time

import torch

from rl_study.algorithms.dpo import evaluate_dpo_preferences, train_dpo
from rl_study.algorithms.reward_model import train_reward_model
from rl_study.algorithms.rlhf_ppo import evaluate_generation, train_rlhf_ppo
from rl_study.algorithms.sft import train_sft
from rl_study.data import (
    TinyReasoningExample,
    build_preferences,
    build_tiny_reasoning,
)
from rl_study.models import TinyTokenizer
from rl_study.models.roles import parameter_sha256


def _shared_prompt_batches(
    *, steps: int, batch_size: int, seed: int, population: int
) -> tuple[tuple[int, ...], ...]:
    batches: list[tuple[int, ...]] = []
    for step in range(steps):
        generator = torch.Generator().manual_seed(seed + step * 1_000_003)
        indices = torch.randint(population, (batch_size,), generator=generator)
        batches.append(tuple(int(index) for index in indices))
    return tuple(batches)


def _ids_hash(values: tuple[str, ...]) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _generation_metrics(
    model: torch.nn.Module, examples: tuple[TinyReasoningExample, ...]
) -> dict[str, float]:
    from rl_study.models import TinyCausalLM

    if not isinstance(model, TinyCausalLM):
        raise TypeError("generation evaluation requires TinyCausalLM")
    exact, format_rate = evaluate_generation(model, examples, tokenizer=TinyTokenizer())
    return {"exact_match": exact, "format_rate": format_rate}


def run_alignment_comparison(
    *, steps: int = 8, batch_size: int = 8, seed: int = 42
) -> dict[str, object]:
    """Run equal-policy-step DPO/RLHF variants from one serialized SFT state."""
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    started = time.perf_counter()
    dataset = build_tiny_reasoning(seed=seed)
    tokenizer = TinyTokenizer()
    prompt_batches = _shared_prompt_batches(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        population=len(dataset.train),
    )

    sft_started = time.perf_counter()
    sft = train_sft(seed=seed)
    sft_seconds = time.perf_counter() - sft_started
    initial_hash = parameter_sha256(sft.model)

    reward_started = time.perf_counter()
    reward = train_reward_model(seed=seed)
    reward_seconds = time.perf_counter() - reward_started

    dpo_initial = copy.deepcopy(sft.model)
    rlhf_rm_initial = copy.deepcopy(sft.model)
    rlhf_verifier_initial = copy.deepcopy(sft.model)
    initial_hashes = {
        "dpo": parameter_sha256(dpo_initial),
        "rlhf_reward_model": parameter_sha256(rlhf_rm_initial),
        "rlhf_verifier": parameter_sha256(rlhf_verifier_initial),
    }
    if set(initial_hashes.values()) != {initial_hash}:
        raise RuntimeError("alignment policies do not share the same initial state")

    dpo_started = time.perf_counter()
    dpo = train_dpo(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        policy=dpo_initial,
        reference=copy.deepcopy(dpo_initial),
        prompt_batches=prompt_batches,
    )
    dpo_seconds = time.perf_counter() - dpo_started

    rlhf_rm_started = time.perf_counter()
    rlhf_rm = train_rlhf_ppo(
        updates=steps,
        batch_size=batch_size,
        seed=seed,
        policy=rlhf_rm_initial,
        reference=copy.deepcopy(rlhf_rm_initial),
        reward_model=copy.deepcopy(reward.model),
        reward_source="reward_model",
        update_epochs=1,
        prompt_batches=prompt_batches,
    )
    rlhf_rm_seconds = time.perf_counter() - rlhf_rm_started

    rlhf_verifier_started = time.perf_counter()
    rlhf_verifier = train_rlhf_ppo(
        updates=steps,
        batch_size=batch_size,
        seed=seed,
        policy=rlhf_verifier_initial,
        reference=copy.deepcopy(rlhf_verifier_initial),
        reward_source="verifier",
        update_epochs=1,
        prompt_batches=prompt_batches,
    )
    rlhf_verifier_seconds = time.perf_counter() - rlhf_verifier_started

    if not (dpo.prompt_uids == rlhf_rm.prompt_uids == rlhf_verifier.prompt_uids):
        raise RuntimeError("alignment algorithms consumed different prompt IDs")
    validation_preferences = build_preferences(dataset.validation)
    dpo_preference_accuracy = evaluate_dpo_preferences(
        dpo.policy,
        dpo.reference,
        validation_preferences,
        tokenizer=tokenizer,
        beta=0.1,
    )
    prompt_uids = dpo.prompt_uids
    diagnostic = reward.validation
    return {
        "schema_version": 1,
        "result_origin": "local_executed",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "fairness_contract": {
            "seed": seed,
            "initial_policy_sha256": initial_hash,
            "per_algorithm_initial_hashes": initial_hashes,
            "split_hash": dataset.split_hash,
            "ordered_prompt_ids_sha256": _ids_hash(prompt_uids),
            "prompt_occurrences": len(prompt_uids),
            "optimizer_steps_per_policy": steps,
            "batch_size": batch_size,
            "stopping_rule": "fixed optimizer-step budget; no metric early stop",
            "evaluation_prompt_ids": [
                example.uid for example in dataset.validation[:32]
            ],
            "evaluation_decoding": {
                "method": "greedy",
                "max_new_tokens": 22,
                "eos_token_id": tokenizer.eos_token_id,
            },
        },
        "shared_stages": {
            "sft": {
                "optimizer_steps": 100,
                "validation_token_accuracy": sft.validation_token_accuracy,
                "model_forwards": 202,
                "wall_seconds": sft_seconds,
            },
            "reward_model": {
                "optimizer_steps": 120,
                "heldout_preference_accuracy": diagnostic.preference_accuracy,
                "numeric_accuracy": diagnostic.numeric_accuracy,
                "format_accuracy": diagnostic.format_accuracy,
                "score_length_correlation": diagnostic.score_length_correlation,
                "model_forwards": 242,
                "wall_seconds": reward_seconds,
            },
        },
        "policies": {
            "sft": {
                "metrics": _generation_metrics(sft.model, dataset.validation[:32]),
                "final_policy_sha256": initial_hash,
                "optimizer_steps_after_sft": 0,
                "generated_tokens_train": 0,
                "processed_response_tokens_train": 0,
                "model_forwards_after_sft": 0,
            },
            "dpo": {
                "metrics": {
                    **_generation_metrics(dpo.policy, dataset.validation[:32]),
                    "preference_accuracy": dpo_preference_accuracy,
                },
                "final_policy_sha256": parameter_sha256(dpo.policy),
                "optimizer_steps_after_sft": steps,
                "generated_tokens_train": 0,
                "processed_response_tokens_train": (dpo.processed_response_tokens),
                "model_forwards_after_sft": dpo.model_forwards,
                "wall_seconds": dpo_seconds,
            },
            "rlhf_reward_model": {
                "metrics": _generation_metrics(rlhf_rm.policy, dataset.validation[:32]),
                "final_policy_sha256": parameter_sha256(rlhf_rm.policy),
                "optimizer_steps_after_sft": steps,
                "generated_tokens_train": rlhf_rm.generated_tokens,
                "processed_response_tokens_train": 0,
                "model_forwards_after_sft": rlhf_rm.model_forwards,
                "wall_seconds": rlhf_rm_seconds,
            },
            "rlhf_verifier_ablation": {
                "metrics": _generation_metrics(
                    rlhf_verifier.policy, dataset.validation[:32]
                ),
                "final_policy_sha256": parameter_sha256(rlhf_verifier.policy),
                "optimizer_steps_after_sft": steps,
                "generated_tokens_train": rlhf_verifier.generated_tokens,
                "processed_response_tokens_train": 0,
                "model_forwards_after_sft": rlhf_verifier.model_forwards,
                "wall_seconds": rlhf_verifier_seconds,
            },
        },
        "interpretation_guardrails": [
            (
                "DPO consumes two offline responses while RLHF generates one "
                "response; token budgets are reported rather than called equal."
            ),
            "Reward-model accuracy and policy exact match measure different objects.",
            "No algorithm is required to win this short stochastic toy run.",
            "Paper-scale results are not reproduced or implied.",
        ],
        "paper_reported": None,
        "upstream_reported": None,
        "wall_seconds_total": time.perf_counter() - started,
    }
