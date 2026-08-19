"""Fair C7 comparison and independent DAPO component ablations."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import time
from typing import Any

import torch

from rl_study.algorithms.group_policy import train_group_policy
from rl_study.algorithms.sft import train_sft
from rl_study.data import build_tiny_reasoning
from rl_study.models.roles import parameter_sha256


def _ids_hash(values: tuple[str, ...]) -> str | None:
    if not values:
        return None
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def run_group_comparison(
    *, steps: int = 3, prompt_batch_size: int = 1, group_size: int = 4, seed: int = 42
) -> dict[str, object]:
    if steps < 1 or prompt_batch_size < 1 or group_size < 2:
        raise ValueError("steps/prompts must be positive and group_size >= 2")
    started = time.perf_counter()
    dataset = build_tiny_reasoning(seed=seed)
    sft_started = time.perf_counter()
    sft = train_sft(seed=seed)
    sft_seconds = time.perf_counter() - sft_started
    initial_hash = parameter_sha256(sft.model)

    variants: dict[str, dict[str, Any]] = {
        "grpo_paper": {"algorithm": "grpo", "kl_coefficient": 0.04},
        "rloo_paper": {"algorithm": "rloo", "kl_coefficient": 0.02},
        "dr_grpo_paper": {"algorithm": "dr_grpo"},
        "gspo_paper": {
            "algorithm": "gspo",
            "clip_low": 0.0003,
            "clip_high": 0.0004,
        },
        "dapo_no_components": {"algorithm": "dapo"},
        "dapo_clip_higher_only": {
            "algorithm": "dapo",
            "clip_high": 0.28,
            "clip_higher": True,
        },
        "dapo_dynamic_sampling_only": {
            "algorithm": "dapo",
            "dynamic_sampling": True,
            "dynamic_sampling_multiplier": 4,
        },
        "dapo_token_loss_only": {
            "algorithm": "dapo",
            "token_level_loss": True,
        },
        "dapo_overlong_only": {
            "algorithm": "dapo",
            "overlong_reward": True,
        },
        "dapo_all_four": {
            "algorithm": "dapo",
            "clip_high": 0.28,
            "clip_higher": True,
            "dynamic_sampling": True,
            "dynamic_sampling_multiplier": 4,
            "token_level_loss": True,
            "overlong_reward": True,
        },
    }
    records: dict[str, object] = {}
    initial_hashes: dict[str, str] = {}
    common_prompt_hashes: dict[str, str | None] = {}
    for name, options in variants.items():
        policy = copy.deepcopy(sft.model)
        reference = copy.deepcopy(sft.model)
        initial_hashes[name] = parameter_sha256(policy)
        variant_started = time.perf_counter()
        result = train_group_policy(
            updates=steps,
            prompt_batch_size=prompt_batch_size,
            group_size=group_size,
            seed=seed,
            policy=policy,
            reference=reference,
            **options,
        )
        common_prompt_hashes[name] = _ids_hash(result.rollout_prompt_uids)
        records[name] = {
            "algorithm": result.algorithm,
            "options": dict(options),
            "final_policy_sha256": parameter_sha256(result.policy),
            "validation_exact_match": result.validation_exact_match,
            "validation_format_rate": result.validation_format_rate,
            "mean_train_reward": sum(result.mean_rewards) / len(result.mean_rewards),
            "mean_informative_group_rate": (
                sum(result.informative_group_rates)
                / len(result.informative_group_rates)
            ),
            "mean_response_length": (
                sum(result.mean_response_lengths) / len(result.mean_response_lengths)
            ),
            "optimizer_steps": result.optimizer_steps,
            "generated_tokens": result.generated_tokens,
            "model_forwards_after_sft": result.model_forwards,
            "rollout_prompt_occurrences": len(result.rollout_prompt_uids),
            "optimized_prompt_occurrences": len(result.optimized_prompt_uids),
            "rollout_prompt_ids_sha256": _ids_hash(result.rollout_prompt_uids),
            "optimized_prompt_ids_sha256": _ids_hash(result.optimized_prompt_uids),
            "rejected_dynamic_groups": result.rejected_dynamic_groups,
            "exhausted_dynamic_updates": result.exhausted_dynamic_updates,
            "wall_seconds": time.perf_counter() - variant_started,
        }
    if set(initial_hashes.values()) != {initial_hash}:
        raise RuntimeError("C7 variants did not share one initial policy state")

    non_dynamic_names = [
        name
        for name, options in variants.items()
        if not bool(options.get("dynamic_sampling", False))
    ]
    non_dynamic_prompt_hashes = {
        common_prompt_hashes[name] for name in non_dynamic_names
    }
    if len(non_dynamic_prompt_hashes) != 1:
        raise RuntimeError("non-dynamic C7 variants consumed different prompt IDs")

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
            "split_hash": dataset.split_hash,
            "initial_policy_sha256": initial_hash,
            "per_variant_initial_hashes": initial_hashes,
            "rollout_steps": steps,
            "prompt_batch_size": prompt_batch_size,
            "group_size": group_size,
            "max_new_tokens": 22,
            "non_dynamic_ordered_prompt_ids_sha256": next(
                iter(non_dynamic_prompt_hashes)
            ),
            "dynamic_sampling_budget_multiplier": 4,
            "stopping_rule": "fixed rollout steps; bounded dynamic candidates",
            "evaluation": "same 32 validation prompts, greedy, max_new_tokens=22",
        },
        "shared_sft": {
            "optimizer_steps": 100,
            "validation_token_accuracy": sft.validation_token_accuracy,
            "model_forwards": 202,
            "wall_seconds": sft_seconds,
        },
        "variants": records,
        "paper_variant_notes": {
            "grpo_paper": (
                "token ratio, per-sequence token mean, group mean/std advantage, "
                "beta=0.04 k3 KL"
            ),
            "rloo_paper": (
                "full sequence as one action, leave-one-out baseline, no PPO clipping"
            ),
            "dr_grpo_paper": "no group reward std and fixed max-token denominator",
            "gspo_paper": (
                "length-normalized sequence likelihood ratio and sequence clipping"
            ),
            "dapo_all_four": (
                "Clip-Higher + bounded Dynamic Sampling + token loss + "
                "soft overlong reward"
            ),
        },
        "interpretation_guardrails": [
            "Dynamic sampling intentionally consumes extra rollout prompts and tokens.",
            "An exhausted bounded dynamic update is reported, never silently replaced.",
            (
                "One tiny seed cannot establish an algorithm ranking or "
                "paper-scale result."
            ),
            (
                "DAPO code is paper-only clean-room; its unlicensed official "
                "repository was not used as implementation source."
            ),
        ],
        "sources": [
            "deepseekmath-grpo-2024",
            "rloo-2024",
            "dapo-2025",
            "dr-grpo-2025",
            "gspo-2025",
        ],
        "paper_reported": None,
        "upstream_reported": None,
        "wall_seconds_total": time.perf_counter() - started,
    }
