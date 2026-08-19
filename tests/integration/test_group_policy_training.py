from __future__ import annotations

import math

import pytest

from rl_study.algorithms.group_policy import train_group_policy


@pytest.mark.parametrize("algorithm", ["grpo", "rloo", "dr_grpo", "gspo"])
def test_group_policy_variants_run_finite_update(algorithm: str) -> None:
    result = train_group_policy(
        algorithm=algorithm,
        updates=1,
        prompt_batch_size=1,
        group_size=2,
        seed=7,
    )
    assert len(result.losses) == 1
    assert all(math.isfinite(value) for value in result.losses)
    assert result.optimizer_steps == 1
    assert result.generated_tokens > 0
    assert result.model_forwards > result.generated_tokens


def test_dapo_all_components_run_with_bounded_dynamic_sampling() -> None:
    result = train_group_policy(
        algorithm="dapo",
        updates=1,
        prompt_batch_size=1,
        group_size=2,
        seed=7,
        clip_high=0.28,
        dynamic_sampling=True,
        dynamic_sampling_multiplier=4,
        token_level_loss=True,
        clip_higher=True,
        overlong_reward=True,
    )
    assert len(result.losses) == 1
    assert math.isfinite(result.losses[0])
    assert result.generated_tokens > 0
    assert result.rejected_dynamic_groups >= 0
