from __future__ import annotations

import copy
import math

from rl_study.algorithms.dpo import train_dpo
from rl_study.algorithms.rlhf_ppo import train_rlhf_ppo
from rl_study.algorithms.sft import train_sft


def test_dpo_and_rlhf_ppo_update_from_same_sft_policy() -> None:
    sft_policy = train_sft(steps=40, batch_size=8, seed=3).model
    dpo = train_dpo(
        steps=8,
        batch_size=4,
        seed=3,
        policy=copy.deepcopy(sft_policy),
        reference=copy.deepcopy(sft_policy),
    )
    assert all(math.isfinite(value) for value in dpo.losses)
    assert 0.0 <= dpo.validation_accuracy <= 1.0

    rlhf = train_rlhf_ppo(
        updates=2,
        batch_size=4,
        seed=3,
        policy=copy.deepcopy(sft_policy),
        reference=copy.deepcopy(sft_policy),
        reward_source="verifier",
        update_epochs=1,
    )
    assert all(math.isfinite(value) for value in rlhf.policy_losses)
    assert len(rlhf.mean_task_rewards) == 2
    assert 0.0 <= rlhf.validation_exact_match <= 1.0
    assert 0.0 <= rlhf.validation_format_rate <= 1.0
