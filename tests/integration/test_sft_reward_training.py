from __future__ import annotations

import math

from rl_study.algorithms.reward_model import train_reward_model
from rl_study.algorithms.sft import train_sft


def test_small_sft_and_reward_model_training() -> None:
    sft = train_sft(steps=30, batch_size=8, seed=7)
    assert all(math.isfinite(value) for value in sft.losses)
    assert sum(sft.losses[-5:]) / 5 < sum(sft.losses[:5]) / 5
    assert 0.0 <= sft.validation_token_accuracy <= 1.0

    reward = train_reward_model(steps=30, batch_size=8, seed=7)
    assert all(math.isfinite(value) for value in reward.losses)
    assert sum(reward.losses[-5:]) / 5 < sum(reward.losses[:5]) / 5
    assert 0.0 <= reward.validation.preference_accuracy <= 1.0
    assert -1.0 <= reward.validation.score_length_correlation <= 1.0
