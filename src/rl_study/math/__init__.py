"""Small, explicit mathematical building blocks used by every trainer."""

from rl_study.math.probability import (
    categorical_cross_entropy,
    categorical_entropy,
    categorical_kl,
    selected_log_probs,
)
from rl_study.math.reductions import masked_mean, masked_sequence_mean, masked_sum
from rl_study.math.returns import discounted_returns, generalized_advantage_estimate

__all__ = [
    "categorical_cross_entropy",
    "categorical_entropy",
    "categorical_kl",
    "discounted_returns",
    "generalized_advantage_estimate",
    "masked_mean",
    "masked_sequence_mean",
    "masked_sum",
    "selected_log_probs",
]
