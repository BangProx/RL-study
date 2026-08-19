from __future__ import annotations

import math

from rl_study.algorithms.dqn import train_dqn
from rl_study.algorithms.policy_gradient import (
    train_actor_critic,
    train_reinforce,
)
from rl_study.algorithms.ppo import train_ppo


def test_classic_trainers_produce_finite_updates() -> None:
    dqn = train_dqn(episodes=30, seed=1)
    reinforce = train_reinforce(episodes=30, seed=1)
    actor_critic = train_actor_critic(episodes=30, seed=1)
    ppo = train_ppo(episodes=20, seed=1, update_epochs=2)
    assert dqn.losses and all(math.isfinite(value) for value in dqn.losses)
    assert reinforce.losses and all(math.isfinite(value) for value in reinforce.losses)
    assert actor_critic.losses and all(
        math.isfinite(value) for value in actor_critic.losses
    )
    assert ppo.policy_losses and all(
        math.isfinite(value) for value in ppo.policy_losses
    )
