from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_study.agentic.credit import assign_step_credit
from rl_study.agentic.envs import CalculatorToolEnv
from rl_study.agentic.trajectory import (
    rollout_episode,
    tool_is_deterministic,
    update_policy,
    validate_trajectory,
)
from rl_study.agentic.types import AgentTrajectory
from rl_study.errors import RetokenizationDriftError, StaleTrajectoryError
from rl_study.models import TinyCausalLM, TinyLMConfig, TinyTokenizer
from rl_study.models.sequence import build_sequence_batch_from_token_ids


def _micro_model() -> TinyCausalLM:
    return TinyCausalLM(
        TinyLMConfig(
            max_sequence_length=128,
            hidden_size=32,
            num_heads=4,
            num_layers=1,
            intermediate_size=64,
        )
    )


def _trajectory() -> tuple[AgentTrajectory, TinyCausalLM, TinyTokenizer]:
    torch.manual_seed(4)
    model = _micro_model()
    tokenizer = TinyTokenizer()
    trajectory, _, _ = rollout_episode(
        model,
        tokenizer,
        CalculatorToolEnv(seed=0, max_steps=2),
        generator=torch.Generator().manual_seed(8),
        policy_version=0,
        task_index=0,
    )
    return trajectory, model, tokenizer


def test_rollout_preserves_original_action_tokens_and_masks_tool_output() -> None:
    trajectory, _, tokenizer = _trajectory()
    for step in trajectory.steps:
        assert step.action_token_ids == tuple(
            tokenizer.encode(step.action.raw_text, add_bos=False, add_eos=True)
        )
        decoded_context = tokenizer.decode(step.context_token_ids)
        if step.observation.step_id > 0:
            assert "O:" in decoded_context
        # The current response is separate; tool/environment text stays in the prompt.
        assert step.action_token_ids == step.candidate_action_token_ids[
            step.chosen_candidate_index
        ]
        batch = build_sequence_batch_from_token_ids(
            (step.context_token_ids,),
            (step.action_token_ids,),
            pad_token_id=tokenizer.pad_token_id,
            max_length=128,
        )
        assert int(batch.action_mask.sum()) == len(step.action_token_ids)
        assert not bool((batch.action_mask & batch.prompt_target_mask).any())


def test_policy_update_changes_parameters_with_finite_loss() -> None:
    trajectory, model, tokenizer = _trajectory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    update = update_policy(
        model,
        tokenizer,
        optimizer,
        trajectory,
        current_policy_version=0,
        credit_mode="discounted_returns",
        gamma=0.9,
    )
    assert torch.isfinite(torch.tensor(update.loss))
    assert update.model_forwards == len(trajectory.steps)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_credit_baseline_and_discounted_return_differ_without_double_counting() -> None:
    trajectory, _, _ = _trajectory()
    broadcast = assign_step_credit(
        trajectory, mode="broadcast_outcome", gamma=0.9
    )
    returns = assign_step_credit(trajectory, mode="discounted_returns", gamma=0.9)
    assert broadcast == (trajectory.steps[-1].outcome_reward,) * len(trajectory.steps)
    expected = trajectory.steps[-1].reward
    assert returns[-1] == pytest.approx(expected)
    if len(returns) > 1:
        assert returns[0] == pytest.approx(
            trajectory.steps[0].reward + 0.9 * returns[1]
        )


def test_stale_and_future_trajectories_are_rejected() -> None:
    trajectory, _, tokenizer = _trajectory()
    with pytest.raises(StaleTrajectoryError):
        validate_trajectory(trajectory, tokenizer, current_policy_version=1)
    with pytest.raises(StaleTrajectoryError):
        validate_trajectory(trajectory, tokenizer, current_policy_version=-1)
    validate_trajectory(
        trajectory, tokenizer, current_policy_version=1, maximum_policy_lag=1
    )


def test_retokenization_drift_is_rejected() -> None:
    trajectory, _, tokenizer = _trajectory()
    first = trajectory.steps[0]
    corrupted = replace(first, action=replace(first.action, raw_text="FINAL changed"))
    drifted = AgentTrajectory(
        episode_id=trajectory.episode_id,
        steps=(corrupted, *trajectory.steps[1:]),
    )
    with pytest.raises(RetokenizationDriftError):
        validate_trajectory(drifted, tokenizer, current_policy_version=0)


def test_context_overflow_is_not_silently_truncated() -> None:
    model = TinyCausalLM(TinyLMConfig.micro())
    with pytest.raises(ValueError, match="exceeds model context"):
        rollout_episode(
            model,
            TinyTokenizer(),
            CalculatorToolEnv(seed=0),
            generator=torch.Generator().manual_seed(0),
            policy_version=0,
            task_index=0,
        )


def test_tool_nondeterminism_detector() -> None:
    values = iter((1, 2, 3))
    assert tool_is_deterministic(lambda: "same")
    assert not tool_is_deterministic(lambda: next(values))
