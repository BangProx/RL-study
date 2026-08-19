"""Finite-action language policy whose scores come only from action tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.models.sequence import (
    build_sequence_batch_from_token_ids,
    response_token_log_probs,
)
from rl_study.models.tiny_lm import TinyCausalLM
from rl_study.models.tiny_tokenizer import TinyTokenizer


@dataclass(frozen=True, slots=True)
class ActionSelection:
    raw_text: str
    context_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    candidate_action_token_ids: tuple[tuple[int, ...], ...]
    chosen_index: int
    behavior_logprob: float


def _candidate_scores_from_ids(
    model: TinyCausalLM,
    *,
    context_token_ids: tuple[int, ...],
    candidate_action_token_ids: tuple[tuple[int, ...], ...],
) -> Tensor:
    if not candidate_action_token_ids:
        raise ValueError("at least one candidate action is required")
    batch = build_sequence_batch_from_token_ids(
        (context_token_ids,) * len(candidate_action_token_ids),
        candidate_action_token_ids,
        pad_token_id=0,
        max_length=model.config.max_sequence_length,
    )
    token_log_probs = response_token_log_probs(model, batch)
    lengths = batch.action_mask.sum(dim=-1).clamp_min(1)
    return (token_log_probs * batch.action_mask).sum(dim=-1) / lengths


def candidate_log_probs(
    model: TinyCausalLM,
    *,
    context_token_ids: tuple[int, ...],
    candidate_action_token_ids: tuple[tuple[int, ...], ...],
) -> Tensor:
    return torch.log_softmax(
        _candidate_scores_from_ids(
            model,
            context_token_ids=context_token_ids,
            candidate_action_token_ids=candidate_action_token_ids,
        ),
        dim=0,
    )


@torch.no_grad()
def select_action(
    model: TinyCausalLM,
    tokenizer: TinyTokenizer,
    *,
    observation_text: str,
    candidates: tuple[str, ...],
    generator: torch.Generator,
    greedy: bool = False,
) -> ActionSelection:
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate actions must be non-empty and unique")
    prompt_text = f"{observation_text}\nA:"
    context_ids = tuple(tokenizer.encode(prompt_text, add_bos=True, add_eos=False))
    candidate_ids = tuple(
        tuple(tokenizer.encode(text, add_bos=False, add_eos=True))
        for text in candidates
    )
    log_probs = candidate_log_probs(
        model,
        context_token_ids=context_ids,
        candidate_action_token_ids=candidate_ids,
    )
    chosen = (
        int(log_probs.argmax())
        if greedy
        else int(
            torch.multinomial(
                log_probs.exp(), num_samples=1, generator=generator
            ).item()
        )
    )
    return ActionSelection(
        raw_text=candidates[chosen],
        context_token_ids=context_ids,
        action_token_ids=candidate_ids[chosen],
        candidate_action_token_ids=candidate_ids,
        chosen_index=chosen,
        behavior_logprob=float(log_probs[chosen]),
    )
