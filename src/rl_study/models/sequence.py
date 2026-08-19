"""Prompt/response packing with masks aligned to causal next-token logits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.math import masked_sum, selected_log_probs
from rl_study.models.tiny_lm import TinyCausalLM
from rl_study.models.tiny_tokenizer import TinyTokenizer


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    input_ids: Tensor
    attention_mask: Tensor
    prompt_target_mask: Tensor
    action_mask: Tensor
    prompt_lengths: Tensor
    response_lengths: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.dtype != torch.int64 or self.input_ids.ndim != 2:
            raise TypeError("input_ids must be int64 [B, L]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        target_shape = (self.input_ids.shape[0], self.input_ids.shape[1] - 1)
        if self.prompt_target_mask.shape != target_shape:
            raise ValueError("prompt_target_mask must be [B, L-1]")
        if self.action_mask.shape != target_shape:
            raise ValueError("action_mask must be [B, L-1]")
        for name in ("attention_mask", "prompt_target_mask", "action_mask"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must be bool")
        if torch.any(self.prompt_target_mask & self.action_mask):
            raise ValueError("prompt and action target masks cannot overlap")
        if self.prompt_lengths.shape != self.response_lengths.shape:
            raise ValueError("prompt_lengths and response_lengths must share [B]")

    @property
    def labels(self) -> Tensor:
        return self.input_ids[:, 1:]


def build_sequence_batch(
    prompts: Iterable[str],
    responses: Iterable[str],
    *,
    tokenizer: TinyTokenizer,
    max_length: int,
) -> SequenceBatch:
    prompt_list = list(prompts)
    response_list = list(responses)
    if not prompt_list or len(prompt_list) != len(response_list):
        raise ValueError("prompts and responses must be non-empty with equal length")
    rows: list[list[int]] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    for prompt, response in zip(prompt_list, response_list, strict=True):
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        response_ids = tokenizer.encode(response, add_bos=False, add_eos=True)
        row = [*prompt_ids, *response_ids]
        if len(row) > max_length:
            raise ValueError(
                f"prompt+response length {len(row)} exceeds max_length={max_length}; "
                "silent truncation is forbidden"
            )
        rows.append(row)
        prompt_lengths.append(len(prompt_ids))
        response_lengths.append(len(response_ids))

    sequence_length = max(len(row) for row in rows)
    input_ids = torch.full(
        (len(rows), sequence_length), tokenizer.pad_token_id, dtype=torch.int64
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    prompt_target_mask = torch.zeros(len(rows), sequence_length - 1, dtype=torch.bool)
    action_mask = torch.zeros_like(prompt_target_mask)
    for index, row in enumerate(rows):
        row_length = len(row)
        prompt_length = prompt_lengths[index]
        input_ids[index, :row_length] = torch.tensor(row, dtype=torch.int64)
        attention_mask[index, :row_length] = True
        prompt_target_mask[index, : prompt_length - 1] = True
        action_mask[index, prompt_length - 1 : row_length - 1] = True
    return SequenceBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_target_mask=prompt_target_mask,
        action_mask=action_mask,
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.int64),
        response_lengths=torch.tensor(response_lengths, dtype=torch.int64),
    )


def build_sequence_batch_from_token_ids(
    prompt_token_ids: Iterable[Iterable[int]],
    response_token_ids: Iterable[Iterable[int]],
    *,
    pad_token_id: int,
    max_length: int,
) -> SequenceBatch:
    prompt_rows = [list(row) for row in prompt_token_ids]
    response_rows = [list(row) for row in response_token_ids]
    if not prompt_rows or len(prompt_rows) != len(response_rows):
        raise ValueError("prompt and response token rows must be non-empty and equal")
    rows: list[list[int]] = []
    for prompt_row, response_row in zip(prompt_rows, response_rows, strict=True):
        if not prompt_row or not response_row:
            raise ValueError("each rollout requires prompt and response tokens")
        row = [*prompt_row, *response_row]
        if len(row) > max_length:
            raise ValueError("rollout token length exceeds model context")
        rows.append(row)
    sequence_length = max(len(row) for row in rows)
    input_ids = torch.full(
        (len(rows), sequence_length), pad_token_id, dtype=torch.int64
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    prompt_target_mask = torch.zeros(len(rows), sequence_length - 1, dtype=torch.bool)
    action_mask = torch.zeros_like(prompt_target_mask)
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    for index, (prompt_row, response_row, row) in enumerate(
        zip(prompt_rows, response_rows, rows, strict=True)
    ):
        prompt_length = len(prompt_row)
        row_length = len(row)
        input_ids[index, :row_length] = torch.tensor(row)
        attention_mask[index, :row_length] = True
        prompt_target_mask[index, : prompt_length - 1] = True
        action_mask[index, prompt_length - 1 : row_length - 1] = True
        prompt_lengths.append(prompt_length)
        response_lengths.append(len(response_row))
    return SequenceBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_target_mask=prompt_target_mask,
        action_mask=action_mask,
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.int64),
        response_lengths=torch.tensor(response_lengths, dtype=torch.int64),
    )


def response_token_log_probs(model: TinyCausalLM, batch: SequenceBatch) -> Tensor:
    logits = model(batch.input_ids, batch.attention_mask).logits[:, :-1]
    return selected_log_probs(logits, batch.labels)


def response_sequence_log_probs(model: TinyCausalLM, batch: SequenceBatch) -> Tensor:
    token_log_probs = response_token_log_probs(model, batch)
    return masked_sum(token_log_probs, batch.action_mask, dim=-1)


@torch.no_grad()
def response_token_accuracy(model: TinyCausalLM, batch: SequenceBatch) -> float:
    logits = model(batch.input_ids, batch.attention_mask).logits[:, :-1]
    correct = logits.argmax(dim=-1).eq(batch.labels) & batch.action_mask
    return float(correct.sum() / batch.action_mask.sum())
