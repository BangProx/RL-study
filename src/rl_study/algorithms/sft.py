"""Response-only supervised fine-tuning for the offline tiny language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_study.data import TinyReasoningExample, build_tiny_reasoning
from rl_study.math import masked_mean
from rl_study.models import TinyCausalLM, TinyTokenizer, build_sequence_batch
from rl_study.models.sequence import SequenceBatch, response_token_log_probs


@dataclass(frozen=True, slots=True)
class SFTLossOutput:
    loss: Tensor
    token_accuracy: Tensor
    valid_tokens: int


@dataclass(frozen=True, slots=True)
class SFTTrainResult:
    model: TinyCausalLM
    optimizer: torch.optim.Optimizer
    losses: tuple[float, ...]
    train_token_accuracy: float
    validation_token_accuracy: float


def sft_loss(model: TinyCausalLM, batch: SequenceBatch) -> SFTLossOutput:
    token_log_probs = response_token_log_probs(model, batch)
    loss = -masked_mean(token_log_probs, batch.action_mask)
    with torch.no_grad():
        predictions = (
            model(batch.input_ids, batch.attention_mask).logits[:, :-1].argmax(dim=-1)
        )
        correct = predictions.eq(batch.labels) & batch.action_mask
        accuracy = correct.sum().to(torch.float32) / batch.action_mask.sum()
    return SFTLossOutput(
        loss=loss,
        token_accuracy=accuracy,
        valid_tokens=int(batch.action_mask.sum()),
    )


def _batch_examples(
    examples: tuple[TinyReasoningExample, ...], indices: Tensor
) -> tuple[TinyReasoningExample, ...]:
    return tuple(examples[int(index)] for index in indices)


def _collate(
    examples: tuple[TinyReasoningExample, ...], tokenizer: TinyTokenizer
) -> SequenceBatch:
    return build_sequence_batch(
        (example.prompt for example in examples),
        (example.target_response for example in examples),
        tokenizer=tokenizer,
        max_length=64,
    )


def train_sft(
    *,
    steps: int = 100,
    batch_size: int = 16,
    seed: int = 42,
    learning_rate: float = 3e-3,
    model: TinyCausalLM | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> SFTTrainResult:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    torch.manual_seed(seed)
    dataset = build_tiny_reasoning(seed=seed)
    tokenizer = TinyTokenizer()
    if model is None:
        model = TinyCausalLM()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer is None:
        raise ValueError("resumed SFT requires an optimizer")
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    last_accuracy = 0.0
    for _ in range(steps):
        indices = torch.randint(len(dataset.train), (batch_size,), generator=generator)
        batch = _collate(_batch_examples(dataset.train, indices), tokenizer)
        output = sft_loss(model, batch)
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(output.loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(output.loss.detach()))
        last_accuracy = float(output.token_accuracy)
    validation_batch = _collate(dataset.validation, tokenizer)
    validation_accuracy = float(sft_loss(model, validation_batch).token_accuracy)
    return SFTTrainResult(
        model=model,
        optimizer=optimizer,
        losses=tuple(losses),
        train_token_accuracy=last_accuracy,
        validation_token_accuracy=validation_accuracy,
    )
