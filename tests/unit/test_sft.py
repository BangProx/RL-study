from __future__ import annotations

import torch

from rl_study.algorithms.sft import sft_loss
from rl_study.data import build_tiny_reasoning
from rl_study.models import TinyCausalLM, TinyTokenizer, build_sequence_batch


def test_sft_loss_is_response_only_and_finite() -> None:
    dataset = build_tiny_reasoning()
    tokenizer = TinyTokenizer()
    examples = dataset.train[:4]
    batch = build_sequence_batch(
        [example.prompt for example in examples],
        [example.target_response for example in examples],
        tokenizer=tokenizer,
        max_length=64,
    )
    model = TinyCausalLM()
    output = sft_loss(model, batch)
    assert torch.isfinite(output.loss)
    assert output.valid_tokens == int(batch.action_mask.sum())
    assert output.valid_tokens == int(batch.response_lengths.sum())
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None
