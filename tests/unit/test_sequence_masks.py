from __future__ import annotations

import pytest
import torch

from rl_study.models import (
    TinyCausalLM,
    TinyLMConfig,
    TinyTokenizer,
    build_sequence_batch,
)
from rl_study.models.sequence import (
    response_sequence_log_probs,
    response_token_log_probs,
)


def test_prompt_and_action_mask_truth_table() -> None:
    tokenizer = TinyTokenizer()
    batch = build_sequence_batch(["a"], ["b"], tokenizer=tokenizer, max_length=8)
    assert batch.input_ids.shape == (1, 4)
    assert batch.labels.tolist() == [
        [
            tokenizer.encode("a", add_bos=False, add_eos=False)[0],
            tokenizer.encode("b", add_bos=False, add_eos=False)[0],
            tokenizer.eos_token_id,
        ]
    ]
    assert batch.prompt_target_mask.tolist() == [[True, False, False]]
    assert batch.action_mask.tolist() == [[False, True, True]]
    assert batch.response_lengths.tolist() == [2]


def test_variable_lengths_mask_padding_and_eos() -> None:
    tokenizer = TinyTokenizer()
    batch = build_sequence_batch(
        ["x", "xy"], ["1", "123"], tokenizer=tokenizer, max_length=12
    )
    assert batch.action_mask.sum(dim=-1).tolist() == [2, 4]
    assert batch.attention_mask.sum(dim=-1).tolist() == [4, 7]
    assert not torch.any(batch.action_mask & batch.prompt_target_mask)
    assert not batch.attention_mask[0, -1]


def test_sequence_log_prob_is_sum_of_action_tokens_only() -> None:
    tokenizer = TinyTokenizer()
    batch = build_sequence_batch(["a"], ["b"], tokenizer=tokenizer, max_length=8)
    model = TinyCausalLM(
        TinyLMConfig(
            vocab_size=128,
            max_sequence_length=8,
            hidden_size=16,
            num_heads=4,
            num_layers=1,
            intermediate_size=32,
        )
    )
    token_log_probs = response_token_log_probs(model, batch)
    sequence_log_probs = response_sequence_log_probs(model, batch)
    torch.testing.assert_close(
        sequence_log_probs,
        (token_log_probs * batch.action_mask).sum(dim=-1),
    )


def test_sequence_builder_forbids_silent_truncation() -> None:
    tokenizer = TinyTokenizer()
    with pytest.raises(ValueError, match="silent truncation is forbidden"):
        build_sequence_batch(["a" * 20], ["b" * 20], tokenizer=tokenizer, max_length=8)
