from __future__ import annotations

import torch

from rl_study.models import TinyCausalLM, TinyLMConfig, TinyTokenizer


def test_tokenizer_round_trip_and_padding_contract() -> None:
    tokenizer = TinyTokenizer()
    text = "Compute: 2 + 3."
    assert tokenizer.decode(tokenizer.encode(text)) == text
    batch = tokenizer.batch_encode(["a", "abc"], max_length=8)
    assert batch.input_ids.dtype == torch.int64
    assert batch.attention_mask.dtype == torch.bool
    assert batch.input_ids.shape == batch.attention_mask.shape == (2, 5)
    assert batch.attention_mask[0].tolist() == [True, True, True, False, False]


def test_tokenizer_marks_non_ascii_as_unknown() -> None:
    tokenizer = TinyTokenizer()
    encoded = tokenizer.encode("한", add_bos=False, add_eos=False)
    assert encoded == [tokenizer.unk_token_id]


def test_canonical_model_parameter_count_and_output_shape() -> None:
    model = TinyCausalLM()
    assert model.parameter_count == 242_976
    inputs = torch.randint(0, 128, (2, 16), dtype=torch.int64)
    output = model(inputs)
    assert output.logits.shape == (2, 16, 128)
    assert output.hidden_states.shape == (2, 16, 96)


def test_causal_mask_prevents_future_token_leakage() -> None:
    torch.manual_seed(42)
    model = TinyCausalLM(TinyLMConfig.micro()).eval()
    first = torch.tensor([[1, 4, 5, 6]], dtype=torch.int64)
    second = torch.tensor([[1, 4, 5, 20]], dtype=torch.int64)
    first_logits = model(first).logits
    second_logits = model(second).logits
    torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])


def test_next_token_log_probs_keep_current_model_gradient() -> None:
    model = TinyCausalLM(TinyLMConfig.micro())
    inputs = torch.randint(0, 64, (2, 8), dtype=torch.int64)
    log_probs = model.next_token_log_probs(inputs)
    assert log_probs.shape == (2, 7)
    loss = -log_probs.mean()
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_generation_respects_max_sequence_length() -> None:
    model = TinyCausalLM(TinyLMConfig.micro()).eval()
    inputs = torch.tensor([[1, 4, 5]], dtype=torch.int64)
    generator = torch.Generator().manual_seed(7)
    generated = model.generate(
        inputs,
        max_new_tokens=100,
        eos_token_id=2,
        generator=generator,
    )
    assert inputs.shape[1] <= generated.shape[1] <= model.config.max_sequence_length
