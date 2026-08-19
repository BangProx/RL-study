from __future__ import annotations

import torch

from rl_study.config import ExperimentConfig
from rl_study.data import build_tiny_reasoning
from rl_study.math import selected_log_probs
from rl_study.models import TinyCausalLM, TinyTokenizer
from rl_study.training import load_checkpoint, save_checkpoint


def test_offline_data_to_model_update_and_resume(tmp_path) -> None:
    config = ExperimentConfig.load("configs/toy/grpo.yaml")
    dataset = build_tiny_reasoning(seed=config.data.seed)
    tokenizer = TinyTokenizer()
    batch = tokenizer.batch_encode(
        [example.target_response for example in dataset.train[:4]], max_length=32
    )
    model = TinyCausalLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(batch.input_ids).logits[:, :-1]
    targets = batch.input_ids[:, 1:]
    action_mask = batch.attention_mask[:, 1:]
    log_probs = selected_log_probs(logits, targets)
    loss = -(log_probs * action_mask).sum() / action_mask.sum()
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

    checkpoint = save_checkpoint(
        tmp_path / "checkpoint-1",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        data_cursor={"prompt_uids": [example.uid for example in dataset.train[:4]]},
    )
    restored = TinyCausalLM()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    result = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        config=config,
        restore_rng=False,
    )
    assert result.step == 1
    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(original, loaded)
