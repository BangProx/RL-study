"""A tiny sequence reward model with an explicit scalar head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rl_study.models.tiny_lm import TinyCausalLM


class TinyRewardModel(nn.Module):
    def __init__(self, backbone: TinyCausalLM | None = None) -> None:
        super().__init__()
        self.backbone = backbone or TinyCausalLM()
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if (
            attention_mask.dtype != torch.bool
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError("attention_mask must be bool and match input_ids")
        if torch.any(attention_mask.sum(dim=-1) == 0):
            raise ValueError("every reward-model sequence must contain a token")
        hidden = self.backbone(input_ids, attention_mask).hidden_states
        last_indices = attention_mask.sum(dim=-1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        last_hidden = hidden[batch_indices, last_indices]
        scores: Tensor = self.reward_head(last_hidden).squeeze(-1)
        return scores
