"""Small decoder-only language model used by every offline LLM-RL lesson."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rl_study.math.probability import selected_log_probs


@dataclass(frozen=True, slots=True)
class TinyLMConfig:
    vocab_size: int = 128
    max_sequence_length: int = 64
    hidden_size: int = 96
    num_heads: int = 4
    num_layers: int = 3
    intermediate_size: int = 192
    dropout: float = 0.0
    tie_embeddings: bool = True

    @classmethod
    def micro(cls) -> TinyLMConfig:
        return cls(
            vocab_size=64,
            max_sequence_length=32,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
        )

    def validate(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (
            min(
                self.vocab_size,
                self.max_sequence_length,
                self.hidden_size,
                self.num_heads,
                self.num_layers,
                self.intermediate_size,
            )
            <= 0
        ):
            raise ValueError("all TinyLM dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TinyLMOutput:
    logits: Tensor
    hidden_states: Tensor


class TinyCausalLM(nn.Module):
    def __init__(self, config: TinyLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyLMConfig()
        self.config.validate()
        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.position_embedding = nn.Embedding(
            self.config.max_sequence_length, self.config.hidden_size
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.intermediate_size,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, self.config.num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> TinyLMOutput:
        if input_ids.dtype != torch.int64 or input_ids.ndim != 2:
            raise ValueError("input_ids must be int64 [B, T]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            maximum = self.config.max_sequence_length
            raise ValueError(f"sequence length {sequence_length} exceeds max {maximum}")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contains an index outside the vocabulary")
        if attention_mask is not None:
            if (
                attention_mask.shape != input_ids.shape
                or attention_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "attention_mask must be bool with the same shape as input_ids"
                )
            padding_mask = ~attention_mask
        else:
            padding_mask = None
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        )
        hidden = self.blocks(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        hidden = self.final_norm(hidden)
        return TinyLMOutput(logits=self.lm_head(hidden), hidden_states=hidden)

    def next_token_log_probs(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        output = self(input_ids, attention_mask)
        return selected_log_probs(output.logits[:, :-1], input_ids[:, 1:])

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        do_sample: bool = True,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        generated = input_ids
        finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_sequence_length:
                break
            logits = self(generated).logits[:, -1] / temperature
            if do_sample:
                next_ids = torch.multinomial(
                    torch.softmax(logits, dim=-1),
                    num_samples=1,
                    generator=generator,
                )
            else:
                next_ids = logits.argmax(dim=-1, keepdim=True)
            next_ids = torch.where(finished.unsqueeze(-1), eos_token_id, next_ids)
            generated = torch.cat((generated, next_ids), dim=1)
            finished |= next_ids.squeeze(-1).eq(eos_token_id)
            if bool(finished.all()):
                break
        return generated
