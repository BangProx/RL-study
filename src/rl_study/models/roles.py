"""Model-role helpers for gradient ownership and reproducibility checks."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor, nn

from rl_study.models.tiny_lm import TinyCausalLM


def freeze_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    module.eval()
    return module


def parameter_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def assert_frozen(module: nn.Module, *, role: str) -> None:
    trainable = [
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    ]
    gradients = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    ]
    if trainable or gradients or module.training:
        state = (
            f"trainable={trainable}, gradients={gradients}, training={module.training}"
        )
        raise RuntimeError(f"{role} must be frozen/eval; {state}")


class TinyValueModel(nn.Module):
    def __init__(self, backbone: TinyCausalLM | None = None) -> None:
        super().__init__()
        self.backbone = backbone or TinyCausalLM()
        self.value_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.backbone(input_ids, attention_mask).hidden_states
        values: Tensor = self.value_head(hidden).squeeze(-1)
        return values
