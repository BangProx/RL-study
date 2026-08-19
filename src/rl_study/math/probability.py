"""Probability functions written to expose the tensor equations."""

from __future__ import annotations

import torch
from torch import Tensor


def _require_floating(logits: Tensor, name: str) -> None:
    if not logits.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if logits.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")


def categorical_entropy(logits: Tensor) -> Tensor:
    """Return ``-sum_a p(a) log p(a)`` along the final dimension."""

    _require_floating(logits, "logits")
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def categorical_cross_entropy(
    target_logits: Tensor, predicted_logits: Tensor
) -> Tensor:
    """Return ``-sum_a p_target(a) log p_predicted(a)``."""

    _require_floating(target_logits, "target_logits")
    _require_floating(predicted_logits, "predicted_logits")
    if target_logits.shape != predicted_logits.shape:
        raise ValueError("target_logits and predicted_logits must have the same shape")
    target_probs = torch.softmax(target_logits, dim=-1)
    predicted_log_probs = torch.log_softmax(predicted_logits, dim=-1)
    return -(target_probs * predicted_log_probs).sum(dim=-1)


def categorical_kl(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """Return forward ``KL(p || q)`` along the final dimension."""

    _require_floating(p_logits, "p_logits")
    _require_floating(q_logits, "q_logits")
    if p_logits.shape != q_logits.shape:
        raise ValueError("p_logits and q_logits must have the same shape")
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def selected_log_probs(logits: Tensor, token_ids: Tensor) -> Tensor:
    """Gather token log-probabilities from ``logits[..., vocabulary]``."""

    _require_floating(logits, "logits")
    if token_ids.dtype != torch.int64:
        raise TypeError("token_ids must have int64 dtype")
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError(
            "token_ids shape must match logits without vocabulary dimension"
        )
    if torch.any(token_ids < 0) or torch.any(token_ids >= logits.shape[-1]):
        raise ValueError("token_ids contains an index outside the vocabulary")
    return (
        torch.log_softmax(logits, dim=-1)
        .gather(-1, token_ids.unsqueeze(-1))
        .squeeze(-1)
    )
