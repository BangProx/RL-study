"""Mask-aware reductions that never hide an empty denominator."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

ZeroPolicy = Literal["error", "zero"]


def _validate(values: Tensor, mask: Tensor) -> Tensor:
    if not values.is_floating_point():
        raise TypeError("values must have a floating dtype")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have bool dtype")
    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape")
    return mask.to(dtype=values.dtype)


def masked_sum(
    values: Tensor, mask: Tensor, *, dim: int | tuple[int, ...] | None = None
) -> Tensor:
    numeric_mask = _validate(values, mask)
    return (values * numeric_mask).sum(dim=dim)


def masked_mean(
    values: Tensor,
    mask: Tensor,
    *,
    dim: int | tuple[int, ...] | None = None,
    zero_policy: ZeroPolicy = "error",
) -> Tensor:
    numeric_mask = _validate(values, mask)
    numerator = (values * numeric_mask).sum(dim=dim)
    denominator = numeric_mask.sum(dim=dim)
    if zero_policy == "error":
        if torch.any(denominator == 0):
            raise ValueError("masked_mean received an empty mask")
        return numerator / denominator
    if zero_policy == "zero":
        safe_denominator = denominator.clamp_min(1)
        return torch.where(
            denominator > 0, numerator / safe_denominator, torch.zeros_like(numerator)
        )
    raise ValueError(f"unknown zero_policy={zero_policy!r}")


def masked_sequence_mean(
    values: Tensor,
    mask: Tensor,
    *,
    zero_policy: ZeroPolicy = "error",
) -> Tensor:
    """Reduce the final token dimension and preserve leading dimensions."""

    if values.ndim < 1:
        raise ValueError("masked_sequence_mean expects at least one dimension")
    return masked_mean(values, mask, dim=-1, zero_policy=zero_policy)
