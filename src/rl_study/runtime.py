"""Seed and device handling with no silent fallback."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from rl_study.errors import PreflightError


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True, slots=True)
class DeviceResolution:
    requested: str
    resolved: torch.device
    fallback_used: bool
    probe_message: str


def _is_available(device: torch.device) -> tuple[bool, str]:
    if device.type == "cpu":
        return True, "CPU is available"
    if device.type == "mps":
        built = torch.backends.mps.is_built()
        available = torch.backends.mps.is_available()
        return available, f"MPS built={built}, available={available}"
    if device.type == "cuda":
        available = torch.cuda.is_available()
        if not available:
            return False, "CUDA is not available"
        if device.index is not None and device.index >= torch.cuda.device_count():
            count = torch.cuda.device_count()
            return (
                False,
                f"CUDA index {device.index} is outside device_count={count}",
            )
        return True, f"CUDA available, device_count={torch.cuda.device_count()}"
    return False, f"unsupported device type: {device.type}"


def _probe(device: torch.device) -> tuple[bool, str]:
    available, message = _is_available(device)
    if not available:
        return False, message
    try:
        value = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
        loss = (value.square()).sum()
        torch.autograd.backward(loss)
        if value.grad is None or not torch.isfinite(value.grad).all():
            return False, f"{device} backward probe produced invalid gradients"
    except (RuntimeError, NotImplementedError) as error:
        return False, f"{device} forward/backward probe failed: {error}"
    return True, f"{message}; forward/backward probe passed"


def resolve_device(requested: str, *, allow_fallback: bool = False) -> DeviceResolution:
    normalized = requested.strip().lower()
    valid = normalized in {"cpu", "mps", "cuda", "auto"} or (
        normalized.startswith("cuda:") and normalized.removeprefix("cuda:").isdigit()
    )
    if not valid:
        raise PreflightError(
            f"unknown device {requested!r}; use cpu, mps, cuda, cuda:N, or auto"
        )

    if normalized == "auto":
        messages: list[str] = []
        for candidate in (
            torch.device("cuda"),
            torch.device("mps"),
            torch.device("cpu"),
        ):
            passed, message = _probe(candidate)
            messages.append(message)
            if passed:
                return DeviceResolution(
                    requested, candidate, candidate.type == "cpu", "; ".join(messages)
                )
        raise PreflightError(
            "no device passed forward/backward probe: " + "; ".join(messages)
        )

    device = torch.device(normalized)
    passed, message = _probe(device)
    if passed:
        return DeviceResolution(requested, device, False, message)
    if not allow_fallback or device.type == "cpu":
        raise PreflightError(message)
    cpu_passed, cpu_message = _probe(torch.device("cpu"))
    if not cpu_passed:
        raise PreflightError(
            f"requested device failed ({message}); CPU fallback failed ({cpu_message})"
        )
    return DeviceResolution(
        requested,
        torch.device("cpu"),
        True,
        f"requested device failed: {message}; {cpu_message}",
    )
