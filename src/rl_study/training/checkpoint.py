"""Integrity-checked atomic checkpoints for repository-created state."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from rl_study.config import ExperimentConfig
from rl_study.errors import CheckpointError

CHECKPOINT_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _rng_payload() -> dict[str, object]:
    numpy_state = cast(
        tuple[str, npt.NDArray[np.uint32], int, int, float],
        np.random.get_state(legacy=True),
    )
    cuda_states = (
        [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": cuda_states,
    }


def _nested_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _restore_rng(payload: Mapping[str, object]) -> None:
    python_state = _nested_tuple(payload["python"])
    if not isinstance(python_state, tuple):
        raise CheckpointError("invalid Python RNG state")
    random.setstate(python_state)
    numpy_state = payload["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise CheckpointError("invalid NumPy RNG state")
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch.tensor(payload["torch_cpu"], dtype=torch.uint8))
    cuda_states = payload.get("torch_cuda", [])
    if torch.cuda.is_available() and isinstance(cuda_states, list):
        torch.cuda.set_rng_state_all(
            [torch.tensor(state, dtype=torch.uint8) for state in cuda_states]
        )


@dataclass(frozen=True, slots=True)
class CheckpointLoadResult:
    step: int
    data_cursor: dict[str, object]
    metrics: list[dict[str, object]]
    manifest: dict[str, object]


def save_checkpoint(
    target: str | Path,
    *,
    model: nn.Module,
    config: ExperimentConfig,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    extra_optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    scheduler: object | None = None,
    data_cursor: Mapping[str, object] | None = None,
    metrics: list[dict[str, object]] | None = None,
) -> Path:
    destination = Path(target)
    if step < 0:
        raise ValueError("step must be non-negative")
    if destination.exists():
        raise CheckpointError(
            f"refusing to overwrite existing checkpoint: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        torch.save(model.state_dict(), temporary / "model.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        for name, extra_optimizer in (extra_optimizers or {}).items():
            if not name.isidentifier():
                raise ValueError(
                    f"extra optimizer name must be an identifier: {name!r}"
                )
            torch.save(extra_optimizer.state_dict(), temporary / f"optimizer-{name}.pt")
        if scheduler is not None:
            state_dict = getattr(scheduler, "state_dict", None)
            if state_dict is None:
                raise TypeError("scheduler must expose state_dict()")
            torch.save(state_dict(), temporary / "scheduler.pt")
        _write_json(temporary / "config.resolved.json", config.to_dict())
        _write_json(temporary / "rng.json", _rng_payload())
        _write_json(temporary / "data_cursor.json", dict(data_cursor or {}))
        _write_json(temporary / "metrics.json", metrics or [])

        files = sorted(path for path in temporary.iterdir() if path.is_file())
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": step,
            "config_hash": config.sha256,
            "resume_hash": config.resume_sha256,
            "files": {path.name: _sha256(path) for path in files},
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_checkpoint(
    source: str | Path,
    *,
    model: nn.Module,
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer | None = None,
    extra_optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    scheduler: object | None = None,
    restore_rng: bool = True,
) -> CheckpointLoadResult:
    checkpoint = Path(source)
    try:
        manifest = json.loads(
            (checkpoint / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(
            f"invalid checkpoint manifest in {checkpoint}: {error}"
        ) from error
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("unsupported checkpoint schema version")
    if manifest.get("resume_hash") != config.resume_sha256:
        checkpoint_hash = manifest.get("resume_hash")
        raise CheckpointError(
            "resume immutable config hash mismatch: "
            f"checkpoint={checkpoint_hash} current={config.resume_sha256}"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise CheckpointError("checkpoint manifest files must be a mapping")
    for name, expected_hash in files.items():
        if not isinstance(name, str) or not isinstance(expected_hash, str):
            raise CheckpointError("checkpoint manifest contains an invalid file entry")
        path = checkpoint / name
        if not path.is_file() or _sha256(path) != expected_hash:
            raise CheckpointError(f"checkpoint integrity failure: {name}")

    try:
        model.load_state_dict(
            torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
        )
        if optimizer is not None:
            optimizer_path = checkpoint / "optimizer.pt"
            if not optimizer_path.is_file():
                raise CheckpointError(
                    "optimizer was requested but optimizer.pt is absent"
                )
            optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu", weights_only=True)
            )
        for name, extra_optimizer in (extra_optimizers or {}).items():
            if not name.isidentifier():
                raise CheckpointError(
                    f"extra optimizer name must be an identifier: {name!r}"
                )
            extra_path = checkpoint / f"optimizer-{name}.pt"
            if not extra_path.is_file():
                raise CheckpointError(
                    f"extra optimizer {name!r} was requested but is absent"
                )
            extra_optimizer.load_state_dict(
                torch.load(extra_path, map_location="cpu", weights_only=True)
            )
        if scheduler is not None:
            scheduler_path = checkpoint / "scheduler.pt"
            load_state_dict = getattr(scheduler, "load_state_dict", None)
            if not scheduler_path.is_file() or load_state_dict is None:
                raise CheckpointError(
                    "scheduler state is absent or scheduler cannot load it"
                )
            load_state_dict(
                torch.load(scheduler_path, map_location="cpu", weights_only=True)
            )
        rng_payload = json.loads((checkpoint / "rng.json").read_text(encoding="utf-8"))
        data_cursor = json.loads(
            (checkpoint / "data_cursor.json").read_text(encoding="utf-8")
        )
        metrics = json.loads((checkpoint / "metrics.json").read_text(encoding="utf-8"))
    except CheckpointError:
        raise
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise CheckpointError(
            f"failed to load checkpoint {checkpoint}: {error}"
        ) from error
    if restore_rng:
        _restore_rng(rng_payload)
    if not isinstance(data_cursor, dict) or not isinstance(metrics, list):
        raise CheckpointError("checkpoint data_cursor or metrics has the wrong type")
    return CheckpointLoadResult(
        step=int(manifest["step"]),
        data_cursor=data_cursor,
        metrics=metrics,
        manifest=dict(manifest),
    )
