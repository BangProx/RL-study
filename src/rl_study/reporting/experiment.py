"""Versioned experiment cards with atomic JSON writes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from rl_study.config import ExperimentConfig
from rl_study.data.tiny_reasoning import GENERATOR_REVISION
from rl_study.platform_metrics import peak_memory_bytes, system_memory_bytes


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(root: Path) -> dict[str, object]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

    commit_result = run("rev-parse", "HEAD")
    status_result = run("status", "--porcelain")
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else "UNBORN"
    status = status_result.stdout
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "diff_sha256": "sha256:" + hashlib.sha256(status.encode()).hexdigest(),
    }


def _device_memory_bytes(device: str) -> int | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.get_device_properties(torch.device(device)).total_memory)
    except (AssertionError, RuntimeError, ValueError):
        return None


def _ordered_ids_sha256(values: Sequence[str]) -> str | None:
    if not values:
        return None
    canonical = json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def build_experiment_card(
    config: ExperimentConfig,
    *,
    run_id: str,
    run_status: str,
    step: int,
    wall_seconds: float,
    metrics: Mapping[str, float],
    data_split_hash: str | None = None,
    prompt_uids: Sequence[str] = (),
    optimized_prompt_uids: Sequence[str] = (),
    optimizer_steps: int | None = None,
    generated_tokens: int = 0,
    processed_tokens: int = 0,
    environment_steps: int = 0,
    model_forwards: int = 0,
    known_deviations: tuple[str, ...] = (),
    failures: tuple[str, ...] = (),
) -> dict[str, object]:
    root = Path.cwd()
    lock_file = (
        root / "requirements" / "laptop.lock"
        if config.profile == "laptop"
        else root / "pyproject.toml"
    )
    device_memory = _device_memory_bytes(config.training.device)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "run_status": run_status,
        "result_origin": "local_executed",
        "git": _git_state(root),
        "config_hash": config.sha256,
        "resume_hash": config.resume_sha256,
        "dependency_lock_hash": _file_sha256(lock_file),
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": config.training.device,
            "dtype": "float32",
            "ram_bytes": system_memory_bytes(),
            "vram_bytes": device_memory,
            "vram_note": (
                "not applicable or unavailable"
                if device_memory is None
                else "CUDA device total memory"
            ),
            "peak_memory_bytes": peak_memory_bytes(),
        },
        "model": {
            "id": config.model.policy,
            "revision": config.model.revision or "repository",
            "license_id": config.model.license_id or "Apache-2.0",
        },
        "data": {
            "id": config.data.id,
            "revision": config.data.revision,
            "generator_revision": (
                GENERATOR_REVISION if config.data.id == "tiny_reasoning" else None
            ),
            "split_hash": data_split_hash,
            "ordered_prompt_ids_sha256": _ordered_ids_sha256(prompt_uids),
            "prompt_occurrences": len(prompt_uids),
            "ordered_optimized_prompt_ids_sha256": _ordered_ids_sha256(
                optimized_prompt_uids
            ),
            "optimized_prompt_occurrences": len(optimized_prompt_uids),
        },
        "algorithm": {
            "name": config.algorithm.name,
            "variant": config.algorithm.variant,
        },
        "seed": config.training.seed,
        "step": step,
        "budgets": {
            "optimizer_steps": step if optimizer_steps is None else optimizer_steps,
            "environment_steps": environment_steps,
            "generated_tokens": generated_tokens,
            "processed_tokens": processed_tokens,
            "model_forwards": model_forwards,
        },
        "timing": {"wall_seconds": wall_seconds},
        "metrics": dict(metrics),
        "paper_reported": None,
        "upstream_reported": None,
        "local_executed": dict(metrics),
        "known_deviations": list(known_deviations),
        "failures": list(failures),
    }


def write_experiment_card(
    run_directory: str | Path, card: Mapping[str, object]
) -> Path:
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "experiment-card.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite experiment card: {target}")
    handle, temporary_name = tempfile.mkstemp(
        prefix=".experiment-card.", suffix=".json", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(card, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
