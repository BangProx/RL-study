"""Audited external-asset manifest, cache probe, and transparent memory estimate."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from rl_study.errors import DownloadApprovalRequired, PreflightError

LARGE_DOWNLOAD_BYTES = 100_000_000


@dataclass(frozen=True, slots=True)
class AssetManifest:
    preset: str
    hub_id: str
    revision: str
    license_id: str
    expected_bytes: int
    source_id: str


@dataclass(frozen=True, slots=True)
class ModelManifest(AssetManifest):
    parameters: int
    hidden_size: int
    layers: int
    expected_lora_parameters: int


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    base_parameters_bytes: int
    adapter_parameters_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    runtime_overhead_bytes: int
    subtotal_bytes: int
    recommended_bytes: int
    assumptions: tuple[str, ...]


MODEL_PRESETS: dict[str, ModelManifest] = {
    "laptop-smoke": ModelManifest(
        preset="laptop-smoke",
        hub_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        license_id="Apache-2.0",
        expected_bytes=269_060_552,
        source_id="model-smollm2-135m-instruct",
        parameters=134_515_008,
        hidden_size=576,
        layers=30,
        expected_lora_parameters=552_960,
    ),
    "laptop-quality": ModelManifest(
        preset="laptop-quality",
        hub_id="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        license_id="Apache-2.0",
        expected_bytes=1_503_300_328,
        source_id="model-qwen3-0.6b",
        parameters=751_632_384,
        hidden_size=1024,
        layers=28,
        expected_lora_parameters=917_504,
    ),
    "server": ModelManifest(
        preset="server",
        hub_id="Qwen/Qwen3-4B",
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
        license_id="Apache-2.0",
        expected_bytes=8_044_982_000,
        source_id="model-qwen3-4b",
        parameters=4_022_468_096,
        hidden_size=2560,
        layers=36,
        expected_lora_parameters=2_949_120,
    ),
}

DATASET_PRESETS: dict[str, AssetManifest] = {
    "gsm8k": AssetManifest(
        preset="gsm8k",
        hub_id="openai/gsm8k",
        revision="740312add88f781978c0658806c59bc2815b9866",
        license_id="MIT",
        expected_bytes=2_725_633,
        source_id="dataset-gsm8k",
    ),
    "ultrafeedback-binarized": AssetManifest(
        preset="ultrafeedback-binarized",
        hub_id="HuggingFaceH4/ultrafeedback_binarized",
        revision="3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
        license_id="MIT",
        expected_bytes=649_967_196,
        source_id="dataset-ultrafeedback-binarized",
    ),
}


def _hub_cache_root(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_cache_name(hub_id: str, *, repo_type: str) -> str:
    prefix = "models" if repo_type == "model" else "datasets"
    return f"{prefix}--{hub_id.replace('/', '--')}"


def model_cache_status(
    manifest: AssetManifest,
    *,
    cache_dir: str | Path | None = None,
    repo_type: str = "model",
) -> tuple[bool, str]:
    """Check exact-revision cache without importing Hub code or touching network."""
    if repo_type not in {"model", "dataset"}:
        raise ValueError("repo_type must be model or dataset")
    root = _hub_cache_root(cache_dir)
    repository = root / _repo_cache_name(manifest.hub_id, repo_type=repo_type)
    snapshot = repository / "snapshots" / manifest.revision
    if snapshot.is_dir():
        files = [path for path in snapshot.rglob("*") if path.is_file()]
        if files:
            return (
                True,
                f"$HF_HUB_CACHE/{repository.name}/snapshots/{manifest.revision}",
            )
    return False, f"$HF_HUB_CACHE/{repository.name}/snapshots/{manifest.revision}"


def enforce_download_guard(
    manifest: AssetManifest,
    *,
    cached: bool,
    accept_download: bool,
) -> None:
    """Fail before an optional framework import or network call for large assets."""
    if cached or manifest.expected_bytes <= LARGE_DOWNLOAD_BYTES or accept_download:
        return
    raise DownloadApprovalRequired(
        f"{manifest.hub_id}@{manifest.revision} needs approximately "
        f"{manifest.expected_bytes:,} bytes; rerun with --accept-download after "
        f"reviewing license={manifest.license_id} and source={manifest.source_id}"
    )


def estimate_training_memory(
    manifest: ModelManifest,
    *,
    adapter: str,
    dtype: str,
    batch_size: int,
    sequence_length: int,
) -> MemoryEstimate:
    """Conservative, inspectable estimate—not a promise that a run will fit."""
    if adapter not in {"lora", "qlora"}:
        raise PreflightError("memory estimator supports lora or qlora")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise PreflightError("unsupported estimator dtype")
    dtype_bytes = 4 if dtype == "float32" else 2
    base_bytes_per_parameter = 0.5 if adapter == "qlora" else dtype_bytes
    base = math.ceil(manifest.parameters * base_bytes_per_parameter)
    adapter_parameters = manifest.expected_lora_parameters
    adapter_bytes = adapter_parameters * dtype_bytes
    gradients = adapter_parameters * 4
    optimizer = adapter_parameters * 8
    activation = (
        batch_size
        * sequence_length
        * manifest.hidden_size
        * manifest.layers
        * dtype_bytes
        * 8
    )
    runtime_overhead = 768 * 1024 * 1024
    subtotal = (
        base + adapter_bytes + gradients + optimizer + activation + runtime_overhead
    )
    recommended = math.ceil(subtotal * 1.2)
    return MemoryEstimate(
        base_parameters_bytes=base,
        adapter_parameters_bytes=adapter_bytes,
        gradient_bytes=gradients,
        optimizer_bytes=optimizer,
        activation_bytes=activation,
        runtime_overhead_bytes=runtime_overhead,
        subtotal_bytes=subtotal,
        recommended_bytes=recommended,
        assumptions=(
            "LoRA trainable count is estimated; run cards store the actual count.",
            "Adam states use 8 bytes and gradients use 4 bytes per trainable value.",
            "Activations use an explicit 8x hidden-state heuristic and 20% margin.",
            "Runtime overhead reserves 768 MiB for Python, PyTorch, and frameworks.",
        ),
    )
