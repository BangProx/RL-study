"""No-network profile preflight with dependency, device, cache, and memory facts."""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict

from rl_study.adapters.manifest import (
    MODEL_PRESETS,
    AssetManifest,
    ModelManifest,
    estimate_training_memory,
    model_cache_status,
)
from rl_study.errors import PreflightError
from rl_study.platform_metrics import system_memory_bytes
from rl_study.runtime import resolve_device

LAPTOP_DEPENDENCIES = {
    "transformers": "4.55<=version<5",
    "peft": "0.17<=version<1",
    "accelerate": "1.10<=version<2",
    "datasets": "4<=version<5",
    "trl": "version==1.10.0",
}
SERVER_DEPENDENCIES = {"verl": "version==0.9.0", "vllm": "required by rollout"}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw in value.split("."):
        digits = "".join(character for character in raw if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _compatible(name: str, version: str | None) -> bool:
    if version is None:
        return False
    parsed = _version_tuple(version)
    constraints = {
        "transformers": ((4, 55), (5, 0)),
        "peft": ((0, 17), (1, 0)),
        "accelerate": ((1, 10), (2, 0)),
        "datasets": ((4, 0), (5, 0)),
        "trl": ((1, 10, 0), (1, 10, 1)),
        "verl": ((0, 9, 0), (0, 9, 1)),
    }
    if name not in constraints:
        return True
    minimum, maximum = constraints[name]
    return minimum <= parsed < maximum


def qlora_capability(device: str) -> tuple[bool, str]:
    resolution = resolve_device(device)
    bitsandbytes = _version("bitsandbytes")
    if resolution.resolved.type != "cuda":
        return (
            False,
            "QLoRA requires a probed CUDA device; CPU/MPS fallback is forbidden",
        )
    if bitsandbytes is None:
        return False, "QLoRA requires the optional bitsandbytes package"
    return True, f"CUDA probe passed and bitsandbytes={bitsandbytes} is installed"


def _custom_model(
    hub_id: str,
    *,
    revision: str | None,
    license_id: str | None,
    expected_weight_bytes: int | None,
) -> ModelManifest:
    if revision is None or license_id is None or expected_weight_bytes is None:
        raise PreflightError(
            "an arbitrary model requires --revision, --license-id, and "
            "--expected-weight-bytes before any Hub lookup"
        )
    return ModelManifest(
        preset="custom-audited",
        hub_id=hub_id,
        revision=revision,
        license_id=license_id,
        expected_bytes=expected_weight_bytes,
        source_id="user-supplied-audit",
        parameters=max(1, expected_weight_bytes // 2),
        hidden_size=1024,
        layers=24,
        expected_lora_parameters=786_432,
    )


def resolve_model_manifest(
    model: str,
    *,
    revision: str | None = None,
    license_id: str | None = None,
    expected_weight_bytes: int | None = None,
) -> ModelManifest:
    if model in MODEL_PRESETS:
        return MODEL_PRESETS[model]
    for manifest in MODEL_PRESETS.values():
        if model == manifest.hub_id:
            if revision is not None and revision != manifest.revision:
                raise PreflightError("model revision conflicts with the audited preset")
            return manifest
    return _custom_model(
        model,
        revision=revision,
        license_id=license_id,
        expected_weight_bytes=expected_weight_bytes,
    )


def _dependency_report(profile: str) -> dict[str, dict[str, object]]:
    requirements = LAPTOP_DEPENDENCIES if profile == "laptop" else SERVER_DEPENDENCIES
    return {
        name: {
            "installed": installed,
            "required": requirement,
            "compatible": _compatible(name, installed),
        }
        for name, requirement in requirements.items()
        for installed in (_version(name),)
    }


def build_profile_preflight(
    *,
    profile: str,
    model: str,
    device: str,
    adapter: str = "lora",
    revision: str | None = None,
    license_id: str | None = None,
    expected_weight_bytes: int | None = None,
    cache_dir: str | None = None,
    batch_size: int = 1,
    sequence_length: int = 128,
    dtype: str = "float32",
) -> dict[str, object]:
    if profile not in {"laptop", "server"}:
        raise PreflightError("external profile preflight requires laptop or server")
    manifest = resolve_model_manifest(
        model,
        revision=revision,
        license_id=license_id,
        expected_weight_bytes=expected_weight_bytes,
    )
    resolution = resolve_device(device)
    cached, cache_hint = model_cache_status(manifest, cache_dir=cache_dir)
    estimate = estimate_training_memory(
        manifest,
        adapter=adapter,
        dtype=dtype,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    dependencies = _dependency_report(profile)
    missing = [name for name, item in dependencies.items() if item["installed"] is None]
    incompatible = [
        name
        for name, item in dependencies.items()
        if item["installed"] is not None and not item["compatible"]
    ]
    qlora: dict[str, object] | None = None
    if adapter == "qlora":
        passed, reason = qlora_capability(str(resolution.resolved))
        qlora = {"supported": passed, "reason": reason}
    physical_memory = system_memory_bytes()
    warnings: list[str] = []
    if (
        physical_memory is not None
        and estimate.recommended_bytes > physical_memory * 0.7
    ):
        warnings.append(
            "estimated training memory exceeds 70% of physical RAM; use a smaller "
            "preset/context or a CUDA server"
        )
    if not cached:
        warnings.append("exact model revision is not complete in the local cache")
    if missing:
        warnings.append("missing optional dependencies: " + ", ".join(missing))
    if incompatible:
        warnings.append(
            "incompatible optional dependencies: " + ", ".join(incompatible)
        )
    if qlora is not None and not qlora["supported"]:
        warnings.append(str(qlora["reason"]))
    status = "ready" if not warnings else "action-required"
    return {
        "status": status,
        "profile": profile,
        "network_attempted": False,
        "model": {
            "preset": manifest.preset,
            "hub_id": manifest.hub_id,
            "revision": manifest.revision,
            "license_id": manifest.license_id,
            "expected_weight_bytes": manifest.expected_bytes,
            "source_id": manifest.source_id,
            "cached": cached,
            "cache_path_hint": cache_hint,
            "requires_accept_download": manifest.expected_bytes > 100_000_000
            and not cached,
        },
        "device": {
            "requested": device,
            "resolved": str(resolution.resolved),
            "probe": resolution.probe_message,
        },
        "adapter": adapter,
        "qlora": qlora,
        "dependencies": dependencies,
        "memory_estimate": asdict(estimate),
        "physical_memory_bytes": physical_memory,
        "warnings": warnings,
    }


def audited_asset_from_config(
    *,
    hub_id: str,
    revision: str | None,
    license_id: str | None,
    expected_bytes: int | None,
) -> AssetManifest:
    if revision is None or license_id is None or expected_bytes is None:
        raise PreflightError("external asset provenance is incomplete")
    return AssetManifest(
        preset="config",
        hub_id=hub_id,
        revision=revision,
        license_id=license_id,
        expected_bytes=expected_bytes,
        source_id="config-audited",
    )
