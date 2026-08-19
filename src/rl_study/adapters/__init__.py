"""Optional real-model and distributed-framework adapter boundaries."""

from rl_study.adapters.alfworld import AlfWorldAdapter, alfworld_preflight
from rl_study.adapters.manifest import (
    DATASET_PRESETS,
    MODEL_PRESETS,
    AssetManifest,
    ModelManifest,
    enforce_download_guard,
    estimate_training_memory,
    model_cache_status,
)
from rl_study.adapters.preflight import build_profile_preflight
from rl_study.adapters.trl_adapter import TRLAdapterSpec
from rl_study.adapters.verl_recipe import render_verl_recipe, validate_verl_recipe

__all__ = [
    "DATASET_PRESETS",
    "MODEL_PRESETS",
    "AlfWorldAdapter",
    "AssetManifest",
    "ModelManifest",
    "TRLAdapterSpec",
    "alfworld_preflight",
    "build_profile_preflight",
    "enforce_download_guard",
    "estimate_training_memory",
    "model_cache_status",
    "render_verl_recipe",
    "validate_verl_recipe",
]
