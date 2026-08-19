"""Pinned TRL adapter contract without importing optional dependencies in core."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

from rl_study.errors import PreflightError

TRL_VERSION = "1.10.0"
TRL_REVISION = "a7be897f5c8d7b52161f9f8a47d8e6242456b898"


@dataclass(frozen=True, slots=True)
class TRLAdapterSpec:
    algorithm: str
    trainer_class: str
    framework_version: str = TRL_VERSION
    framework_revision: str = TRL_REVISION

    @classmethod
    def for_algorithm(cls, algorithm: str) -> TRLAdapterSpec:
        trainers = {
            "sft": "SFTTrainer",
            "reward_model": "RewardTrainer",
            "dpo": "DPOTrainer",
            "rlhf_ppo": "PPOTrainer",
            "grpo": "GRPOTrainer",
            "rloo": "RLOOTrainer",
        }
        if algorithm not in trainers:
            raise PreflightError(
                f"TRL {TRL_VERSION} laptop adapter does not map {algorithm!r}; "
                "use the toy clean-room trainer or pinned verl server recipe"
            )
        return cls(algorithm=algorithm, trainer_class=trainers[algorithm])

    def validate_installed(self) -> None:
        try:
            installed = importlib.metadata.version("trl")
        except importlib.metadata.PackageNotFoundError as error:
            raise PreflightError(
                "TRL is optional; install the repository laptop extra"
            ) from error
        if installed != self.framework_version:
            raise PreflightError(
                f"TRL version mismatch: installed={installed}, "
                f"required={self.framework_version}"
            )

    def metadata(self) -> dict[str, str]:
        return {
            "framework": "trl",
            "version": self.framework_version,
            "revision": self.framework_revision,
            "algorithm": self.algorithm,
            "trainer_class": self.trainer_class,
        }
