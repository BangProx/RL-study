"""Strict, immutable experiment configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from rl_study.errors import ConfigError

SUPPORTED_ALGORITHMS = frozenset(
    {
        "q_learning",
        "sft",
        "dqn",
        "reinforce",
        "actor_critic",
        "ppo",
        "reward_model",
        "rlhf_ppo",
        "dpo",
        "grpo",
        "rloo",
        "dr_grpo",
        "dapo",
        "gspo",
        "agentic_reinforce",
    }
)
SUPPORTED_PROFILES = frozenset({"toy", "laptop", "server"})


def _as_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{context} must be a mapping with string keys")
    return value


def _check_keys(
    data: Mapping[str, Any],
    *,
    context: str,
    allowed: set[str],
    required: set[str],
) -> None:
    unknown = set(data) - allowed
    missing = required - set(data)
    if unknown:
        raise ConfigError(f"{context} has unknown keys: {sorted(unknown)}")
    if missing:
        raise ConfigError(f"{context} is missing required keys: {sorted(missing)}")


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context} must be a finite number")
    converted = float(value)
    if converted != converted or converted in (float("inf"), float("-inf")):
        raise ConfigError(f"{context} must be a finite number")
    return converted


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty string")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be true or false")
    return value


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    name: str
    variant: str = "paper"
    gamma: float = 0.99
    clip_low: float = 0.2
    clip_high: float = 0.2
    kl_coefficient: float = 0.0
    advantage_normalization: str = "none"
    loss_reduction: str = "masked_mean"
    beta: float = 0.1
    label_smoothing: float = 0.0
    update_epochs: int = 2
    reward_source: str = "verifier"
    clip_higher: bool = False
    dynamic_sampling: bool = False
    dynamic_sampling_multiplier: int = 4
    token_level_loss: bool = False
    overlong_reward_shaping: bool = False
    overlong_buffer_length: int = 4
    overlong_penalty_scale: float = 1.0
    credit_assignment: str = "discounted_returns"

    @classmethod
    def from_mapping(cls, value: object) -> AlgorithmConfig:
        data = _as_mapping(value, "algorithm")
        allowed = {
            "name",
            "variant",
            "gamma",
            "clip_low",
            "clip_high",
            "kl_coefficient",
            "advantage_normalization",
            "loss_reduction",
            "beta",
            "label_smoothing",
            "update_epochs",
            "reward_source",
            "clip_higher",
            "dynamic_sampling",
            "dynamic_sampling_multiplier",
            "token_level_loss",
            "overlong_reward_shaping",
            "overlong_buffer_length",
            "overlong_penalty_scale",
            "credit_assignment",
        }
        _check_keys(data, context="algorithm", allowed=allowed, required={"name"})
        config = cls(
            name=_string(data["name"], "algorithm.name"),
            variant=_string(data.get("variant", "paper"), "algorithm.variant"),
            gamma=_number(data.get("gamma", 0.99), "algorithm.gamma"),
            clip_low=_number(data.get("clip_low", 0.2), "algorithm.clip_low"),
            clip_high=_number(data.get("clip_high", 0.2), "algorithm.clip_high"),
            kl_coefficient=_number(
                data.get("kl_coefficient", 0.0), "algorithm.kl_coefficient"
            ),
            advantage_normalization=_string(
                data.get("advantage_normalization", "none"),
                "algorithm.advantage_normalization",
            ),
            loss_reduction=_string(
                data.get("loss_reduction", "masked_mean"), "algorithm.loss_reduction"
            ),
            beta=_number(data.get("beta", 0.1), "algorithm.beta"),
            label_smoothing=_number(
                data.get("label_smoothing", 0.0), "algorithm.label_smoothing"
            ),
            update_epochs=_integer(
                data.get("update_epochs", 2),
                "algorithm.update_epochs",
                minimum=1,
            ),
            reward_source=_string(
                data.get("reward_source", "verifier"), "algorithm.reward_source"
            ),
            clip_higher=_boolean(
                data.get("clip_higher", False), "algorithm.clip_higher"
            ),
            dynamic_sampling=_boolean(
                data.get("dynamic_sampling", False), "algorithm.dynamic_sampling"
            ),
            dynamic_sampling_multiplier=_integer(
                data.get("dynamic_sampling_multiplier", 4),
                "algorithm.dynamic_sampling_multiplier",
                minimum=1,
            ),
            token_level_loss=_boolean(
                data.get("token_level_loss", False), "algorithm.token_level_loss"
            ),
            overlong_reward_shaping=_boolean(
                data.get("overlong_reward_shaping", False),
                "algorithm.overlong_reward_shaping",
            ),
            overlong_buffer_length=_integer(
                data.get("overlong_buffer_length", 4),
                "algorithm.overlong_buffer_length",
                minimum=1,
            ),
            overlong_penalty_scale=_number(
                data.get("overlong_penalty_scale", 1.0),
                "algorithm.overlong_penalty_scale",
            ),
            credit_assignment=_string(
                data.get("credit_assignment", "discounted_returns"),
                "algorithm.credit_assignment",
            ),
        )
        if config.name not in SUPPORTED_ALGORITHMS:
            raise ConfigError(
                f"algorithm.name must be one of {sorted(SUPPORTED_ALGORITHMS)}"
            )
        if not 0.0 <= config.gamma <= 1.0:
            raise ConfigError("algorithm.gamma must be between 0 and 1")
        if config.clip_low < 0 or config.clip_high < 0:
            raise ConfigError("algorithm clip bounds must be non-negative")
        if config.kl_coefficient < 0:
            raise ConfigError("algorithm.kl_coefficient must be non-negative")
        if config.beta <= 0:
            raise ConfigError("algorithm.beta must be positive")
        if not 0.0 <= config.label_smoothing < 0.5:
            raise ConfigError("algorithm.label_smoothing must be in [0, 0.5)")
        if config.reward_source not in {"verifier", "reward_model"}:
            raise ConfigError(
                "algorithm.reward_source must be verifier or reward_model"
            )
        if config.overlong_penalty_scale < 0:
            raise ConfigError("algorithm.overlong_penalty_scale must be non-negative")
        if config.credit_assignment not in {
            "broadcast_outcome",
            "discounted_returns",
        }:
            raise ConfigError(
                "algorithm.credit_assignment must be broadcast_outcome or "
                "discounted_returns"
            )
        if config.name != "agentic_reinforce" and "credit_assignment" in data:
            raise ConfigError(
                "algorithm.credit_assignment requires name=agentic_reinforce"
            )
        dapo_toggles = (
            config.clip_higher,
            config.dynamic_sampling,
            config.token_level_loss,
            config.overlong_reward_shaping,
        )
        if config.name != "dapo" and any(dapo_toggles):
            raise ConfigError("DAPO component toggles require algorithm.name=dapo")
        return config


@dataclass(frozen=True, slots=True)
class DataConfig:
    id: str
    revision: str
    split: str = "train"
    seed: int = 42
    subset: str | None = None
    license_id: str | None = None
    expected_download_bytes: int | None = None

    @classmethod
    def from_mapping(cls, value: object) -> DataConfig:
        data = _as_mapping(value, "data")
        _check_keys(
            data,
            context="data",
            allowed={
                "id",
                "revision",
                "split",
                "seed",
                "subset",
                "license_id",
                "expected_download_bytes",
            },
            required={"id", "revision"},
        )
        subset = data.get("subset")
        license_id = data.get("license_id")
        expected_download_bytes = data.get("expected_download_bytes")
        config = cls(
            id=_string(data["id"], "data.id"),
            revision=_string(data["revision"], "data.revision"),
            split=_string(data.get("split", "train"), "data.split"),
            seed=_integer(data.get("seed", 42), "data.seed"),
            subset=None if subset is None else _string(subset, "data.subset"),
            license_id=None
            if license_id is None
            else _string(license_id, "data.license_id"),
            expected_download_bytes=None
            if expected_download_bytes is None
            else _integer(
                expected_download_bytes,
                "data.expected_download_bytes",
                minimum=0,
            ),
        )
        return config


@dataclass(frozen=True, slots=True)
class ModelConfig:
    policy: str
    reference: str | None = None
    reward: str | None = None
    trust_remote_code: bool = False
    revision: str | None = None
    license_id: str | None = None
    expected_weight_bytes: int | None = None
    adapter: str = "none"
    dtype: str = "float32"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,v_proj"
    gradient_checkpointing: bool = False

    @classmethod
    def from_mapping(cls, value: object) -> ModelConfig:
        data = _as_mapping(value, "model")
        _check_keys(
            data,
            context="model",
            allowed={
                "policy",
                "reference",
                "reward",
                "trust_remote_code",
                "revision",
                "license_id",
                "expected_weight_bytes",
                "adapter",
                "dtype",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "lora_target_modules",
                "gradient_checkpointing",
            },
            required={"policy"},
        )
        reference = data.get("reference")
        reward = data.get("reward")
        revision = data.get("revision")
        license_id = data.get("license_id")
        expected_weight_bytes = data.get("expected_weight_bytes")
        config = cls(
            policy=_string(data["policy"], "model.policy"),
            reference=None
            if reference is None
            else _string(reference, "model.reference"),
            reward=None if reward is None else _string(reward, "model.reward"),
            trust_remote_code=_boolean(
                data.get("trust_remote_code", False), "model.trust_remote_code"
            ),
            revision=None if revision is None else _string(revision, "model.revision"),
            license_id=None
            if license_id is None
            else _string(license_id, "model.license_id"),
            expected_weight_bytes=None
            if expected_weight_bytes is None
            else _integer(
                expected_weight_bytes,
                "model.expected_weight_bytes",
                minimum=0,
            ),
            adapter=_string(data.get("adapter", "none"), "model.adapter"),
            dtype=_string(data.get("dtype", "float32"), "model.dtype"),
            lora_rank=_integer(data.get("lora_rank", 8), "model.lora_rank", minimum=1),
            lora_alpha=_integer(
                data.get("lora_alpha", 16), "model.lora_alpha", minimum=1
            ),
            lora_dropout=_number(data.get("lora_dropout", 0.0), "model.lora_dropout"),
            lora_target_modules=_string(
                data.get("lora_target_modules", "q_proj,v_proj"),
                "model.lora_target_modules",
            ),
            gradient_checkpointing=_boolean(
                data.get("gradient_checkpointing", False),
                "model.gradient_checkpointing",
            ),
        )
        if config.adapter not in {"none", "lora", "qlora"}:
            raise ConfigError("model.adapter must be none, lora, or qlora")
        if config.dtype not in {"float32", "float16", "bfloat16"}:
            raise ConfigError("model.dtype must be float32, float16, or bfloat16")
        if not 0.0 <= config.lora_dropout < 1.0:
            raise ConfigError("model.lora_dropout must be in [0, 1)")
        targets = [item.strip() for item in config.lora_target_modules.split(",")]
        if not targets or any(not item for item in targets):
            raise ConfigError("model.lora_target_modules must be comma-separated names")
        return config


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    steps: int = 100
    batch_size: int = 16
    group_size: int = 4
    response_token_budget: int = 32768
    max_new_tokens: int = 22
    max_sequence_length: int = 128
    learning_rate: float = 0.0005
    gradient_accumulation_steps: int = 1
    device: str = "cpu"
    allow_device_fallback: bool = False

    @classmethod
    def from_mapping(cls, value: object) -> TrainingConfig:
        data = _as_mapping(value, "training")
        allowed = {
            "seed",
            "steps",
            "batch_size",
            "group_size",
            "response_token_budget",
            "max_new_tokens",
            "max_sequence_length",
            "learning_rate",
            "gradient_accumulation_steps",
            "device",
            "allow_device_fallback",
        }
        _check_keys(data, context="training", allowed=allowed, required=set())
        config = cls(
            seed=_integer(data.get("seed", 42), "training.seed"),
            steps=_integer(data.get("steps", 100), "training.steps", minimum=1),
            batch_size=_integer(
                data.get("batch_size", 16), "training.batch_size", minimum=1
            ),
            group_size=_integer(
                data.get("group_size", 4), "training.group_size", minimum=2
            ),
            response_token_budget=_integer(
                data.get("response_token_budget", 32768),
                "training.response_token_budget",
                minimum=1,
            ),
            max_new_tokens=_integer(
                data.get("max_new_tokens", 22),
                "training.max_new_tokens",
                minimum=1,
            ),
            max_sequence_length=_integer(
                data.get("max_sequence_length", 128),
                "training.max_sequence_length",
                minimum=8,
            ),
            learning_rate=_number(
                data.get("learning_rate", 0.0005), "training.learning_rate"
            ),
            gradient_accumulation_steps=_integer(
                data.get("gradient_accumulation_steps", 1),
                "training.gradient_accumulation_steps",
                minimum=1,
            ),
            device=_string(data.get("device", "cpu"), "training.device"),
            allow_device_fallback=_boolean(
                data.get("allow_device_fallback", False),
                "training.allow_device_fallback",
            ),
        )
        if config.learning_rate <= 0:
            raise ConfigError("training.learning_rate must be positive")
        return config


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    every_steps: int = 25
    split: str = "validation"

    @classmethod
    def from_mapping(cls, value: object) -> EvaluationConfig:
        data = _as_mapping(value, "evaluation")
        _check_keys(
            data,
            context="evaluation",
            allowed={"every_steps", "split"},
            required=set(),
        )
        config = cls(
            every_steps=_integer(
                data.get("every_steps", 25), "evaluation.every_steps", minimum=1
            ),
            split=_string(data.get("split", "validation"), "evaluation.split"),
        )
        if config.split == "train":
            raise ConfigError("evaluation.split must not be train")
        return config


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root: str = "artifacts"

    @classmethod
    def from_mapping(cls, value: object) -> OutputConfig:
        data = _as_mapping(value, "output")
        _check_keys(data, context="output", allowed={"root"}, required=set())
        return cls(root=_string(data.get("root", "artifacts"), "output.root"))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: int
    profile: str
    algorithm: AlgorithmConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig

    @classmethod
    def from_mapping(cls, value: object) -> ExperimentConfig:
        data = _as_mapping(value, "config")
        required = {
            "schema_version",
            "profile",
            "algorithm",
            "data",
            "model",
            "training",
            "evaluation",
            "output",
        }
        _check_keys(data, context="config", allowed=required, required=required)
        schema_version = _integer(data["schema_version"], "schema_version", minimum=1)
        if schema_version != 1:
            raise ConfigError(
                f"unsupported schema_version={schema_version}; expected 1"
            )
        profile = _string(data["profile"], "profile")
        if profile not in SUPPORTED_PROFILES:
            raise ConfigError(f"profile must be one of {sorted(SUPPORTED_PROFILES)}")
        config = cls(
            schema_version=schema_version,
            profile=profile,
            algorithm=AlgorithmConfig.from_mapping(data["algorithm"]),
            data=DataConfig.from_mapping(data["data"]),
            model=ModelConfig.from_mapping(data["model"]),
            training=TrainingConfig.from_mapping(data["training"]),
            evaluation=EvaluationConfig.from_mapping(data["evaluation"]),
            output=OutputConfig.from_mapping(data["output"]),
        )
        if config.profile == "toy" and not config.model.policy.startswith("tiny-"):
            raise ConfigError("toy profile requires a repository tiny-* policy")
        if config.model.trust_remote_code:
            raise ConfigError(
                "trust_remote_code=true requires a separate explicit approval"
            )
        if config.profile == "toy" and config.model.adapter != "none":
            raise ConfigError("toy profile uses repository models and adapter=none")
        if config.profile in {"laptop", "server"}:
            missing_provenance = [
                name
                for name, field in (
                    ("model.revision", config.model.revision),
                    ("model.license_id", config.model.license_id),
                    ("model.expected_weight_bytes", config.model.expected_weight_bytes),
                    ("data.license_id", config.data.license_id),
                    (
                        "data.expected_download_bytes",
                        config.data.expected_download_bytes,
                    ),
                )
                if field is None
            ]
            if missing_provenance:
                raise ConfigError(
                    "external profiles require audited provenance fields: "
                    + ", ".join(missing_provenance)
                )
        if config.profile == "laptop" and config.model.adapter not in {
            "lora",
            "qlora",
        }:
            raise ConfigError("laptop profile requires model.adapter=lora or qlora")
        if config.profile == "server" and config.training.device == "cpu":
            raise ConfigError("server profile cannot claim a CPU distributed run")
        if config.algorithm.overlong_buffer_length > config.training.max_new_tokens:
            raise ConfigError(
                "algorithm.overlong_buffer_length cannot exceed training.max_new_tokens"
            )
        return config

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        source = Path(path)
        try:
            loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f"failed to read config {source}: {error}") from error
        return cls.from_mapping(loaded)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        algorithm = payload.get("algorithm")
        if (
            self.algorithm.name != "agentic_reinforce"
            and isinstance(algorithm, dict)
        ):
            algorithm.pop("credit_assignment", None)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def resume_immutable_dict(self) -> dict[str, object]:
        payload = self.to_dict()
        training = payload["training"]
        evaluation = payload["evaluation"]
        if not isinstance(training, dict) or not isinstance(evaluation, dict):
            raise ConfigError("internal config serialization error")
        training.pop("steps")
        evaluation.pop("every_steps")
        payload.pop("output")
        return payload

    @property
    def resume_sha256(self) -> str:
        canonical = json.dumps(
            self.resume_immutable_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"
