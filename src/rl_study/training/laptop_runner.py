"""Actual public-model LoRA SFT smoke runner behind explicit download approval."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rl_study.adapters.manifest import enforce_download_guard, model_cache_status
from rl_study.adapters.preflight import audited_asset_from_config, qlora_capability
from rl_study.adapters.trl_adapter import TRLAdapterSpec
from rl_study.config import ExperimentConfig
from rl_study.data.gsm8k import GSM8KExample, GSM8KSplit, load_gsm8k
from rl_study.errors import CheckpointError, ConfigError, NumericError, PreflightError
from rl_study.reporting import build_experiment_card, write_experiment_card
from rl_study.runtime import resolve_device, seed_everything

_VALIDATION_EXAMPLES = 8


@dataclass(frozen=True, slots=True)
class LaptopRunResult:
    step: int
    checkpoint: Path
    experiment_card: Path
    metrics: dict[str, float]
    trainable_parameters: int
    total_parameters: int


def _config_from_checkpoint(checkpoint: Path) -> ExperimentConfig:
    try:
        payload = json.loads(
            (checkpoint / "config.resolved.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(
            f"cannot read laptop checkpoint config: {error}"
        ) from error
    return ExperimentConfig.from_mapping(payload)


def _verify_checkpoint_files(checkpoint: Path) -> None:
    try:
        manifest = json.loads(
            (checkpoint / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(
            f"cannot read laptop checkpoint manifest: {error}"
        ) from error
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise CheckpointError("laptop checkpoint file manifest is invalid")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise CheckpointError("laptop checkpoint hash entry is invalid")
        source = checkpoint / name
        if not source.is_file() or _sha256(source) != expected:
            raise CheckpointError(f"laptop checkpoint integrity mismatch: {name}")


@dataclass(frozen=True, slots=True)
class _OptionalStack:
    transformers: Any
    peft: Any


def _optional_stack() -> _OptionalStack:
    TRLAdapterSpec.for_algorithm("sft").validate_installed()
    try:
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
    except ImportError as error:
        raise PreflightError(
            "real-model LoRA requires the repository laptop extra"
        ) from error
    return _OptionalStack(transformers=transformers, peft=peft)


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _model_asset(config: ExperimentConfig):  # type: ignore[no-untyped-def]
    return audited_asset_from_config(
        hub_id=config.model.policy,
        revision=config.model.revision,
        license_id=config.model.license_id,
        expected_bytes=config.model.expected_weight_bytes,
    )


def _check_laptop_config(config: ExperimentConfig) -> None:
    if config.profile != "laptop" or config.algorithm.name != "sft":
        raise ConfigError("laptop runner requires profile=laptop and algorithm=sft")
    if config.data.id != "openai/gsm8k" or config.data.subset != "main":
        raise ConfigError("C8 laptop runner requires audited openai/gsm8k main")
    if config.model.revision is None:
        raise ConfigError("laptop model revision is required")


def _load_base_and_tokenizer(
    config: ExperimentConfig,
    stack: _OptionalStack,
    *,
    device: torch.device,
    cache_dir: str | None,
    local_files_only: bool,
) -> tuple[Any, Any]:
    tokenizer = stack.transformers.AutoTokenizer.from_pretrained(
        config.model.policy,
        revision=config.model.revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None:
        raise PreflightError("tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_arguments: dict[str, object] = {
        "revision": config.model.revision,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
        "dtype": _dtype(config.model.dtype),
    }
    if config.model.adapter == "qlora":
        supported, reason = qlora_capability(str(device))
        if not supported:
            raise PreflightError(reason)
        load_arguments["quantization_config"] = stack.transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_dtype(config.model.dtype),
        )
        load_arguments["device_map"] = {"": str(device)}
    base = stack.transformers.AutoModelForCausalLM.from_pretrained(
        config.model.policy, **load_arguments
    )
    if config.model.adapter != "qlora":
        base.to(device)
    if config.model.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
    return base, tokenizer


def _attach_adapter(
    config: ExperimentConfig,
    stack: _OptionalStack,
    base: Any,
    *,
    resume_checkpoint: Path | None,
) -> Any:
    if resume_checkpoint is not None:
        model = stack.peft.PeftModel.from_pretrained(
            base, resume_checkpoint / "adapter", is_trainable=True
        )
    else:
        lora_config = stack.peft.LoraConfig(
            task_type="CAUSAL_LM",
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=[
                item.strip() for item in config.model.lora_target_modules.split(",")
            ],
            bias="none",
        )
        model = stack.peft.get_peft_model(base, lora_config)
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name in trainable):
        raise PreflightError(
            "LoRA ownership check failed: only lora_* parameters may be trainable"
        )
    return model


def _encode_example(
    tokenizer: Any,
    example: GSM8KExample,
    *,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    prompt = f"Question: {example.question}\nAnswer:"
    completion = f" {example.answer}{tokenizer.eos_token}"
    encoded = tokenizer(
        prompt + completion,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_ids = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )["input_ids"]
    batch = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    labels = batch["input_ids"].clone()
    prompt_length = min(prompt_ids.shape[1], labels.shape[1])
    labels[:, :prompt_length] = -100
    if tokenizer.pad_token_id is not None:
        labels[batch["input_ids"] == tokenizer.pad_token_id] = -100
    action_tokens = int((labels != -100).sum())
    if action_tokens == 0:
        raise PreflightError(
            "max_sequence_length truncated every completion token; increase it"
        )
    batch["labels"] = labels
    return batch, action_tokens


def _load_resume_state(checkpoint: Path, config: ExperimentConfig) -> dict[str, object]:
    try:
        state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(
            f"cannot read laptop checkpoint state: {error}"
        ) from error
    if not isinstance(state, dict) or state.get("resume_hash") != config.resume_sha256:
        raise CheckpointError("laptop checkpoint immutable config hash mismatch")
    step = state.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise CheckpointError("laptop checkpoint step is invalid")
    return state


def _state_int(state: dict[str, object], name: str, *, minimum: int = 0) -> int:
    value = state.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckpointError(f"laptop checkpoint {name} is invalid")
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _save_adapter_checkpoint(
    target: Path,
    *,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    state: dict[str, object],
) -> None:
    if target.exists():
        raise CheckpointError(f"refusing to overwrite laptop checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        model.save_pretrained(
            temporary / "adapter",
            safe_serialization=True,
            save_embedding_layers=False,
        )
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        (temporary / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "config.resolved.json").write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, allow_nan=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        files = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file()
        )
        manifest = {
            "schema_version": 1,
            "step": state["step"],
            "resume_hash": config.resume_sha256,
            "files": {
                name: _sha256(temporary / name)
                for name in files
                if name != "manifest.json"
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@torch.no_grad()
def _validation_loss(
    model: Any,
    tokenizer: Any,
    examples: tuple[GSM8KExample, ...],
    *,
    max_length: int,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for example in examples[:_VALIDATION_EXAMPLES]:
        batch, _ = _encode_example(
            tokenizer, example, max_length=max_length, device=device
        )
        losses.append(float(model(**batch).loss.detach().cpu()))
    loss = sum(losses) / len(losses)
    model.train()
    if not torch.isfinite(torch.tensor(loss)):
        raise NumericError("laptop validation loss is not finite")
    return loss


def train_laptop_sft(
    config: ExperimentConfig,
    *,
    accept_download: bool,
    cache_dir: str | None = None,
    output_root: str | Path | None = None,
    resume: str | Path | None = None,
    stop_after: int | None = None,
) -> LaptopRunResult:
    _check_laptop_config(config)
    model_asset = _model_asset(config)
    cached, _ = model_cache_status(model_asset, cache_dir=cache_dir)
    enforce_download_guard(model_asset, cached=cached, accept_download=accept_download)
    stack = _optional_stack()
    resolution = resolve_device(
        config.training.device,
        allow_fallback=config.training.allow_device_fallback,
    )
    if config.model.dtype == "float16" and resolution.resolved.type == "cpu":
        raise PreflightError("float16 CPU training is unsupported; use float32")
    seed_everything(config.training.seed)
    target_step = config.training.steps if stop_after is None else stop_after
    if not 1 <= target_step <= config.training.steps:
        raise ConfigError("stop_after must be within configured training.steps")
    resume_checkpoint = None if resume is None else Path(resume)
    resume_state: dict[str, object] = {}
    start_step = 0
    processed_tokens = 0
    prior_model_forwards = 0
    prompt_uids: list[str] = []
    if resume_checkpoint is not None:
        _verify_checkpoint_files(resume_checkpoint)
        resume_state = _load_resume_state(resume_checkpoint, config)
        start_step = _state_int(resume_state, "step", minimum=1)
        processed_tokens = _state_int(resume_state, "processed_tokens")
        prior_model_forwards = _state_int(resume_state, "model_forwards")
        stored_uids = resume_state.get("prompt_uids", [])
        if not isinstance(stored_uids, list) or not all(
            isinstance(item, str) for item in stored_uids
        ):
            raise CheckpointError("laptop checkpoint prompt_uids are invalid")
        prompt_uids = list(stored_uids)
    if start_step >= target_step:
        raise ConfigError("resume checkpoint must be before the target step")

    started = time.perf_counter()
    base, tokenizer = _load_base_and_tokenizer(
        config,
        stack,
        device=resolution.resolved,
        cache_dir=cache_dir,
        local_files_only=not accept_download,
    )
    model = _attach_adapter(config, stack, base, resume_checkpoint=resume_checkpoint)
    model.train()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=0.0,
    )
    if resume_checkpoint is not None:
        try:
            optimizer_state = torch.load(
                resume_checkpoint / "optimizer.pt",
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, RuntimeError) as error:
            raise CheckpointError(f"cannot load laptop optimizer: {error}") from error
        optimizer.load_state_dict(optimizer_state)

    loaded = load_gsm8k(
        seed=config.data.seed,
        validation_size=256,
        cache_dir=cache_dir,
    )
    if not isinstance(loaded, GSM8KSplit):
        raise PreflightError("GSM8K train loader returned the wrong split type")
    losses: list[float] = []
    for step in range(start_step, target_step):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for micro_step in range(config.training.gradient_accumulation_steps):
            index = (
                step * config.training.gradient_accumulation_steps + micro_step
            ) % len(loaded.train)
            example = loaded.train[index]
            batch, action_tokens = _encode_example(
                tokenizer,
                example,
                max_length=config.training.max_sequence_length,
                device=resolution.resolved,
            )
            output = model(**batch)
            loss = output.loss / config.training.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise NumericError("laptop LoRA loss is not finite")
            loss.backward()
            accumulated_loss += float(loss.detach().cpu())
            processed_tokens += action_tokens
            prompt_uids.append(example.uid)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and "lora_" not in name:
                raise PreflightError(f"unexpected base-model gradient: {name}")
        optimizer.step()
        losses.append(accumulated_loss)

    if processed_tokens > config.training.response_token_budget:
        raise ConfigError(
            "response_token_budget exceeded: "
            f"processed={processed_tokens}, "
            f"budget={config.training.response_token_budget}"
        )

    validation_loss = _validation_loss(
        model,
        tokenizer,
        loaded.validation,
        max_length=config.training.max_sequence_length,
        device=resolution.resolved,
    )
    root = Path(output_root or config.output.root)
    checkpoint = (
        root
        / f"laptop-sft-seed{config.training.seed}"
        / f"checkpoint-{target_step:06d}"
    )
    model_forwards = (
        prior_model_forwards
        + (target_step - start_step) * config.training.gradient_accumulation_steps
        + _VALIDATION_EXAMPLES
    )
    state = {
        "schema_version": 1,
        "step": target_step,
        "resume_hash": config.resume_sha256,
        "split_hash": loaded.split_hash,
        "prompt_uids": prompt_uids,
        "processed_tokens": processed_tokens,
        "model_forwards": model_forwards,
        "last_train_loss": losses[-1],
        "validation_loss": validation_loss,
        "validation_examples": _VALIDATION_EXAMPLES,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }
    _save_adapter_checkpoint(
        checkpoint,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        config=config,
        state=state,
    )
    metrics = {
        "last_train_loss": losses[-1],
        "validation_loss": validation_loss,
        "validation_examples": float(_VALIDATION_EXAMPLES),
        "trainable_fraction": trainable_parameters / total_parameters,
    }
    card = build_experiment_card(
        config,
        run_id=f"laptop-sft-seed{config.training.seed}-step{target_step}",
        run_status="completed",
        step=target_step,
        wall_seconds=time.perf_counter() - started,
        metrics=metrics,
        data_split_hash=loaded.split_hash,
        prompt_uids=prompt_uids,
        optimized_prompt_uids=prompt_uids,
        optimizer_steps=target_step,
        processed_tokens=processed_tokens,
        model_forwards=model_forwards,
        known_deviations=(
            "C8 laptop smoke is LoRA SFT, not a paper-scale RL result",
            "GSM8K official test is not loaded; validation is derived from train",
        ),
    )
    card["adapter"] = {
        "type": config.model.adapter,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "target_modules": config.model.lora_target_modules.split(","),
        "trl": asdict(TRLAdapterSpec.for_algorithm("sft")),
    }
    experiment_card = write_experiment_card(checkpoint, card)
    return LaptopRunResult(
        step=target_step,
        checkpoint=checkpoint,
        experiment_card=experiment_card,
        metrics=metrics,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
    )


def evaluate_laptop_checkpoint(
    checkpoint: str | Path, *, cache_dir: str | None = None
) -> tuple[str, int, dict[str, float]]:
    source = Path(checkpoint)
    _verify_checkpoint_files(source)
    config = _config_from_checkpoint(source)
    _check_laptop_config(config)
    state = _load_resume_state(source, config)
    stack = _optional_stack()
    resolution = resolve_device(
        config.training.device,
        allow_fallback=config.training.allow_device_fallback,
    )
    base, tokenizer = _load_base_and_tokenizer(
        config,
        stack,
        device=resolution.resolved,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model = _attach_adapter(config, stack, base, resume_checkpoint=source)
    loaded = load_gsm8k(
        seed=config.data.seed,
        validation_size=256,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    if not isinstance(loaded, GSM8KSplit):
        raise PreflightError("GSM8K evaluator loaded the wrong split type")
    validation_loss = _validation_loss(
        model,
        tokenizer,
        loaded.validation,
        max_length=config.training.max_sequence_length,
        device=resolution.resolved,
    )
    return (
        "sft",
        _state_int(state, "step", minimum=1),
        {
            "validation_loss": validation_loss,
            "validation_examples": float(_VALIDATION_EXAMPLES),
        },
    )
