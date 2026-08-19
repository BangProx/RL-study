"""Pinned GSM8K adapter with deterministic train-derived validation."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rl_study.adapters.manifest import DATASET_PRESETS
from rl_study.errors import PreflightError


@dataclass(frozen=True, slots=True)
class GSM8KExample:
    uid: str
    question: str
    answer: str
    final_answer: str


@dataclass(frozen=True, slots=True)
class GSM8KSplit:
    train: tuple[GSM8KExample, ...]
    validation: tuple[GSM8KExample, ...]
    split_hash: str


def extract_gsm8k_final_answer(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        raise ValueError("GSM8K answer is missing the #### final-answer marker")
    final = answer.rsplit(marker, maxsplit=1)[1].strip().replace(",", "")
    if not final:
        raise ValueError("GSM8K final answer is empty")
    return final


def _example(row: Mapping[str, object]) -> GSM8KExample:
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("GSM8K question must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("GSM8K answer must be a non-empty string")
    uid = "gsm8k-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return GSM8KExample(
        uid=uid,
        question=question.strip(),
        answer=answer.strip(),
        final_answer=extract_gsm8k_final_answer(answer),
    )


def prepare_gsm8k_train_validation(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int = 42,
    validation_size: int = 256,
) -> GSM8KSplit:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not 1 <= validation_size < len(rows):
        raise ValueError("validation_size must leave at least one training example")
    examples = tuple(_example(row) for row in rows)
    if len({item.uid for item in examples}) != len(examples):
        raise ValueError("duplicate GSM8K question IDs detected")
    ordered = tuple(
        sorted(
            examples,
            key=lambda item: hashlib.sha256(f"{seed}:{item.uid}".encode()).hexdigest(),
        )
    )
    validation = ordered[:validation_size]
    train = ordered[validation_size:]
    canonical = json.dumps(
        {
            "revision": DATASET_PRESETS["gsm8k"].revision,
            "seed": seed,
            "train": [item.uid for item in train],
            "validation": [item.uid for item in validation],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    split_hash = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    return GSM8KSplit(train=train, validation=validation, split_hash=split_hash)


def load_gsm8k(
    *,
    seed: int = 42,
    validation_size: int = 256,
    cache_dir: str | None = None,
    official_test: bool = False,
    purpose: str = "training",
    local_files_only: bool = False,
) -> GSM8KSplit | tuple[GSM8KExample, ...]:
    """Load the exact audited revision; official test is final-evaluation only."""
    if official_test and purpose != "final_evaluation":
        raise PreflightError(
            "GSM8K official test is blocked during training/tuning; "
            "set purpose=final_evaluation in an evaluator-only process"
        )
    try:
        datasets = importlib.import_module("datasets")
    except ImportError as error:
        raise PreflightError(
            "GSM8K loading requires the repository laptop extra"
        ) from error
    manifest = DATASET_PRESETS["gsm8k"]
    split_name = "test" if official_test else "train"
    download_config = datasets.DownloadConfig(local_files_only=local_files_only)
    loaded: Any = datasets.load_dataset(
        manifest.hub_id,
        "main",
        split=split_name,
        revision=manifest.revision,
        cache_dir=cache_dir,
        download_config=download_config,
    )
    rows = [dict(row) for row in loaded]
    if official_test:
        return tuple(_example(row) for row in rows)
    return prepare_gsm8k_train_validation(
        rows, seed=seed, validation_size=validation_size
    )
