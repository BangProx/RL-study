"""Deterministic arithmetic lineage for SFT, preference, and verifiable RL."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass

GENERATOR_REVISION = "generated-v1"
SPLIT_SIZES = {"train": 256, "validation": 64, "test": 128}
_ANSWER_PATTERN = re.compile(r"\s*<answer>(-?\d+)</answer>\s*")


@dataclass(frozen=True, slots=True)
class TinyReasoningExample:
    uid: str
    split: str
    prompt: str
    answer: int
    operation: str
    left: int
    right: int

    @property
    def target_response(self) -> str:
        return f"<answer>{self.answer}</answer>"


@dataclass(frozen=True, slots=True)
class PreferenceExample:
    uid: str
    prompt_uid: str
    prompt: str
    chosen: str
    rejected: str
    reason: str


@dataclass(frozen=True, slots=True)
class TinyReasoningDataset:
    train: tuple[TinyReasoningExample, ...]
    validation: tuple[TinyReasoningExample, ...]
    test: tuple[TinyReasoningExample, ...]
    preferences: tuple[PreferenceExample, ...]
    generator_revision: str
    seed: int
    split_hash: str

    def split(self, name: str) -> tuple[TinyReasoningExample, ...]:
        if name == "train":
            return self.train
        if name == "validation":
            return self.validation
        if name == "test":
            return self.test
        raise ValueError(f"unknown split {name!r}; expected {sorted(SPLIT_SIZES)}")


def _task_pool() -> list[tuple[str, int, int, int]]:
    tasks: list[tuple[str, int, int, int]] = []
    for left in range(32):
        for right in range(32):
            tasks.extend(
                (
                    ("+", left, right, left + right),
                    ("-", left, right, left - right),
                    ("*", left, right, left * right),
                )
            )
    return tasks


def _task_uid(operation: str, left: int, right: int) -> str:
    payload = f"{GENERATOR_REVISION}|{operation}|{left}|{right}".encode()
    return "tr-" + hashlib.sha256(payload).hexdigest()[:16]


def _make_example(split: str, task: tuple[str, int, int, int]) -> TinyReasoningExample:
    operation, left, right, answer = task
    uid = _task_uid(operation, left, right)
    prompt = f"{left} {operation} {right} = ? Answer: <answer>N</answer>"
    return TinyReasoningExample(
        uid=uid,
        split=split,
        prompt=prompt,
        answer=answer,
        operation=operation,
        left=left,
        right=right,
    )


def verify_response(example: TinyReasoningExample, response: str) -> bool:
    match = _ANSWER_PATTERN.fullmatch(response)
    return match is not None and int(match.group(1)) == example.answer


def has_valid_format(response: str) -> bool:
    return _ANSWER_PATTERN.fullmatch(response) is not None


def verifier_reward(example: TinyReasoningExample, response: str) -> float:
    if verify_response(example, response):
        return 1.0
    if has_valid_format(response):
        return 0.1
    return 0.0


def _preference_pairs(
    train: tuple[TinyReasoningExample, ...],
) -> tuple[PreferenceExample, ...]:
    pairs: list[PreferenceExample] = []
    deltas = (-3, -2, -1, 1, 2, 3)
    for example in train:
        delta_index = int(hashlib.sha256(example.uid.encode()).hexdigest()[:8], 16)
        wrong_answer = example.answer + deltas[delta_index % len(deltas)]
        pairs.append(
            PreferenceExample(
                uid=f"{example.uid}-numeric",
                prompt_uid=example.uid,
                prompt=example.prompt,
                chosen=example.target_response,
                rejected=f"<answer>{wrong_answer}</answer>",
                reason="incorrect_numeric_answer",
            )
        )
        pairs.append(
            PreferenceExample(
                uid=f"{example.uid}-format",
                prompt_uid=example.uid,
                prompt=example.prompt,
                chosen=example.target_response,
                rejected=f"The answer is {example.answer}.",
                reason="invalid_required_format",
            )
        )
    return tuple(pairs)


def build_preferences(
    examples: tuple[TinyReasoningExample, ...],
) -> tuple[PreferenceExample, ...]:
    return _preference_pairs(examples)


def _split_hash(splits: dict[str, tuple[TinyReasoningExample, ...]]) -> str:
    payload = {
        split: [asdict(example) for example in examples]
        for split, examples in sorted(splits.items())
    }
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def build_tiny_reasoning(*, seed: int = 42) -> TinyReasoningDataset:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    tasks = _task_pool()
    random.Random(seed).shuffle(tasks)
    cursor = 0
    splits: dict[str, tuple[TinyReasoningExample, ...]] = {}
    for split, size in SPLIT_SIZES.items():
        selected = tasks[cursor : cursor + size]
        splits[split] = tuple(_make_example(split, task) for task in selected)
        cursor += size
    preferences = _preference_pairs(splits["train"])
    return TinyReasoningDataset(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        preferences=preferences,
        generator_revision=GENERATOR_REVISION,
        seed=seed,
        split_hash=_split_hash(splits),
    )
