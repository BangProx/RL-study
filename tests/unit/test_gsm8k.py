from __future__ import annotations

import pytest

from rl_study.data.gsm8k import (
    extract_gsm8k_final_answer,
    load_gsm8k,
    prepare_gsm8k_train_validation,
)
from rl_study.errors import PreflightError


def _rows() -> list[dict[str, object]]:
    return [
        {
            "question": f"What is {index}+1?",
            "answer": f"Add one. #### {index + 1:,}",
        }
        for index in range(8)
    ]


def test_gsm8k_final_answer_and_deterministic_split() -> None:
    assert extract_gsm8k_final_answer("reasoning\n#### 1,234") == "1234"
    first = prepare_gsm8k_train_validation(_rows(), seed=42, validation_size=2)
    second = prepare_gsm8k_train_validation(_rows(), seed=42, validation_size=2)
    changed = prepare_gsm8k_train_validation(_rows(), seed=7, validation_size=2)
    assert first == second
    assert first.split_hash.startswith("sha256:")
    assert first.split_hash != changed.split_hash
    assert {item.uid for item in first.train}.isdisjoint(
        item.uid for item in first.validation
    )


def test_gsm8k_rejects_bad_schema_duplicates_and_test_contamination() -> None:
    with pytest.raises(ValueError, match="####"):
        extract_gsm8k_final_answer("no final marker")
    duplicated = [_rows()[0], _rows()[0], *_rows()[1:]]
    with pytest.raises(ValueError, match="duplicate"):
        prepare_gsm8k_train_validation(duplicated, validation_size=2)
    with pytest.raises(PreflightError, match="official test is blocked"):
        load_gsm8k(official_test=True, purpose="training")
