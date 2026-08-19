#!/usr/bin/env python3
"""Check semantic structure and executable identity of Korean/English mirrors."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def _stable_map(notebook: nbformat.NotebookNode) -> dict[str, nbformat.NotebookNode]:
    return {
        cell.metadata["rl_study"]["stable_id"]: cell for cell in notebook.cells
    }


def _equations(source: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", equation).strip()
        for equation in re.findall(r"\$\$(.*?)\$\$", source, flags=re.DOTALL)
    ]


def main() -> None:
    errors: list[str] = []
    for ko_path in sorted((ROOT / "notebooks/ko").glob("L*.ipynb")):
        en_path = ROOT / "notebooks/en" / ko_path.name
        if not en_path.is_file():
            errors.append(f"missing mirror: {en_path.relative_to(ROOT)}")
            continue
        ko = nbformat.read(ko_path, as_version=4)
        en = nbformat.read(en_path, as_version=4)
        ko_map, en_map = _stable_map(ko), _stable_map(en)
        if list(ko_map) != list(en_map):
            errors.append(f"stable ID order mismatch: {ko_path.name}")
            continue
        root_fields = (
            "lesson_id",
            "title_key",
            "profile",
            "estimated_minutes_full",
            "estimated_minutes_fast",
            "prerequisites",
            "source_ids",
            "network_required",
            "seed",
        )
        for field in root_fields:
            if ko.metadata["rl_study"][field] != en.metadata["rl_study"][field]:
                errors.append(f"root {field} mismatch: {ko_path.name}")
        for stable_id in ko_map:
            left, right = ko_map[stable_id], en_map[stable_id]
            if left.cell_type != right.cell_type or left.id != right.id:
                errors.append(f"cell identity mismatch: {ko_path.name} {stable_id}")
            left_meta = left.metadata["rl_study"]
            right_meta = right.metadata["rl_study"]
            for field in ("kind", "path", "concept_ids", "source_ids", "test_ids"):
                if left_meta[field] != right_meta[field]:
                    errors.append(f"{field} mismatch: {ko_path.name} {stable_id}")
            if left.cell_type == "code" and (
                left.source != right.source
                or left_meta["code_hash"] != right_meta["code_hash"]
            ):
                errors.append(f"code mismatch: {ko_path.name} {stable_id}")
            if left.cell_type == "markdown" and _equations(
                left.source
            ) != _equations(right.source):
                errors.append(f"equation mismatch: {ko_path.name} {stable_id}")
        ko_asserts = sum(
            len(re.findall(r"(?m)^\s*assert ", cell.source))
            for cell in ko.cells
            if cell.cell_type == "code"
        )
        en_asserts = sum(
            len(re.findall(r"(?m)^\s*assert ", cell.source))
            for cell in en.cells
            if cell.cell_type == "code"
        )
        if ko_asserts != en_asserts:
            errors.append(f"assertion count mismatch: {ko_path.name}")
    if errors:
        raise SystemExit("Bilingual parity failed:\n- " + "\n- ".join(errors))
    print("Bilingual parity: PASS (17 pairs)")


if __name__ == "__main__":
    main()
