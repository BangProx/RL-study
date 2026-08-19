#!/usr/bin/env python3
"""Validate the Colab quickstart without inventing hosted-runtime evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/colab/RL_study_quickstart.ipynb"


def _hash(source: str) -> str:
    normalized = (
        "\n".join(line.rstrip() for line in source.splitlines()).rstrip() + "\n"
    )
    return "sha256:" + hashlib.sha256(normalized.encode()).hexdigest()


def main() -> None:
    notebook = nbformat.read(PATH, as_version=4)
    errors: list[str] = []
    metadata = notebook.metadata.get("rl_study", {})
    if metadata.get("order") != ["clone", "install", "toy", "real-model-optional"]:
        errors.append("root execution order is not canonical")
    sources = [cell.source for cell in notebook.cells]
    joined = "\n".join(sources)
    needles = [
        "git\", \"clone",
        "pip\", \"install",
        "rl_study.demo",
        "RUN_REAL_MODEL_SMOKE",
    ]
    positions = [joined.find(needle) for needle in needles]
    if -1 in positions or positions != sorted(positions):
        errors.append(f"clone/install/toy/optional order mismatch: {positions}")
    stable_ids: set[str] = set()
    for cell in notebook.cells:
        cell_meta = cell.metadata.get("rl_study", {})
        stable_id = cell_meta.get("stable_id")
        if not isinstance(stable_id, str) or stable_id in stable_ids:
            errors.append(f"invalid or duplicate stable ID: {stable_id!r}")
        stable_ids.add(str(stable_id))
        if cell_meta.get("kind") != cell.cell_type:
            errors.append(f"kind mismatch: {cell.id}")
        if cell.cell_type == "code":
            try:
                compile(cell.source, f"{PATH.name}:{cell.id}", "exec")
            except SyntaxError as error:
                errors.append(f"invalid Python in {cell.id}: {error}")
            if cell_meta.get("code_hash") != _hash(cell.source):
                errors.append(f"code hash mismatch: {cell.id}")
            if cell.get("outputs") or cell.get("execution_count") is not None:
                errors.append(f"unverified hosted output must be empty: {cell.id}")
    optional = next(cell for cell in notebook.cells if cell.id == "colab-real-model")
    if set(optional.metadata.get("tags", [])) != {
        "rl-study-network",
        "rl-study-server",
    }:
        errors.append("real-model cell needs network and server tags")
    if "--accept-download" not in optional.source:
        errors.append("real-model download requires explicit approval flag")
    if "assert evidence[\"toy_demo\"] == 'passed'" not in joined:
        errors.append("final evidence assertion is missing")
    if errors:
        raise SystemExit("Colab contract failed:\n- " + "\n- ".join(errors))
    print("Colab contract: PASS (hosted execution still requires populated remote)")


if __name__ == "__main__":
    main()
