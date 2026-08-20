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
    install = next(cell for cell in notebook.cells if cell.id == "colab-install")
    if "https://download.pytorch.org/whl/cpu" not in install.source:
        errors.append("free CPU runtime must use the official PyTorch CPU wheel index")
    if "torch==2.13.0" not in install.source:
        errors.append("Colab must install the C1-audited PyTorch version")
    toy = next(cell for cell in notebook.cells if cell.id == "colab-toy")
    if 'toy_payload["status"] == "completed"' not in toy.source:
        errors.append("toy cell must assert the current demo completion contract")
    if 'toy_payload["result_origin"] == "local_executed"' not in toy.source:
        errors.append("toy cell must reject non-executed or fallback results")
    if "assert evidence[\"toy_demo\"] == 'passed'" not in joined:
        errors.append("final evidence assertion is missing")
    if errors:
        raise SystemExit("Colab contract failed:\n- " + "\n- ".join(errors))
    print("Colab source contract: PASS")


if __name__ == "__main__":
    main()
