#!/usr/bin/env python3
"""Render the README's real one-update diagnostic figure from the toy demo."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs/assets/one-update-diagnostics.png"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib-readme"))

from rl_study.reporting.demo import run_demo  # noqa: E402


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rl-study-readme-") as temporary:
        artifacts = run_demo(output_root=Path(temporary), device="cpu")
        shutil.copyfile(artifacts.figure_png, TARGET)
    print(TARGET.relative_to(ROOT))


if __name__ == "__main__":
    main()
