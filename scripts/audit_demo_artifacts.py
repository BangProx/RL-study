#!/usr/bin/env python3
"""Audit one real demo run and optionally write a compact C11 evidence record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ALGORITHMS = [
    "dpo",
    "rlhf_ppo",
    "grpo",
    "dapo",
    "agentic_reinforce",
]
CARD_FIELDS = {
    "schema_version",
    "run_id",
    "run_status",
    "result_origin",
    "git",
    "config_hash",
    "dependency_lock_hash",
    "environment",
    "model",
    "data",
    "algorithm",
    "seed",
    "step",
    "budgets",
    "timing",
    "metrics",
    "known_deviations",
    "failures",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def _assert_finite(value: object, context: str = "root") -> None:
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite number at {context}"
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{context}[{index}]")


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header.startswith(b"\x89PNG\r\n\x1a\n"), "invalid PNG signature"
    width, height = struct.unpack(">II", header[16:24])
    assert width >= 600 and height >= 400
    return width, height


def _checkpoint(checkpoint: Path) -> dict[str, object]:
    manifest = _json(checkpoint / "manifest.json")
    files = manifest["files"]
    assert isinstance(files, dict)
    for name, expected in files.items():
        assert isinstance(name, str) and isinstance(expected, str)
        assert _sha256(checkpoint / name) == expected, f"integrity failure: {name}"
    card_path = checkpoint / "experiment-card.json"
    card = _json(card_path)
    assert set(card) >= CARD_FIELDS, f"card fields missing in {card_path}"
    assert card["run_status"] == "completed"
    assert card["result_origin"] == "local_executed"
    assert card["paper_reported"] is None
    assert card["upstream_reported"] is None
    environment = card["environment"]
    assert isinstance(environment, dict)
    assert isinstance(environment["ram_bytes"], int)
    assert "vram_bytes" in environment
    return {
        "path": str(checkpoint.relative_to(ROOT)),
        "algorithm": card["algorithm"],
        "step": manifest["step"],
        "manifest_sha256": _sha256(checkpoint / "manifest.json"),
        "experiment_card_sha256": _sha256(card_path),
        "verified_files": len(files),
    }


def audit(
    run_directory: Path, *, command_wall_seconds: float | None
) -> dict[str, object]:
    run = run_directory.resolve()
    assert run.is_relative_to(ROOT), "demo run must be under the repository"
    summary_path = run / "summary.json"
    png_path = run / "comparison.png"
    report_path = run / "report.html"
    interactive_path = run / "compare.html"
    summary = _json(summary_path)
    _assert_finite(summary)
    assert summary["schema_version"] == 1
    assert summary["status"] == "completed"
    assert summary["result_origin"] == "local_executed"
    assert summary["paper_reported"] is None
    assert summary["upstream_reported"] is None
    records = summary["records"]
    assert isinstance(records, list)
    assert [record["algorithm"] for record in records] == EXPECTED_ALGORITHMS
    diagnostics = [record["diagnostic"] for record in records[:4]]
    assert all(isinstance(value, dict) for value in diagnostics)
    assert records[4]["diagnostic"] is None

    width, height = _png_dimensions(png_path)
    report = report_path.read_text(encoding="utf-8")
    interactive = interactive_path.read_text(encoding="utf-8")
    assert "<caption>" in report and "alt=" in report
    assert 'aria-live="polite"' in interactive
    assert "textContent" in interactive
    assert "innerHTML" not in interactive
    assert "https://" not in interactive and "http://" not in interactive

    checkpoint_paths = sorted((run / "checkpoints").glob("**/manifest.json"))
    assert len(checkpoint_paths) == 5
    checkpoints = [_checkpoint(path.parent) for path in checkpoint_paths]
    if command_wall_seconds is not None:
        assert 0 < command_wall_seconds < 600

    artifacts = []
    for path in (summary_path, png_path, report_path, interactive_path):
        artifacts.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "checkpoint": "C11",
        "result_origin": "local_executed",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "command": (
            "python -m rl_study.demo --profile toy --non-interactive "
            "--output-dir artifacts/demo --json"
        ),
        "run_directory": str(run.relative_to(ROOT)),
        "environment": summary["environment"],
        "timing": {
            **summary["timing"],
            "command_wall_seconds": command_wall_seconds,
        },
        "algorithms": EXPECTED_ALGORITHMS,
        "artifacts": artifacts,
        "png_dimensions": {"width": width, "height": height},
        "checkpoints": checkpoints,
        "checks": [
            "all JSON numbers finite",
            "four same-task policies expose reward/KL/entropy/clip diagnostics",
            "Agentic metrics remain semantically separate",
            "PNG signature and dimensions valid",
            "static report has data table and image alternative text",
            "interactive UI uses textContent, has aria-live, and has no network URL",
            "five checkpoint file manifests and experiment cards verified",
            "paper/upstream results remain null",
        ],
        "retention": (
            "demo artifacts are gitignored local evidence; rerun the command to "
            "regenerate them from source"
        ),
        "limitations": [
            "one tiny seed and one update do not rank algorithms",
            "hosted CI and Colab are not implied by this local audit",
        ],
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--command-wall-seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = audit(
        args.run_directory, command_wall_seconds=args.command_wall_seconds
    )
    if args.output is not None:
        _atomic_json(args.output, evidence)
        print(f"Demo artifact audit: PASS -> {args.output}")
    else:
        print(json.dumps(evidence, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
