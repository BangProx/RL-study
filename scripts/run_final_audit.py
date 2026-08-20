#!/usr/bin/env python3
"""Run the C12 clean-clone command set and atomically preserve every result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import torch

from rl_study.platform_metrics import peak_memory_bytes

ROOT = Path(__file__).resolve().parents[1]


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(arguments: Sequence[str]) -> dict[str, object]:
    display = ["<python>" if value == sys.executable else value for value in arguments]
    started = time.perf_counter()
    completed = subprocess.run(
        list(arguments),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(ROOT / ".cache" / "matplotlib-c12"),
        },
    )
    output = completed.stdout + completed.stderr
    return {
        "argv": display,
        "returncode": completed.returncode,
        "status": "passed" if completed.returncode == 0 else "failed",
        "wall_seconds": time.perf_counter() - started,
        "output_sha256": _sha256_text(output),
        "output_tail": output[-4000:],
    }


def _atomic_json(path: Path, value: object) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--allow-hosted-pending", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    initial_status = _git("status", "--porcelain")
    assert not initial_status, "C12 final audit must start from a clean clone"
    python = sys.executable
    commands: list[list[str]] = [
        [python, "-m", "pip", "check"],
        [python, "-m", "rl_study.cli", "--version"],
        [
            python,
            "-m",
            "rl_study.demo",
            "--profile",
            "toy",
            "--non-interactive",
            "--output-dir",
            "artifacts/c12-demo",
            "--json",
        ],
        [python, "-m", "pytest", "-q"],
        [python, "-m", "ruff", "check", "."],
        [python, "-m", "mypy", "src"],
        [python, "scripts/check_design_contract.py"],
        [python, "scripts/check_provenance.py"],
        [python, "scripts/check_links.py", "--local"],
        [
            python,
            "scripts/check_notebook_contract.py",
            "--language",
            "all",
            "--require-executed",
        ],
        [python, "scripts/check_bilingual_parity.py"],
        [python, "scripts/check_colab_contract.py"],
        [python, "scripts/check_hosted_evidence.py"],
        [python, "-m", "mkdocs", "build", "--strict"],
        [
            python,
            "scripts/execute_notebooks.py",
            "--language",
            "all",
            "--kernel-name",
            args.kernel_name,
        ],
        [
            python,
            "scripts/check_notebook_contract.py",
            "--language",
            "all",
            "--require-executed",
        ],
        [python, "scripts/check_bilingual_parity.py"],
    ]
    release_command = [python, "scripts/check_release_contract.py"]
    if args.allow_hosted_pending:
        release_command.append("--allow-hosted-pending")
    commands.insert(9, release_command)

    results = [_run(command) for command in commands]
    passed = all(result["returncode"] == 0 for result in results)
    evidence = {
        "schema_version": 1,
        "checkpoint": "C12",
        "status": (
            "local_pass_hosted_pending"
            if passed and args.allow_hosted_pending
            else "passed"
            if passed
            else "failed"
        ),
        "result_origin": "local_executed",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "clean_clone": {
            "path_at_execution": str(ROOT),
            "initial_git_status": initial_status,
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "origin": _git("remote", "get-url", "origin"),
        },
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "interpreter": sys.executable,
            "peak_auditor_memory_bytes": peak_memory_bytes(),
        },
        "hosted_pending_allowed": args.allow_hosted_pending,
        "commands": results,
        "summary": {
            "passed": sum(result["returncode"] == 0 for result in results),
            "failed": sum(result["returncode"] != 0 for result in results),
            "wall_seconds": time.perf_counter() - started,
        },
        "working_tree_after": _git("status", "--porcelain"),
        "limitations": (
            [
                "GitHub-hosted CI and fresh hosted Colab remain pending and are "
                "not implied by this local clean-clone audit."
            ]
            if args.allow_hosted_pending
            else []
        ),
    }
    _atomic_json(args.output.resolve(), evidence)
    print(
        f"C12 audit: {evidence['status']} "
        f"({evidence['summary']['passed']}/{len(results)} commands passed) -> "
        f"{args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
