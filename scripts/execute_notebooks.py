#!/usr/bin/env python3
"""Execute notebooks in fresh kernels and append truthful JSONL run records."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import psutil
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/research/C10_NOTEBOOK_EXECUTIONS.jsonl"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        commit.stdout.strip() if commit.returncode == 0 else "UNBORN",
        bool(status.stdout.strip()),
    )


def _monitor_memory(stop: threading.Event, result: list[int]) -> None:
    process = psutil.Process()
    peak = 0
    while not stop.wait(0.02):
        candidates = [process]
        with contextlib.suppress(
            psutil.NoSuchProcess, psutil.AccessDenied, PermissionError
        ):
            candidates.extend(process.children(recursive=True))
        total = 0
        for candidate in candidates:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                total += candidate.memory_info().rss
        peak = max(peak, total)
    result.append(peak)


def _stream_text(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(
        str(output.get("text", ""))
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def _runtime_versions(notebook: nbformat.NotebookNode) -> tuple[str, str]:
    output = _stream_text(notebook)
    for line in output.splitlines():
        if line.startswith("python=") and " torch=" in line:
            fields = dict(item.split("=", 1) for item in line.split())
            return fields["python"], fields["torch"]
    return platform.python_version(), "unknown"


def execute(path: Path, *, timeout: int, kernel_name: str) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    metadata = notebook.metadata["rl_study"]
    language = str(metadata["language"])
    os.environ["RL_STUDY_NOTEBOOK_LANGUAGE"] = language
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    stop = threading.Event()
    memory: list[int] = []
    monitor = threading.Thread(target=_monitor_memory, args=(stop, memory), daemon=True)
    monitor.start()
    error: str | None = None
    success = False
    try:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        client.execute()
        nbformat.write(notebook, path)
        success = True
    except Exception as caught:  # record every failed attempt; never erase evidence
        error = f"{type(caught).__name__}: {caught}"
    finally:
        stop.set()
        monitor.join(timeout=2)
    python_version, torch_version = _runtime_versions(notebook)
    commit, dirty = _git_state()
    record: dict[str, object] = {
        "lesson_id": metadata["lesson_id"],
        "language": language,
        "profile": metadata["profile"],
        "started_at": started_at,
        "duration_seconds": time.perf_counter() - started,
        "peak_rss_bytes": max(memory, default=0),
        "python": python_version,
        "torch": torch_version,
        "platform": f"{platform.system()}-{platform.machine()}",
        "git_commit": commit,
        "dirty": dirty,
        "notebook": str(path.relative_to(ROOT)),
        "notebook_sha256": _sha256(path),
        "success": success,
        "error": error,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("ko", "en", "all"), default="ko")
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--kernel-name", default="rl-study")
    args = parser.parse_args()
    if args.notebook is not None:
        paths = [args.notebook.resolve()]
    else:
        languages = ("ko", "en") if args.language == "all" else (args.language,)
        paths = [
            path
            for language in languages
            for path in sorted((ROOT / "notebooks" / language).glob("L*.ipynb"))
        ]
    failed = 0
    for index, path in enumerate(paths, start=1):
        record = execute(path, timeout=args.timeout, kernel_name=args.kernel_name)
        failed += int(not record["success"])
        print(
            f"[{index}/{len(paths)}] {record['lesson_id']} {record['language']} "
            f"success={record['success']} duration={record['duration_seconds']:.2f}s"
        )
        if not record["success"]:
            print(f"  error={record['error']}")
    if failed:
        raise SystemExit(f"notebook execution failed: {failed}/{len(paths)}")


if __name__ == "__main__":
    main()
