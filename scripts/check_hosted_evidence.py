#!/usr/bin/env python3
"""Validate durable Colab and GitHub Actions evidence contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPECTED_MATRIX = {
    f"{operating_system}-latest / Python {python} / CPU"
    for operating_system in ("ubuntu", "macos", "windows")
    for python in ("3.10", "3.12")
}


def _load(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict), relative
    return value


def main() -> None:
    colab = _load("docs/research/C10_COLAB_EVIDENCE.json")
    hosted = _load("docs/research/C12_HOSTED_AUDIT.json")

    assert colab["status"] == "passed"
    assert colab["result_origin"] == "hosted_executed"
    colab_sha = colab["notebook"]["source_commit"]
    assert SHA_PATTERN.fullmatch(colab_sha)
    toy_step = next(step for step in colab["steps"] if "toy training" in step["name"])
    assert toy_step["status"] == "passed"
    assert toy_step["toy_demo"] == "passed"
    assert toy_step["demo_status"] == "completed"
    assert toy_step["demo_result_origin"] == "local_executed"
    assert toy_step["fallback_used"] is False

    assert hosted["status"] == "passed"
    hosted_sha = hosted["verified_commit"]
    assert SHA_PATTERN.fullmatch(hosted_sha)
    assert hosted_sha == colab_sha
    for run_name in ("ci", "scheduled_audit"):
        run = hosted[run_name]
        assert run["status"] == "completed", run_name
        assert run["conclusion"] == "success", run_name
        assert run["head_sha"] == hosted_sha, run_name
        assert run["url"].startswith(
            "https://github.com/BangProx/RL-study/actions/runs/"
        )

    jobs = hosted["ci"]["jobs"]
    assert {job["name"] for job in jobs if job["name"] in EXPECTED_MATRIX} == (
        EXPECTED_MATRIX
    )
    assert len(jobs) == 7
    assert all(job["conclusion"] == "success" for job in jobs)
    scheduled_job = hosted["scheduled_audit"]["job"]
    assert scheduled_job["name"] == "links-and-notebooks"
    assert scheduled_job["conclusion"] == "success"
    artifact = hosted["scheduled_audit"]["artifact"]
    assert artifact["name"] == (
        f"scheduled-learning-audit-{hosted['scheduled_audit']['run_id']}"
    )
    assert DIGEST_PATTERN.fullmatch(artifact["digest"])
    assert artifact["size_in_bytes"] > 0
    assert hosted["colab"]["source_commit"] == hosted_sha
    assert hosted["colab"]["toy_demo"] == "passed"
    assert hosted["release_boundary"]["remaining_designed_or_pending"] == []
    print(
        "Hosted evidence: PASS "
        f"(commit={hosted_sha[:12]}, matrix=6, scheduled=1, colab=1)"
    )


if __name__ == "__main__":
    main()
