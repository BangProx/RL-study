#!/usr/bin/env python3
"""Audit release metadata, claims, secrets, file sizes, Git, and traceability."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAX_REPOSITORY_FILE_BYTES = 5 * 1024 * 1024
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "site",
}
REQUIRED_PUBLIC_FILES = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "GOAL.md",
    "LICENSE",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    "docs/known-limitations.md",
    "docs/provenance.md",
    "mkdocs.yml",
}
SECRET_PATTERNS = {
    "AWS access key": re.compile("AKIA" + r"[0-9A-Z]{16}"),
    "GitHub token": re.compile("ghp_" + r"[A-Za-z0-9]{30,}"),
    "Hugging Face token": re.compile("hf_" + r"[A-Za-z0-9]{30,}"),
    "OpenAI-style key": re.compile("sk-" + r"(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "Slack token": re.compile("xox" + r"[abprs]-[A-Za-z0-9-]{20,}"),
    "private key": re.compile("BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY"),
}
EXPECTED_REPOSITORY_URL = "https://github.com/BangProx/RL-study"


def _candidate_files() -> list[Path]:
    return [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and not any(part in EXCLUDED_PARTS for part in path.parts)
    ]


def _version() -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package = (ROOT / "src/rl_study/_version.py").read_text(encoding="utf-8")
    project_match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    package_match = re.search(r'^__version__ = "([^"]+)"$', package, re.MULTILINE)
    assert project_match is not None and package_match is not None
    assert project_match.group(1) == package_match.group(1)
    return project_match.group(1)


def _traceability(allow_hosted_pending: bool) -> dict[str, str]:
    text = (ROOT / "docs/design/traceability.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| (R\d{2}) \|.*?\| ([^|]+) \|$", text, flags=re.MULTILINE
    )
    assert len(rows) == 20, f"expected 20 traceability rows, found {len(rows)}"
    statuses = {requirement: status.strip() for requirement, status in rows}
    allowed = {"verified", "external-manual"}
    pending = {
        requirement
        for requirement, status in statuses.items()
        if status not in allowed
    }
    if allow_hosted_pending:
        assert pending <= {"R11", "R17", "R19", "R20"}, sorted(pending)
    else:
        assert not pending, f"unfinished traceability rows: {sorted(pending)}"
    return statuses


def _json_finite(value: object, context: str) -> None:
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite JSON value: {context}"
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _json_finite(child, f"{context}.{key}")
        if value.get("result_origin") == "not_executed":
            assert value.get("local_executed") is None, context
        if value.get("run_status") == "external-manual":
            assert value.get("local_executed") is None, context
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _json_finite(child, f"{context}[{index}]")


def _research_claims() -> int:
    count = 0
    for path in sorted((ROOT / "docs/research").glob("*.json")):
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        _json_finite(value, str(path.relative_to(ROOT)))
        count += 1
    return count


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _canonical_repository_url(value: str) -> str:
    canonical = value.strip().rstrip("/")
    if canonical.endswith(".git"):
        canonical = canonical[:-4]
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-hosted-pending", action="store_true")
    args = parser.parse_args()

    missing = [
        relative
        for relative in REQUIRED_PUBLIC_FILES
        if not (ROOT / relative).is_file()
    ]
    assert not missing, f"missing public files: {sorted(missing)}"
    version = _version()
    assert version == "0.1.0"
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{version}]" in changelog
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert str(citation["version"]) == version
    assert citation["license"] == "Apache-2.0"

    files = _candidate_files()
    too_large = [
        (str(path.relative_to(ROOT)), path.stat().st_size)
        for path in files
        if path.stat().st_size > MAX_REPOSITORY_FILE_BYTES
    ]
    assert not too_large, f"repository files over 5 MiB: {too_large}"
    symlinks = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_symlink()
        and not any(part in EXCLUDED_PARTS for part in path.parts)
    ]
    assert not symlinks, f"unexpected symlinks: {symlinks}"

    findings: list[tuple[str, str]] = []
    text_files = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text_files += 1
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append((str(path.relative_to(ROOT)), label))
    assert not findings, f"possible secrets: {findings}"

    statuses = _traceability(args.allow_hosted_pending)
    research_reports = _research_claims()
    assert _git("branch", "--show-current") == "main"
    assert _canonical_repository_url(
        _git("remote", "get-url", "origin")
    ) == EXPECTED_REPOSITORY_URL
    assert "force" not in (ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    ).lower()

    pending = [
        key
        for key, value in statuses.items()
        if value not in {"verified", "external-manual"}
    ]
    print(
        "Release contract: PASS "
        f"(version={version}, files={len(files)}, text_files={text_files}, "
        f"research_json={research_reports}, largest_bytes="
        f"{max(path.stat().st_size for path in files)}, pending={pending})"
    )


if __name__ == "__main__":
    main()
