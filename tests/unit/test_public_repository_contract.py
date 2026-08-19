from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: Path) -> dict[str, object]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(value, dict)
    return value


def test_public_docs_and_community_files_exist() -> None:
    required = (
        "README.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "SUPPORT.md",
        "docs/index.md",
        "docs/getting-started.md",
        "docs/course-map.md",
        "docs/math.md",
        "docs/glossary.md",
        "docs/hardware.md",
        "docs/troubleshooting.md",
        "docs/provenance.md",
        "docs/algorithms/cards.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/lesson_feedback.yml",
        ".github/pull_request_template.md",
    )
    for relative in required:
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 100, relative


def test_ci_covers_three_operating_systems_and_document_contracts() -> None:
    workflow = _yaml(ROOT / ".github/workflows/ci.yml")
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    cpu = jobs["cpu-tests"]
    assert isinstance(cpu, dict)
    strategy = cpu["strategy"]
    assert isinstance(strategy, dict)
    matrix = strategy["matrix"]
    assert isinstance(matrix, dict)
    assert set(matrix["os"]) == {
        "ubuntu-latest",
        "macos-latest",
        "windows-latest",
    }
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for command in (
        "python -m pytest",
        "python -m ruff check .",
        "python -m mypy src",
        "python scripts/check_links.py --local",
        "python scripts/check_provenance.py",
        "python -m mkdocs build --strict",
    ):
        assert command in text
    assert "actions/checkout@v7" in text
    assert "actions/setup-python@v7" in text


def test_scheduled_job_checks_network_and_fresh_notebooks() -> None:
    workflow = _yaml(ROOT / ".github/workflows/scheduled.yml")
    triggers = workflow["on"]
    assert isinstance(triggers, dict)
    assert "schedule" in triggers
    text = (ROOT / ".github/workflows/scheduled.yml").read_text(encoding="utf-8")
    assert "python scripts/check_links.py --network" in text
    assert "python scripts/execute_notebooks.py --language all" in text
    assert "actions/upload-artifact@v7" in text


def test_mkdocs_navigation_targets_exist() -> None:
    config = _yaml(ROOT / "mkdocs.yml")
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert config["strict"] == "true"
    for required in (
        "getting-started.md",
        "course-map.md",
        "math.md",
        "algorithms/cards.md",
        "hardware.md",
        "troubleshooting.md",
        "provenance.md",
    ):
        assert required in text
        assert (ROOT / "docs" / required).is_file()


def test_release_contract_allows_only_declared_hosted_pending_rows() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_release_contract.py",
            "--allow-hosted-pending",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "Release contract: PASS" in completed.stdout
