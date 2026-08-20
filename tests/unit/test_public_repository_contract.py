from __future__ import annotations

import re
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
        "README.en.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "SUPPORT.md",
        "docs/index.md",
        "docs/getting-started.md",
        "docs/course-map.md",
        "docs/course-map.en.md",
        "docs/math.md",
        "docs/glossary.md",
        "docs/hardware.md",
        "docs/troubleshooting.md",
        "docs/provenance.md",
        "docs/research/README.md",
        "docs/algorithms/cards.md",
        "docs/assets/alignment-loss-map.svg",
        "docs/assets/one-update-diagnostics.png",
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
    assert "python scripts/summarize_notebook_runs.py" in text
    assert ".internal/evidence/C10_NOTEBOOK_REPORT.json" in text
    assert "docs/research/C10_" not in text
    assert "actions/upload-artifact@v7" in text
    assert "include-hidden-files: true" in text


def test_mkdocs_navigation_targets_exist() -> None:
    config = _yaml(ROOT / "mkdocs.yml")
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert config["strict"] == "true"
    assert "scripts/mkdocs_hooks.py" in text
    for required in (
        "getting-started.md",
        "course-map.md",
        "course-map.en.md",
        "math.md",
        "algorithms/cards.md",
        "hardware.md",
        "troubleshooting.md",
        "provenance.md",
    ):
        assert required in text
        assert (ROOT / "docs" / required).is_file()


def test_public_research_excludes_maintainer_audit_logs() -> None:
    names = {path.name for path in (ROOT / "docs/research").iterdir()}
    assert not any(name.startswith(("C10_", "C11_", "C12_")) for name in names)


def test_course_maps_link_every_language_lesson_notebook() -> None:
    expected = [f"{index:02d}" for index in range(17)]
    for language, course_map in (
        ("ko", "docs/course-map.md"),
        ("en", "docs/course-map.en.md"),
    ):
        text = (ROOT / course_map).read_text(encoding="utf-8")
        matches = re.findall(
            rf"\[L(\d{{2}}) · [^\]]+\]\(\.\./"
            rf"(notebooks/{language}/L\d{{2}}_[^)]+\.ipynb)\)",
            text,
        )
        assert [lesson for lesson, _ in matches] == expected
        for lesson, relative in matches:
            assert Path(relative).stem.startswith(f"L{lesson}_")
            assert (ROOT / relative).is_file()


def test_readme_has_one_primary_start_and_real_visual_assets() -> None:
    korean = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README.en.md").read_text(encoding="utf-8")
    assert korean.count("> **처음 시작하기 →") == 1
    assert english.count("> **Start here →") == 1
    figure = ROOT / "docs/assets/one-update-diagnostics.png"
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure.stat().st_size > 100_000
    loss_map = (ROOT / "docs/assets/alignment-loss-map.svg").read_text(
        encoding="utf-8"
    )
    assert "PPO → DPO → GRPO → DAPO" in loss_map
    assert "role=\"img\"" in loss_map


def test_algorithm_notes_link_course_notebooks_and_neighbors() -> None:
    for note in sorted((ROOT / "docs/algorithms").glob("*.md")):
        text = note.read_text(encoding="utf-8")
        assert "../course-map.md" in text, note.name
        assert "../../notebooks/ko/" in text, note.name


def test_release_contract_passes_for_public_repository() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_release_contract.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "Release contract: PASS" in completed.stdout
