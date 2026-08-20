#!/usr/bin/env python3
"""Fail on structural, metadata, source, output, or safety contract drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TAGS = {
    "rl-study-core",
    "rl-study-deep-dive",
    "rl-study-network",
    "rl-study-server",
    "rl-study-slow",
    "rl-study-hide-solution",
}
HEADINGS = {
    "ko": [
        "## Goal",
        "## Setup",
        "## Steps",
        "## Checks",
        "## 내가 자주 틀리는 것",
        "## 60초 요약",
        "## Next Steps",
        "## Sources",
    ],
    "en": [
        "## Goal",
        "## Setup",
        "## Steps",
        "## Checks",
        "## Mistakes I Revisit",
        "## 60-Second Recap",
        "## Next Steps",
        "## Sources",
    ],
}


def _code_hash(source: str) -> str:
    normalized = (
        "\n".join(line.rstrip() for line in source.splitlines()).rstrip() + "\n"
    )
    return "sha256:" + hashlib.sha256(normalized.encode()).hexdigest()


def _fail(errors: list[str], path: Path, message: str) -> None:
    errors.append(f"{path.relative_to(ROOT)}: {message}")


def _output_text(cell: nbformat.NotebookNode) -> str:
    pieces: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            pieces.append(str(output.get("text", "")))
        elif output.get("output_type") == "error":
            pieces.extend(str(item) for item in output.get("traceback", []))
        else:
            data = output.get("data", {})
            pieces.extend(str(value) for value in data.values())
    return "".join(pieces)


def check(
    path: Path, source_ids: set[str], *, require_executed: bool = False
) -> list[str]:
    errors: list[str] = []
    notebook = nbformat.read(path, as_version=4)
    metadata = notebook.metadata.get("rl_study", {})
    language = metadata.get("language")
    lesson = metadata.get("lesson_id")
    if language not in HEADINGS:
        _fail(errors, path, "language must be ko or en")
        return errors
    if not isinstance(lesson, str) or not re.fullmatch(r"L\d{2}", lesson):
        _fail(errors, path, "invalid lesson_id")
    if metadata.get("schema_version") != 1 or metadata.get("profile") != "toy":
        _fail(errors, path, "invalid root schema/profile")
    if metadata.get("network_required") is not False or metadata.get("seed") != 42:
        _fail(errors, path, "core notebook must be offline with seed 42")
    declared_sources = metadata.get("source_ids", [])
    if not isinstance(declared_sources, list) or not declared_sources:
        _fail(errors, path, "source_ids must be a non-empty list")
        declared_sources = []
    unknown = set(declared_sources) - source_ids
    if unknown:
        _fail(errors, path, f"unknown root source IDs: {sorted(unknown)}")
    learning_doc = metadata.get("learning_doc")
    if not isinstance(learning_doc, str) or not learning_doc.startswith("docs/"):
        _fail(errors, path, "learning_doc must be a repository docs path")
    elif not (ROOT / learning_doc).is_file():
        _fail(errors, path, f"learning_doc does not exist: {learning_doc}")

    headings = [
        line
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("## ")
    ]
    if headings != HEADINGS[language]:
        _fail(errors, path, f"wrong section order: {headings}")
    goal_cell = next(
        (cell for cell in notebook.cells if cell.source.startswith("## Goal")), None
    )
    goals = 0 if goal_cell is None else sum(
        line.startswith("- ") for line in goal_cell.source.splitlines()
    )
    if not 1 <= goals <= 3:
        _fail(errors, path, f"Goal must contain 1-3 bullets, found {goals}")

    cell_ids: set[str] = set()
    stable_ids: set[str] = set()
    assertion_count = 0
    execution_counts: list[int | None] = []
    combined_markdown = ""
    for cell in notebook.cells:
        combined_markdown += cell.source if cell.cell_type == "markdown" else ""
        if not re.fullmatch(r"[a-z0-9-]+", cell.id):
            _fail(errors, path, f"invalid cell id {cell.id!r}")
        if cell.id in cell_ids:
            _fail(errors, path, f"duplicate cell id {cell.id}")
        cell_ids.add(cell.id)
        cell_meta = cell.metadata.get("rl_study", {})
        stable_id = cell_meta.get("stable_id")
        if not isinstance(stable_id, str) or not re.fullmatch(
            rf"{lesson}\.S\d{{2}}\.C\d{{2}}", stable_id
        ):
            _fail(errors, path, f"invalid stable_id {stable_id!r}")
        elif stable_id in stable_ids:
            _fail(errors, path, f"duplicate stable_id {stable_id}")
        stable_ids.add(str(stable_id))
        if cell_meta.get("kind") != cell.cell_type:
            _fail(errors, path, f"kind mismatch in {cell.id}")
        cell_sources = cell_meta.get("source_ids", [])
        if not isinstance(cell_sources, list) or set(cell_sources) - source_ids:
            _fail(errors, path, f"invalid cell source_ids in {cell.id}")
        tags = set(cell.metadata.get("tags", []))
        if not tags or tags - ALLOWED_TAGS:
            _fail(errors, path, f"invalid/missing tags in {cell.id}")
        if cell.cell_type == "code":
            execution_counts.append(cell.get("execution_count"))
            if cell_meta.get("code_hash") != _code_hash(cell.source):
                _fail(errors, path, f"code hash mismatch in {cell.id}")
            assertion_count += sum(
                line.lstrip().startswith("assert ")
                for line in cell.source.splitlines()
            )
            output = _output_text(cell)
            if len(output.encode()) > 100_000:
                _fail(errors, path, f"output exceeds 100KB in {cell.id}")
            if any(
                item.get("output_type") == "error"
                for item in cell.get("outputs", [])
            ):
                _fail(errors, path, f"traceback output in {cell.id}")
            if re.search(r"\b\d{1,3}%\|", output):
                _fail(errors, path, f"progress spam in {cell.id}")
            if re.search(r"(?:UserWarning|RuntimeWarning|Traceback)", output):
                _fail(errors, path, f"warning/traceback text in {cell.id}")
            if require_executed and not cell.get("outputs"):
                _fail(errors, path, f"executed notebook has no output in {cell.id}")
    if assertion_count < 1:
        _fail(errors, path, "Checks requires at least one executable assertion")
    required_words = (
        ("먼저 예측", "회상 문제", "내가 자주 틀리는 것", "60초 요약")
        if language == "ko"
        else ("Predict first", "Recall", "Mistakes I Revisit", "60-Second Recap")
    )
    for word in required_words:
        if word not in combined_markdown:
            _fail(errors, path, f"missing learning-rhythm marker {word!r}")
    if (
        isinstance(learning_doc, str)
        and f"../../{learning_doc}" not in combined_markdown
    ):
        _fail(errors, path, "Next Steps must link the declared learning_doc")
    expected_course_map = (
        "../../docs/course-map.md"
        if language == "ko"
        else "../../docs/course-map.en.md"
    )
    if expected_course_map not in combined_markdown:
        _fail(errors, path, "Next Steps must link the language course map")
    if combined_markdown.count("⏱") != 3:
        _fail(errors, path, "expected exactly three timed micro-sections")
    labels = (
        ("[필수/CORE]", "[심화/DEEP DIVE]")
        if language == "ko"
        else ("[CORE]", "[DEEP DIVE]")
    )
    if combined_markdown.count(labels[0]) < 2 or labels[1] not in combined_markdown:
        _fail(errors, path, "missing fast/full path labels")
    if lesson == "L00" and not all(
        marker in combined_markdown for marker in ("```mermaid", "```text")
    ):
        _fail(errors, path, "L00 requires Mermaid and ASCII concept maps")
    if require_executed:
        expected_counts = list(range(1, len(execution_counts) + 1))
        if execution_counts != expected_counts:
            _fail(
                errors,
                path,
                f"execution counts are not a clean sequence: {execution_counts}",
            )
    all_source = "\n".join(cell.source for cell in notebook.cells)
    if re.search(r"/Users/|[A-Za-z]:\\\\", all_source):
        _fail(errors, path, "absolute local path found")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("ko", "en", "all"), default="all")
    parser.add_argument("--require-executed", action="store_true")
    args = parser.parse_args()
    sources = json.loads((ROOT / "docs/sources.yml").read_text(encoding="utf-8"))
    known_source_ids = {
        item["id"]
        for section in ("papers", "repositories", "models", "datasets")
        for item in sources[section]
    }
    languages = ("ko", "en") if args.language == "all" else (args.language,)
    paths = [
        path
        for language in languages
        for path in sorted((ROOT / "notebooks" / language).glob("L*.ipynb"))
    ]
    errors: list[str] = []
    expected = 17 * len(languages)
    if len(paths) != expected:
        errors.append(f"expected {expected} lesson notebooks, found {len(paths)}")
    for path in paths:
        errors.extend(
            check(path, known_source_ids, require_executed=args.require_executed)
        )
    if errors:
        raise SystemExit("Notebook contract failed:\n- " + "\n- ".join(errors))
    print(f"Notebook contract: PASS ({len(paths)} notebooks)")


if __name__ == "__main__":
    main()
