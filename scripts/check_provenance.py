#!/usr/bin/env python3
"""Validate source IDs, exact revisions, licenses, and no-copy declarations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = re.compile(r"`([a-z0-9][a-z0-9.-]+)`")
HEX_REVISION = re.compile(r"[0-9a-f]{40}")
ALLOWED_PROVENANCE = {
    "copied",
    "adapted",
    "clean-room-reimplemented",
    "architecture-reference",
    "optional-runtime-dependency",
}
MANIFEST_PREFIXES = (
    "repo-",
    "framework-",
    "benchmark-",
    "model-",
    "dataset-",
)


def _manifest() -> dict[str, Any]:
    value = json.loads((ROOT / "docs/sources.yml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("source manifest must be an object")
    return value


def _doc_source_ids() -> set[str]:
    values: set[str] = set()
    for path in sorted((ROOT / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for source_id in SOURCE_ID.findall(text):
            if (
                source_id.startswith(MANIFEST_PREFIXES)
                or source_id == "sutton-barto-rl2"
                or re.search(r"-\d{4}$", source_id)
            ):
                values.add(source_id)
    return values


def main() -> None:
    manifest = _manifest()
    policy = manifest["policy"]
    assert policy["repository_license"] == "Apache-2.0"
    assert policy["default_trust_remote_code"] is False
    assert set(policy["provenance_classes"]) == ALLOWED_PROVENANCE

    sections = ("papers", "repositories", "models", "datasets")
    entries = [entry for section in sections for entry in manifest[section]]
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids)), "source IDs must be globally unique"

    for paper in manifest["papers"]:
        for key in ("title", "authors", "status", "version", "url", "used_for"):
            assert paper.get(key), f"{paper['id']} missing {key}"
        assert paper["provenance"] in ALLOWED_PROVENANCE
        assert str(paper["url"]).startswith(("http://", "https://"))

    copied_or_adapted: list[str] = []
    for repository in manifest["repositories"]:
        assert HEX_REVISION.fullmatch(repository["revision"]), repository["id"]
        assert repository["license"], repository["id"]
        assert repository["provenance"] in ALLOWED_PROVENANCE
        copied_files = repository["copied_files"]
        assert isinstance(copied_files, list), repository["id"]
        if repository["provenance"] in {"copied", "adapted"}:
            copied_or_adapted.append(repository["id"])
            assert copied_files, repository["id"]
            assert repository["license"] != "NOASSERTION", repository["id"]
            assert repository["license_path"], repository["id"]
        if repository["license"] == "NOASSERTION":
            assert not copied_files, repository["id"]
            assert repository["license_path"] is None, repository["id"]

    for asset in [*manifest["models"], *manifest["datasets"]]:
        assert asset["license"], asset["id"]
        checksum = str(asset["manifest_checksum"])
        if asset["id"] == "dataset-tinyreasoning":
            assert asset["revision"] == "generated-by-repository-code"
            assert checksum == "recorded per generation config and seed"
        else:
            assert HEX_REVISION.fullmatch(asset["revision"]), asset["id"]
            assert checksum == f"hf_revision:{asset['revision']}", asset["id"]
        assert asset["redistribution"], asset["id"]

    unknown = _doc_source_ids() - set(ids)
    assert not unknown, f"unknown source IDs in docs: {sorted(unknown)}"
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    if not copied_or_adapted:
        assert "No third-party source file is currently vendored" in notice
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text and "Version 2.0" in license_text

    print(
        "Provenance contract: PASS "
        f"({len(ids)} sources, {len(_doc_source_ids())} documented source IDs, "
        f"{len(copied_or_adapted)} copied/adapted repositories)"
    )


if __name__ == "__main__":
    main()
