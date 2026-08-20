#!/usr/bin/env python3
"""Summarize the append-only C10 notebook evidence without hiding failures."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / ".internal/evidence/C10_NOTEBOOK_EXECUTIONS.jsonl"
REPORT = ROOT / ".internal/evidence/C10_NOTEBOOK_REPORT.json"


def main() -> None:
    raw = MANIFEST.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    latest: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        latest[(str(row["language"]), str(row["lesson_id"]))] = row
    if len(latest) != 34:
        raise SystemExit(f"expected 34 language/lesson pairs, found {len(latest)}")
    failed_latest = [row for row in latest.values() if not row["success"]]
    if failed_latest:
        raise SystemExit(f"latest evidence contains failures: {failed_latest}")
    failure_types = Counter(
        str(row.get("error", "unknown")).split(":", 1)[0]
        for row in rows
        if not row["success"]
    )
    language_summary = {}
    for language in ("ko", "en"):
        selected = [
            row for (row_language, _), row in latest.items() if row_language == language
        ]
        language_summary[language] = {
            "notebooks": len(selected),
            "success": sum(bool(row["success"]) for row in selected),
            "total_duration_seconds": sum(
                float(row["duration_seconds"]) for row in selected
            ),
            "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in selected),
        }
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_origin": "local_executed",
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "selection_rule": "last append-only record per (language, lesson_id)",
        "attempts": {
            "total": len(rows),
            "success": sum(bool(row["success"]) for row in rows),
            "failed": sum(not row["success"] for row in rows),
            "failure_types": dict(sorted(failure_types.items())),
            "failures_are_preserved": True,
        },
        "latest_clean_set": {
            "notebooks": len(latest),
            "success": sum(bool(row["success"]) for row in latest.values()),
            "languages": language_summary,
            "python_versions": sorted({str(row["python"]) for row in latest.values()}),
            "torch_versions": sorted({str(row["torch"]) for row in latest.values()}),
            "platforms": sorted({str(row["platform"]) for row in latest.values()}),
            "git_commits": sorted({str(row["git_commit"]) for row in latest.values()}),
            "dirty": any(bool(row["dirty"]) for row in latest.values()),
            "records": [latest[key] for key in sorted(latest)],
        },
        "interpretation": (
            "Durations measure code execution in fresh local kernels, not "
            "learner reading time. Failed attempts remain in the JSONL manifest. "
            "UNBORN/dirty records are pre-release implementation evidence and "
            "must be replaced in the final clean-clone audit."
        ),
        "hosted_colab": {
            "status": "pending-populated-remote",
            "reason": (
                "The required GitHub origin has no refs; hosted clone execution "
                "would fail. Remote push requires separate user approval."
            ),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
