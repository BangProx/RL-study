#!/usr/bin/env python3
"""Check repository-local Markdown targets and optionally probe external URLs."""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^)]*['\"])?\)")
ANGLE_URL = re.compile(r"<(https?://[^>]+)>")
HTTP_SCHEMES = ("http://", "https://")
RESTRICTED_STATUSES = {401, 403, 429}


@dataclass(frozen=True, slots=True)
class LinkOccurrence:
    source: Path
    target: str
    line: int


@dataclass(frozen=True, slots=True)
class NetworkResult:
    url: str
    status: int | None
    detail: str
    ok: bool
    restricted: bool = False


def _occurrences(root: Path) -> list[LinkOccurrence]:
    records: list[LinkOccurrence] = []
    for path in sorted(root.rglob("*.md")):
        if any(part in {".git", ".venv", "site", "artifacts"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            targets = [*MARKDOWN_LINK.findall(line), *ANGLE_URL.findall(line)]
            records.extend(
                LinkOccurrence(path.relative_to(root), target, line_number)
                for target in targets
            )
    return records


def _external_urls_from_manifest(root: Path) -> set[str]:
    manifest = root / "docs" / "sources.yml"
    payload: Any = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    urls: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str) and value.startswith(HTTP_SCHEMES):
            urls.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return urls


def check_local(root: Path) -> tuple[list[LinkOccurrence], int]:
    occurrences = _occurrences(root)
    failures: list[LinkOccurrence] = []
    checked = 0
    for record in occurrences:
        target = record.target
        if target.startswith(HTTP_SCHEMES) or target.startswith(("mailto:", "data:")):
            continue
        path_text = urllib.parse.unquote(target.split("#", maxsplit=1)[0])
        if not path_text:
            continue
        checked += 1
        if path_text.startswith("/"):
            resolved = root / path_text.lstrip("/")
        else:
            resolved = root / record.source.parent / path_text
        if not resolved.resolve().exists():
            failures.append(record)
    return failures, checked


def _probe(url: str, *, timeout: float) -> NetworkResult:
    headers = {
        "User-Agent": "RL-study-link-audit/1.0 (+https://github.com/BangProx/RL-study)",
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.1",
    }
    for method in ("HEAD", "GET"):
        request_headers = dict(headers)
        if method == "GET":
            request_headers["Range"] = "bytes=0-0"
        request = urllib.request.Request(url, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                return NetworkResult(url, status, "reachable", 200 <= status < 400)
        except urllib.error.HTTPError as error:
            if error.code in RESTRICTED_STATUSES:
                return NetworkResult(
                    url,
                    error.code,
                    "server reachable but automated access restricted",
                    True,
                    restricted=True,
                )
            if method == "HEAD" and error.code in {400, 405, 501}:
                continue
            return NetworkResult(url, error.code, f"HTTP {error.code}", False)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if method == "HEAD":
                continue
            return NetworkResult(url, None, str(error), False)
    return NetworkResult(url, None, "no probe completed", False)


def check_network(
    root: Path, *, timeout: float, workers: int
) -> list[NetworkResult]:
    urls = {
        record.target
        for record in _occurrences(root)
        if record.target.startswith(HTTP_SCHEMES)
    }
    urls.update(_external_urls_from_manifest(root))
    results: list[NetworkResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_probe, url, timeout=timeout): url for url in sorted(urls)
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: result.url)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--network", action="store_true")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    local_failures, local_count = check_local(root)
    if local_failures:
        for failure in local_failures:
            print(
                f"BROKEN local {failure.source}:{failure.line}: {failure.target}",
                file=sys.stderr,
            )
        return 1
    print(f"Local links: PASS ({local_count} file targets)")
    if not args.network:
        return 0
    if args.timeout <= 0 or args.workers < 1:
        print("timeout and workers must be positive", file=sys.stderr)
        return 2
    results = check_network(root, timeout=args.timeout, workers=args.workers)
    failures = [result for result in results if not result.ok]
    restricted = [result for result in results if result.restricted]
    for result in restricted:
        print(f"RESTRICTED {result.status} {result.url}")
    for result in failures:
        print(
            f"BROKEN external {result.status or '-'} {result.url}: {result.detail}",
            file=sys.stderr,
        )
    if failures:
        return 1
    print(
        f"External links: PASS ({len(results)} URLs; "
        f"{len(restricted)} access-restricted)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
