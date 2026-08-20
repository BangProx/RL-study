"""Adapt repository-local notebook links for the hosted documentation site."""

from __future__ import annotations

import re
from typing import Any

NOTEBOOK_LINK = re.compile(
    r"\]\((?:\.\./)+(notebooks/(?:ko|en)/[^)]+\.ipynb)\)"
)
GITHUB_BLOB_ROOT = "https://github.com/BangProx/RL-study/blob/main"


def on_page_markdown(
    markdown: str,
    *,
    page: Any,
    config: Any,
    files: Any,
) -> str:
    """Point course notebook links to GitHub only in the rendered website."""
    del page, config, files
    return NOTEBOOK_LINK.sub(
        lambda match: f"]({GITHUB_BLOB_ROOT}/{match.group(1)})",
        markdown,
    )
